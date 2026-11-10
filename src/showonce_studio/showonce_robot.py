"""Explicit short SDK commands. Import and --help perform no hardware I/O."""
import argparse
import fcntl
import json
from pathlib import Path
import re
import sys
import time
from .robot_sdk_support import (MODES,SELECTABLE,fresh,positive,request_mode,set_motor_power,snapshot,wait_until)

ACTIONS={'read','enable','disable','stop','reset','mode'}

def write_report(folder,value):
    if folder:
        folder.mkdir(parents=True,exist_ok=True);temp=folder/'report.tmp';temp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2));temp.replace(folder/'report.json')

def acquire_channel(channel):
    lock=Path('/tmp',f'show100-{channel}.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:lock.close();raise RuntimeError('같은 CAN 채널에서 동작 중입니다. 정책 중단 후 실행하세요.')
    return lock

def run_command(action,channel='can_piper',mode=None,timeout=3.,force=.5,robot_factory=None,width_mm=None):
    if action not in ACTIONS|{'gripper'} or not re.fullmatch(r'[A-Za-z0-9_-]{1,40}',channel):raise ValueError('지원하지 않는 SDK 명령 또는 CAN 채널입니다.')
    if action=='gripper':
        from .robot_actions import manual_parameters
        manual_parameters({'width_mm':width_mm,'force':force},action)
    if action=='mode'and mode not in SELECTABLE:raise ValueError('SDK 선택 가능 모드는 0, 1, 3, 4, 7입니다. 티칭 모드는 로봇 버튼으로 전환합니다.')
    if not 0<float(timeout)<=5 or not .5<=float(force)<=3:raise ValueError('잘못된 timeout/force입니다.')
    report={'kind':'sdk','action':action,'channel':channel,'started_s':time.time(),'success':False,'request_sent':False,'hardware_commands_sent':False,'calls':[]}
    robot=gripper=lock=None
    def invoke(obj,method,**kwargs):
        report['calls'].append({'method':method,'arguments':kwargs});report['hardware_commands_sent']=True;report['request_sent']=True
        print(json.dumps({'event':'sdk_call','method':method,'arguments':kwargs},ensure_ascii=False),flush=True)
        result=getattr(obj,method)(**kwargs)
        report['calls'][-1]['sdk_return']=result if result is None or isinstance(result,(bool,int,float,str))else repr(result)
        # Power return values describe the immediate feedback, not final acknowledgement.
        # set_motor_power verifies fresh arm/gripper feedback after both requests.
        return result
    try:
        if action in ('enable','reset','mode','gripper'):lock=acquire_channel(channel)
        if robot_factory is None:
            from pyAgxArm import AgxArmFactory,ArmModel,PiperFW,create_agx_arm_config
            config=create_agx_arm_config(robot=ArmModel.PIPER,firmeware_version=PiperFW.V189,interface='socketcan',channel=channel,bitrate=1000000,auto_connect=False,log_level='ERROR')
            robot=AgxArmFactory.create_arm(config)
        else:robot=robot_factory()
        gripper=robot.init_effector('agx_gripper');robot.connect()
        if action=='stop':invoke(robot,'electronic_emergency_stop')
        elif action in ('enable','disable'):
            power_result=set_motor_power(robot,gripper,action,invoke,timeout,force=force)
            if action=='enable':report['gripper_enable']=power_result
            if action=='disable':
                # Reassert after another channel owner has exited, as in the existing CLI.
                deadline=time.monotonic()+2.
                while lock is None and time.monotonic()<deadline:
                    try:lock=acquire_channel(channel)
                    except RuntimeError:time.sleep(.02)
                if lock is None:raise RuntimeError('disable 요청 후에도 다른 동작 프로세스가 채널을 사용합니다. 최종 전원 상태는 미확인입니다.')
                set_motor_power(robot,gripper,action,invoke,timeout,force=force)
        elif action=='reset':invoke(robot,'reset')
        elif action=='gripper':
            from .robot_sdk_support import require_gripper_can_control,gripper_enable_target
            wait_until(lambda:fresh(robot.get_arm_status())and fresh(gripper.get_gripper_status()),timeout,'최신 팔/그리퍼 피드백이 필요합니다.')
            require_gripper_can_control(robot.get_arm_status());gripper_enable_target(gripper)
            status=robot.get_arm_status()
            if int(status.msg.arm_status)!=0 or int(status.msg.err_code)!=0:raise RuntimeError('팔 오류/정지 상태입니다.')
            if not int(gripper.get_gripper_status().msg.status_code)&64:raise RuntimeError('그리퍼 활성화 후 실행하세요.')
            invoke(gripper,'move_gripper_m',value=width_mm/1000,force=force)
        elif action=='mode':
            def requested():report.update(hardware_commands_sent=True,request_sent=True)
            request_mode(robot,mode,timeout,on_request=requested)
        if action in ('read','reset'):wait_until(lambda:fresh(robot.get_arm_status()),timeout,'최신 팔 상태를 수신하지 못했습니다.')
        time.sleep(.15 if action in ('enable','disable','mode')else .35)
        report['state']=snapshot(robot,gripper);report['feedback_fresh']=fresh(robot.get_arm_status())
        report.update(success=True,confirmation='motor_feedback'if action in ('enable','disable')else 'mode_feedback'if action=='mode'else 'request_sent_and_observed'if action!='read'else 'read_only')
        if action=='mode':report.update(selected_mode=mode,selected_mode_name=MODES[mode])
    except Exception as error:
        report['error']=f'{type(error).__name__}: {error}'
        if robot is not None and gripper is not None:
            try:report['state']=snapshot(robot,gripper)
            except Exception:pass
    finally:
        if robot is not None:
            try:robot.disconnect()
            except Exception as error:report['disconnect_error']=str(error)
        if lock:lock.close()
    report['completed_s']=time.time();return report

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=sorted(ACTIONS|{'home','manual','jog','rotate','gripper','unwind'}));p.add_argument('--channel',default='can_piper');p.add_argument('--timeout',type=positive,default=3.);p.add_argument('--force',type=float,default=.5);p.add_argument('--value',type=int);p.add_argument('--out',type=Path);p.add_argument('--job-token');p.add_argument('--policy-id')
    p.add_argument('--axis',choices=['x','y','z','roll','pitch','yaw']);p.add_argument('--direction',type=int,choices=[-1,1]);p.add_argument('--speed-mm-s',type=float,default=10.);p.add_argument('--seconds',type=float);p.add_argument('--calib',type=Path);p.add_argument('--leased',action='store_true');p.add_argument('--width-mm',type=float);p.add_argument('--plan-only',action='store_true')
    args=p.parse_args(argv)
    if args.plan_only and args.command!='unwind':p.error('--plan-only는 unwind에서만 사용할 수 있습니다.')
    args.seconds=args.seconds if args.seconds is not None else {'home':60.,'manual':600.,'rotate':30.,'unwind':60.}.get(args.command,2.)
    if args.command=='home':
        import math
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,40}',args.channel)or not math.isfinite(args.seconds)or not 0<args.seconds<=60 or not .5<=args.force<=3:p.error('잘못된 자세 A 복귀 채널/시간/force입니다.')
        from .home_control import run_home
        from .config import ROOT
        args.out=args.out or ROOT/'runs/showonce-home'/str(time.time_ns())
        try:report=run_home(args)
        except Exception as error:report=dict(success=False,error=f'{type(error).__name__}: {error}',hardware_commands_sent=False,completed_s=time.time())
        write_report(args.out,report);print(json.dumps(report,ensure_ascii=False,allow_nan=False),flush=True)
        return 0 if report.get('success')or report.get('stopped_by_request')else 1
    if args.command in ('manual','jog','rotate','unwind'):
        import math
        from .robot_actions import manual_parameters
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,40}',args.channel):p.error('잘못된 CAN 채널입니다.')
        if not math.isfinite(args.seconds)or not 0<args.seconds<=600:p.error('수동 제어 시간은 0초 초과, 최대 600초입니다.')
        if args.command=='manual'and(not args.leased or args.out is None):p.error('manual은 웹앱 연결용입니다. 터미널에서는 jog/rotate를 사용하세요.')
        if args.command in ('jog','rotate'):manual_parameters(vars(args),args.command)
        from .config import ROOT
        from .manual_control import run_manual
        args.out=args.out or ROOT/'runs/showonce-manual'/str(time.time_ns())
        try:report=run_manual(args)
        except Exception as error:report=dict(success=False,error=f'{type(error).__name__}: {error}',hardware_commands_sent=False,completed_s=time.time())
        write_report(args.out,report);print(json.dumps(report,ensure_ascii=False,allow_nan=False),flush=True)
        return 0 if report['success']else 1
    if args.command=='mode'and args.value is None:
        for number,label in MODES.items():print(f'{number}: {label}'+(' (로봇 버튼/상태 표시)'if number not in SELECTABLE else ''))
        if not sys.stdin.isatty():p.error('비대화 실행의 mode에는 --value 번호가 필요합니다.')
        raw=input('모드 번호 (엔터: 취소): ').strip()
        if not raw:return 0
        args.value=int(raw)
    report=run_command(args.command,args.channel,args.value,args.timeout,args.force,width_mm=args.width_mm);write_report(args.out,report)
    print(json.dumps(report,ensure_ascii=False,allow_nan=False),flush=True)
    return 0 if report['success']else 1
if __name__=='__main__':raise SystemExit(main())
