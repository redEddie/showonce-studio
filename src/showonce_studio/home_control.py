"""Studio-owned posture-A execution. Importing never connects to hardware."""
import fcntl
import json
import os
from pathlib import Path
import signal
import time

from .config import ROOT,default_calibration
from .showonce_stages import read_json,write_json


def run_home(args,robot_factory=None):
    from .showonce_policy_runtime import CancellableTransport,ExecutionCancelled,serial
    from .motion_config import motion_adapter
    from . import posture_control
    from piper_policy_runtime.geometry_guard import GeometryGuard
    from piper_policy_runtime.control100 import ControlFault
    import numpy as np

    folder=args.out or ROOT/'runs/showonce-home'/str(time.time_ns())
    motion=folder/'motion';motion.mkdir(parents=True,exist_ok=False)
    calibration=args.calib or default_calibration()
    value=read_json(calibration,{})
    table=value.get('table_check',{}).get('height_m',value.get('table_height_prior_m'))
    if not value.get('trusted') or table is None or not np.isfinite(table):
        raise ValueError('시작 자세 A 복귀에는 신뢰된 테이블 보정 파일이 필요합니다.')
    cancelled=False
    def stop(*_):
        nonlocal cancelled
        cancelled=True
    def check():
        if cancelled or (folder/'halt.now').exists():raise ExecutionCancelled('시작 자세 A 중단: 새 목표 전송을 중지합니다.')
    def progress(phase):write_json(folder/'progress.json',{'phase':phase,'updated_s':time.time()})
    report=dict(kind='sdk',action='home',success=False,hardware_commands_sent=False,
        implementation='studio_native_posture_a_v1',policy_weights_loaded=False,
        calibration=str(calibration),gripper_force_sdk=args.force,
        interruption_semantics='new_targets_stopped_last_sdk_target_unchanged')
    robot=lock=None
    handlers={code:signal.signal(code,stop)for code in (signal.SIGINT,signal.SIGTERM)}
    affinity=os.sched_getaffinity(0) if hasattr(os,'sched_getaffinity') else None
    event_log=(motion/'events.jsonl').open('w')
    def emit(event,**data):
        check()
        event_log.write(json.dumps(serial(dict(event=event,time_s=time.time(),**data)),allow_nan=False)+'\n')
        event_log.flush()
    class HomeTransport(CancellableTransport):
        def send_joints(self,target):check();return self.target.send_joints(target)
        def send_gripper(self,width,force):check();return self.target.send_gripper(width,force)
        def speed(self,percent):check();return self.target.speed(percent)
        def acceleration_limits(self):check();return self.target.acceleration_limits()
    try:
        progress('preflight');check()
        adapter=motion_adapter(getattr(args,'policy_id',None))
        write_json(motion/'calibration.json',value)
        report.update(posture_target_rad=list(adapter.environment['nominal_qpos']),
            tcp_local_position_m=adapter.model.tcp_local_position.tolist(),
            tcp_floor_clearance_m=.01,arm_link_floor_clearance_m=.01)
        from piper_runtime.resources import GEOMETRY
        guard=GeometryGuard(GEOMETRY,table_z=float(table),link_floor=.01)
        if affinity:
            control=sorted(affinity)[-2:];os.sched_setaffinity(0,control);report['control_cpus']=control
        # Compile, warm and benchmark before opening SDK or acquiring any motion target.
        controller=posture_control.make_controller(adapter,guard,'show103',posture=True,jerk=100.)
        timings=posture_control.benchmark(controller,np.asarray(adapter.environment['nominal_qpos']),guard,.1)
        report['cpu_controller_and_geometry_benchmark']=timings
        if timings['p99_ms']>5.:raise ControlFault('CPU QP/geometry p99 exceeds 5ms; inspect report before motion')
        check()
        lock=Path('/tmp',f'show100-{args.channel.replace("/","_")}.lock').open('a')
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ControlFault('Another motion runner owns this CAN channel')
        if robot_factory is None:
            from piper_runtime.transports.robot_io import SDKRobot
            robot_factory=SDKRobot
        robot=HomeTransport(robot_factory(args.channel),check)
        check();progress('homing')
        report.update(posture_control.home(robot,adapter,guard,emit,timeout=args.seconds,
            jerk=100.,command_hz=200,force=args.force))
        report['success']=True
    except ExecutionCancelled as error:
        report.update(stopped_by_request=True,error=str(error))
    except Exception as error:
        report['error']=f'{type(error).__name__}: {error}'.replace('show78_sdk_demo.py enable','showonce-robot enable')
    finally:
        if robot is not None:
            report.update(joint_writes=robot.sent_joints,gripper_writes=robot.sent_gripper,
                hardware_commands_sent=bool(robot.sent_joints or robot.sent_gripper),
                gripper_zero_offset_clamps=robot.gripper_zero_clamps,
                gripper_last_raw_width_m=robot.last_raw_width)
            try:robot.close()
            except Exception as error:report.update(success=False,disconnect_error=str(error))
        if lock is not None:lock.close()
        if affinity is not None:os.sched_setaffinity(0,affinity)
        for code,handler in handlers.items():signal.signal(code,handler)
        event_log.close();progress('finished')
        report.update(completed_s=time.time(),stopped_by_request=cancelled or (folder/'halt.now').exists() or report.get('stopped_by_request',False))
        write_json(motion/'report.json',report)
    return report
