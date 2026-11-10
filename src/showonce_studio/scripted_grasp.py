"""Standalone scripted grasp tutorial using an independently running perception node.

No browser, Studio web server, actor weights or external staged runner required.
Plan uses only a mock plant; run asks before each selected stage by default.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
from .config import ROOT
from .live_perception import LivePerception
from .showonce_stages import read_json,write_json
from piper_policy_runtime.control100 import ControlFault


def studio_snapshot(folder,calibration,now=None):
    """Adapt the live file protocol to the previous runner's vision protocol."""
    now=time.time()if now is None else now
    manifest=read_json(folder/'manifest.json',{})
    source=manifest.get('calibration',{})
    path=Path(source.get('path','')).expanduser()
    if (manifest.get('mode')!='live_camera_only' or not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest()!=source.get('sha256')
            or read_json(path)!=read_json(calibration)):
        raise ControlFault('Studio 인식 노드와 스크립트의 보정 파일이 일치하지 않습니다.')
    data=read_json(folder/'tracked_poses.json',{})
    if data.get('offline_replay')is not False or data.get('frame')!='piper_base_link' or data.get('units')!='m':
        raise ControlFault('실시간 로봇 베이스 좌표의 인식 데이터가 필요합니다.')
    age=now-float(data.get('sample_time_s',0.))
    if not np.isfinite(age) or not -.01<=age<=1.:
        raise ControlFault('Studio 인식 노드의 응답이 만료되었습니다.')
    fresh=[]
    for item in data.get('objects',[]):
        age=now-float(item.get('source_time_s',0.))
        if np.isfinite(age)and -.01<=age<=.25:fresh.append(item['id'])
    diagnostics=read_json(folder/'diagnostics.json',{})
    return dict(service_alive=True,phase='tracking_active',active_calibration=str(path),
        output_dir=str(folder),objects=data.get('objects',[]),fresh_ids=fresh,
        status=diagnostics.get('object_status',{}))


def load_script():
    # A fresh instance prevents per-run hooks from accumulating across invocations.
    path=Path(__file__).with_name('tutorial_scripted_control.py')
    spec=importlib.util.spec_from_file_location('showonce_studio._scripted_tutorial_run',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    from .showonce_policy_runtime import CancellableTransport
    sdk=module.SDKRobot
    module.SDKRobot=lambda channel:CancellableTransport(sdk(channel),lambda:None)
    return module


def configure(module,folder,calibration,model_folder=None,through='lift'):
    from piper_runtime.resources import GEOMETRY
    module.GEOMETRY=GEOMETRY
    module.vision_state=lambda _:studio_snapshot(folder,calibration)
    def orientation(adapter,guard,targets,cube_yaw):
        # A pick-only test must not require an unselected route over the bowl.
        from scipy.spatial.transform import Rotation
        stages=('approach','descend','close','lift','transport','release','retreat')
        route=[s for s in stages[:stages.index(through)+1]if s in targets]
        model=adapter.model;nominal=np.asarray(adapter.environment['nominal_qpos'])
        quat=model.forward(nominal)['link6_quaternion_wxyz']
        nominal_rotation=Rotation.from_quat(quat[[1,2,3,0]])
        candidates=[]
        for pitch in (90,85,80,75,70,65):
            for turn in range(-2,3):
                yaw=float(cube_yaw+turn*np.pi/2)
                desired=Rotation.from_euler('z',yaw)*Rotation.from_euler('y',pitch,degrees=True)*nominal_rotation
                seed=nominal.copy();goals={};margins=[]
                try:
                    for name in route:
                        width=0. if name in ('lift','transport')else .1
                        seed=module.solve_joint_goal(model,guard,targets[name],seed,desired,width)
                        goals[name]=seed.copy()
                        margins.append(float(np.rad2deg(np.min(np.minimum(seed-model.lower,model.upper-seed)))))
                except ControlFault:continue
                margin=min(margins)
                if margin>=8.:
                    candidates.append((pitch,margin,-float(np.linalg.norm(goals['approach']-nominal)),yaw,goals))
        if not candidates:raise ControlFault('선택한 파지 경로에 관절 여유8°를 확보하지 못했습니다. 큐브 위치를 조정하세요.')
        pitch,margin,_,yaw,goals=max(candidates,key=lambda v:v[:3])
        return yaw,pitch,margin,goals
    module.choose_grasp_orientation=orientation
    original_stage=module.execute_stage

    def stage(name,target,robot,adapter,guard,log,**kwargs):
        if name!='close':return original_stage(name,target,robot,adapter,guard,log,**kwargs)
        before=module.read_robot(robot,adapter.model,np.asarray(adapter.environment['nominal_qpos']),
            real=not kwargs['mock'])
        origin=np.asarray(adapter.model.forward(before['q_rad'])['tcp_position'])
        samples=[]
        def capture(event):
            log(event)
            if event.get('event')=='tick'and event.get('stage')=='close':
                delta=float(np.linalg.norm(np.asarray(event['tcp_m'])-origin));samples.append(delta)
                if delta>.005:raise ControlFault(f'close: TCP drift {delta*1000:.1f}mm > 5mm; lift cancelled')
        result=original_stage(name,target,robot,adapter,guard,capture,**kwargs)
        final=float(np.linalg.norm(np.asarray(result['tcp_m'])-origin));maximum=max([final,*samples])
        if maximum>.005:raise ControlFault(f'close: TCP drift {maximum*1000:.1f}mm > 5mm; lift cancelled')
        result['close_tcp_max_sampled_drift_m']=maximum
        log(dict(event='close_static_check',tcp_before_m=origin,tcp_after_m=result['tcp_m'],
            max_sampled_drift_m=maximum,sample_period_s=.1,gripper_width_m=result['gripper_width_m']))
        print(f"닫기 중 TCP 이동: 최대 {maximum*1000:.1f}mm · 실측 폭 {result['gripper_width_m']*1000:.1f}mm",flush=True)
        return result
    module.execute_stage=stage
    return module


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('plan','run'))
    p.add_argument('--through',choices=('approach','descend','close','lift','transport','release','retreat'),default='lift')
    p.add_argument('--policy-id',help='Optional registered robot configuration; no actor weights are read')
    p.add_argument('--auto',action='store_true',help='Explicitly skip terminal stage prompts')
    p.add_argument('--perception-dir',type=Path)
    p.add_argument('--calib',type=Path)
    p.add_argument('--channel',default='can_piper')
    p.add_argument('--gripper-force',type=float,default=.5)
    p.add_argument('--out',type=Path)
    args=p.parse_args()
    if not .5<=args.gripper_force<=3:p.error('gripper force: 0.5..3')
    if args.perception_dir:
        folder=args.perception_dir.expanduser().resolve()
        calibration=args.calib or Path(read_json(folder/'manifest.json',{}).get('calibration',{}).get('path',''))
    else:
        monitor=LivePerception(ROOT)
        current=monitor.state()
        if not current.get('connected'):p.error('실시간 인식 노드를 실행하고 --perception-dir를 지정하세요.')
        folder=monitor.folder(current['folder']);calibration=args.calib or Path(current['calibration'])
    studio_snapshot(folder,calibration)
    out=args.out or ROOT/'runs/showonce-scripted'/str(time.time_ns())
    module=configure(load_script(),folder,calibration,through=args.through)
    from .motion_config import motion_adapter
    module.REFERENCE_ADAPTER=motion_adapter(args.policy_id)
    argv=sys.argv
    print('독립 인식 노드:',folder,flush=True)
    print('현재 보정:',calibration,flush=True)
    print('비교 실험: 접근 → 하강·정지 → 닫기·정지 → 들어올리기 (선택 단계까지)',flush=True)
    try:
        sys.argv=['showonce-scripted',args.command,'--through',args.through,'--calib',str(calibration),
            '--channel',args.channel,'--gripper-force',str(args.gripper_force),'--out',str(out)]
        if args.auto:sys.argv.append('--auto')
        module.main()
    finally:
        sys.argv=argv
        report=read_json(out/'summary.json')
        if report:
            report.update(studio_perception_directory=str(folder),policy_id_for_model_config=args.policy_id,
                bridge_revision='standalone_scripted_tutorial_v2',rl_policy_changed=False)
            write_json(out/'summary.json',report)
    if not report or not report.get('success'):raise SystemExit(1)


if __name__=='__main__':main()
