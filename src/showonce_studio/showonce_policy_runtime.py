"""Policy validation/replay/mock and explicit managed or terminal shadow/run.

SDK construction is confined to the explicit shadow/run branch. Recorded
video is never a live robot input. Page loads never start this worker.
"""
from .config import ROOT
from pathlib import Path
import argparse
import bisect
import fcntl
import json
import os
import signal
import sys
import time

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from showonce_studio.showonce_policies import PolicyLibrary,PROFILE,GEOMETRY,contract,layout,sha
from showonce_studio.showonce_stages import read_json,write_json


from piper_policy_runtime.policy import Policy
from piper_policy_runtime.kinematics import PiperKinematics
from piper_policy_runtime.show100 import assemble_observation
from piper_policy_runtime.control103 import Controller103
from piper_policy_runtime.control108 import Controller108
from piper_policy_runtime.geometry_guard import GeometryGuard


def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:serial(v)for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serial(v)for v in value]
    return value


class Adapter:
    def __init__(self,folder):
        self.folder=Path(folder);self.policy=Policy(folder);self.meta=self.policy.metadata;self.contract=contract(self.meta)
        if self.meta.get('runtime_identity'):
            from piper_runtime.identity import identity
            actual=identity();expected=self.meta['runtime_identity']
            if actual['version']!=expected['version']or actual['sha256']!=expected['sha256']:raise ValueError('Policy was exported with a different piper-runtime implementation')
        size=self.meta['observation_size']
        if any(self.policy.weights[key].shape!=(size,)for key in ('normalizer/mean','normalizer/std')):raise ValueError('Normalizer dimensions disagree with observation contract')
        if sha(self.folder/'piper.xml')!=self.meta['kinematics']['piper_xml_sha256']:raise ValueError('Robot XML mismatch')
        self.profile=self.contract['runtime_profile']
        self.model=PiperKinematics(self.folder/'piper.xml');self.model.tcp_local_position=np.array(self.profile['tcp_local_position_m'])
        if self.model.robot_model!=self.meta['environment']['robot_model']:raise ValueError('Robot model mismatch')
        self.environment=dict(self.meta['environment'])

    def observation(self,sensors,reference,goal,previous,phase=0):
        obs=assemble_observation(sensors,self.model,self.environment,joint_reference_rad=reference,
            gripper_final_goal_width_m=goal,previous_raw_action=previous)
        if self.meta['observation_size']==50:
            if phase not in (0,1,2):raise ValueError('Unknown controller phase')
            obs=np.r_[obs[:42],np.float32(phase/2.),obs[42:]].astype(np.float32)
        return obs

    def step(self,sensors,controller,reset=False,reference=None):
        obs=self.observation(sensors,controller.reference if reference is None else reference,controller.gripper_goal,controller.previous_action,controller.phase)
        return obs,self.policy.step(obs,reset=reset)

    def controller(self,table_z,floor_normal=None):
        if self.profile['controller']=='show108_checked_plan_independent':
            return Controller108(self.model,self.environment,floor_offset=table_z+self.profile['tcp_floor_clearance_m'],
                **({'floor_normal':floor_normal}if floor_normal is not None else {}))
        # The legacy controller selects resolved settings through this tag.
        # Only its private configuration copy changes; actor metadata/weights,
        # normalization and action scales remain exactly as exported.
        env=dict(self.environment,task='scripted_control',**{k:PROFILE[k]for k in ('max_joint_vel','control_acceleration','control_jerk')})
        result=Controller103(self.model,env,floor_offset=table_z+PROFILE['tcp_floor_clearance_m'],**({'floor_normal':floor_normal}if floor_normal is not None else {}))
        if not np.allclose([result.speed,result.acceleration,result.jerk],[1.,5.,100.]):raise ValueError('Resolved controller limits differ')
        return result


def validate(folder):
    from piper_policy_runtime.check import check
    adapter=Adapter(folder)
    with np.load(Path(folder)/'golden.npz',allow_pickle=False)as data:
        obs,resets,actions=(data[name]for name in ('observations','resets','actions'))
        if obs.ndim!=3 or min(obs.shape[:2])<=0 or obs.shape[2]!=adapter.meta['observation_size']or resets.shape!=obs.shape[:2]or actions.shape!=(*obs.shape[:2],7):raise ValueError('Invalid golden-vector shapes')
        if not all(np.isfinite(v).all()for v in (obs,resets,actions)):raise ValueError('Nonfinite golden vectors')
    parity=check(folder)
    recorded=None
    if (Path(folder)/'recorded-golden.npz').is_file():
        # Reuse the locked Runtime checker, including logits and both LSTM
        # states, without changing its identity or any registered file.
        import tempfile
        with tempfile.TemporaryDirectory(prefix='showonce-recorded-')as directory:
            replay=Path(directory)
            for name in ('policy.json','weights.npz'):
                (replay/name).symlink_to((Path(folder)/name).resolve())
            (replay/'golden.npz').symlink_to((Path(folder)/'recorded-golden.npz').resolve())
            recorded=check(replay)
        recorded['scope']='Recorded simulation observations; offline arithmetic and LSTM replay, not a new rollout or real-robot success evaluation'
    sources={name:sha(Path(__import__('piper_policy_runtime').__file__).parent/name)for name in ('control108.py','control103.py','control100.py','plan103.cpp','show100.py','kinematics.py','scene_state.py','geometry_guard.py')}
    return {'status':'passed','contract':adapter.contract,'parity':parity,'recorded_replay':recorded,'runtime_source_sha256':sources,'hardware_commands_sent':False,
        'scope':'Policy arithmetic and declared observation/action contracts, not task success or hardware validation.'}


def recorded_inputs(folder,recording):
    from scipy.spatial.transform import Rotation
    import trimesh
    adapter=Adapter(folder);recording=Path(recording);tracks=read_json(recording/'geometry/foundationpose_v1/tracking.json')
    assets=read_json(recording/'geometry/sam3d_assets_v1/manifest.json')
    if not tracks or not tracks.get('complete')or not assets:raise ValueError('FoundationPose 결과가 있는 영상을 선택하세요.')
    by_id={a['id']:a for a in assets['assets']};models={oid:trimesh.load(recording/'geometry/sam3d_assets_v1'/by_id[oid]['mesh_ply'],force='mesh',process=False)for oid in ('green_block','bowl')}
    cube_center=models['green_block'].bounds.mean(0);half=float(np.mean(models['green_block'].extents)/2)
    bowl_origin=models['bowl'].bounds.mean(0);bowl_origin[2]=models['bowl'].bounds[0,2]
    q=np.array(adapter.environment['nominal_qpos']);previous=np.zeros(7,np.float32);rows=[];skipped=0
    for frame in tracks['frames']:
        block=frame['objects']['green_block'].get('pose_base');bowl=frame['objects']['bowl'].get('pose_base')
        if block is None or bowl is None:skipped+=1;continue
        block=np.asarray(block);bowl=np.asarray(bowl);xyzw=Rotation.from_matrix(block[:3,:3]).as_quat()
        sensors={'frame':'piper_base_link','units':'SI','q_rad':q,'qvel_rad_s':np.zeros(6),'gripper_full_width_m':.1,
            'block_pos_m':block[:3,:3]@cube_center+block[:3,3],'block_quat_wxyz':np.r_[xyzw[3],xyzw[:3]],
            'bowl_pos_m':bowl[:3,:3]@bowl_origin+bowl[:3,3],'block_half_size_m':half}
        obs=adapter.observation(sensors,q,.1,previous,phase=0);action=adapter.policy.step(obs,reset=not rows)
        rows.append({'frame_index':frame['index'],'time_s':frame['time_s'],'observation':obs.tolist(),'raw_action':action.tolist()});previous=action
    if not rows:raise ValueError('No valid recorded object poses')
    return {'status':'passed','frames':len(rows),'unresolved_frames_skipped':skipped,'samples':rows,'hardware_commands_sent':False,
        'synthetic_robot_state':True,'controller_phase':'synthetic MOVE for input-format inspection only',
        'origins':{'block':'reference mesh bounding-box centre','bowl':'reference mesh outer-bottom centre'},
        'cube_half_size_m_from_asset':half,'cube_half_size_m_in_training':adapter.environment['block_half'],
        'asset_size_matches_training':bool(np.isclose(half,adapter.environment['block_half'],atol=.001)),
        'scope':'Recorded object poses plus synthetic stationary arm feedback; not a robot rollout, closed-loop simulation or task success.'}


class ExecutionCancelled(Exception):pass

class CancellableTransport:
    def __init__(self,target,check):
        self.target=target;self.check=check;self.gripper_zero_clamps=0;self.last_raw_width=None
    def __getattr__(self,name):
        method=getattr(self.target,name)
        if name not in ('read','observations'):return method
        def checked(*args,**kwargs):
            self.check();value=method(*args,**kwargs)
            if name=='read'and isinstance(value,dict)and 'gripper_full_width_m'in value:
                # Same closed-stop tolerance as SDK Enable, not a travel-wide
                # encoder offset. Keep the signed source reading for diagnosis.
                from .robot_sdk_support import GRIPPER_ZERO_TOLERANCE_M
                raw=float(value['gripper_full_width_m']);self.last_raw_width=raw if np.isfinite(raw)else None
                clamped=-GRIPPER_ZERO_TOLERANCE_M<=raw<0.
                value=dict(value,gripper_raw_full_width_m=raw,
                    gripper_full_width_m=0. if clamped else raw,gripper_zero_offset_clamped=clamped)
                self.gripper_zero_clamps+=int(clamped)
            return value
        return checked

def execute(folder,mode,seconds,out,calibration=None,perception_dir=None,channel='can_piper'):
    from piper_runtime.transports.robot_io import MockRobot,MockVision
    from piper_runtime.preflight import validate_state,validate_objects,wait_feedback,wait_vision
    from piper_policy_runtime.scene_state import SceneState
    from piper_policy_runtime.control100 import ReferenceHistory
    adapter=Adapter(folder);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    episode_seconds=float(adapter.environment.get('episode_length',200))*adapter.profile['policy_dt_s']
    if (mode in ('shadow','run')and adapter.profile['controller']=='show108_checked_plan_independent'
            and seconds>episode_seconds):
        raise ValueError(f'이 정책의 한 에피소드는 {episode_seconds:g}초입니다. 실행 시간을 그 이하로 설정하세요.')
    calibration=read_json(Path(calibration))if calibration else {'trusted':True,'table_check':{'height_m':-.015}}
    table=float(calibration['table_check']['height_m']);controller=adapter.controller(table)
    write_json(out/'calibration.json',calibration)
    guard=GeometryGuard(GEOMETRY,table_z=table,link_floor=.01)
    robot=vision=motion_lock=None;estimator=None;stopping=False;episode_started=False
    def stop(*_):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def check_cancel():
        if (out/'halt.now').exists()or(stopping and not episode_started):raise ExecutionCancelled('Execution cancelled before new targets')
    def progress(phase):write_json(out/'progress.json',{'phase':phase,'updated_s':time.time()})
    progress('connecting')
    report={'mode':mode,'profile':adapter.profile,'policy_contract':adapter.contract,'hardware_commands_sent':False,'success':False,
        'task_success_evaluated':False,'mock_is_physics_simulation':False,'gripper_force_sdk':.5}
    log=(out/'events.jsonl').open('w')
    trajectory=(out/'policy-inputs.jsonl').open('w')
    def emit(event,**values):log.write(json.dumps(serial({'event':event,'wall_time_s':time.time(),**values}),ensure_ascii=False,allow_nan=False)+'\n')
    try:
        allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,allowed[-2:]);report['control_cpus']=allowed[-2:]
        if mode=='mock':robot=MockRobot(adapter.environment['nominal_qpos']);vision=MockVision()
        else:
            if not perception_dir:raise ValueError('Live perception directory required; recording data is not accepted')
            if not calibration.get('trusted'):raise ValueError('Trusted calibration required')
            if mode=='run':
                motion_lock=Path('/tmp',f'show100-{channel.replace("/","_")}.lock').open('a');fcntl.flock(motion_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            # This branch is never entered by a web validation/mock request.
            from piper_runtime.transports.robot_io import SDKRobot,PerceptionProcess
            robot=SDKRobot(channel)
            vision=PerceptionProcess(out/'perception',out/'calibration.json',allowed[:-2],attached_directory=perception_dir)
            estimator_type=SceneState
            if adapter.profile['controller']=='show108_checked_plan_independent':
                from .scene_estimation import DeploymentSceneState
                estimator_type=DeploymentSceneState
            estimator=estimator_type(adapter.model,adapter.environment,table_offset=table)
            if adapter.profile['controller']=='show108_checked_plan_independent':
                estimator.episode_horizon=episode_seconds
            from inspect import getfile
            report['deployment_state_estimator']={
                'schema':estimator.schema,'source_sha256':sha(Path(getfile(estimator_type))),
                'episode_horizon_s':estimator.episode_horizon,
                'scope':'deployment object-state inference; actor weights and controller unchanged'}
        robot=CancellableTransport(robot,check_cancel);vision=CancellableTransport(vision,check_cancel)
        progress('preflight')
        state=wait_feedback(robot,adapter,4.,moving=mode!='shadow')
        if mode!='mock':state,objects=wait_vision(robot,vision,adapter,120.,mode=='run',estimator=estimator)
        else:objects=vision.read(time.time());validate_objects(objects,adapter.environment)
        if 'support_plane'in objects:
            plane=objects['support_plane'];guard.set_plane(plane['normal'],plane['offset'])
            controller=adapter.controller(guard.floor_offset,floor_normal=guard.normal)
        guard.check(state['q_rad'],state['q_rad'],state['gripper_full_width_m'],state['gripper_full_width_m'])
        report['floor_plane_base']={'normal':guard.normal.tolist(),'offset':guard.floor_offset}
        if mode=='run':
            if np.min(robot.acceleration_limits())<5.:raise ValueError('Firmware acceleration below runtime profile')
            state=wait_feedback(robot,adapter,4.,moving=True);validate_state(state,adapter,start=True);robot.speed(100)
        controller.reset(state['q_rad'],state['gripper_full_width_m'],.1)
        history=ReferenceHistory(float(np.min(state['joint_timestamps']))-.1,controller.reference)
        if estimator is not None:estimator.start_episode(time.time())
        episode_started=True;progress('running')
        started=time.monotonic();write_json(out/'run-clock.json',{'started_s':time.time(),'seconds':seconds});next_tick=started;next_policy=started;last_tick=started;last_width=controller.gripper_reference;calls=ticks=0;maxima=np.zeros(3)
        last_scene_phase=estimator.phase if estimator is not None else None
        while time.monotonic()-started<seconds and not stopping:
            now=time.monotonic()
            if now<next_tick:time.sleep(next_tick-now)
            now=time.monotonic()
            if ticks and now-last_tick>.05:raise ValueError('Control cycle gap exceeds 50ms')
            last_tick=now
            if mode=='mock':robot.advance(controller.dt)
            check_cancel()
            state=robot.read(require_enabled=mode!='shadow',require_can=mode!='shadow');validate_state(state,adapter)
            if now>=next_policy:
                if estimator is None:objects=vision.read(time.time())
                else:
                    wall=time.time();estimator.ingest(vision.observations(wall),wall)
                    objects=estimator.estimate(dict(state,gripper_reference_width_m=controller.gripper_reference if mode=='run'else None),wall,controller.gripper_goal)
                    if estimator.phase!=last_scene_phase:
                        emit('scene_transition',previous_phase=last_scene_phase,phase=estimator.phase,
                            state_estimation=objects['state_estimation'])
                        if estimator.phase=='held_tcp':
                            print(f"파지 추정: 폭 {state['gripper_full_width_m']*1000:.1f}mm 유지 → 큐브 위치를 TCP로 갱신합니다.",flush=True)
                        elif estimator.phase.startswith('released'):
                            print('해제 추정: '+estimator.phase,flush=True)
                        last_scene_phase=estimator.phase
                reference=controller.reference if mode=='shadow'else history.at(state['joint_timestamps'])
                obs,action=adapter.step(dict(state,**objects),controller,reset=calls==0,reference=reference)
                input_state=serial({'updated_s':time.time(),'mode':mode,'input_kind':'mock'if mode=='mock'else'live',
                    'perception_dir':str(perception_dir)if perception_dir else None,
                    'block_pos_m':objects['block_pos_m'],'bowl_pos_m':objects['bowl_pos_m'],
                    'block_quat_wxyz':objects['block_quat_wxyz'],'tcp_pos_m':obs[18:21],
                    'state_estimation':objects.get('state_estimation',{}),'raw_action':action})
                input_state.update(gripper_width_m=state['gripper_full_width_m'],
                    gripper_raw_width_m=state.get('gripper_raw_full_width_m',state['gripper_full_width_m']))
                write_json(out/'policy-input.json',input_state)
                trajectory.write(json.dumps(input_state,ensure_ascii=False,allow_nan=False)+'\n');trajectory.flush()
                controller.set_action(state['q_rad'],action,qdot=state['qvel_rad_s'],width=state['gripper_full_width_m'])
                emit('policy',observation=obs,raw_action=action,controller_phase=controller.phase,object_input=input_state);calls+=1;next_policy=now+controller.policy_dt
            command=controller.step(state['q_rad'],state['qvel_rad_s'],width=state['gripper_full_width_m'])
            guard.check(state['q_rad'],command['joint_target_rad'],state['gripper_full_width_m'],controller.gripper_reference)
            if mode!='shadow':
                check_cancel()
                robot.send_joints(command['joint_target_rad'])
                history.append(time.time(),command['joint_target_rad'])
                if abs(controller.gripper_reference-last_width)>1e-9:robot.send_gripper(controller.gripper_reference,.5);last_width=controller.gripper_reference
            maxima=np.maximum(maxima,[np.max(np.abs(controller.velocity)),np.max(np.abs(controller.reference_acceleration)),np.max(np.abs(command['reference_jerk']))])
            emit('control',**command,gripper_width=controller.gripper_reference);ticks+=1;next_tick=max(next_tick+controller.dt,now+controller.dt)
        check_cancel()
        if mode!='shadow':
            progress('braking')
            controller.begin_brake();deadline=time.monotonic()+1.
            while not controller.stopped:
                if time.monotonic()>deadline:raise ValueError('Reference braking timeout')
                if mode=='mock':robot.advance(controller.dt)
                check_cancel()
                state=robot.read();validate_state(state,adapter)
                command=controller.step(state['q_rad'],state['qvel_rad_s'],width=state['gripper_full_width_m'])
                guard.check(state['q_rad'],command['joint_target_rad'],state['gripper_full_width_m'],controller.gripper_reference)
                robot.send_joints(command['joint_target_rad']);emit('brake',**command);time.sleep(controller.dt)
        report.update(success=True,stopped_by_request=stopping,policy_calls=calls,control_ticks=ticks,maximum_reference_velocity_rad_s=maxima[0],maximum_reference_acceleration_rad_s2=maxima[1],maximum_reference_jerk_rad_s3=maxima[2])
    except ExecutionCancelled:
        report.update(success=True,stopped_by_request=True,fast_halt=(out/'halt.now').exists(),fault_response='new_targets_stopped_last_sdk_target_unchanged')
    except Exception as error:
        report.update(error=f'{type(error).__name__}: {error}',fault_response='new_targets_stopped_last_sdk_target_unchanged');emit('fault',error=report['error'])
    finally:
        if estimator is not None:
            report['scene_estimation_summary']={'schema':estimator.schema,'phase':estimator.phase,
                'source_counts':dict(estimator.counts),'last_grasp_evidence':getattr(estimator,'grasp_evidence',None),
                'success_verified':False}
        if robot is not None:
            report.update(hardware_commands_sent=mode=='run'and bool(robot.sent_joints or robot.sent_gripper),joint_writes=robot.sent_joints,gripper_writes=robot.sent_gripper)
            report.update(gripper_zero_offset_clamps=getattr(robot,'gripper_zero_clamps',0),gripper_last_raw_width_m=getattr(robot,'last_raw_width',None))
            try:robot.close()
            except Exception as error:report.update(success=False,cleanup_error=str(error))
        if vision is not None:
            try:vision.close()
            except Exception as error:report.update(success=False,cleanup_error=str(error))
        if motion_lock:motion_lock.close()
        log.close();trajectory.close()
    report['completed_s']=time.time();progress('finished')
    return serial(report)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['validate','replay','mock','shadow','run'])
    p.add_argument('--policy-id',required=True);p.add_argument('--recording',type=Path);p.add_argument('--out',type=Path)
    p.add_argument('--seconds',type=float,default=2.);p.add_argument('--calib',type=Path);p.add_argument('--perception-dir',type=Path);p.add_argument('--channel',default='can_piper');p.add_argument('--job-token')
    args=p.parse_args();args.out=args.out or ROOT/'runs/showonce-policy'/str(time.time_ns());args.out.mkdir(parents=True,exist_ok=True)
    if not np.isfinite(args.seconds)or not 0<args.seconds<=60:raise ValueError('Duration must be 0..60 seconds')
    library=PolicyLibrary();manifest=library.get(args.policy_id);folder=library.folder(args.policy_id)
    if args.mode in ('shadow','run')and(not args.calib or not args.perception_dir):p.error('shadow/run require explicit --calib and --perception-dir')
    try:
        write_json(args.out/'progress.json',{'phase':'validating','updated_s':time.time()})
        validated=validate(folder)
        if args.mode=='validate':result=validated
        elif args.mode=='replay':
            if not args.recording:raise ValueError('Recording required')
            result=recorded_inputs(folder,args.recording)
        else:result=execute(folder,args.mode,args.seconds,args.out,args.calib,args.perception_dir,args.channel)
        result.update(policy_id=manifest['id'],mode=args.mode,completed_s=time.time(),output_directory=str(args.out),package_validation=validated['parity'],package_recorded_replay=validated.get('recorded_replay'),runtime_source_sha256=validated['runtime_source_sha256'])
        write_json(args.out/'report.json',result);print(json.dumps({k:v for k,v in result.items()if k!='samples'},ensure_ascii=False))
        if result.get('success')is False:sys.exit(1)
    except Exception as error:
        write_json(args.out/'report.json',{'status':'error','mode':args.mode,'error':f'{type(error).__name__}: {error}','hardware_commands_sent':False});raise

if __name__=='__main__':main()
