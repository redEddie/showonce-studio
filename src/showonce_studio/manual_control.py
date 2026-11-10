"""Manual TCP control using the shared QP. No SDK connection on import.

Button availability is a local kinematic/table check, not a full collision or
physical safety guarantee. A leased jog never falls back to terminal timing.
"""
from pathlib import Path
import math
import signal
import time
import warnings
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from piper_policy_runtime.kinematics import PiperKinematics
from piper_policy_runtime.control100 import ControlFault
from piper_policy_runtime.control103 import Controller103
from piper_policy_runtime.geometry_guard import GeometryGuard
from piper_runtime.resources import MODEL_XML, GEOMETRY
from .config import default_calibration
from .showonce_stages import read_json, write_json

ENV = dict(task='scripted_control', sim_dt=.005, ctrl_dt=.1, control_mode='qp',
           gripper_mode='ternary_latched', max_joint_vel=1., control_acceleration=5.,
           control_jerk=100., max_target_lag=.15, tracking_extra=.02, tracking_margin=.02,
           tracking_horizon=.05, tracking_weight=100., tracking_timeout=.5,
           gripper_rate=.01, gripper_max=.05, gripper_threshold=.5, ik_damping=.05,
           max_lin_vel=.05, max_ang_vel=.5, base_height=.015)
XYZ = ('x', 'y', 'z')
RPY = ('roll', 'pitch', 'yaw')

def rotation_of(model, q):
    wxyz = model.forward(q)['link6_quaternion_wxyz']
    return Rotation.from_quat(wxyz[[1, 2, 3, 0]])

def pose(model, q):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        rpy = rotation_of(model, q).as_euler('xyz', degrees=True)
    return dict(xyz_mm=(model.forward(q)['tcp_position']*1000).tolist(),
                rpy_deg=rpy.tolist(), frame='piper_base_link', tcp_mm=138.5)

class ManualModel:
    def __init__(self, calibration=None):
        path = Path(calibration) if calibration else default_calibration()
        if not path.is_file():
            raise ControlFault(f'수동 TCP 보정 파일을 찾을 수 없습니다: {path}. 웹앱의 현재 보정 경로를 확인하세요.')
        document = read_json(path)
        if not document or not document.get('trusted'):
            raise ControlFault(f'수동 TCP 보정 파일이 신뢰 판정을 통과하지 않았습니다: {path}')
        table = document.get('table_check', {}).get('height_m')
        if table is None:
            raise ControlFault('보정 파일에 테이블 높이가 없습니다.')
        self.model = PiperKinematics(MODEL_XML)
        self.model.tcp_local_position = np.array([0., 0., .1385])
        self.guard = GeometryGuard(GEOMETRY, table_z=float(table), link_floor=.01)
        check = document.get('table_check', {})
        if 'table_normal_base' in check and 'table_plane_offset_m' in check:
            self.guard.set_plane(check['table_normal_base'], check['table_plane_offset_m'])
        self.calibration = str(path.resolve())

    def forward_positions(self,qs):
        """Vectorized version of the pinned runtime FK, including hinge pivots."""
        qs=np.asarray(qs,float)
        if qs.ndim!=2 or qs.shape[1]!=6 or not np.isfinite(qs).all():raise ValueError('Expected finite joint samples (N,6)')
        count=len(qs);current=np.broadcast_to(np.eye(4),(count,4,4)).copy()
        for index,((offset,wxyz),(pivot,axis,reference,_))in enumerate(zip(self.model._bodies,self.model._joints)):
            body=np.eye(4);body[:3,:3]=Rotation.from_quat(wxyz[[1,2,3,0]]).as_matrix();body[:3,3]=offset
            before=np.eye(4);before[:3,3]=pivot;after=np.eye(4);after[:3,3]=-pivot
            axis=axis/np.linalg.norm(axis);x,y,z=axis;skew=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
            angle=qs[:,index]-reference;sine=np.sin(angle)[:,None,None];cosine=np.cos(angle)[:,None,None]
            spin=np.broadcast_to(np.eye(4),(count,4,4)).copy();spin[:,:3,:3]=np.eye(3)+sine*skew+(1-cosine)*(skew@skew)
            current=current@body@before@spin@after
        return np.einsum('nij,j->ni',current[:,:3,:3],self.model.tcp_local_position)+current[:,:3,3]

    def check(self, q, target, width):
        model = self.model
        if not np.isfinite(target).all() or np.any(target < model.lower-1e-8) or np.any(target > model.upper+1e-8):
            bad=[f'J{i+1}={np.degrees(target[i]):.3f}° (허용 {np.degrees(model.lower[i]):.2f}..{np.degrees(model.upper[i]):.2f}°)'
                 for i in range(6)if not np.isfinite(target[i])or target[i]<model.lower[i]-1e-8 or target[i]>model.upper[i]+1e-8]
            raise ControlFault('관절 목표가 모델 한계 밖입니다: '+', '.join(bad))
        samples = max(2, int(math.ceil(np.max(np.abs(np.asarray(target)-q))/np.deg2rad(1))))
        self.guard.check(q, target, width, width, samples=samples)
        for fraction in np.linspace(0., 1., samples+1):
            point = model.forward(np.asarray(q)+fraction*(np.asarray(target)-q))['tcp_position']
            if self.guard.normal @ point < self.guard.floor_offset+.01-1e-6:
                raise ControlFault('TCP 테이블 간격 10mm 미만')

    def inverse(self, q, position, rotation, width):
        model = self.model
        def residual(candidate):
            fk = model.forward(candidate)
            return np.r_[(fk['tcp_position']-position)/.01,
                         (rotation*rotation_of(model, candidate).inv()).as_rotvec()/.2,
                         (candidate-q)*.001]
        result = least_squares(residual, np.clip(q, model.lower+1e-9, model.upper-1e-9),
                               bounds=(model.lower, model.upper), max_nfev=45,
                               ftol=1e-7, xtol=1e-7, gtol=1e-7)
        target = result.x
        position_error = np.linalg.norm(model.forward(target)['tcp_position']-position)
        rotation_error = (rotation*rotation_of(model, target).inv()).magnitude()
        if position_error > .00025 or rotation_error > np.deg2rad(.5):
            raise ControlFault('관절 한계·특이점 또는 도달 불가 (위치/자세 유지 불가)')
        self.check(q, target, width)
        return target

    def rotation_path(self, q, width, axis, direction):
        # R/P/Y buttons rotate around base X/Y/Z, keeping the TCP position.
        position = self.model.forward(q)['tcp_position']
        orientation = rotation_of(self.model, q); vector = np.eye(3)[RPY.index(axis)]
        route = []
        for angle in np.linspace(0., direction*np.pi/2, 19)[1:]:
            target_rotation = Rotation.from_rotvec(vector*angle)*orientation
            q = self.inverse(q, position, target_rotation, width)
            route.append(q.copy())
        return route

    def availability(self, q, width, *, rotations=True):
        result = {}
        position = self.model.forward(q)['tcp_position']; orientation = rotation_of(self.model, q)
        for index, axis in enumerate(XYZ):
            for direction in (-1, 1):
                key = axis+('+' if direction > 0 else '-')
                try:
                    self.inverse(q, position+np.eye(3)[index]*direction*.001, orientation, width)
                    result[key] = {'allowed': True, 'reason': ''}
                except (ValueError, ControlFault) as error:
                    result[key] = {'allowed': False, 'reason': str(error)}
        if rotations:
            for axis in RPY:
                for direction in (-1, 1):
                    key = axis+('+' if direction > 0 else '-')
                    try:
                        self.rotation_path(q, width, axis, direction)
                        result[key] = {'allowed': True, 'reason': ''}
                    except (ValueError, ControlFault) as error:
                        result[key] = {'allowed': False, 'reason': str(error)}
        return result

    def check_stop_plan(self, reference, plan, width, *, position_anchor=None, position_tolerance=.001):
        """Batch the retained hulls for every reference and midpoint of its stop.

        The native QP already checks TCP and joint bounds at all reference
        samples. Geometry batching keeps this check out of Python's 200Hz path.
        """
        qs=np.vstack([reference,np.asarray(plan)[:,:6]])
        if not np.isfinite(qs).all()or np.any(qs<self.model.lower-2e-5)or np.any(qs>self.model.upper+2e-5):
            raise ControlFault('QP 정지 경로의 관절 한계 위반')
        qs=np.vstack([qs,(qs[:-1]+qs[1:])/2]);count=len(qs)
        current=np.broadcast_to(np.eye(4),(count,4,4)).copy();transforms=[current]
        for index,(origin,axis)in enumerate(zip(self.guard.origins,self.guard.axes)):
            axis=axis/np.linalg.norm(axis);x,y,z=axis
            skew=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
            sine=np.sin(qs[:,index])[:,None,None];cosine=np.cos(qs[:,index])[:,None,None]
            rotation=np.broadcast_to(np.eye(4),(count,4,4)).copy()
            rotation[:,:3,:3]=np.eye(3)+sine*skew+(1-cosine)*(skew@skew)
            current=current@origin@rotation;transforms.append(current)
        for index,sign in enumerate((1.,-1.)):
            slide=np.eye(4);slide[:3,3]=self.guard.finger_axes[index]*sign*width/2
            transforms.append(current@self.guard.finger_origins[index]@slide)
        for index in range(2,9):
            transform=transforms[index]
            points=self.guard.points[index]@np.swapaxes(transform[:,:3,:3],1,2)+transform[:,:3,3][:,None,:]
            minimum=self.guard.floor_offset+(self.guard.link_floor if index<6 else self.guard.grip_floor)
            if np.min(points@self.guard.normal)<minimum-1e-8:
                raise ControlFault('QP 가속·정지 경로가 팔/그리퍼 테이블 간격을 침범합니다.')
        maximum=0.
        if position_anchor is not None:
            maximum=float(np.max(np.linalg.norm(self.forward_positions(qs)-position_anchor,axis=1)))
            if maximum>position_tolerance:raise ControlFault(f'TCP 위치 고정 참조 오차 {maximum*1000:.2f}mm > {position_tolerance*1000:g}mm')
        return maximum

def checked_state(robot, context):
    state = robot.read()
    q = np.asarray(state['q_rad'], float); velocity = np.asarray(state['qvel_rad_s'], float)
    if q.shape != (6,) or velocity.shape != (6,) or not np.isfinite([q, velocity]).all():
        raise ControlFault('유효한 관절 위치·속도가 필요합니다.')
    if np.max(np.abs(velocity)) > 2.5:
        raise ControlFault('관절 속도 피드백이 허용 범위 밖입니다.')
    raw = float(state['gripper_full_width_m'])
    if not np.isfinite(raw) or not -.003 <= raw <= .1:
        raise ControlFault('유효한 그리퍼 폭 피드백이 필요합니다.')
    state['gripper_full_width_m'] = max(0., raw)
    context.check(q, q, state['gripper_full_width_m'])
    return state

def new_controller(context, state):
    control = Controller103(context.model, ENV, floor_normal=context.guard.normal,
                            floor_offset=context.guard.floor_offset+.01)
    width = state['gripper_full_width_m']
    control.reset(state['q_rad'], width, .1)
    # Hold current width; zero gripper action makes no new stroke request.
    control.gripper_goal = control.actuated_goal = width
    return control

def run_manual(args, robot_factory=None):
    """Leased browser session or bounded terminal jog/rotation. Never auto-enable."""
    from .showonce_robot import acquire_channel
    context = ManualModel(args.calib)
    folder = Path(args.out); folder.mkdir(parents=True, exist_ok=True)
    report = dict(kind='sdk', action=args.command, success=False, hardware_commands_sent=False,
                  started_s=time.time(), calibration=context.calibration)
    robot = lock = None; stopping = False; state = None; control = None
    requests = folder/'manual-input.json'; lease = folder/'manual-lease.json'
    operation = None; completed = ''; error = ''; route = []; cursor = 0
    route_summary = None
    availability = {}; last_probe_q = None; last_probe_width = None
    started = time.monotonic(); next_tick = started; last_tick = started
    last_progress = 0.; last_action = 0.; brake_started = None
    previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    def stop(*_):
        nonlocal stopping
        stopping = True
    for sig in previous_handlers: signal.signal(sig, stop)
    def cancelled():
        return stopping or (folder/'halt.now').exists()
    def leased():
        value = read_json(lease, {})
        return time.monotonic() <= value.get('expires_monotonic', 0.)
    def emit(phase):
        value = dict(phase=phase, updated_s=time.time(), available=availability, operation_error=error,
                     active_operation=operation['id'] if operation else None, completed_operation=completed,
                     unwind_plan=route_summary,jog_speed_limit_mm_s=ENV['max_lin_vel']*1000,
                     unwind_position_fixed=True)
        if state is not None:
            value.update(pose=pose(context.model, state['q_rad']),
                         gripper_width_mm=state['gripper_full_width_m']*1000,
                         feedback_s=state['sample_time_s'])
        write_json(folder/'progress.json', value)
    def check_plan():
        # Every accepted QP reference AND its checked stop must clear the table.
        holding=operation is not None and operation['action']=='unwind'
        anchor=np.asarray(route_summary['position_anchor_m'])if holding else None
        tolerance=route_summary['reference_position_tolerance_m']if holding else .001
        maximum=context.check_stop_plan(control.reference,control.plan[control.cursor:],state['gripper_full_width_m'],position_anchor=anchor,position_tolerance=tolerance)
        if holding:report['max_unwind_reference_tcp_error_m']=max(report.get('max_unwind_reference_tcp_error_m',0.),maximum)
    def begin_stop():
        nonlocal brake_started
        if control is not None and not control.finishing:
            control.begin_brake(); brake_started = time.monotonic()
    try:
        lock = acquire_channel(args.channel)
        if robot_factory is None:
            from piper_runtime.transports.robot_io import SDKRobot
            robot = SDKRobot(args.channel)
        else: robot = robot_factory()
        if hasattr(robot, 'joint_limits'):
            context.model.lower = np.maximum(context.model.lower, robot.joint_limits[:, 0])
            context.model.upper = np.minimum(context.model.upper, robot.joint_limits[:, 1])
        emit('connecting')
        deadline = time.monotonic()+4
        while True:
            if cancelled() or (args.command == 'manual' and not leased()):
                raise ControlFault('수동 연결이 중단되었거나 브라우저 연결 기한이 만료되었습니다.')
            try: state = checked_state(robot, context); break
            except ControlFault:
                if time.monotonic() > deadline: raise
                time.sleep(.02)
        # SDK speed is set explicitly before the first joint target, not on connect.
        speed_set = False
        first = True
        started = time.monotonic(); next_tick = last_tick = started
        while time.monotonic()-started < args.seconds or control is not None:
            now = time.monotonic()
            if now < next_tick: time.sleep(next_tick-now)
            now = time.monotonic()
            if control is not None and now-last_tick > .05:
                raise ControlFault('수동 제어 주기 지연이 50ms를 초과했습니다.')
            last_tick = now
            if hasattr(robot, 'advance'): robot.advance(.005)
            state = checked_state(robot, context)
            if operation is not None and operation['action']=='unwind':
                error_m=float(np.linalg.norm(context.model.forward(state['q_rad'])['tcp_position']-np.asarray(route_summary['position_anchor_m'])))
                report['max_unwind_measured_tcp_error_m']=max(report.get('max_unwind_measured_tcp_error_m',0.),error_m)
                if error_m>route_summary['measured_position_tolerance_m']:
                    raise ControlFault(f'TCP 위치 고정 실측 오차 {error_m*1000:.2f}mm 초과. 새 목표 전송을 중단했습니다.')
            if (folder/'halt.now').exists(): break
            expired = args.command == 'manual' and not leased()
            ending = cancelled() or expired or now-started >= args.seconds
            incoming = read_json(requests, {}) if args.command == 'manual' else {}
            if args.command != 'manual' and first:
                incoming = dict(id='terminal', action=args.command, axis=args.axis,
                                direction=args.direction, speed_mm_s=args.speed_mm_s,
                                plan_only=getattr(args,'plan_only',False),
                                expires_monotonic=started+args.seconds)
            first = False
            if operation is not None:
                if operation['action'] == 'jog':
                    if args.command == 'manual':
                        valid = incoming.get('id') == operation['id'] and incoming.get('action') == 'jog' and not incoming.get('released') and now <= incoming.get('expires_monotonic', 0.)
                    else: valid = not ending
                    if not valid: begin_stop()
                elif now-operation['started_monotonic']>(60 if operation['action']=='unwind'else 30):
                    error='계획한 자세에 도달하는 제한 시간을 초과했습니다.';begin_stop()
                if ending:
                    if operation['action']in ('rotate','unwind'):error='계획한 경로 완료 전 중단되었습니다.'
                    begin_stop()
            if ending and control is None: break
            if operation is None and incoming.get('id') and incoming['id'] != completed and not incoming.get('released') and not ending:
                error = ''
                try:
                    action = incoming['action']; q = state['q_rad']; width = state['gripper_full_width_m']
                    if action == 'jog':
                        if now > incoming.get('expires_monotonic', 0.): raise ControlFault('이동 버튼 입력이 만료되었습니다.')
                        index = XYZ.index(incoming['axis'])
                        context.inverse(q, context.model.forward(q)['tcp_position']+np.eye(3)[index]*incoming['direction']*.001, rotation_of(context.model, q), width)
                        route = []
                    elif action == 'rotate':
                        emit('planning')
                        route = context.rotation_path(q, width, incoming['axis'], incoming['direction'])
                        # Re-read after planning; no targets were sent during IK.
                        fresh = checked_state(robot, context)
                        if np.max(np.abs(fresh['q_rad']-q)) > .01: raise ControlFault('경로 계산 중 팔 자세가 바뀌었습니다. 다시 시도하세요.')
                        state = fresh
                    elif action == 'unwind':
                        from .manual_planner import unwind_plan
                        emit('planning')
                        route_summary=unwind_plan(context,q,width,cancel=lambda:cancelled()or(args.command=='manual'and not leased()))
                        write_json(folder/'manual-route.json',route_summary)
                        fresh=checked_state(robot,context)
                        if np.max(np.abs(fresh['q_rad']-q))>.01 or abs(fresh['gripper_full_width_m']-width)>.002:
                            raise ControlFault('경로 계산 중 팔/그리퍼 상태가 바뀌었습니다. 다시 계산하세요.')
                        if np.linalg.norm(context.model.forward(fresh['q_rad'])['tcp_position']-np.asarray(route_summary['position_anchor_m']))>.0005:
                            raise ControlFault('경로 계산 중 TCP가 0.5mm 이상 움직였습니다. 다시 계산하세요.')
                        state=fresh
                        if incoming.get('plan_only'):
                            completed=incoming['id'];last_probe_q=None
                            if args.command!='manual':break
                            continue
                        route=[np.asarray(point,float)for point in route_summary['joint_waypoints_rad'][1:]]
                    elif action == 'gripper':
                        # Keep arm targets untouched; check all intermediate finger widths.
                        target = float(incoming['width_mm'])/1000; force = float(incoming.get('force', .5))
                        if not 0 <= target <= .1 or not .5 <= force <= 3: raise ControlFault('잘못된 그리퍼 폭/force입니다.')
                        for width_sample in np.linspace(width, target, 21): context.check(q, q, width_sample)
                        if args.command == 'manual' and (cancelled() or not leased()): raise ControlFault('수동 연결이 중단되었습니다.')
                        robot.send_gripper(target, force); completed = incoming['id']; last_probe_q = None
                        continue
                    else: raise ControlFault('지원하지 않는 수동 명령입니다.')
                    if cancelled() or (args.command == 'manual' and not leased()): raise ControlFault('수동 연결이 중단되었습니다.')
                    if not speed_set: robot.speed(100); speed_set = True
                    operation = dict(incoming,started_monotonic=time.monotonic()); cursor = 0; control = new_controller(context, state)
                    last_action = 0.; brake_started = None
                except (ValueError, KeyError, ControlFault, InterruptedError) as exc:
                    error = str(exc); completed = incoming['id']; operation = control = None
                    if args.command != 'manual': raise
                last_tick = next_tick = time.monotonic()
            now = time.monotonic()
            if control is not None:
                if not control.finishing and now-last_action >= .1:
                    q = state['q_rad']; qdot = state['qvel_rad_s']; width = state['gripper_full_width_m']
                    saved={key:getattr(control,key)for key in ('plan','cursor','previous_target','target','previous_action','phase','timer','arm_stable','gripper_goal','actuated_goal','measured_width','opening_unlocked','plan_detail','candidate_rejections','latched')}
                    try:
                        if operation['action'] == 'jog':
                            index = XYZ.index(operation['axis']); direction = operation['direction']
                            action = np.zeros(7); action[index] = direction*operation['speed_mm_s']/1000/ENV['max_lin_vel']
                            control.set_action(q, action, qdot=qdot, width=width)
                            if now-operation['started_monotonic']>.4:
                                future=context.model.forward(control.plan[min(control.steps-1,len(control.plan)-1),:6])['tcp_position']
                                current=context.model.forward(control.reference)['tcp_position']
                                if direction*(future[index]-current[index])<operation['speed_mm_s']/1000*.1*.1:
                                    raise ControlFault('요청 방향의 이동 여유가 부족합니다.')
                        else:
                            settled=np.max(np.abs(q-route[cursor])) < (.012 if operation['action']=='unwind'else .025)
                            if cursor < len(route)-1 and settled:cursor += 1
                            control.set_joint_goal(q, route[cursor], qdot=qdot, width=width, speed_cap=.12 if operation['action']=='unwind'else .4, gain=3., gripper_action=0.)
                            if cursor == len(route)-1 and np.max(np.abs(q-route[-1])) < .005: begin_stop()
                        check_plan()
                        if control.candidate_rejections > saved['candidate_rejections']: raise ControlFault('QP가 요청 방향의 가속·정지 경로를 허용하지 않습니다.')
                    except (ValueError, ControlFault) as exc:
                        error = str(exc)
                        # Continue only the previously validated plan + full stop.
                        # No newly rejected candidate is ever transmitted.
                        for key,value in saved.items():setattr(control,key,value)
                        if operation.get('axis')is not None:
                            key=operation['axis']+('+'if operation['direction']>0 else '-')
                            availability[key]={'allowed':False,'reason':error}
                        begin_stop()
                    last_action = now
                if not speed_set: robot.speed(100); speed_set = True
                command = control.step(state['q_rad'], state['qvel_rad_s'], width=state['gripper_full_width_m'])
                context.check(state['q_rad'], command['joint_target_rad'], state['gripper_full_width_m'])
                if cancelled() and (folder/'halt.now').exists(): break
                robot.send_joints(command['joint_target_rad'])
                if control.finishing and control.stopped:
                    completed = operation['id']; operation = control = None; last_probe_q = None
                    if args.command != 'manual': break
                elif brake_started and now-brake_started > 2: raise ControlFault('수동 QP 정지 시간 초과')
            elif not ending:
                q = state['q_rad']; width = state['gripper_full_width_m']
                if last_probe_q is None or np.max(np.abs(q-last_probe_q)) > .01 or abs(width-last_probe_width) > .002:
                    emit('planning')
                    availability = context.availability(q, width)
                    last_probe_q = q.copy(); last_probe_width = width
                    # Probe computation is never part of a moving 200Hz loop.
                    last_tick = next_tick = time.monotonic()
            if now-last_progress >= .15:
                emit('moving' if control is not None else 'ready'); last_progress = now
            next_tick = max(next_tick+(.005 if control is not None else .03), time.monotonic())
        report.update(success=not bool(error), stopped_by_request=stopping, error=error or None)
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        if route_summary is not None:report['unwind_plan']=route_summary
        if state is not None: report['pose'] = pose(context.model, state['q_rad'])
        if robot is not None:
            report.update(hardware_commands_sent=bool(robot.sent_joints or robot.sent_gripper), joint_writes=robot.sent_joints, gripper_writes=robot.sent_gripper)
            try:robot.close()
            except Exception as exc:report.update(success=False,cleanup_error=str(exc))
        if lock is not None: lock.close()
        for sig, handler in previous_handlers.items(): signal.signal(sig, handler)
        report['completed_s'] = time.time(); emit('finished')
    return report
