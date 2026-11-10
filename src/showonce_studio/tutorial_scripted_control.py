"""Internal seven-stage controller for the standalone scripted grasp tutorial.

Use showonce-scripted / showonce_studio.scripted_grasp, which supplies measured
file inputs and optional robot configuration. This is not a webapp backend.
"""
import argparse
from collections import Counter, deque
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from piper_policy_runtime.kinematics import PiperKinematics
from piper_policy_runtime.control100 import ControlFault
from piper_policy_runtime.control103 import Controller103, MOVE
from piper_policy_runtime.geometry_guard import GeometryGuard
from piper_policy_runtime.show100 import task_objects
from piper_runtime.transports.robot_io import SDKRobot, MockRobot

STAGES = ('approach', 'descend', 'close', 'lift', 'transport', 'release', 'retreat')
from .config import ROOT, default_calibration
from piper_runtime.resources import MODEL_XML, GEOMETRY
CALIB = default_calibration()
MODEL = MODEL_XML
REFERENCE_ADAPTER = None


def dump(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k:dump(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [dump(v) for v in value]
    return value


def load_controller(calibration):
    # Only the exported *configuration* is read; weights.npz is never opened.
    from .motion_config import motion_adapter
    adapter = REFERENCE_ADAPTER or motion_adapter()
    env = dict(adapter.environment, task='scripted_control',
               max_joint_vel=1., control_acceleration=5., control_jerk=100.,
               max_lin_vel=.05, max_ang_vel=.5, smooth_stop_steps=64)
    model = adapter.model
    table = calibration.get('table_check', {}).get('height_m', calibration.get('table_height_prior_m'))
    if table is None or not calibration.get('trusted'):
        raise ControlFault('Trusted calibration with table height required')
    guard = GeometryGuard(GEOMETRY, table_z=float(table), link_floor=.01)
    return SimpleNamespace(model=model, environment=env), guard


def vision_state(_):
    raise ControlFault('Use showonce-scripted with --perception-dir to configure the input source')


def freeze_scene(url, calib_path, *, seconds=1.2):
    samples = {'green_block': [], 'bowl': []}
    reasons = {'green_block': Counter(), 'bowl': Counter()}
    last_frames = {'green_block': None, 'bowl': None}
    latest = {}
    deadline = time.monotonic()+max(3., seconds+2.)
    start = time.monotonic()
    while time.monotonic() < deadline:
        state = vision_state(url)
        if not state.get('service_alive') or state.get('phase') != 'tracking_active':
            raise ControlFault('Vision node is not tracking: '+str(state.get('phase')))
        active = Path(state['active_calibration']).resolve()
        if json.loads(active.read_text()) != json.loads(calib_path.read_text()):
            raise ControlFault('Vision node and scripted runner calibration differ')
        now = time.time()
        for oid in samples:
            if oid not in state.get('fresh_ids', []):
                reasons[oid][state.get('status', {}).get(oid, 'not_fresh')] += 1
        for item in state.get('objects', []):
            oid = item['id']
            if oid not in samples or oid not in state.get('fresh_ids', []):
                continue
            if not -.01 <= now-float(item['source_time_s']) <= .25:
                continue
            frame = item['source_frame']
            if frame == last_frames[oid]:
                continue
            pos = np.asarray(item['pos'], float)
            if pos.shape != (3,) or not np.isfinite(pos).all():
                continue
            samples[oid].append((now, pos))
            samples[oid] = samples[oid][-30:]
            last_frames[oid] = frame
            latest[oid] = item
        if (time.monotonic()-start >= seconds and all(len(v) >= 5 for v in samples.values())
                and all(now-v[-1][0] < .3 for v in samples.values())):
            break
        time.sleep(.04)
    if not all(len(v) >= 5 and now-v[-1][0] < .3 for v in samples.values()):
        details = []
        for oid, rows in samples.items():
            reason = reasons[oid].most_common(1)
            details.append(f'{oid} {len(rows)}/5 fresh frames'
                           +(f', main rejection={reason[0][0]}' if reason else ''))
        raise ControlFault('Perception preflight incomplete: '+'; '.join(details))
    positions = {}
    for oid, rows in samples.items():
        recent = np.stack([row[1] for row in rows[-10:]])
        median = np.median(recent, axis=0)
        if np.max(np.linalg.norm(recent[:,:2]-median[:2], axis=1)) > .005:
            raise ControlFault(f'{oid} moved during preflight; place it still and retry')
        positions[oid] = median
    document = dict(frame='piper_base_link', units='m', objects=[dict(latest[oid], pos=positions[oid].tolist())
                                                                  for oid in ('green_block', 'bowl')])
    parsed = task_objects(document)
    if not np.isclose(parsed['block_half_size_m'], .025, atol=.001):
        raise ControlFault('Expected measured 50mm cube')
    bowl_checks = latest['bowl'].get('checks', {})
    return parsed, dict(directory=state['output_dir'], frames=last_frames,
                        positions_m=positions, support_plane=dict(
                            normal=bowl_checks.get('table_normal_base', [0.,0.,1.]),
                            offset=bowl_checks.get('table_plane_offset_m')),
                        cube_yaw_rad=latest['green_block'].get('checks', {}).get('yaw_rad'))


def make_targets(objects):
    block = objects['block_pos_m']
    bowl = objects['bowl_pos_m']
    if np.linalg.norm(block[:2]-bowl[:2]) < .11:
        raise ControlFault('Cube and bowl need at least 11cm horizontal separation for this staged path')
    approach = block+np.array([0., 0., .085])
    descend = block+np.array([0., 0., .006])
    lift = block+np.array([0., 0., .12])
    transport = np.array([bowl[0], bowl[1], max(bowl[2]+.13, lift[2])])
    retreat = transport+np.array([0.,0.,.05])
    return dict(approach=approach, descend=descend, lift=lift,
                transport=transport, retreat=retreat)


def read_robot(robot, model, nominal, *, start=False, real=False, state=None):
    state = robot.read(require_enabled=real, require_can=real) if state is None else state
    q = np.asarray(state['q_rad'], float)
    qdot = np.asarray(state['qvel_rad_s'], float)
    if (q.shape != (6,) or qdot.shape != (6,) or not np.isfinite(q).all()
            or not np.isfinite(qdot).all() or np.max(np.abs(qdot)) > 2.5):
        raise ControlFault('Invalid joint feedback')
    if np.any(q < model.lower-.02) or np.any(q > model.upper+.02):
        raise ControlFault('Measured joints outside model range')
    if start:
        posture_error_deg = float(np.rad2deg(np.max(np.abs(q-nominal))))
        if posture_error_deg > 3.:
            raise ControlFault(f'Start posture A error {posture_error_deg:.2f}deg > 3deg; '
                               'run showonce-robot home first')
        if state['gripper_full_width_m'] < .09:
            raise ControlFault(f'Start gripper width {state["gripper_full_width_m"]*1000:.1f}mm < 90mm; '
                               'run showonce-robot home first')
    return state


def wait_initial_feedback(robot, model, nominal, *, real, timeout=4.):
    """Check posture and a short stationary position window after CAN connects."""
    deadline = time.monotonic()+timeout
    last_error = None
    samples = deque()
    position_range_deg = float('inf')
    median_speed = float('inf')
    while time.monotonic() < deadline:
        try:
            state = robot.read(require_enabled=real, require_can=real)
            state = read_robot(robot, model, nominal, start=True, real=real, state=state)
        except ControlFault as exc:
            last_error = exc
            if str(exc).startswith(('Start posture A error', 'Start gripper width')):
                raise
            time.sleep(.02)
            continue
        if not real:
            return state
        now = time.monotonic()
        samples.append((now, np.asarray(state['q_rad']).copy(),
                        float(np.max(np.abs(state['qvel_rad_s'])))))
        while samples and now-samples[0][0] > .35:
            samples.popleft()
        if len(samples) >= 4 and now-samples[0][0] >= .25:
            positions = np.stack([sample[1] for sample in samples])
            position_range_deg = float(np.rad2deg(np.max(np.ptp(positions, axis=0))))
            median_speed = float(np.median([sample[2] for sample in samples]))
            if position_range_deg <= .2 and median_speed <= .05:
                return state
        time.sleep(.02)
    if samples:
        raise ControlFault(f'Start posture A is not stationary: position range '
                           f'{position_range_deg:.2f}deg, median reported speed '
                           f'{median_speed:.3f}rad/s in 0.35s')
    raise ControlFault(str(last_error) if last_error else 'Initial feedback timeout')


def solve_joint_goal(model, guard, target, seed, desired_rotation, width):
    """Resolve one measured TCP target before the QP moves the arm."""
    seed = np.asarray(seed, float)
    target = np.asarray(target, float)
    def residual(q):
        fk = model.forward(q)
        quaternion = fk['link6_quaternion_wxyz']
        actual = Rotation.from_quat(quaternion[[1,2,3,0]])
        orientation_error = (desired_rotation*actual.inv()).as_rotvec()
        return np.r_[(fk['tcp_position']-target)/.01,
                     orientation_error/.3, (q-seed)*.05]
    joint_margin = np.deg2rad(.5)
    result = least_squares(residual, np.clip(seed, model.lower+joint_margin, model.upper-joint_margin),
                           bounds=(model.lower+joint_margin, model.upper-joint_margin),
                           max_nfev=100, ftol=1e-7, xtol=1e-7)
    goal = result.x
    pose = model.forward(goal)
    error = float(np.linalg.norm(pose['tcp_position']-target))
    actual = Rotation.from_quat(pose['link6_quaternion_wxyz'][[1,2,3,0]])
    orientation_error = float(np.linalg.norm((desired_rotation*actual.inv()).as_rotvec()))
    if error > .003 or orientation_error > .55:
        raise ControlFault(f'IK target not reachable: TCP error {error*1000:.1f}mm, '
                           f'orientation error {np.degrees(orientation_error):.1f}deg')
    guard.check(goal, goal, width, width)
    return goal


def choose_grasp_orientation(adapter, guard, targets, cube_yaw):
    """Find a near-downward grasp whose complete route avoids joint limits."""
    model = adapter.model
    nominal = np.asarray(adapter.environment['nominal_qpos'], float)
    quat = model.forward(nominal)['link6_quaternion_wxyz']
    nominal_rotation = Rotation.from_quat(quat[[1, 2, 3, 0]])
    candidates = []
    for pitch_deg in (90, 85, 80, 75, 70, 65):
        for quarter_turns in range(-2, 3):
            yaw = float(cube_yaw + quarter_turns*np.pi/2)
            rotation = (Rotation.from_euler('z', yaw) *
                        Rotation.from_euler('y', pitch_deg, degrees=True) *
                        nominal_rotation)
            seed = nominal.copy()
            margins = []
            goals = {}
            try:
                for stage in ('approach', 'descend', 'lift', 'transport', 'retreat'):
                    width = .1 if stage in ('approach', 'descend', 'retreat') else 0.
                    seed = solve_joint_goal(model, guard, targets[stage], seed, rotation, width)
                    goals[stage] = seed.copy()
                    margins.append(float(np.rad2deg(np.min(np.minimum(
                        seed-model.lower, model.upper-seed)))))
            except ControlFault:
                continue
            minimum_margin = min(margins)
            if minimum_margin < 8.:
                continue
            # Prefer a vertical tool, then limit margin and a short initial move.
            candidates.append((pitch_deg, minimum_margin,
                               -float(np.linalg.norm(goals['approach']-nominal)),
                               yaw, goals))
    if not candidates:
        raise ControlFault('No near-vertical grasp route with >=8deg joint-limit margin; '
                           'move cube/bowl to a more reachable position')
    pitch_deg, minimum_margin, _, yaw, goals = max(candidates, key=lambda c: c[:3])
    return yaw, pitch_deg, minimum_margin, goals


def execute_stage(name, target, robot, adapter, guard, log, *, mock, force,
                  grasp_yaw=0., grasp_pitch_deg=75., max_seconds=18.):
    model, env = adapter.model, adapter.environment
    state = read_robot(robot, model, np.asarray(env['nominal_qpos']), real=not mock)
    width = state['gripper_full_width_m']
    closed = name in ('lift','transport')
    final_goal = 0. if closed else .1
    if name == 'close':
        final_goal = .1
    if name == 'release':
        final_goal = 0.
    controller = Controller103(model, env, floor_normal=guard.normal, floor_offset=guard.floor_offset+.01)
    measured_q = np.asarray(state['q_rad'], float)
    initial_q = np.clip(measured_q, model.lower, model.upper)
    boundary_error = float(np.max(np.abs(measured_q-initial_q)))
    if boundary_error > np.deg2rad(.1):
        raise ControlFault(f'{name}: measured joint exceeds model limit by '
                           f'{np.rad2deg(boundary_error):.2f}deg')
    controller.reset(initial_q, width, final_goal)
    target = None if target is None else np.asarray(target, float)
    nominal_q = np.asarray(env['nominal_qpos'])
    nominal_quat = model.forward(nominal_q)['link6_quaternion_wxyz']
    nominal_rotation = Rotation.from_quat(nominal_quat[[1,2,3,0]])
    desired_rotation = (Rotation.from_euler('z', grasp_yaw) *
                        Rotation.from_euler('y', grasp_pitch_deg, degrees=True) *
                        nominal_rotation)
    if target is not None and guard.normal @ target < guard.floor_offset+.01:
        raise ControlFault(f'{name}: target TCP below table +10mm')
    joint_goal = None if target is None else solve_joint_goal(
        model, guard, target, state['q_rad'], desired_rotation, width)
    start = time.monotonic()
    next_tick = next_action = start
    previous_tick = start
    settled_since = None
    settled_ticks = 0
    recent_motion = deque()
    last_width = controller.gripper_reference
    ticks = 0
    while (ticks < int(max_seconds/controller.dt) if mock
           else time.monotonic()-start < max_seconds):
        now = time.monotonic()
        if not mock and now < next_tick:
            time.sleep(next_tick-now)
            now = time.monotonic()
        if not mock and ticks and now-previous_tick > .05:
            raise ControlFault(f'{name}: 200Hz control deadline missed by {(now-previous_tick)*1000:.1f}ms')
        previous_tick = now
        if mock:
            robot.advance(controller.dt)
        state = read_robot(robot, model, np.asarray(env['nominal_qpos']), real=not mock)
        q, qdot, width = state['q_rad'], state['qvel_rad_s'], state['gripper_full_width_m']
        if not mock:
            recent_motion.append((now, q.copy(), float(np.max(np.abs(qdot)))))
            while recent_motion and now-recent_motion[0][0] > .35:
                recent_motion.popleft()
        tcp = model.forward(q)['tcp_position']
        if now >= next_action or mock and ticks % controller.steps == 0:
            if name in ('close','release'):
                action = np.array([0.,0.,0.,0.,0.,0., -1. if name=='close' else 1.])
                controller.set_action(q, action, qdot=qdot, width=width)
            else:
                controller.set_joint_goal(q, joint_goal, qdot=qdot, width=width,
                                          speed_cap=.4, gain=2.,
                                          gripper_action=-1. if closed else 1.)
            next_action = now+controller.policy_dt
        command = controller.step(q, qdot, width=width)
        guard.check(q, command['joint_target_rad'], width, controller.gripper_reference)
        robot.send_joints(command['joint_target_rad'])
        if abs(controller.gripper_reference-last_width) > 1e-9:
            robot.send_gripper(controller.gripper_reference, force)
            last_width = controller.gripper_reference
        if ticks % 20 == 0:
            log(dict(event='tick', stage=name, elapsed_s=now-start,
                     tcp_m=tcp, tcp_goal_m=target, joint_rad=q,
                     joint_target_rad=command['joint_target_rad'],
                     measured_speed_rad_s=float(np.max(np.abs(qdot))),
                     reference_speed_rad_s=float(np.max(np.abs(controller.velocity))),
                     gripper_width_m=width, phase=command['coord_phase_name'],
                     target_error_m=None if target is None else float(np.linalg.norm(target-tcp))))
        if target is None:
            reached = controller.operations_completed >= 1 and controller.phase == MOVE
        else:
            tol = .003 if name == 'descend' else .009
            stationary = np.max(np.abs(qdot)) <= .05 if mock else False
            if not mock and len(recent_motion) >= 4 and now-recent_motion[0][0] >= .25:
                positions = np.stack([sample[1] for sample in recent_motion])
                # CAN velocity reports pulse while the encoder position is fixed.
                # The position window, TCP error and stopped QP reference together
                # establish arrival without relying on those velocity pulses.
                stationary = np.rad2deg(np.max(np.ptp(positions, axis=0))) <= .2
            reached = (np.linalg.norm(target-tcp) <= tol and
                       np.rad2deg(np.max(np.abs(joint_goal-q))) <= 1.5 and stationary and
                       np.max(np.abs(controller.velocity)) <= .05)
        if reached:
            settled_since = now if settled_since is None else settled_since
            settled_ticks += 1
            if (settled_ticks >= 30 if mock else now-settled_since >= .15):
                break
        else:
            settled_since = None
            settled_ticks = 0
        ticks += 1
        next_tick = max(next_tick+controller.dt, now+controller.dt)
    else:
        raise ControlFault(f'{name}: target did not settle within {max_seconds:g}s; '
                           f'TCP={np.round(tcp,3)}, goal={np.round(target,3) if target is not None else None}')
    if name == 'close' and not mock and not .035 <= width <= .066:
        raise ControlFault(f'Grasp width {width*1000:.1f}mm does not indicate a 50mm cube')
    if name == 'release' and width < .09:
        raise ControlFault('Gripper did not reopen to >=90mm')
    return dict(stage=name, elapsed_s=ticks*controller.dt if mock else now-start, ticks=ticks,
                tcp_m=tcp, tcp_goal_m=target, gripper_width_m=width,
                joint_goal_rad=joint_goal,
                error_m=None if target is None else float(np.linalg.norm(target-tcp)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=('plan','run'))
    ap.add_argument('--calib', type=Path, default=CALIB)
    ap.add_argument('--channel', default='can_piper')
    ap.add_argument('--through', choices=STAGES, default='retreat')
    ap.add_argument('--auto', action='store_true', help='Run all selected stages without Enter prompts')
    ap.add_argument('--gripper-force', type=float, default=.5,
                    help='SDK gripper force argument; temporary default: 0.5')
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()
    if not .5 <= args.gripper_force <= 3.:
        ap.error('gripper force must be 0.5..3')
    if args.command == 'run' and not args.auto and not sys.stdin.isatty():
        ap.error('Interactive run needs a terminal; --auto explicitly enables unattended stages')
    output = args.out or ROOT/'runs'/('show100-scripted-'+dt.datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    output.mkdir(parents=True, exist_ok=False)
    event_log = (output/'events.jsonl').open('w')
    def log(value):
        event_log.write(json.dumps(dump(dict(time_s=time.time(), **value)), allow_nan=False)+'\n')
        event_log.flush()
    lock = robot = None
    report = dict(command=args.command, success=False, hardware_motion_requested=args.command=='run',
                  policy_weights_loaded=False, gripper_force_sdk=args.gripper_force,
                  stages=[], output=str(output))
    try:
        calib = json.loads(args.calib.read_text())
        adapter, guard = load_controller(calib)
        scene, provenance = freeze_scene(None, args.calib)
        support = provenance['support_plane']
        if support['offset'] is not None:
            guard.set_plane(support['normal'], support['offset'])
        targets = make_targets(scene)
        cube_yaw = provenance['cube_yaw_rad']
        if cube_yaw is None or not np.isfinite(cube_yaw):
            raise ControlFault('Measured cube orientation is required for the scripted grasp')
        grasp_yaw, grasp_pitch_deg, joint_margin_deg, grasp_goals = (
            choose_grasp_orientation(adapter, guard, targets, float(cube_yaw)))
        report.update(calibration=str(args.calib.resolve()), perception=provenance,
                      frozen_objects=scene, targets_m=targets, grasp_yaw_rad=grasp_yaw,
                      grasp_pitch_deg=grasp_pitch_deg,
                      grasp_joint_margin_deg=joint_margin_deg,
                      grasp_joint_goals_rad=grasp_goals,
                      controller_limits=dict(
                          joint_speed_rad_s=1., joint_acceleration_rad_s2=5.,
                          reference_jerk_rad_s3=100., tcp_floor_clearance_m=.01,
                          arm_link_floor_clearance_m=guard.link_floor,
                          ik_joint_limit_margin_deg=.5))
        print('정책 없는 단계 제어 | 인식 좌표 저장:', {k:np.round(v,3).tolist() for k,v in provenance['positions_m'].items()}, flush=True)
        print('단계:', ' → '.join(STAGES[:STAGES.index(args.through)+1]), flush=True)
        print('목표 TCP(m):', {k:np.round(v,3).tolist() for k,v in targets.items()}, flush=True)
        print(f'파지 자세: 기울기 {grasp_pitch_deg}° / 손목 한계 최소 여유 '
              f'{joint_margin_deg:.1f}°', flush=True)
        if args.command == 'run':
            print('실기 명령 전 전체 선택 단계의 모의 도달성 검사...', flush=True)
            dry_robot = MockRobot(adapter.environment['nominal_qpos'])
            report['preflight_plan'] = []
            for stage in STAGES[:STAGES.index(args.through)+1]:
                result = execute_stage(stage, targets.get(stage), dry_robot, adapter, guard, log,
                                       mock=True, force=args.gripper_force, grasp_yaw=grasp_yaw,
                                       grasp_pitch_deg=grasp_pitch_deg)
                report['preflight_plan'].append(result)
            control_cpus = sorted(os.sched_getaffinity(0))[-2:]
            os.sched_setaffinity(0, control_cpus)
            report['control_cpus'] = control_cpus
            lock = Path('/tmp',f'show100-{args.channel.replace("/","_")}.lock').open('a')
            try:
                fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                raise ControlFault('Another SHOW-100 motion runner owns this CAN channel')
            robot = SDKRobot(args.channel)
            sdk_limits = np.asarray(robot.joint_limits, float)
            for planned in report['preflight_plan']:
                goal = planned['joint_goal_rad']
                if goal is not None and (np.any(goal < sdk_limits[:,0]+1e-4)
                                         or np.any(goal > sdk_limits[:,1]-1e-4)):
                    raise ControlFault(planned['stage']+' IK goal reaches an SDK joint limit')
            firmware_acceleration = robot.acceleration_limits()
            report['firmware_joint_acceleration_rad_s2'] = firmware_acceleration
            if np.any(firmware_acceleration < 5.-1e-6):
                raise ControlFault('Firmware joint acceleration is below scripted 5rad/s² setting')
            robot.speed(100)
        else:
            robot = MockRobot(adapter.environment['nominal_qpos'])
        wait_initial_feedback(robot, adapter.model, np.asarray(adapter.environment['nominal_qpos']),
                              real=args.command=='run')
        for stage in STAGES[:STAGES.index(args.through)+1]:
            if args.command == 'run' and not args.auto:
                answer = input(f'{stage} 단계 시작: Enter / 중단: q > ').strip().lower()
                if answer in ('q','quit','stop'):
                    raise ControlFault('Operator stopped before '+stage)
                if answer:
                    raise ControlFault('Unknown response; stage not started')
            target = targets.get(stage)
            print('진행:',stage,flush=True)
            log(dict(event='stage_begin', stage=stage, target_m=target))
            result = execute_stage(stage, target, robot, adapter, guard, log,
                                   mock=args.command=='plan', force=args.gripper_force,
                                   grasp_yaw=grasp_yaw, grasp_pitch_deg=grasp_pitch_deg)
            report['stages'].append(result)
            log(dict(event='stage_complete', **result))
            print('완료:',stage,'TCP',np.round(result['tcp_m'],3),'오차(mm)',
                  None if result['error_m'] is None else round(result['error_m']*1000,1),flush=True)
        report['success'] = True
    except (ControlFault, OSError, ValueError, KeyboardInterrupt) as exc:
        report['error'] = str(exc)
        print(type(exc).__name__+': '+str(exc), file=sys.stderr, flush=True)
        if args.command == 'run' and robot is not None and robot.sent_joints:
            print('새 목표 전송을 중단했습니다. 마지막 SDK 목표까지 움직임이 남을 수 있습니다.',
                  file=sys.stderr, flush=True)
    finally:
        if robot is not None and hasattr(robot,'close'):
            try:
                robot.close()
            except Exception as close_error:
                report['close_error'] = str(close_error)
                if report['success']:
                    report['success'] = False
                    report['error'] = 'Robot connection close failed: '+str(close_error)
        if lock is not None:
            lock.close()
        report['joint_writes'] = getattr(robot,'sent_joints',0)
        report['gripper_writes'] = getattr(robot,'sent_gripper',0)
        report['hardware_commands_sent'] = args.command=='run' and bool(report['joint_writes'] or report['gripper_writes'])
        (output/'summary.json').write_text(json.dumps(dump(report),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        event_log.close()
    print('기록:',output,flush=True)
    return 0 if report['success'] else 1
