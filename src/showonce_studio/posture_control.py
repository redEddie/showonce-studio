"""Preserved 200Hz posture-A reference + QP controller; no SDK import on load."""
from collections import deque
import time
import numpy as np
from piper_policy_runtime.control100 import Controller100, ControlFault
from piper_policy_runtime.control103 import Controller103, PHASE_NAMES
from piper_runtime.transports.robot_io import PerceptionFailed

def validate_state(state, adapter, *, start=False, recovery_limits=None):
    q, qdot = state['q_rad'], state['qvel_rad_s']
    if not np.isfinite(q).all() or not np.isfinite(qdot).all() or np.max(np.abs(qdot)) > 2.5:
        raise ControlFault('Invalid/overspeed measured joints')
    lower, upper = (adapter.model.lower, adapter.model.upper) if recovery_limits is None else (
        np.asarray(recovery_limits)[:, 0], np.asarray(recovery_limits)[:, 1])
    if np.any(q < lower-.05) or np.any(q > upper+.05):
        offending = [f'J{i+1}={np.rad2deg(q[i]):.2f}° outside {np.rad2deg(lower[i]):.2f}..{np.rad2deg(upper[i]):.2f}°'
                     for i in range(6) if q[i] < lower[i]-.05 or q[i] > upper[i]+.05]
        raise ControlFault(('Measured joints outside SDK recovery limits: ' if recovery_limits is not None
                            else 'Measured joints outside model limits: ')+', '.join(offending))
    if not np.isfinite(state['gripper_full_width_m']) or not 0 <= state['gripper_full_width_m'] <= .101:
        raise ControlFault('Invalid gripper aperture')
    if start:
        if np.max(np.abs(q-np.array(adapter.environment['nominal_qpos']))) > np.deg2rad(3.):
            raise ControlFault('Start pose not reached: run show100_robot.py home first')
        if state['gripper_full_width_m'] < .09 or np.max(np.abs(qdot)) > .05:
            raise ControlFault('Start requires open gripper >=90mm and stationary arm')


def wait_feedback(robot, adapter, timeout, *, moving=False, recovery_limits=None):
    deadline, last = time.monotonic()+timeout, None
    while time.monotonic() < deadline:
        try:
            state = robot.read(require_enabled=moving, require_can=moving)
            validate_state(state, adapter, recovery_limits=recovery_limits)
            return state
        except PerceptionFailed:
            raise
        except ControlFault as exc:
            last = str(exc)
            time.sleep(.02)
    raise ControlFault(last or 'Feedback startup timeout')


def benchmark(controller, q, guard, width):
    controller.reset(q, width, .1)
    times = []
    for _ in range(160):
        start = time.perf_counter()
        if isinstance(controller, Controller103) and _ % 20 == 0:
            if controller.posture_mode:
                controller.set_joint_goal(q, q+np.array([.01,0,0,0,0,0]), qdot=np.zeros(6), width=width, gripper_action=0.)
            else:
                controller.set_action(q, [.02, 0, 0, 0, 0, 0, 0], qdot=np.zeros(6), width=width)
        command = controller.step(q, np.zeros(6), width=width) if isinstance(controller, Controller103) else controller.step(q, np.zeros(6))
        guard.check(q, command['joint_target_rad'], width, width)
        times.append((time.perf_counter()-start)*1000)
    return dict(p50_ms=float(np.percentile(times, 50)), p99_ms=float(np.percentile(times, 99)))


def make_controller(adapter, guard, kind, *, posture=False, jerk=80., acceleration=5., shaper_hz=None):
    if kind not in ('show103', 'show100'):
        raise ControlFault('Unknown controller: '+str(kind))
    cls = Controller103 if kind == 'show103' else Controller100
    extra = dict(posture_mode=posture, posture_jerk=jerk, posture_acceleration=acceleration,
                 posture_shaper_hz=shaper_hz) if kind == 'show103' else {}
    return cls(adapter.model, adapter.environment, **extra,
               floor_normal=guard.normal, floor_offset=guard.floor_offset+.01)


def move_posture(robot, adapter, guard, emit, *, posture='home', timeout=20., jerk=80., command_hz=200,
                 shaper_hz=None, force=.5):
    if command_hz != 200 or shaper_hz is not None:
        raise ControlFault("Studio posture-A uses the deployed unshaped 200Hz controller")
    recovery_limits = getattr(robot, 'joint_limits', None)
    state = wait_feedback(robot, adapter, 4., moving=True, recovery_limits=recovery_limits)
    firmware_acceleration = robot.acceleration_limits()
    # The six firmware queries take CAN cycles; preflight the current pose.
    state = wait_feedback(robot, adapter, 2., moving=True, recovery_limits=recovery_limits)
    acceleration = float(np.min(firmware_acceleration))
    label = posture.upper()
    target = (np.deg2rad([0, 0, 0, 0, 22, 0]) if posture == 'fold'
              else np.array(adapter.environment['nominal_qpos']))
    measured_origin = state['q_rad']
    width = state['gripper_full_width_m']
    if posture == 'fold' and width < .06:
        raise ControlFault(f'Fold requires open gripper >=60mm, as in Jisu fold; actual {width*1000:.1f}mm. Release the object before folding.')
    end_width = width if posture == 'fold' else .1
    emit(posture+'_preflight_pose', measured=measured_origin,
         measured_qvel=state['qvel_rad_s'], gripper_full_width_m=width,
         geometry_extrema_m=guard.extremes(measured_origin, width), target=target)
    guard.check(measured_origin, target, width, end_width, samples=120)
    deadline = time.monotonic()+timeout
    clipped = np.clip(measured_origin, adapter.model.lower, adapter.model.upper)
    boundary_error = float(np.max(np.abs(clipped-measured_origin)))
    # CAN joint positions are quantized; a few hundredths of a degree beyond
    # a model boundary should move inward with the normal QP, not leave the
    # SDK at 1% speed waiting for noisy reported velocities to reach zero.
    recovery = boundary_error > np.deg2rad(.1)
    if boundary_error > 1e-4 and not recovery:
        emit(posture+'_limit_quantization', measured=measured_origin,
             clipped_reference=clipped, boundary_error_deg=np.rad2deg(boundary_error))
    if recovery:
        # Position commands outside the SDK limits cannot be sent. Make this
        # exceptional return explicit, rather than hiding a2-degree clip in the
        # first supposedly smooth reference. No limit bypass or outward command.
        print(f'{label}: 실측 관절이 명령 범위를 벗어나 있어 SDK 1% 속도로 가까운 경계에 복귀합니다.', flush=True)
        robot.speed(1)
        guard.check(measured_origin, clipped, width, width, samples=20)
        robot.send_joints(clipped)
        recovery_deadline = min(deadline, time.monotonic()+30.)
        stationary = deque()
        while time.monotonic() < recovery_deadline:
            state = robot.read()
            validate_state(state, adapter, recovery_limits=recovery_limits)
            emit(posture+'_limit_recovery', reference=clipped, measured=state['q_rad'],
                 measured_qvel=state['qvel_rad_s'], sdk_speed_percent=1,
                 reference_jerk_limited=False)
            now = time.monotonic()
            stationary.append((now, state['q_rad'].copy(),
                               float(np.max(np.abs(state['qvel_rad_s'])))))
            while stationary and now-stationary[0][0] > .35:
                stationary.popleft()
            if (np.max(np.abs(clipped-state['q_rad'])) <= .002 and len(stationary) >= 4
                    and now-stationary[0][0] >= .25):
                positions = np.stack([sample[1] for sample in stationary])
                position_range = float(np.rad2deg(np.max(np.ptp(positions, axis=0))))
                median_speed = float(np.median([sample[2] for sample in stationary]))
                if position_range <= .2 and median_speed <= .05:
                    break
            time.sleep(.005)
        else:
            raise ControlFault(label+' joint-limit recovery did not settle')
    controller = make_controller(adapter, guard, 'show103', posture=True, jerk=jerk,
                                 acceleration=acceleration, shaper_hz=shaper_hz)
    origin = np.clip(state['q_rad'], adapter.model.lower, adapter.model.upper)
    width = min(.1, state['gripper_full_width_m'])
    nearly_open = posture == 'fold' or width >= .099
    controller.reset(origin, width if posture == 'fold' else (.1 if nearly_open else width), .1 if nearly_open else 0.)
    print(f'{label}: 시간 최적 참조 + QP로 {"접힌 자세 [0,0,0,0,22,0]°" if posture == "fold" else "자세 A"}를 추종합니다 (관절속도 ≤{controller.speed:g}rad/s, 가속도 ≤{controller.acceleration:g}rad/s², jerk ≤{controller.jerk:g}rad/s³, 입력 성형 {shaper_hz or "꺼짐"}Hz).', flush=True)
    # Firmware performs its internal interpolation at full configured speed;
    # the externally generated references impose the1rad/s operating bound.
    robot.speed(100)
    start = next_tick = next_goal = time.monotonic()
    previous_tick, last_width, maximum_jerk = start, controller.gripper_reference, 0.
    gaps, previous_phase = [], None
    braking = False
    ticks = 0
    last_progress_tick = 0
    best_goal_error = float(np.max(np.abs(controller.reference-target)))
    while time.monotonic() < deadline:
        now = time.monotonic()
        if now < next_tick:
            time.sleep(next_tick-now)
        now = time.monotonic()
        if gaps and now-previous_tick > .05:
            raise ControlFault(label+' control deadline missed')
        gaps.append(now-previous_tick)
        previous_tick = now
        state = robot.read()
        validate_state(state, adapter)
        if not braking and (ticks % controller.steps == 0 if shaper_hz is not None else now >= next_goal):
            controller.set_joint_goal(state['q_rad'], target, qdot=state['qvel_rad_s'],
                                      width=state['gripper_full_width_m'],
                                      gripper_action=0. if posture == 'fold' else 1.)
            next_goal = now+controller.policy_dt
        command = controller.step(state['q_rad'], state['qvel_rad_s'], width=state['gripper_full_width_m'])
        sent_target = command['joint_target_rad']
        guard.check(state['q_rad'], sent_target, state['gripper_full_width_m'], controller.gripper_reference)
        robot.send_joints(sent_target)
        ticks += 1
        if abs(controller.gripper_reference-last_width) > 1e-9:
            robot.send_gripper(controller.gripper_reference, force)
            last_width = controller.gripper_reference
        maximum_jerk = max(maximum_jerk, float(np.max(np.abs(command['reference_jerk']))))
        if shaper_hz is not None:
            goal_error = float(np.max(np.abs(controller.reference-target)))
            if goal_error < best_goal_error-.001:
                best_goal_error, last_progress_tick = goal_error, ticks
            if controller.candidate_rejections >= 5 and ticks-last_progress_tick > 200:
                raise ControlFault(label+' shaped reference stalled after repeated QP rejection')
        emit(posture, reference=command['joint_target_rad'], measured=state['q_rad'],
             measured_qvel=state['qvel_rad_s'],
             sent_target_rad=sent_target,
             joint_feedback_time_s=state.get('joint_timestamps'),
             velocity_feedback_time_s=state.get('velocity_timestamps_s'),
             nominal_goal_remaining_s=controller.nominal_goal_duration_s, **command)
        if controller.phase != previous_phase:
            emit(posture+'_coord_phase', phase=PHASE_NAMES[controller.phase])
            previous_phase = controller.phase
        close = (np.max(np.abs(controller.reference-target)) < 1e-4
                 and np.max(np.abs(state['q_rad']-target)) <= .01
                 and (abs(state['gripper_full_width_m']-end_width) <= .005 if posture == 'fold'
                      else state['gripper_full_width_m'] >= .09 and controller.gripper_reference >= .099)
                 and controller.phase == 0)
        if close and not braking:
            controller.begin_brake()
            braking = True
        if braking and controller.stopped and np.max(np.abs(state['qvel_rad_s'])) <= .02:
            print(f'{label} 완료: 목표 자세·참조 감속·실측 정지를 확인했습니다.', flush=True)
            return {posture+'_'+k:v for k,v in dict(controller='show103', limit_recovery=bool(recovery),
                joint_speed_limit_rad_s=controller.speed, acceleration_limit_rad_s2=controller.acceleration,
                firmware_acceleration_limits_rad_s2=firmware_acceleration.tolist(), sdk_speed_percent=100,
                jerk_limit_rad_s3=controller.jerk, stop_horizon_s=controller.stop_steps*controller.dt,
                goal_planner='synchronized_time_optimal_reference_then_QP',
                input_shaper_hz=shaper_hz,
                input_shaper_delay_s=controller.input_shaper.delay_s if controller.input_shaper is not None else None,
                command_hz=command_hz,
                maximum_quantized_wire_jerk_rad_s3=None,
                quantized_wire_writes=None,
                qp_candidate_rejections=controller.candidate_rejections,
                target_rad=target.tolist(), maximum_reference_jerk_rad_s3=maximum_jerk,
                control_period_p50_ms=float(np.percentile(gaps[1:], 50)*1000) if len(gaps)>1 else None,
                control_period_p99_ms=float(np.percentile(gaps[1:], 99)*1000) if len(gaps)>1 else None,
                elapsed_s=time.monotonic()-start,
                final_measured_error_rad=float(np.max(np.abs(state['q_rad']-target)))).items()}
        next_tick = max(next_tick+controller.dt, now+controller.dt)
    raise ControlFault(label+' did not settle at target/open gripper within timeout')


def home(robot, adapter, guard, emit, timeout=20., jerk=80., command_hz=200, shaper_hz=None, force=.5):
    return move_posture(robot, adapter, guard, emit, posture='home', timeout=timeout, jerk=jerk,
                        command_hz=command_hz, shaper_hz=shaper_hz, force=force)
