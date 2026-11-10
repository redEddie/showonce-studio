"""Power/mode semantics extracted from the locally verified SHOW-78/100 utilities.
No import connects to a device; SDK-specific mode messages are loaded on demand.
"""
import argparse
import math
import time

def finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("유한한 숫자를 입력하세요")
    return number

def positive(value):
    number = finite_float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("0보다 큰 숫자를 입력하세요")
    return number

def reading(record, fields=None):
    if record is None:
        return None
    payload = list(record.msg) if fields is None else {
        key: getattr(record.msg, key, None) for key in fields
    }
    return {"msg": payload, "timestamp": record.timestamp, "hz": record.hz}

def snapshot(robot, gripper):
    enabled = robot.get_joints_enable_status_list()
    return {
        "get_joint_angles_rad": reading(robot.get_joint_angles()),
        "get_flange_pose_m_rad": reading(robot.get_flange_pose()),
        "get_arm_status": reading(robot.get_arm_status(), (
            "ctrl_mode", "arm_status", "mode_feedback", "teach_status", "motion_status", "err_code",
        )),
        "get_joints_enable_status_list": [
            enabled[i - 1] if robot.get_driver_states(i) is not None else None
            for i in range(1, 7)
        ],
        "get_motor_states": {
            str(i): reading(robot.get_motor_states(i), ("position", "velocity", "current", "torque"))
            for i in range(1, 7)
        },
        "get_gripper_status": reading(gripper.get_gripper_status(), (
            "value", "force", "mode", "status_code",
        )),
    }

def control_mode_summary(code):
    return {
        0: "대기", 1: "CAN 제어", 2: "티칭 모드", 3: "이더넷 제어",
        4: "Wi-Fi 제어", 5: "리모컨 제어", 6: "연동 티칭 입력", 7: "오프라인 궤적",
    }.get(code, f"알 수 없음({code})")

def require_gripper_can_control(status):
    if status is None:
        raise RuntimeError("팔 제어 상태를 수신하지 못해 그리퍼 명령을 보내지 않았습니다")
    code = int(status.msg.ctrl_mode)
    if code != 1:
        current = control_mode_summary(code)
        guidance = "로봇에서 티칭 모드를 종료한 뒤" if code == 2 else "show78_sdk_demo.py can 실행 뒤"
        raise RuntimeError(f"현재 제어 상태: {current}. {guidance} read로 확인하세요. 그리퍼 명령은 보내지 않았습니다")

def wait_until(predicate, timeout, message):
    deadline = time.monotonic() + timeout
    while True:
        if predicate():
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(message)
        time.sleep(0.02)

POWER_COMMANDS = {
    "enable": (True, True, True),
    "disable": (False, True, True),
    "enable-arm": (True, True, False),
    "disable-arm": (False, True, False),
    "enable-gripper": (True, False, True),
    "disable-gripper": (False, False, True),
}

def fresh(record, *, since=None):
    return (record is not None and math.isfinite(record.timestamp)
            and 0 <= time.time() - record.timestamp <= 0.5
            and (since is None or record.timestamp >= since))

def ensure_can_control(robot, invoke, timeout):
    wait_until(lambda: fresh(robot.get_arm_status()), timeout,
               "최신 팔 제어 상태를 수신하지 못했습니다")
    status = robot.get_arm_status()
    if int(status.msg.arm_status) != 0 or int(status.msg.err_code) != 0:
        raise RuntimeError("팔 오류/전자 정지 상태입니다. enable은 정지 상태를 리셋하지 않습니다. read로 확인하세요")
    if int(status.msg.ctrl_mode) == 2:
        require_gripper_can_control(status)
    if int(status.msg.ctrl_mode) == 1:
        return
    # Song Jisu's sender calls MotionCtrl_2(0x01, 0x01, speed, 0x00)
    # before JointCtrl. In pyAgxArm V189 this is the ordinary J mode setter
    # (the newly-created driver's mode message has ctrl_mode=1).
    requested_at = time.time()
    invoke(robot, "set_motion_mode", motion_mode="j")

    def in_can():
        feedback = robot.get_arm_status()
        return (fresh(feedback, since=requested_at) and int(feedback.msg.ctrl_mode) == 1
                and int(feedback.msg.arm_status) == 0 and int(feedback.msg.err_code) == 0)

    wait_until(in_can, timeout, "CAN 모드 전환 후 새 피드백에서 CAN 제어 상태를 확인하지 못했습니다")

GRIPPER_ZERO_TOLERANCE_M = .003

def gripper_enable_target(gripper):
    """Accept a small signed zero offset, while keeping the command in 0..100mm.

    This is an enable-time tolerance, not a new encoder zero or a correction
    applied to policy observations. Missing/stale/error feedback still fails.
    """
    feedback = gripper.get_gripper_status()
    if not fresh(feedback):
        raise RuntimeError("그리퍼 폭 피드백이 없거나 오래되어 활성화하지 않았습니다")
    if feedback.msg.mode != "width" or int(feedback.msg.status_code) & 0x3F:
        raise RuntimeError("그리퍼 폭 모드/오류 상태를 확인하세요. 활성화 명령을 보내지 않았습니다")
    measured = float(feedback.msg.value)
    if not math.isfinite(measured) or not -GRIPPER_ZERO_TOLERANCE_M <= measured <= .10:
        raise RuntimeError(f"그리퍼 실측 폭 {measured * 1000:g}mm가 활성화 허용 범위 -3..100mm 밖입니다")
    return {"measured_width_m": measured, "target_width_m": max(0., measured),
            "zero_offset_clamped": measured < 0, "zero_tolerance_m": GRIPPER_ZERO_TOLERANCE_M}

def set_motor_power(robot, gripper, command, invoke, timeout, force=0.5):
    """Send both requests before waiting; CAN frames are not atomic/simultaneous.

    There is no enable_gripper API in this SDK. Enabling uses the measured
    current aperture as its width target; up to 3mm of negative encoder offset
    is clamped to zero. Never substitute zero/open for missing feedback.
    No reset, zero calibration, or automatic rollback is performed.
    """
    enabled, use_arm, use_gripper = POWER_COMMANDS[command]
    width = enable_target = None
    if enabled:
        def ready():
            return (fresh(robot.get_arm_status())
                    and (not use_gripper or fresh(gripper.get_gripper_status())))

        wait_until(ready, timeout, "활성화 전 최신 팔/그리퍼 피드백을 확인하지 못했습니다")
        if use_gripper:
            gripper_enable_target(gripper)
        ensure_can_control(robot, invoke, timeout)
        # Mode acknowledgement may have taken time; refresh the aperture
        # immediately before the enable request rather than reuse an old width.
        if use_gripper:
            enable_target = gripper_enable_target(gripper)
            width = enable_target["target_width_m"]
            if enable_target["zero_offset_clamped"]:
                print(f'그리퍼 영점 부근 피드백: {enable_target["measured_width_m"] * 1000:g}mm → 활성화 목표 0mm (허용 오차 3mm, force={force:g})', flush=True)

    requested, errors = {}, []
    # Send consecutive frames without waiting for either acknowledgement.
    # Even if one request fails, attempt the other explicitly requested device.
    if use_gripper:
        requested["gripper"] = time.time()
        try:
            if enabled:
                invoke(gripper, "move_gripper_m", value=width, force=force)
            else:
                invoke(gripper, "disable_gripper")
        except Exception as exc:
            errors.append(f"그리퍼 요청: {exc}")
    if use_arm:
        requested["arm"] = time.time()
        try:
            invoke(robot, "enable" if enabled else "disable")
        except Exception as exc:
            errors.append(f"팔 요청: {exc}")

    confirmed = {}
    def acknowledged():
        if use_arm:
            drivers = [robot.get_driver_states(i) for i in range(1, 7)]
            states = robot.get_joints_enable_status_list()
            confirmed["팔 6축"] = (all(fresh(d, since=requested["arm"]) for d in drivers)
                                      and len(states) == 6
                                      and all(x is not None and bool(x) == enabled for x in states))
        if use_gripper:
            feedback = gripper.get_gripper_status()
            confirmed["그리퍼"] = (fresh(feedback, since=requested["gripper"])
                                      and bool(int(feedback.msg.status_code) & 0x40) == enabled
                                      and (not enabled or not int(feedback.msg.status_code) & 0x3F))
        return all(confirmed.values())

    try:
        wait_until(acknowledged, timeout, "요청 후 갱신된 모터 상태를 확인하지 못했습니다")
    except TimeoutError:
        pending = ", ".join(name for name, ok in confirmed.items() if not ok)
        errors.append(f"새 피드백 확인 실패: {pending}")
    if errors:
        raise RuntimeError("; ".join(errors) + " (일부 장치만 상태가 바뀌었을 수 있습니다. read로 확인하세요)")
    return enable_target

MODES = {0:'대기', 1:'CAN 제어 (일반 관절 J 모드)', 2:'티칭 (로봇 버튼)',
         3:'이더넷 제어', 4:'Wi-Fi 제어', 5:'리모컨 제어',
         6:'연동 티칭 입력', 7:'오프라인 궤적 모드'}

SELECTABLE = (0, 1, 3, 4, 7)

def end_teaching_record(robot):
    from pyAgxArm.protocols.can_protocol.msgs.piper.default.transmit.arm_motion_ctrl import ArmMsgMotionCtrl
    # byte0=0: no emergency-stop/reset; byte1=0: no trajectory command;
    # byte2=2: end teaching record. Never execute a recorded trajectory.
    robot._send_msg(ArmMsgMotionCtrl(grag_teach_ctrl=2))

def request_mode(robot, selection, timeout=3., on_request=None):
    if selection not in SELECTABLE:
        raise ValueError('이 모드는 SDK 설정 명령을 지원하지 않습니다. 티칭은 로봇 버튼으로 전환하세요.')
    wait_until(lambda: fresh(robot.get_arm_status()), timeout, '최신 팔 상태를 수신하지 못했습니다')
    status = robot.get_arm_status()
    if int(status.msg.err_code) != 0 or int(status.msg.arm_status) not in (0, 11):
        raise RuntimeError('팔 오류 상태입니다. mode는 정지/오류를 리셋하지 않습니다.')
    if int(getattr(status.msg, 'teach_status', 0)) == 1 or int(status.msg.arm_status) == 11:
        if on_request:on_request()
        end_teaching_record(robot)
        print('티칭 기록 종료 요청을 보냈습니다.', flush=True)
    requested_at = time.time()
    message = robot._MSG_ModeCtrl(ctrl_mode=selection, move_mode=1,
                                  move_spd_rate_ctrl=10, mit_mode=0)
    robot._msg_mode = message
    if on_request:on_request()
    robot._send_msg(message)
    def applied():
        state = robot.get_arm_status()
        return (fresh(state, since=requested_at) and int(state.msg.ctrl_mode) == selection
                and int(state.msg.arm_status) == 0 and int(state.msg.err_code) == 0)
    try:
        wait_until(applied, timeout, '모드 전환 후 새 피드백에서 선택한 모드를 확인하지 못했습니다')
    except TimeoutError:
        state = robot.get_arm_status()
        if state is not None and int(state.msg.ctrl_mode) == 2:
            raise RuntimeError('로봇 버튼의 티칭 모드가 유지되고 있습니다. 버튼으로 티칭을 종료한 뒤 mode에서 1을 선택하세요.')
        raise
    print(f'확인: {selection} — {MODES[selection]}', flush=True)
