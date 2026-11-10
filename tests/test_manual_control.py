"""Reachability and leased control tests; no SDK, CAN, camera, or real motion."""
from pathlib import Path
from types import SimpleNamespace
import json
import threading
import time
import numpy as np
import pytest
from piper_policy_runtime.control100 import ControlFault
from piper_runtime.transports.robot_io import MockRobot
from showonce_studio.manual_control import ManualModel, run_manual, new_controller
from showonce_studio.robot_actions import manual_parameters
from showonce_studio.showonce_stages import read_json, write_json

Q = np.array([0., .523, -.45, 0., 0., 0.])

@pytest.fixture
def calibration(tmp_path):
    path=tmp_path/'calibration.json'
    path.write_text(json.dumps({'trusted':True,'table_check':{'height_m':-.015}}))
    return path

def test_impossible_fixed_tcp_rotation_rejected_before_any_motor_target(calibration):
    context=ManualModel(calibration)
    with pytest.raises(ControlFault):context.rotation_path(Q,.1,'roll',1)
    available=context.availability(Q,.1)
    assert available['x-']['allowed'] and not available['roll+']['allowed']
    assert available['roll+']['reason']

def test_reachable_yaw_rotation_keeps_tcp_and_respects_all_limits(calibration):
    from scipy.spatial.transform import Rotation
    from showonce_studio.manual_control import rotation_of
    context=ManualModel(calibration)
    q=np.array([.4381678659,1.4384644,-.8544142,-.3279305,.8252013,.9244296])
    start=context.model.forward(q)['tcp_position']
    expected=Rotation.from_euler('z',90,degrees=True)*rotation_of(context.model,q)
    route=context.rotation_path(q,.1,'yaw',1)
    for target in route:
        context.check(q,target,.1)
        assert np.linalg.norm(context.model.forward(target)['tcp_position']-start)<.00025
        q=target
    assert (expected*rotation_of(context.model,q).inv()).magnitude()<np.deg2rad(.5)

def test_manual_requires_trusted_calibration_and_initial_joint_limits(tmp_path,calibration):
    with pytest.raises(ControlFault):ManualModel(tmp_path/'missing.json')
    context=ManualModel(calibration);outside=Q.copy();outside[1]=context.model.lower[1]-.001
    with pytest.raises(ControlFault):context.check(outside,outside,.1)

def test_vectorized_stop_guard_matches_retained_geometry(calibration):
    context=ManualModel(calibration);robot=MockRobot(Q);control=new_controller(context,robot.read())
    control.set_action(Q,[-.5,0,0,0,0,0,0],qdot=np.zeros(6),width=.1)
    context.check_stop_plan(Q,control.plan,.1)
    for sample in control.plan:context.check(sample[:6],sample[:6],.1)
    # Lifting the table into the arm must reject the same trajectory in both paths.
    context.guard.floor_offset=float(context.guard.extremes(Q,.1)[3])+.001
    with pytest.raises(ControlFault):context.check_stop_plan(Q,control.plan,.1)
    with pytest.raises(ControlFault):context.check(Q,Q,.1)

def test_down_button_can_be_blocked_while_up_stays_available(calibration):
    from showonce_studio.manual_control import rotation_of
    context=ManualModel(calibration)
    q=np.array([.4381678659,1.6147664868,-.6739958764,-.5189959984,.4989747119,1.1637510162])
    for _ in range(40):
        target=context.model.forward(q)['tcp_position']-np.array([0,0,.001])
        try:q=context.inverse(q,target,rotation_of(context.model,q),.1)
        except ControlFault:break
    else:pytest.fail('The fixture never approached its table boundary')
    context.check(q,q,.1)
    available=context.availability(q,.1,rotations=False)
    assert not available['z-']['allowed']
    assert available['z+']['allowed']

@pytest.mark.parametrize('speed',[10.,50.])
def test_bounded_terminal_jog_moves_requested_axis_and_holds_gripper(tmp_path,calibration,speed):
    robot=MockRobot(Q);context=ManualModel(calibration);before=context.model.forward(Q)['tcp_position']
    args=SimpleNamespace(command='jog',out=tmp_path/'jog',calib=calibration,channel='test-jog',seconds=.65,axis='x',direction=-1,speed_mm_s=speed)
    report=run_manual(args,lambda:robot)
    assert report['success'],report
    after=context.model.forward(robot.q)['tcp_position']
    assert -.06<after[0]-before[0]<(-.015 if speed == 50. else -.001)
    # Damped velocity IK has a residual that scales with travel distance.
    assert abs(after[1]-before[1])<.0005
    assert abs(after[2]-before[2])<.0005+.05*abs(after[0]-before[0])
    assert robot.sent_joints>10 and robot.sent_gripper==0

def test_opening_manual_session_never_sends_motion(tmp_path,calibration):
    folder=tmp_path/'read-only-session';folder.mkdir();robot=MockRobot(Q)
    write_json(folder/'manual-lease.json',{'expires_monotonic':time.monotonic()+5})
    args=SimpleNamespace(command='manual',out=folder,calib=calibration,channel='test-manual-idle',seconds=.2,axis=None,direction=None,speed_mm_s=10.)
    report=run_manual(args,lambda:robot)
    assert report['success']and not report['hardware_commands_sent'],report
    assert robot.sent_joints==robot.sent_gripper==0

def test_browser_jog_expires_without_more_renewals(tmp_path,calibration):
    robot=MockRobot(Q);folder=tmp_path/'session';folder.mkdir();token='a'*32
    write_json(folder/'manual-lease.json',{'expires_monotonic':time.monotonic()+6})
    args=SimpleNamespace(command='manual',out=folder,calib=calibration,channel='test-lease',seconds=4.,axis=None,direction=None,speed_mm_s=10.)
    result={}
    def browser():
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            progress=read_json(folder/'progress.json',{})
            if progress.get('phase')=='ready':break
            time.sleep(.01)
        write_json(folder/'manual-input.json',dict(id=token,action='jog',axis='x',direction=-1,speed_mm_s=10.,released=False,expires_monotonic=time.monotonic()+.25))
        while time.monotonic()<deadline:
            progress=read_json(folder/'progress.json',{})
            if progress.get('completed_operation')==token:
                result['stopped']=True;result['writes_at_stop']=robot.sent_joints
                time.sleep(.15);result['writes_later']=robot.sent_joints
                break
            time.sleep(.01)
        write_json(folder/'manual-lease.json',{'expires_monotonic':0})
    thread=threading.Thread(target=browser);thread.start()
    try:report=run_manual(args,lambda:robot)
    finally:thread.join(timeout=7)
    assert report['success'],report
    assert result.get('stopped')and result['writes_at_stop']>0
    assert result['writes_later']==result['writes_at_stop']

@pytest.mark.parametrize('body,action',[({'axis':'x','direction':1,'speed_mm_s':float('nan')},'jog'),({'axis':'x','direction':1,'speed_mm_s':50.01},'jog'),({'axis':'yaw','direction':1},'jog'),({'axis':'roll','direction':True},'rotate'),({'width_mm':101},'gripper'),({'width_mm':50,'force':0},'gripper')])
def test_malformed_manual_commands_rejected(body,action):
    with pytest.raises(ValueError):manual_parameters(body,action)
