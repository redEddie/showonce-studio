"""Manual unwind geometry and bounded route search, without any devices."""
from types import SimpleNamespace
import json
import numpy as np
import pytest
from piper_policy_runtime.control100 import ControlFault
from piper_runtime.transports.robot_io import MockRobot
from showonce_studio.manual_control import ManualModel,run_manual
from showonce_studio.manual_planner import unwind_plan

@pytest.fixture
def calibration(tmp_path):
    path=tmp_path/'cal.json';path.write_text(json.dumps({'trusted':True,'table_check':{'height_m':-.015}}))
    return path

def test_unwind_neutralizes_twisted_wrist_and_checks_every_segment(calibration):
    context=ManualModel(calibration);q=np.array([0.,1.3,-1.1,1.4,.9,-1.8])
    plan=unwind_plan(context,q,.1)
    assert not plan['hardware_commands_sent']and plan['tcp_fixed']and not plan['orientation_fixed']
    route=np.asarray(plan['joint_waypoints_rad'])
    np.testing.assert_allclose(route[0],q)
    assert np.linalg.norm(route[-1,[3,5]])<.01
    assert plan['goal_joint_margin_deg']>10
    assert plan['posture_cost_after']<.2*plan['posture_cost_before']
    for start,end in zip(route,route[1:]):
        context.check(start,end,.1)
        positions=context.forward_positions(np.linspace(start,end,25))
        assert np.max(np.linalg.norm(positions-np.asarray(plan['position_anchor_m']),axis=1))<=.0002

def test_fixed_position_unwind_handles_nonzero_base_heading(calibration):
    context=ManualModel(calibration);q=np.array([1.8,1.3,-1.1,1.4,.9,-1.8])
    plan=unwind_plan(context,q,.1)
    assert plan['tcp_fixed']
    np.testing.assert_allclose(plan['goal_pose']['xyz_mm'],plan['start_pose']['xyz_mm'],atol=.003)

def test_batch_forward_matches_runtime_kinematics(calibration):
    context=ManualModel(calibration);rng=np.random.default_rng(71)
    qs=rng.uniform(context.model.lower,context.model.upper,(25,6))
    expected=np.array([context.model.forward(q)['tcp_position']for q in qs])
    np.testing.assert_allclose(context.forward_positions(qs),expected,atol=1e-12)

def test_reference_stop_guard_rejects_tcp_translation(calibration):
    context=ManualModel(calibration);q=np.array([0.,.523,-.45,0.,0.,0.]);plan=np.zeros((5,18));plan[:,:6]=q
    context.check_stop_plan(q,plan,.1,position_anchor=context.model.forward(q)['tcp_position'])
    plan[:,2]+=.02
    with pytest.raises(ControlFault,match='위치 고정 참조'):context.check_stop_plan(q,plan,.1,position_anchor=context.model.forward(q)['tcp_position'])

def test_plan_only_worker_does_not_send_any_target(tmp_path,calibration):
    robot=MockRobot([0.,1.3,-1.1,1.4,.9,-1.8])
    args=SimpleNamespace(command='unwind',out=tmp_path/'preview',calib=calibration,channel='test-unwind-preview',seconds=5.,axis=None,direction=None,speed_mm_s=10.,plan_only=True)
    report=run_manual(args,lambda:robot)
    assert report['success'],report
    assert report['unwind_plan']['segments']>=1
    assert not report['hardware_commands_sent']
    assert robot.sent_joints==robot.sent_gripper==0

def test_feedback_leaving_position_tube_stops_further_targets(tmp_path,calibration):
    class DisturbedRobot(MockRobot):
        def read(self,**kwargs):
            if self.sent_joints:self.q[1]+=.1
            return super().read(**kwargs)
    robot=DisturbedRobot([0.,1.3,-1.1,1.4,.9,-1.8])
    args=SimpleNamespace(command='unwind',out=tmp_path/'disturbed',calib=calibration,channel='test-unwind-disturbed',seconds=5.,axis=None,direction=None,speed_mm_s=10.,plan_only=False)
    report=run_manual(args,lambda:robot)
    assert not report['success']and '위치 고정 실측'in report['error'],report
    assert robot.sent_joints==1 and robot.sent_gripper==0

def test_qp_executes_position_fixed_unwind_without_gripper_motion(tmp_path,calibration):
    robot=MockRobot([.4,1.25,-1.25,.15,.5,-.2]);context=ManualModel(calibration)
    initial=context.model.forward(robot.q)['tcp_position'].copy()
    args=SimpleNamespace(command='unwind',out=tmp_path/'fixed-execute',calib=calibration,channel='test-fixed-execute',seconds=10.,axis=None,direction=None,speed_mm_s=10.,plan_only=False)
    report=run_manual(args,lambda:robot)
    assert report['success'],report
    assert report['max_unwind_reference_tcp_error_m']<=.001
    assert report['max_unwind_measured_tcp_error_m']<=.002
    assert np.linalg.norm(context.model.forward(robot.q)['tcp_position']-initial)<.002
    assert np.linalg.norm(robot.q[[3,5]])<.25
    assert robot.sent_joints>10 and robot.sent_gripper==0

def test_fixed_position_search_can_be_cancelled(calibration):
    context=ManualModel(calibration)
    with pytest.raises(InterruptedError):unwind_plan(context,np.array([0.,1.3,-1.1,1.4,.9,-1.8]),.1,cancel=lambda:True)

def test_planner_does_not_silently_clamp_an_out_of_range_start(calibration):
    context=ManualModel(calibration);q=np.deg2rad([7.137,-1.996,2.333,-31.984,24.931,32.523])
    with pytest.raises(ControlFault,match='J2=')as error:unwind_plan(context,q,.1)
    assert 'J3='in str(error.value)

def test_plan_only_flag_cannot_accidentally_start_a_jog():
    from showonce_studio.showonce_robot import main
    with pytest.raises(SystemExit)as error:main(['jog','--axis','z','--direction','1','--plan-only'])
    assert error.value.code==2
