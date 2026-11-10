"""Projection and alignment checks require neither a robot nor a camera."""
from types import SimpleNamespace
import numpy as np
import pytest
from showonce_studio.tcp_diagnostic import inspect_frame,project


def fixture():
    packet=dict(schema='showonce_live_rgbd_v1',offline_replay=False,aligned_to='color',
        sequence=7,stream_id='test',camera={'serial':'123'},
        stamps={'host_wall_time_s':100.,'host_monotonic_s':50.})
    feedback=dict(source='passive_socketcan_receive_only',telemetry_fresh_at_frame=True,
        q_rad=[0]*6,gripper_full_width_m=.05,feedback_components={
            f'0x{i:03X}':{'host_monotonic_ns':int(49.99*1e9)}for i in (0x2A5,0x2A6,0x2A7)})
    calib=dict(trusted=True,serial='123',T_base_cam=np.eye(4).tolist())
    model=SimpleNamespace(forward=lambda q:dict(tcp_position=np.array([.1,.2,1.]),
        link6_quaternion_wxyz=np.array([1.,0,0,0])))
    return packet,feedback,calib,model


def test_receive_only_feedback_aligned_before_frame():
    packet,feedback,calib,model=fixture()
    row=inspect_frame(packet,feedback,calib,model,100.02)
    assert row['hardware_commands_sent'] is False
    assert row['independent_tcp_measurement'] is False
    assert row['joint_age_at_frame_ms']==pytest.approx([10.,10.,10.])


@pytest.mark.parametrize('case',['serial','replay','expired','future','untrusted','stale',
    'feedback_after_frame','pair_skew','old_joints'])
def test_misaligned_or_wrong_camera_is_rejected(case):
    packet,feedback,calib,model=fixture()
    if case=='serial':packet['camera']['serial']='456'
    if case=='replay':packet['offline_replay']=True
    if case=='expired':packet['stamps']['host_wall_time_s']=99.
    if case=='future':packet['stamps']['host_wall_time_s']=101.
    if case=='untrusted':calib['trusted']=False
    if case=='stale':feedback['telemetry_fresh_at_frame']=False
    if case=='feedback_after_frame':feedback['feedback_components']['0x2A5']['host_monotonic_ns']=int(50.001*1e9)
    if case=='pair_skew':feedback['feedback_components']['0x2A5']['host_monotonic_ns']=int(49.96*1e9)
    if case=='old_joints':feedback['feedback_components']['0x2A5']['host_monotonic_ns']=int(49.9*1e9)
    with pytest.raises(ValueError):inspect_frame(packet,feedback,calib,model,100.02)


def test_camera_to_base_is_inverted_and_points_behind_camera_are_hidden():
    calib={'T_base_cam':np.eye(4).tolist()};calib['T_base_cam'][0][3]=1.
    intr=dict(fx=100.,fy=100.,cx=320.,cy=240.)
    pixels=project([[1,0,2],[2,0,2],[1,0,-1]],calib,intr)
    np.testing.assert_allclose(pixels[:2],[[320,240],[370,240]])
    assert np.isnan(pixels[2]).all()
    distorted=dict(intr,distortion_coefficients=[.01,0,0,0,0])
    with pytest.raises(ValueError):project([[1,0,2]],calib,distorted)


def test_rotated_camera_axes_are_respected():
    # Camera +Z points along base +X, camera +X along base -Z.
    calib={'T_base_cam':[[0,0,1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]]}
    pixels=project([[2,0,-1]],calib,dict(fx=100.,fy=100.,cx=320.,cy=240.))
    np.testing.assert_allclose(pixels,[[370,240]])
