"""Native posture-A controller tests; never open SDK/CAN or move hardware."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import fcntl
import json
import os
import uuid

import numpy as np
import pytest

from showonce_studio.home_control import run_home
from showonce_studio.motion_config import motion_adapter
from showonce_studio.showonce_stages import read_json,write_json
from piper_policy_runtime.control100 import ControlFault


def arguments(tmp_path):
    calibration=tmp_path/'calibration.json'
    write_json(calibration,{'trusted':True,'table_check':{'height_m':-.015}})
    return SimpleNamespace(out=tmp_path/'job',calib=calibration,channel='test-home-'+uuid.uuid4().hex[:8],seconds=60.,force=.5,policy_id=None)


class FakeRobot:
    sent_joints=sent_gripper=0
    def __init__(self):self.closed=False
    def read(self):return {'gripper_full_width_m':-.0018}
    def send_joints(self,target):self.sent_joints+=1
    def send_gripper(self,width,force):self.sent_gripper+=1
    def close(self):self.closed=True


def test_home_runs_owned_controller_and_preserves_limits(tmp_path):
    args=arguments(tmp_path);robot=FakeRobot();seen={};before=os.sched_getaffinity(0)
    def home(transport,adapter,guard,emit,**kw):
        seen.update(kw);seen['link_floor']=guard.link_floor
        assert transport.read()['gripper_full_width_m']==0.
        transport.send_joints(adapter.environment['nominal_qpos'])
        transport.send_gripper(.1,kw['force'])
        emit('home',reference=np.array(adapter.environment['nominal_qpos']))
        return {'home_controller':'show103','home_jerk_limit_rad_s3':kw['jerk']}
    with patch('showonce_studio.posture_control.benchmark',return_value={'p99_ms':.1}),patch('showonce_studio.posture_control.home',side_effect=home):
        report=run_home(args,robot_factory=lambda _:robot)
    assert report['success'] and report['hardware_commands_sent'] and robot.closed
    assert seen['jerk']==100 and seen['force']==.5 and seen['command_hz']==200 and seen['link_floor']==.01
    assert report['joint_writes']==report['gripper_writes']==1
    assert report['implementation']=='studio_native_posture_a_v1' and not report['policy_weights_loaded']
    assert 'home_runner' not in report
    assert os.sched_getaffinity(0)==before
    assert read_json(args.out/'progress.json')['phase']=='finished'
    assert read_json(args.out/'motion/report.json')['success']


def test_cancel_before_configuration_or_sdk(tmp_path):
    args=arguments(tmp_path);args.out.mkdir();(args.out/'halt.now').touch()
    with patch('showonce_studio.motion_config.motion_adapter') as adapter:
        result=run_home(args,robot_factory=lambda _:pytest.fail('SDK must not open'))
    adapter.assert_not_called()
    assert result['stopped_by_request'] and not result['hardware_commands_sent']


def test_cancellation_after_partial_motion_reports_writes_and_closes(tmp_path):
    args=arguments(tmp_path);robot=FakeRobot()
    def home(transport,adapter,guard,emit,**kw):
        transport.send_joints(adapter.environment['nominal_qpos'])
        (args.out/'halt.now').touch()
        transport.send_joints(adapter.environment['nominal_qpos'])
    with patch('showonce_studio.posture_control.benchmark',return_value={'p99_ms':.1}),patch('showonce_studio.posture_control.home',side_effect=home):
        report=run_home(args,robot_factory=lambda _:robot)
    assert report['stopped_by_request'] and not report['success']
    assert report['hardware_commands_sent'] and report['joint_writes']==1 and robot.closed


def test_failed_controller_preflight_never_opens_sdk(tmp_path):
    args=arguments(tmp_path)
    with patch('showonce_studio.posture_control.benchmark',return_value={'p99_ms':6.}):
        report=run_home(args,robot_factory=lambda _:pytest.fail('SDK must not open'))
    assert not report['success'] and not report['hardware_commands_sent'] and 'p99' in report['error']


def test_motion_lock_prevents_concurrent_sdk_connection(tmp_path):
    args=arguments(tmp_path)
    with Path('/tmp',f'show100-{args.channel}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with patch('showonce_studio.posture_control.benchmark',return_value={'p99_ms':.1}):
            report=run_home(args,robot_factory=lambda _:pytest.fail('SDK must not open'))
    assert not report['success'] and 'owns this CAN' in report['error']


def test_default_posture_needs_no_policy_package():
    with patch('showonce_studio.showonce_policies.PolicyLibrary',side_effect=AssertionError('no policy needed')):
        adapter=motion_adapter()
    np.testing.assert_allclose(adapter.environment['nominal_qpos'],[0,.523,-.45,0,0,0])
    np.testing.assert_allclose(adapter.model.tcp_local_position,[0,0,.1385])


def test_native_posture_mock_plant_converges_without_sdk(monkeypatch):
    from showonce_studio import posture_control
    from piper_runtime.transports.robot_io import MockRobot
    from piper_runtime.resources import GEOMETRY
    from piper_policy_runtime.geometry_guard import GeometryGuard
    class Clock:
        t=0.
        def monotonic(self):return self.t
        def sleep(self,dt):self.t+=max(0.,dt)
    class Plant(MockRobot):
        def read(self,**kw):
            self.advance(.005)
            return super().read(**kw)
    adapter=motion_adapter();goal=np.asarray(adapter.environment['nominal_qpos'])
    robot=Plant(goal+np.array([.15,.1,-.1,.1,.1,-.1]))
    guard=GeometryGuard(GEOMETRY,table_z=-.015,link_floor=.01)
    monkeypatch.setattr(posture_control,'time',Clock())
    report=posture_control.home(robot,adapter,guard,lambda *a,**kw:None,timeout=60.,jerk=100.)
    assert report['home_final_measured_error_rad']<=.01
    assert report['home_maximum_reference_jerk_rad_s3']<=100.+1e-3
    assert robot.sent_joints>0 and not robot.stops
    json.dumps(report,allow_nan=False)


def test_policy_target_reads_only_config_and_model(tmp_path,monkeypatch):
    import hashlib
    from piper_runtime.resources import MODEL_XML
    from showonce_studio.showonce_policies import PolicyLibrary
    root=tmp_path/'policies';policy_id='a'*20;folder=root/policy_id;folder.mkdir(parents=True)
    (folder/'piper.xml').write_bytes(MODEL_XML.read_bytes())
    target=[.1,.623,-.55,0,0,0]
    meta=dict(environment={'nominal_qpos':target},kinematics=dict(
        piper_xml_sha256=hashlib.sha256(MODEL_XML.read_bytes()).hexdigest(),tcp_local_position_m=[0,0,.1385]))
    (folder/'policy.json').write_text(json.dumps(meta))
    write_json(folder/'manifest.json',{'files':{name:hashlib.sha256((folder/name).read_bytes()).hexdigest()for name in ('piper.xml','policy.json')}})
    monkeypatch.setattr('showonce_studio.showonce_policies.PolicyLibrary',lambda:PolicyLibrary(root))
    adapter=motion_adapter(policy_id)
    np.testing.assert_allclose(adapter.environment['nominal_qpos'],target)
    assert not (folder/'weights.npz').exists()
