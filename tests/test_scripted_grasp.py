import hashlib
from types import SimpleNamespace
import numpy as np
import pytest
from piper_policy_runtime.control100 import ControlFault
from showonce_studio.scripted_grasp import studio_snapshot,configure
from showonce_studio.showonce_stages import write_json


def fixture(tmp_path):
    calib=tmp_path/'calibration.json';write_json(calib,{'trusted':True,'serial':'fixture'})
    folder=tmp_path/'vision';folder.mkdir()
    write_json(folder/'manifest.json',{'mode':'live_camera_only','calibration':{
        'path':str(calib),'sha256':hashlib.sha256(calib.read_bytes()).hexdigest()}})
    write_json(folder/'tracked_poses.json',dict(frame='piper_base_link',units='m',
        offline_replay=False,sample_time_s=100.,objects=[
            dict(id='green_block',source_time_s=99.95),dict(id='bowl',source_time_s=90.)]))
    return folder,calib


def test_old_observation_is_not_made_fresh_by_live_heartbeat(tmp_path):
    folder,calib=fixture(tmp_path)
    result=studio_snapshot(folder,calib,100.)
    assert result['service_alive']
    assert result['fresh_ids']==['green_block']
    assert result['objects'][1]['source_time_s']==90.


@pytest.mark.parametrize('case',['calibration_mismatch','calibration_tampered','replay','heartbeat_stale'])
def test_inconsistent_or_stale_inputs_rejected(tmp_path,case):
    folder,calib=fixture(tmp_path)
    if case=='calibration_mismatch':
        other=tmp_path/'other.json';write_json(other,{'trusted':True,'serial':'other'});calib=other
    if case=='calibration_tampered':write_json(calib,{'trusted':True,'serial':'changed'})
    if case=='replay':write_json(folder/'tracked_poses.json',{'offline_replay':True})
    with pytest.raises(ControlFault):studio_snapshot(folder,calib,102. if case=='heartbeat_stale'else 100.)


def test_static_close_records_drift_and_does_not_add_motion_commands(tmp_path):
    seen=[]
    def original(name,target,robot,adapter,guard,log,**kwargs):
        log(dict(event='tick',stage=name,tcp_m=np.array([.3,0,.02])))
        return dict(tcp_m=np.array([.3,0,.021]),gripper_width_m=.052)
    module=SimpleNamespace(execute_stage=original,read_robot=lambda *a,**kw:{'q_rad':[0]*6})
    adapter=SimpleNamespace(environment={'nominal_qpos':[0]*6},model=SimpleNamespace(
        forward=lambda q:{'tcp_position':np.array([.3,0,.02])}))
    module=configure(module,tmp_path,tmp_path/'calib.json',tmp_path/'model')
    result=module.execute_stage('close',None,object(),adapter,object(),seen.append,mock=False,force=.5)
    assert result['close_tcp_max_sampled_drift_m']==pytest.approx(.001)
    assert seen[-1]['event']=='close_static_check'


def test_close_drift_fault_prevents_progress_to_lift(tmp_path):
    def original(name,target,robot,adapter,guard,log,**kwargs):
        log(dict(event='tick',stage=name,tcp_m=np.array([.3,.006,.02])))
        raise AssertionError('drift should have faulted')
    module=SimpleNamespace(execute_stage=original,read_robot=lambda *a,**kw:{'q_rad':[0]*6})
    adapter=SimpleNamespace(environment={'nominal_qpos':[0]*6},model=SimpleNamespace(
        forward=lambda q:{'tcp_position':np.array([.3,0,.02])}))
    configure(module,tmp_path,tmp_path/'calib.json',tmp_path/'model')
    with pytest.raises(ControlFault,match='lift cancelled'):
        module.execute_stage('close',None,object(),adapter,object(),lambda _:None,mock=False,force=.5)


def test_pick_only_does_not_require_unselected_transport_route(tmp_path):
    visited=[]
    def solve(model,guard,target,seed,desired,width):
        visited.append(target[0])
        if target[0]>=4:raise ControlFault('unreachable bowl route')
        return np.zeros(6)
    module=SimpleNamespace(execute_stage=lambda *a,**kw:None,solve_joint_goal=solve)
    configure(module,tmp_path,tmp_path/'calib.json',tmp_path/'model',through='lift')
    model=SimpleNamespace(lower=np.full(6,-2.),upper=np.full(6,2.),
        forward=lambda q:{'link6_quaternion_wxyz':np.array([1.,0,0,0])})
    adapter=SimpleNamespace(model=model,environment={'nominal_qpos':[0]*6})
    targets={k:np.array([i,0,0])for i,k in enumerate(('approach','descend','lift','transport','retreat'),1)}
    _,_,margin,goals=module.choose_grasp_orientation(adapter,None,targets,0.)
    assert margin>=8. and set(goals)=={'approach','descend','lift'}
    assert set(visited)=={1,2,3}
