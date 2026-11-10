import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from piper_policy_runtime.control100 import ControlFault
from piper_policy_runtime.scene_state import SceneState
from showonce_studio.scene_estimation import DeploymentSceneState


ENV=dict(block_half=.025,bowl_r_rim=.075,bowl_r_bottom=.045,bowl_height=.071)
MODEL=SimpleNamespace(forward=lambda q:dict(tcp_position=np.asarray(q[:3]),
    link6_quaternion_wxyz=np.array([1.,0,0,0])))


def prime(kind=DeploymentSceneState,model=MODEL,block=(.3,0,.01),bowl=(.3,.18,-.015)):
    scene=kind(model,ENV)
    scene.rest={'green_block':np.array(block),'bowl':np.array(bowl)}
    scene.rest_time={o:9.9 for o in scene.rest}
    scene.latest={o:dict(time=9.9,frame=1,pos=p.copy(),quat=np.array([1.,0,0,0]))for o,p in scene.rest.items()}
    scene.heartbeat=10.;scene.start_episode(10.)
    return scene


def step(scene,now,tcp,width,goal=0.,reference=0.):
    scene.heartbeat=now
    robot=dict(q_rad=[*tcp,0,0,0],sample_time_s=now,gripper_full_width_m=width,
        gripper_reference_width_m=reference)
    return scene.estimate(robot,now,goal)


def close_then_move(scene):
    step(scene,10.,(.3,.01,.035),.1,.1)
    step(scene,10.1,(.315,.018,.03),.086)
    step(scene,10.2,(.335,.036,.025),.052)
    return step(scene,10.31,(.36,.05,.04),.0525)


def test_grasp_is_detected_after_leaving_old_proximity_gate():
    scene=prime();result=close_then_move(scene)
    assert result['state_estimation']['cube_source']=='held_tcp'
    assert result['state_estimation']['grasp_evidence']['reason']=='width_plateau_after_recent_approach'
    assert not result['state_estimation']['success_verified']
    np.testing.assert_allclose(result['block_pos_m'],[.36,.05,.04])
    # The same trace fails to transition in the previous estimator.
    old=prime(SceneState);assert close_then_move(old)['state_estimation']['phase']=='stationary'


def test_empty_closure_only_passes_through_cube_width():
    scene=prime()
    step(scene,10.,(.3,0,.03),.1,.1)
    for i,width in enumerate((.086,.066,.047,.027,.008,.001,.001)):
        result=step(scene,10.1+.1*i,(.35,.04,.03),width)
        assert result['state_estimation']['phase']=='stationary'


def test_pause_at_requested_reference_is_not_inferred_as_contact():
    scene=prime();step(scene,10.,(.3,0,.03),.1,.1)
    step(scene,10.1,(.36,.05,.04),.052,reference=.052)
    result=step(scene,10.21,(.36,.05,.04),.052,reference=.052)
    assert result['state_estimation']['phase']=='stationary'


@pytest.mark.parametrize('case',['no_approach','expired_approach','closing_timeout','too_far','open_goal','not_stalled'])
def test_plateau_without_matching_approach_and_closing_is_not_a_grasp(case):
    scene=prime()
    if case!='no_approach':step(scene,10.,(.3,0,.03),.1,.1)
    start=10.6 if case=='expired_approach' else 10.1
    location=(.6,.1,.03)if case=='too_far'else(.36,.05,.04)
    if case=='closing_timeout':
        step(scene,10.05,(.315,.01,.03),.086)
        start=11.2
    if case=='not_stalled':
        step(scene,start,location,.055)
        result=step(scene,start+.11,location,.051)
    else:
        goal=.1 if case=='open_goal'else 0.
        step(scene,start,location,.052,goal)
        result=step(scene,start+.11,location,.052,goal)
    assert result['state_estimation']['phase']=='stationary'


def test_held_position_follows_tcp_and_table_projection_does_not_pull_it_back():
    scene=prime();close_then_move(scene)
    source_time=10.4
    scene.ingest(dict(frame='piper_base_link',units='m',offline_replay=False,
        sample_time_s=source_time,objects=[dict(id='green_block',pos=[.31,0,.01],
            quat_wxyz=[1,0,0,0],source_time_s=source_time,source_frame=2,
            size={'type':'box','half_extents':[.025]*3})]),source_time)
    result=step(scene,10.41,(.37,.09,.15),.0525)
    np.testing.assert_allclose(result['block_pos_m'],[.37,.09,.15])
    assert result['state_estimation']['ignored_observations']['green_block']=='held_or_release_prediction'
    assert result['state_estimation']['raw_observation_time_s']['green_block']==source_time
    assert result['state_estimation']['raw_observation_time_s']['bowl']==9.9
    # An open goal alone does not invent release before the jaws actually open.
    assert step(scene,10.51,(.37,.09,.15),.0525,.1)['state_estimation']['phase']=='held_tcp'
    released=step(scene,10.61,(.37,.09,.15),.075,.1)
    assert released['state_estimation']['phase']=='released_prediction'


def test_invalid_feedback_does_not_promote_grasp():
    scene=prime();step(scene,10.,(.3,0,.03),.1,.1)
    step(scene,10.1,(.335,.036,.025),.052)
    scene.heartbeat=10.21
    with pytest.raises(ControlFault,match='current robot feedback'):
        scene.estimate(dict(q_rad=[.36,.05,.04,0,0,0],sample_time_s=9.,gripper_full_width_m=.052),10.21,0.)
    assert scene.phase=='stationary'


def test_recorded_width_plateau_transitions_before_recorded_reopen():
    from piper_policy_runtime.kinematics import PiperKinematics
    from piper_runtime.resources import MODEL_XML
    rows=json.loads((Path(__file__).parent/'data/grasp-plateau.json').read_text())['samples']
    model=PiperKinematics(MODEL_XML);model.tcp_local_position=np.array([0.,0.,.1385])
    scenes=[prime(kind,model,rows[0]['cube'],rows[0]['bowl'])for kind in (SceneState,DeploymentSceneState)]
    phases=[[],[]]
    for row in rows:
        now=10.+row['time_s']
        for index,scene in enumerate(scenes):
            scene.heartbeat=now
            if scene.phase=='stationary':scene.rest['green_block']=np.array(row['cube'])
            state=dict(q_rad=row['q'],sample_time_s=now,gripper_full_width_m=row['width'],
                gripper_reference_width_m=row['reference'])
            result=scene.estimate(state,now,row['goal'])
            phases[index].append((row['time_s'],result['state_estimation']['phase']))
    assert not any(phase=='held_tcp'for _,phase in phases[0])
    confirmed=[t for t,phase in phases[1]if phase=='held_tcp']
    assert confirmed and min(confirmed)<3.793-rows[0]['original_time_s']
