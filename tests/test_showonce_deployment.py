"""Policy contracts, offline isolation and deployment preflight boundaries."""
from pathlib import Path
import copy
import json
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
from showonce_studio.showonce_policies import PolicyLibrary,PROFILE,MODEL_XML,layout,contract,sha,POLICIES
from showonce_studio.showonce_policy_runtime import Adapter,validate
from showonce_studio.showonce_policy_runtime import CancellableTransport
from showonce_studio.showonce_deployment_api import DeploymentManager,live_inputs
from showonce_studio.showonce_stages import read_json,write_json

FIXTURE=Path(__file__).with_name('policy-fixture.json')

class DeploymentTests(unittest.TestCase):
    def test_wide_spawn_uses_new_environment_without_widening_old_policy(self):
        from piper_runtime.preflight import validate_objects
        from piper_policy_runtime.control100 import ControlFault
        env=copy.deepcopy(self.meta['environment'])
        old=dict(env,block_x=[.25,.35],bowl_x=[.25,.35],block_y=[-.2,.2],bowl_y=[-.2,.2],spawn_max_radius=.4)
        new=dict(old,block_x=[.1,.5],bowl_x=[.1,.5],block_y=[-.3,.3],bowl_y=[-.3,.3],spawn_max_radius=.6)
        objects=dict(block_pos_m=np.array([.45,-.25,.01]),bowl_pos_m=np.array([.16,.2,-.015]),block_half_size_m=new['block_half'])
        validate_objects(objects,new)
        with self.assertRaises(ControlFault):validate_objects(objects,old)

    def test_training_conditions_are_read_from_selected_package_without_rewriting_manifest(self):
        item=self.library.register(self.folder,'original')
        path=self.library.folder(item['id'])/'manifest.json';before=path.read_bytes()
        original=self.library.catalog()[0]
        self.assertEqual(original['training_conditions']['spawn_ranges_m']['block']['x'],self.meta['environment']['block_x'])
        self.assertEqual(before,path.read_bytes())

    def test_show108_episode_limit_is_rejected_before_sdk_construction(self):
        from types import SimpleNamespace
        from showonce_studio.showonce_policy_runtime import execute
        adapter=SimpleNamespace(environment={'episode_length':200},profile={'policy_dt_s':.1,'controller':'show108_checked_plan_independent'})
        with patch('showonce_studio.showonce_policy_runtime.Adapter',return_value=adapter),patch('piper_runtime.transports.robot_io.SDKRobot')as sdk:
            with self.assertRaisesRegex(ValueError,'20초'):execute(self.folder,'run',21.,self.root/'too-long')
            sdk.assert_not_called()
    def test_policy_feedback_closed_stop_tolerance_keeps_raw_and_rejects_larger_errors(self):
        class Feedback:
            width=-.0018
            def read(self):return {'gripper_full_width_m':self.width}
        feedback=Feedback();transport=CancellableTransport(feedback,lambda:None)
        value=transport.read();self.assertEqual(value['gripper_full_width_m'],0.)
        self.assertEqual(value['gripper_raw_full_width_m'],-.0018)
        self.assertEqual(transport.gripper_zero_clamps,1)
        feedback.width=.05;self.assertEqual(transport.read()['gripper_full_width_m'],.05)
        feedback.width=-.004;self.assertEqual(transport.read()['gripper_full_width_m'],-.004)
        self.assertEqual(transport.gripper_zero_clamps,1)
        feedback.width=float('nan');self.assertTrue(np.isnan(transport.read()['gripper_full_width_m']))
        self.assertIsNone(transport.last_raw_width)
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.folder=self.root/'export';self.folder.mkdir();shutil.copy2(MODEL_XML,self.folder/'piper.xml')
        meta=read_json(FIXTURE);weights={'normalizer/mean':np.zeros(49,np.float32),'normalizer/std':np.ones(49,np.float32),'params/hidden_0/kernel':np.zeros((49,14),np.float32),'params/hidden_0/bias':np.zeros(14,np.float32)}
        np.savez(self.folder/'weights.npz',**weights);meta['weights_sha256']=sha(self.folder/'weights.npz');write_json(self.folder/'policy.json',meta)
        np.savez(self.folder/'golden.npz',observations=np.zeros((3,1,49),np.float32),resets=np.zeros((3,1),bool),actions=np.zeros((3,1,7),np.float32))
        self.meta=read_json(self.folder/'policy.json')
        self.library=PolicyLibrary(self.root/'library')
    def tearDown(self):self.temp.cleanup()
    def synthetic(self,size=50):
        meta=copy.deepcopy(self.meta);meta.update(observation_size=size,observation_layout=layout(size),head_sizes=[],recurrent_width=0)
        weights={'normalizer/mean':np.zeros(size,np.float32),'normalizer/std':np.ones(size,np.float32),
            'params/hidden_0/kernel':np.zeros((size,14),np.float32),'params/hidden_0/bias':np.zeros(14,np.float32)}
        np.savez(self.folder/'weights.npz',**weights);meta['weights_sha256']=sha(self.folder/'weights.npz');write_json(self.folder/'policy.json',meta)
        np.savez(self.folder/'golden.npz',observations=np.zeros((3,1,size),np.float32),resets=np.zeros((3,1),bool),actions=np.zeros((3,1,7),np.float32))
        return meta,weights
    def test_register_is_immutable_checks_source_and_path(self):
        original=sha(self.folder/'weights.npz');item=self.library.register(self.folder,'baseline')
        again=self.library.register(self.folder,'new name');self.assertEqual(item,again)
        target=self.library.folder(item['id']);self.assertEqual(sha(target/'weights.npz'),original)
        (target/'weights.npz').write_bytes(b'tamper')
        with self.assertRaises(ValueError):self.library.get(item['id'])
        self.assertEqual(sha(self.folder/'weights.npz'),original)
        with self.assertRaises(ValueError):self.library.folder('../escape')
    def test_phase_layout_matches_runtime_indices(self):
        self.synthetic();adapter=Adapter(self.folder);q=np.asarray(adapter.environment['nominal_qpos'])
        sensors={'frame':'piper_base_link','units':'SI','q_rad':q,'qvel_rad_s':np.zeros(6),'gripper_full_width_m':.1,
            'block_pos_m':np.array([.3,.1,.02]),'block_quat_wxyz':np.array([1.,0,0,0]),'bowl_pos_m':np.array([.3,-.15,-.015]),'block_half_size_m':.025}
        previous=np.arange(7,dtype=np.float32)/7
        for phase in (0,1,2):
            obs=adapter.observation(sensors,q,.1,previous,phase)
            self.assertEqual(obs.shape,(50,));self.assertEqual(obs[42],phase/2);np.testing.assert_equal(obs[43:],previous)
        with self.assertRaises(ValueError):adapter.observation(sensors,q,.1,previous,3)
        self.assertEqual(validate(self.folder)['status'],'passed')
    def test_wrong_observation_action_tcp_and_direct_gripper_rejected(self):
        for mutate in (lambda m:m['observation_layout'].reverse(),lambda m:m.update(action_size=6),
            lambda m:m['kinematics'].update(tcp_local_position_m=[0,0,.12]),
            lambda m:m['environment'].update(independent_gripper=True),
            lambda m:m['kinematics'].update(observation_frame='camera')):
            meta=copy.deepcopy(self.meta);mutate(meta)
            with self.assertRaises(ValueError):contract(meta)
    def test_normalizer_shape_and_bad_golden_are_rejected(self):
        meta,weights=self.synthetic();weights['normalizer/mean']=np.zeros(49,np.float32)
        np.savez(self.folder/'weights.npz',**weights);meta['weights_sha256']=sha(self.folder/'weights.npz');write_json(self.folder/'policy.json',meta)
        with self.assertRaises(ValueError):Adapter(self.folder)
        self.synthetic();np.savez(self.folder/'golden.npz',observations=np.zeros((3,1,50)),resets=np.zeros((3,1)),actions=np.ones((3,1,7)))
        with self.assertRaises(AssertionError):validate(self.folder)
        np.savez(self.folder/'golden.npz',observations=np.full((3,1,50),np.nan),resets=np.zeros((3,1)),actions=np.zeros((3,1,7)))
        with self.assertRaises(ValueError):validate(self.folder)
    def test_web_cannot_launch_hardware_and_plan_only_writes_json(self):
        item=self.library.register(self.folder,'baseline');manager=DeploymentManager(self.root,library=self.library,jobs=self.root/'jobs')
        with patch('showonce_studio.showonce_deployment_api.subprocess.Popen')as spawn:
            for mode in ('run','shadow'):
                with self.assertRaises(ValueError):manager.start(item['id'],mode)
            plan=manager.plan(item['id'],'run');spawn.assert_not_called()
        self.assertFalse(plan['hardware_motion_requested']);self.assertFalse(plan['live_inputs']['inputs_ready'])
        self.assertIn('showonce_studio.showonce_policy_runtime',plan['command']);self.assertEqual(plan['runtime_profile'],PROFILE)
    def test_live_contract_rejects_recordings_stale_or_mismatched_calibration(self):
        live=self.root/'live';live.mkdir();cal=self.root/'cal.json';value={'trusted':True,'table_check':{'height_m':-.015}};write_json(cal,value)
        manifest={'mode':'live_camera_only','calibration':{'path':str(cal),'sha256':sha(cal)}}
        snapshot={'offline_replay':False,'sample_time_s':time.time()}
        write_json(live/'manifest.json',manifest);write_json(live/'tracked_poses.json',snapshot)
        self.assertTrue(live_inputs(live,cal)['inputs_ready'])
        for stamp in (time.time()-2,time.time()+1):
            write_json(live/'tracked_poses.json',dict(snapshot,sample_time_s=stamp));self.assertFalse(live_inputs(live,cal)['inputs_ready'])
        write_json(live/'tracked_poses.json',dict(snapshot,offline_replay=True));self.assertFalse(live_inputs(live,cal)['inputs_ready'])
        write_json(live/'tracked_poses.json',snapshot);write_json(cal,dict(value,trusted=False));self.assertFalse(live_inputs(live,cal)['inputs_ready'])
    def test_resolved_controller_uses_agreed_limits(self):
        adapter=Adapter(self.folder);controller=adapter.controller(-.015)
        np.testing.assert_equal([controller.speed,controller.acceleration,controller.jerk],[1,5,100])
        self.assertEqual(adapter.meta['environment']['task'],self.meta['environment']['task'])
        self.assertNotEqual(adapter.environment['max_joint_vel'],controller.speed)

    def test_catalog_preserves_policy_specific_checks_and_discovers_only_live_inputs(self):
        baseline=self.library.register(self.folder,'baseline')
        meta=read_json(self.folder/'policy.json');meta['display']={'recommended':True,'role':'trained','architecture':'test actor'}
        write_json(self.folder/'policy.json',meta);trained=self.library.register(self.folder,'trained')
        for index,policy in enumerate((baseline,trained)):
            job=self.root/'runs/showonce-robot'/str(index);job.mkdir(parents=True)
            write_json(job/'job.json',{'id':str(index),'policy_id':policy['id'],'mode':'validate','started_s':index+1})
            write_json(job/'report.json',{'status':'error'if index==0 else 'passed','hardware_commands_sent':False})
        live=self.root/'runs/show100-vision-current';live.mkdir();calib=self.root/'cal.json'
        write_json(calib,{'trusted':True});write_json(live/'manifest.json',{'mode':'live_camera_only',
            'calibration':{'path':str(calib),'sha256':sha(calib)}})
        write_json(live/'tracked_poses.json',{'offline_replay':False,'sample_time_s':time.time()})
        manager=DeploymentManager(self.root,library=self.library,jobs=self.root/'jobs')
        with patch('showonce_studio.showonce_deployment_api.ROOT',self.root),patch('subprocess.Popen')as spawn:
            value=manager.catalog();spawn.assert_not_called()
        self.assertEqual([p['id']for p in value['policies']],[trained['id'],baseline['id']])
        self.assertEqual([p['validation']for p in value['policies']],['passed','failed'])
        self.assertEqual(len(value['live_inputs']),1)
        write_json(live/'tracked_poses.json',{'offline_replay':True,'sample_time_s':time.time()})
        with patch('showonce_studio.showonce_deployment_api.ROOT',self.root):self.assertFalse(manager.catalog()['live_inputs'])

if __name__=='__main__':unittest.main()
