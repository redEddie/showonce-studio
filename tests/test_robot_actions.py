"""SDK semantics via injected fake robot; subprocess ownership without CAN/camera."""
from pathlib import Path
from types import SimpleNamespace
import json
import signal
import sys
import time
from unittest.mock import patch
import pytest
from showonce_studio.showonce_robot import run_command
from showonce_studio.robot_actions import RobotActions

class FakeRobot:
    def __init__(self,stale=False):
        self.stale=stale;self.enabled=[False]*6;self.width=.063;self.code=0;self.control=1;self.calls=[];self._MSG_ModeCtrl=lambda **k:SimpleNamespace(**k)
    def record(self,msg):return SimpleNamespace(msg=msg,timestamp=time.time()-10 if self.stale else time.time(),hz=100.)
    def connect(self):self.calls.append(('connect',));return True
    def disconnect(self):self.calls.append(('disconnect',))
    def init_effector(self,name):return self
    def get_arm_status(self):return self.record(SimpleNamespace(ctrl_mode=self.control,arm_status=0,err_code=0,teach_status=0,motion_status=0,mode_feedback=1))
    def get_gripper_status(self):return self.record(SimpleNamespace(value=self.width,mode='width',status_code=self.code,force=.5))
    def get_joints_enable_status_list(self):return self.enabled
    def get_driver_states(self,i):return self.record(SimpleNamespace())
    def get_joint_angles(self):return self.record([0.]*6)
    def get_flange_pose(self):return self.record([0.]*6)
    def get_motor_states(self,i):return None
    def move_gripper_m(self,value,force):self.calls.append(('gripper',value,force));self.width=value;self.code=64;return True
    def disable_gripper(self):self.calls.append(('disable_gripper',));self.code=0;return True
    def enable(self):self.calls.append(('enable',));self.enabled=[True]*6;return True
    def disable(self):self.calls.append(('disable',));self.enabled=[False]*6;return True
    def reset(self):self.calls.append(('reset',));return True
    def electronic_emergency_stop(self):self.calls.append(('electronic_emergency_stop',));return True
    def set_motion_mode(self,**kw):self.calls.append(('set_motion_mode',));self.control=1;return True
    def _send_msg(self,msg):self.control=msg.ctrl_mode;self.calls.append(('mode',msg.ctrl_mode));return True

@pytest.mark.parametrize('action',['enable','disable','stop','reset','mode','read'])
def test_sdk_request_uses_only_explicit_semantics(action):
    robot=FakeRobot();report=run_command(action,'test-sdk-'+action,mode=1,timeout=.2,robot_factory=lambda:robot)
    assert report['success'],report
    if action=='enable':assert ('gripper',.063,.5)in robot.calls and robot.enabled==[True]*6
    if action=='disable':assert ('disable_gripper',)in robot.calls and ('disable',)in robot.calls
    if action=='stop':assert ('electronic_emergency_stop',)in robot.calls and not any(c[0]=='gripper'for c in robot.calls)
    if action=='mode':assert ('mode',1)in robot.calls
    if action=='read':assert not report['hardware_commands_sent']and not any(c[0]in ('enable','gripper','reset')for c in robot.calls)

def test_stale_gripper_never_gets_guessed_width():
    robot=FakeRobot(stale=True);report=run_command('enable','test-stale',timeout=.03,robot_factory=lambda:robot)
    assert not report['success']and not report['hardware_commands_sent'];assert not any(c[0]in ('enable','gripper')for c in robot.calls)

@pytest.mark.parametrize('measured,target',[(-.0023,0.),(-.003,0.),(0.,0.),(.063,.063),(.10,.10)])
def test_enable_handles_signed_zero_feedback_without_changing_positive_width(measured,target):
    robot=FakeRobot();robot.width=measured
    report=run_command('enable','test-zero-offset',timeout=.2,robot_factory=lambda:robot)
    assert report['success'],report
    assert ('gripper',target,.5)in robot.calls and robot.enabled==[True]*6
    assert report['gripper_enable']['measured_width_m']==measured
    assert report['gripper_enable']['target_width_m']==target
    assert report['gripper_enable']['zero_offset_clamped']==(measured<0)
    assert not any(c[0]in ('reset','calibrate_gripper')for c in robot.calls)

@pytest.mark.parametrize('measured',[-.00301,-.02,.10001,float('nan'),float('inf')])
def test_invalid_aperture_is_not_replaced_with_zero(measured):
    robot=FakeRobot();robot.width=measured
    report=run_command('enable','test-invalid-width',timeout=.2,robot_factory=lambda:robot)
    assert not report['success']and not report['hardware_commands_sent']
    assert not any(c[0]in ('enable','gripper')for c in robot.calls)
    assert '허용 범위'in report['error']

@pytest.mark.parametrize('state',['missing','stale','angle','fault','outside'])
def test_gripper_is_revalidated_after_can_mode_change(state):
    class ChangedFeedback(FakeRobot):
        def __init__(self):super().__init__();self.control=0
        def get_gripper_status(self):
            record=super().get_gripper_status()
            if self.control==1:
                if state=='missing':return None
                if state=='stale':record.timestamp-=10
                if state=='angle':record.msg.mode='angle'
                if state=='fault':record.msg.status_code=1
                if state=='outside':record.msg.value=-.02
            return record
    robot=ChangedFeedback();report=run_command('enable','test-can-width',timeout=.2,robot_factory=lambda:robot)
    assert not report['success']
    assert ('set_motion_mode',)in robot.calls
    assert not any(c[0]in ('enable','gripper')for c in robot.calls)

def test_unsupported_mode_rejected_before_sdk_factory():
    factory=lambda:pytest.fail('factory must not run')
    with pytest.raises(ValueError):run_command('mode','test-mode',mode=2,robot_factory=factory)

class Library:
    def get(self,policy_id):
        if policy_id!='fixture':raise ValueError('unknown policy')
        return {}

@pytest.fixture
def manager(tmp_path):
    worker=tmp_path/'fake_worker.py';worker.write_text('''import argparse,json,signal,time\nfrom pathlib import Path\np=argparse.ArgumentParser();p.add_argument('mode');p.add_argument('--out',type=Path,required=True);p.add_argument('--seconds',type=float,default=.15);args,_=p.parse_known_args();args.out.mkdir(parents=True,exist_ok=True)\ndef done(*_):\n (args.out/'report.tmp').write_text(json.dumps({'success':True,'stopped_by_request':True,'completed_s':time.time(),'hardware_commands_sent':False}));(args.out/'report.tmp').replace(args.out/'report.json');raise SystemExit(0)\nsignal.signal(signal.SIGINT,done);print('fake backend running',flush=True);time.sleep(args.seconds);(args.out/'report.tmp').write_text(json.dumps({'success':True,'completed_s':time.time(),'hardware_commands_sent':False}));(args.out/'report.tmp').replace(args.out/'report.json')\n''')
    command=[sys.executable,'-u',str(worker)]
    return RobotActions(tmp_path,Library(),jobs=tmp_path/'jobs',sdk_command=command,policy_command=command)

def wait_done(manager,job_id):
    end=time.monotonic()+4
    while time.monotonic()<end:
        job=manager.job(job_id)
        if not job['alive']:return job
        time.sleep(.03)
    pytest.fail('worker did not stop')

def test_command_choice_never_spawns_or_creates_job(manager):
    with patch('showonce_studio.robot_actions.subprocess.Popen')as popen:
        result=manager.perform({'kind':'sdk','action':'mode','mode_value':1,'execution':'command'})
    popen.assert_not_called();assert not result['process_started']and '--value 1'in result['command'];assert not list(manager.jobs.iterdir())
    assert result['display_command']=='python fake_worker.py mode --value 1'
    assert '--out'not in result['display_command']and '--job-token'not in result['display_command']
    assert result['execution_metadata']['job_token']==result['id']
    assert result['execution_metadata']['output_directory']==str(manager.folder(result['id']))

def test_home_command_is_managed_motion_and_can_be_interrupted(manager):
    with patch('showonce_studio.robot_actions.subprocess.Popen')as spawn:
        value=manager.perform({'kind':'sdk','action':'home','calibration':'/current.json','execution':'command'})
    spawn.assert_not_called();assert value['seconds']==60 and value['hardware_motion_requested']
    assert '--calib /current.json'in value['display_command']
    assert '--out'not in value['display_command']and '--job-token'not in value['display_command']
    job=manager.perform({'kind':'sdk','action':'home','seconds':4,'execution':'direct'})['job']
    time.sleep(.1);manager.interrupt(job['id']);result=wait_done(manager,job['id'])
    assert result['status']=='cancelled'and not result['report']['hardware_commands_sent']

def test_invalid_requests_and_live_inputs_rejected_without_process(manager):
    for body in [{'kind':'sdk','action':'joints'},{'kind':'sdk','action':'mode','mode_value':2},{'kind':'sdk','action':'enable','channel':'can;evil'},{'kind':'policy','policy_id':'fixture','mode':'run','execution':'direct'}]:
        with patch('showonce_studio.robot_actions.subprocess.Popen')as popen:
            with pytest.raises(ValueError):manager.perform(body)
        popen.assert_not_called()

def test_missing_live_connection_does_not_offer_a_placeholder_command(manager):
    with patch('showonce_studio.robot_actions.subprocess.Popen')as spawn:
        plan=manager.perform({'kind':'policy','policy_id':'fixture','mode':'run','execution':'command'})
    spawn.assert_not_called()
    assert not plan['command_ready']
    assert plan['display_command']is None
    assert plan['live_inputs']['reasons']

def test_connected_command_uses_the_exact_live_folder_without_starting_worker(manager,tmp_path):
    from showonce_studio.showonce_policies import sha
    from showonce_studio.showonce_stages import write_json
    folder=tmp_path/'live folder';folder.mkdir();calibration=tmp_path/'calibration.json';write_json(calibration,{'trusted':True})
    write_json(folder/'manifest.json',{'mode':'live_camera_only','calibration':{'path':str(calibration),'sha256':sha(calibration)}})
    write_json(folder/'tracked_poses.json',{'offline_replay':False,'sample_time_s':time.time()})
    with patch('showonce_studio.robot_actions.subprocess.Popen')as spawn:
        plan=manager.perform({'kind':'policy','policy_id':'fixture','mode':'run','execution':'command',
            'perception_dir':str(folder),'calibration':str(calibration)})
    spawn.assert_not_called();assert plan['command_ready']
    assert str(folder)in plan['display_command']and '<실시간 인식 폴더>'not in plan['display_command']
    assert '--out'not in plan['display_command']and '--job-token'not in plan['display_command']

def test_live_trajectory_cursor_keeps_partial_lines_and_excludes_mock(manager):
    job_id='a'*32;folder=manager.folder(job_id);folder.mkdir()
    live={'input_kind':'live','updated_s':1.,'block_pos_m':[.3,.1,.01],'tcp_pos_m':[.3,0,.1]}
    path=folder/'policy-inputs.jsonl'
    path.write_bytes((json.dumps(live)+'\n'+json.dumps({'input_kind':'mock'})+'\n'+json.dumps(live)[:12]).encode())
    job={'id':job_id,'kind':'policy','mode':'run','policy_id':'fixture','alive':True}
    with patch.object(manager,'job',return_value=job):
        first=manager.trajectory(job_id)
        assert first['samples']==[live]
        second=manager.trajectory(job_id,first['next_offset']);assert not second['samples']
        with path.open('ab')as stream:stream.write((json.dumps(live)[12:]+'\n').encode())
        third=manager.trajectory(job_id,second['next_offset']);assert third['samples']==[live]
        with pytest.raises(ValueError):manager.trajectory(job_id,-1)
        with pytest.raises(ValueError):manager.trajectory(job_id,path.stat().st_size+10)
    with patch.object(manager,'job',return_value=dict(job,mode='mock')):
        assert not manager.trajectory(job_id)['samples']

def test_refresh_recovers_process_and_ctrl_c_stops_it(manager):
    value=manager.perform({'kind':'policy','policy_id':'fixture','mode':'mock','execution':'direct','seconds':4})
    job=value['job'];time.sleep(.1)
    restored=RobotActions(manager.recordings,manager.library,jobs=manager.jobs)
    assert restored.job(job['id'])['alive'];assert 'fake backend'in restored.logs(job['id'])['text']
    restored.interrupt(job['id']);result=wait_done(manager,job['id'])
    assert result['status']=='cancelled'and result['report']['stopped_by_request'];assert not result['report']['hardware_commands_sent']

def test_pid_reuse_does_not_signal_an_unowned_process(manager):
    value=manager.perform({'kind':'policy','policy_id':'fixture','mode':'mock','execution':'direct','seconds':4});job=value['job'];time.sleep(.1)
    path=manager.folder(job['id'])/'job.json';meta=json.loads(path.read_text());bad=dict(meta,process_birth='incorrect');path.write_text(json.dumps(bad))
    with patch('showonce_studio.robot_actions.os.killpg')as kill:manager.interrupt(job['id'])
    kill.assert_not_called();path.write_text(json.dumps(meta));manager.interrupt(job['id']);wait_done(manager,job['id'])

def test_sdk_stop_preempts_owned_policy_sender(manager,tmp_path):
    cal=tmp_path/'cal.json';cal.write_text(json.dumps({'trusted':True}));live=tmp_path/'live';live.mkdir()
    from showonce_studio.showonce_policies import sha
    (live/'manifest.json').write_text(json.dumps({'mode':'live_camera_only','calibration':{'path':str(cal),'sha256':sha(cal)}}))
    (live/'tracked_poses.json').write_text(json.dumps({'offline_replay':False,'sample_time_s':time.time()}))
    first=manager.perform({'kind':'policy','policy_id':'fixture','mode':'run','execution':'direct','seconds':4,'calibration':str(cal),'perception_dir':str(live)})['job'];time.sleep(.1)
    with pytest.raises(ValueError):manager.perform({'kind':'sdk','action':'enable','execution':'direct'})
    stop=manager.perform({'kind':'sdk','action':'stop','execution':'direct'})['job'];assert not manager.job(first['id'])['alive']
    assert wait_done(manager,stop['id'])['status']=='complete'

def test_policy_ctrl_c_is_responsive_during_perception_preflight(tmp_path,monkeypatch):
    import os,threading
    from piper_runtime.transports import robot_io
    from piper_runtime.resources import MODEL_XML
    from showonce_studio.showonce_policy_runtime import Adapter,execute
    from showonce_studio.showonce_policies import sha
    folder=tmp_path/'policy';folder.mkdir();meta=json.loads(Path(__file__).with_name('policy-fixture.json').read_text())
    import numpy as np
    np.savez(folder/'weights.npz',**{'normalizer/mean':np.zeros(49,np.float32),'normalizer/std':np.ones(49,np.float32),'params/hidden_0/kernel':np.zeros((49,14),np.float32),'params/hidden_0/bias':np.zeros(14,np.float32)})
    meta['weights_sha256']=sha(folder/'weights.npz');(folder/'policy.json').write_text(json.dumps(meta));(folder/'piper.xml').write_bytes(MODEL_XML.read_bytes())
    adapter=Adapter(folder);monkeypatch.setattr(robot_io,'SDKRobot',lambda channel:robot_io.MockRobot(adapter.environment['nominal_qpos']))
    monkeypatch.setattr(robot_io,'PerceptionProcess',lambda *args,**kwargs:robot_io.MockVision())
    cal=tmp_path/'cal.json';cal.write_text(json.dumps({'trusted':True,'table_check':{'height_m':-.015}}));out=tmp_path/'out'
    original_affinity=os.sched_getaffinity(0);handlers={s:signal.getsignal(s)for s in (signal.SIGINT,signal.SIGTERM)}
    timer=threading.Timer(.15,lambda:os.kill(os.getpid(),signal.SIGINT));timer.start();start=time.monotonic()
    try:result=execute(folder,'shadow',2,out,cal,tmp_path)
    finally:
        timer.join();os.sched_setaffinity(0,original_affinity)
        for sig,handler in handlers.items():signal.signal(sig,handler)
    assert time.monotonic()-start<2
    assert result['success']and result['stopped_by_request']and not result['hardware_commands_sent']


def test_immediate_false_sdk_return_uses_fresh_feedback_confirmation():
    class ImmediateFalse(FakeRobot):
        def enable(self):super().enable();return False
        def move_gripper_m(self,value,force):super().move_gripper_m(value,force);return False
    robot=ImmediateFalse();report=run_command('enable','test-early-ack',timeout=.2,robot_factory=lambda:robot)
    assert report['success'],report
    assert any(call.get('sdk_return')is False for call in report['calls'])
    assert report['state']['get_joints_enable_status_list']==[True]*6

def test_manual_direction_checks_and_release_cannot_be_rearmed(manager):
    job=manager.perform({'kind':'sdk','action':'manual','seconds':4.,'execution':'direct'})['job']
    folder=manager.folder(job['id']);token='a'*32
    from showonce_studio.showonce_stages import write_json,read_json
    write_json(folder/'progress.json',{'phase':'ready','updated_s':time.time(),'available':{'z-':{'allowed':False,'reason':'table'},'z+':{'allowed':True}}})
    try:
        body=dict(id=job['id'],operation_id=token,action='jog',axis='z',direction=-1,speed_mm_s=10.)
        with pytest.raises(ValueError,match='table'):manager.manual_input(body)
        body['direction']=1
        assert manager.manual_input(body)['accepted']
        manager.manual_input({'id':job['id'],'action':'release','operation_id':token})
        assert manager.manual_input(body)['released']
        assert read_json(folder/'manual-input.json')['released']
        delayed='c'*32
        manager.manual_input({'id':job['id'],'action':'release','operation_id':delayed})
        assert manager.manual_input(dict(body,operation_id=delayed))['released']
        write_json(folder/'manual-lease.json',{'expires_monotonic':0})
        with pytest.raises(ValueError,match='만료'):manager.manual_input({'id':job['id'],'action':'ping'})
    finally:manager.interrupt(job['id']);wait_done(manager,job['id'])

def test_manual_command_views_keep_actual_axes_width_and_force(manager):
    values=[{'action':'jog','axis':'z','direction':-1,'speed_mm_s':8.}, {'action':'rotate','axis':'yaw','direction':1}, {'action':'gripper','width_mm':37.,'force':.5}]
    for body in values:
        with patch('showonce_studio.robot_actions.subprocess.Popen')as popen:
            result=manager.perform(dict(kind='sdk',execution='command',**body))
        popen.assert_not_called()
        assert '--out'not in result['display_command']and '--job-token'not in result['display_command']
        assert result['mode']==body['action']

def test_standalone_gripper_sets_only_explicit_width_and_force():
    robot=FakeRobot();robot.code=64
    report=run_command('gripper','test-width-control',width_mm=35.,force=.5,timeout=.2,robot_factory=lambda:robot)
    assert report['success'],report
    assert ('gripper',.035,.5)in robot.calls
    assert not any(call[0]in ('enable','set_motion_mode','reset')for call in robot.calls)

def test_manual_worker_receives_web_camera_calibration_without_client_path(manager,tmp_path):
    camera_calibration=tmp_path/'camera-current.json'
    camera_calibration.write_text(json.dumps({'trusted':True,'table_check':{'height_m':-.0136}}))
    manager.calibration=camera_calibration
    assert manager.state()['calibration']==str(camera_calibration)
    for action in ('manual','jog','rotate'):
        body=dict(kind='sdk',action=action,execution='command')
        if action!='manual':body.update(axis='z'if action=='jog'else 'yaw',direction=1)
        plan=manager.perform(body)
        assert plan['argv'][plan['argv'].index('--calib')+1]==str(camera_calibration)
    override=tmp_path/'explicit.json'
    plan=manager.perform(dict(body,calibration=str(override)))
    assert plan['argv'][plan['argv'].index('--calib')+1]==str(override)

def test_jog_accepts_50_and_rejects_old_session_scale(manager):
    plan=manager.perform(dict(kind='sdk',action='jog',axis='x',direction=1,speed_mm_s=50.,execution='command'))
    assert plan['argv'][plan['argv'].index('--speed-mm-s')+1]=='50.0'
    job=manager.perform(dict(kind='sdk',action='manual',seconds=4.,execution='direct'))['job']
    from showonce_studio.showonce_stages import write_json
    progress=dict(phase='ready',updated_s=time.time(),available={'x+':{'allowed':True}})
    folder=manager.folder(job['id']);write_json(folder/'progress.json',progress)
    body=dict(id=job['id'],operation_id='d'*32,action='jog',axis='x',direction=1,speed_mm_s=50.)
    try:
        with pytest.raises(ValueError,match='이전 속도'):manager.manual_input(body)
        write_json(folder/'progress.json',dict(progress,jog_speed_limit_mm_s=50.))
        assert manager.manual_input(body)['accepted']
    finally:manager.interrupt(job['id']);wait_done(manager,job['id'])

def test_unwind_command_and_preview_preserve_execution_choice(manager):
    for preview in (True,False):
        with patch('showonce_studio.robot_actions.subprocess.Popen')as popen:
            result=manager.perform(dict(kind='sdk',action='unwind',plan_only=preview,execution='command',seconds=60))
        popen.assert_not_called()
        assert ('--plan-only'in result['argv'])==preview
        assert result['hardware_motion_requested']==(not preview)

def test_old_manual_worker_cannot_receive_position_fixed_unwind(manager):
    job=manager.perform(dict(kind='sdk',action='manual',seconds=4.,execution='direct'))['job']
    from showonce_studio.showonce_stages import write_json,read_json
    progress=dict(phase='ready',updated_s=time.time())
    folder=manager.folder(job['id']);write_json(folder/'progress.json',progress)
    body=dict(id=job['id'],operation_id='e'*32,action='unwind',plan_only=False)
    try:
        with pytest.raises(ValueError,match='연결 종료'):manager.manual_input(body)
        assert not (folder/'manual-input.json').exists()
        write_json(folder/'progress.json',dict(progress,unwind_position_fixed=True))
        assert manager.manual_input(body)['accepted']
        assert read_json(folder/'manual-input.json')['keep_tcp_position']is True
    finally:manager.interrupt(job['id']);wait_done(manager,job['id'])
