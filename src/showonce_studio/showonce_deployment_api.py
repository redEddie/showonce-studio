from showonce_studio.config import worker_python
"""Web-to-policy workers. Web operations cannot launch robot shadow/run."""
from pathlib import Path
import datetime as dt
import json
import math
import os
import re
import shlex
import signal
import subprocess
import threading
import time
import uuid
from showonce_studio.config import ROOT,runtime_env
from showonce_studio.showonce_policies import PolicyLibrary,PROFILE,MODEL_XML,sha
from showonce_studio.showonce_stages import recording_path,read_json,write_json
from showonce_studio.live_perception import LivePerception

JOBS=ROOT/'runs/showonce-deployment'

def live_inputs(perception_dir,calibration):
    reasons=[]
    try:
        folder=Path(perception_dir);calib=Path(calibration);manifest=read_json(folder/'manifest.json',{})
        snapshot=read_json(folder/'tracked_poses.json',{});value=read_json(calib,{})
        if manifest.get('mode')!='live_camera_only'or snapshot.get('offline_replay',True):reasons.append('실행 중인 실시간 인식 출력이 필요합니다.')
        stamp=float(snapshot.get('sample_time_s',0));age=time.time()-stamp
        if not -.01<=age<=1.:reasons.append('실시간 인식 신호가 최신 상태가 아닙니다.')
        service_path=Path(manifest.get('calibration',{}).get('path',''))
        if not value.get('trusted')or not service_path.is_file()or read_json(service_path)!=value:reasons.append('실시간 인식과 현재 카메라 보정이 일치해야 합니다.')
        if service_path.is_file()and sha(service_path)!=manifest.get('calibration',{}).get('sha256'):reasons.append('인식 노드의 보정 파일이 변경됐습니다.')
    except (OSError,TypeError,ValueError):reasons.append('실시간 인식 폴더와 현재 보정 파일을 지정하세요.')
    return {'inputs_ready':not reasons,'reasons':reasons,'robot_feedback_checked':False}


class DeploymentManager:
    def __init__(self,recordings,library=None,jobs=JOBS):
        self.root=Path(recordings);self.library=library or PolicyLibrary();self.jobs=Path(jobs);self.lock=threading.Lock();self.processes={}
        self.perception=LivePerception(ROOT)
    def catalog(self):
        result=[]
        for policy in self.library.catalog():
            latest=read_json(self.jobs/policy['id']/'latest.json',{})
            history={mode:self.job(policy['id'],job_id)for mode,job_id in latest.items()}
            # The SDK/policy process manager owns newer jobs; include its persisted
            # reports so changing policy never displays another policy's checks.
            for path in (ROOT/'runs/showonce-robot').glob('*/job.json'):
                job=read_json(path,{})
                if job.get('policy_id')!=policy['id']or job.get('mode')not in ('validate','mock','replay','shadow','run'):continue
                report=read_json(path.parent/'report.json')
                if report and job.get('started_s',0)>history.get(job['mode'],{}).get('started_s',0):
                    history[job['mode']]={**job,'status':'error'if report.get('status')=='error'or report.get('success')is False else 'complete',
                        'report':{k:v for k,v in report.items()if k!='samples'}}
            validation=history.get('validate',{})
            result.append({**policy,'latest_jobs':history,'validation':
                'passed'if validation.get('status')=='complete'else 'failed'if validation.get('status')=='error'else 'pending'})
        result.sort(key=lambda value:(not value.get('is_final',False),not value.get('recommended',False),-value.get('created_s',0)))
        # Recommendation is presentation state; immutable old package manifests
        # remain untouched when a newer recommended release is registered.
        recommended=next((p['id']for p in result if p.get('recommended')),None)
        for item in result:item['recommended']=item['id']==recommended
        candidates=[]
        for path in (ROOT/'runs').glob('show100-vision-*/manifest.json'):
            manifest=read_json(path,{})
            calib=manifest.get('calibration',{}).get('path')
            if manifest.get('mode')!='live_camera_only'or not calib:continue
            state=live_inputs(path.parent,calib)
            if state['inputs_ready']:candidates.append({'perception_dir':str(path.parent),'calibration':calib,**state})
        return {'policies':result,'runtime_profile':PROFILE,'live_inputs':candidates,
            'registration_example':str(ROOT/'artifacts/exported-policy')}
    def folder(self,policy_id,job_id):
        self.library.folder(policy_id)
        if not isinstance(job_id,str)or not re.fullmatch(r'[a-f0-9]{32}',job_id):raise ValueError('잘못된 실행 ID입니다.')
        path=(self.jobs/policy_id/job_id).resolve()
        if self.jobs.resolve()not in path.parents:raise ValueError('실행 기록 폴더 밖입니다.')
        return path
    def job(self,policy_id,job_id):
        folder=self.folder(policy_id,job_id);job=read_json(folder/'job.json')
        if not job:raise ValueError('실행 기록이 없습니다.')
        report=read_json(folder/'report.json')
        if report:
            status='error'if report.get('status')=='error'or report.get('success')is False else 'complete'
            return {**job,'status':status,'report':{k:v for k,v in report.items()if k!='samples'}}
        process=self.processes.get(job_id)
        command=Path(f'/proc/{job["pid"]}/cmdline')
        if (process is not None and process.poll()is not None)or not command.is_file()or b'showonce_studio.showonce_policy_runtime'not in command.read_bytes():return {**job,'status':'error','error':'작업이 종료됐지만 결과 파일이 없습니다.'}
        return job
    def register(self,source,name,model_xml=None):
        with self.lock:value=self.library.register(source,name,model_xml or MODEL_XML)
        return value
    def start(self,policy_id,mode,seconds=2.,recording_id=None):
        if mode not in ('validate','mock','replay'):raise ValueError('웹에서는 정책 검증·입력 재생·가짜 로봇 모의 실행만 시작할 수 있습니다.')
        seconds=float(seconds)
        if not math.isfinite(seconds)or not 0<seconds<=10:raise ValueError('웹 모의 실행 시간은 0..10초입니다.')
        self.library.get(policy_id)
        recording=recording_path(self.root,recording_id)[0]if mode=='replay'else None
        with self.lock:
            job_id=uuid.uuid4().hex;folder=self.folder(policy_id,job_id);folder.mkdir(parents=True)
            command=[str(worker_python('policy')),'-P','-u','-m', 'showonce_studio.showonce_policy_runtime',mode,'--policy-id',policy_id,'--out',str(folder),'--seconds',str(seconds)]
            if recording:command+=['--recording',str(recording)]
            with (folder/'console.log').open('w')as log:
                process=subprocess.Popen(command,cwd=ROOT,env=dict(runtime_env(),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            self.processes[job_id]=process
            write_json(folder/'job.json',{'id':job_id,'policy_id':policy_id,'mode':mode,'status':'running','pid':process.pid,'started_s':time.time(),'hardware_motion_requested':False})
            latest_path=self.jobs/policy_id/'latest.json';latest=read_json(latest_path,{})
            latest[mode]=job_id;write_json(latest_path,latest);return self.job(policy_id,job_id)
    def stop(self,policy_id,job_id):
        job=self.job(policy_id,job_id)
        if job['mode']not in ('mock','replay')or job['status']!='running':return job
        command=Path(f'/proc/{job["pid"]}/cmdline')
        if command.is_file()and b'showonce_studio.showonce_policy_runtime'in command.read_bytes():os.kill(job['pid'],signal.SIGTERM)
        return self.job(policy_id,job_id)
    def readiness(self,recording_id):
        recording,_=recording_path(self.root,recording_id);task=read_json(recording/'task_definition.json',{})
        asset=read_json(recording/'geometry/sam3d_assets_v1/manifest.json',{})
        return {'recording_id':recording_id,'training_ready_in_studio':False,'items':[
            {'name':'RGB-D·에셋·포즈·관계 분석','state':'ready'if(recording/'relations/scene_graph_v1/scene_graph.json').is_file()else 'pending'},
            {'name':'작업 스테이지 확정','state':'ready'if task.get('review_status')=='confirmed'else 'pending'},
            {'name':'실측 치수·질량·관성·마찰 및 로봇 포함 학습 환경','state':'external','detail':'형상 미리보기와 물체 낙하 검증만으로 학습 환경 검증이 끝나지는 않습니다.'},
            {'name':'관계 → 보상·종료·리셋 조건 연결','state':'external','detail':'별도 강화학습 프로젝트에서 정의·검증합니다.'},
            {'name':'체크포인트 내보내기','state':'external','detail':'관측 순서·정규화·행동 계약과 golden 검증 벡터를 포함해야 합니다.'},
            {'name':'실시간 인식·실측 로봇 피드백 연결 및 실기 검증','state':'pending','detail':'Studio의 저장 영상 분석과 별도의 실행 경로입니다.'}],
            'policy_execution_preparation':'registration, parity checks, recorded-input inspection, mock QP execution and terminal command plans',
            'physics_parameters_measured':bool(asset.get('physics_parameters_measured',False))}
    def plan(self,policy_id,mode='shadow',seconds=2.,calibration=None,perception_dir=None,channel='can_piper'):
        selected=self.library.get(policy_id)
        if mode not in ('shadow','run'):raise ValueError('shadow 또는 run 계획을 선택하세요.')
        seconds=float(seconds)
        if not math.isfinite(seconds)or not 0<seconds<=60:raise ValueError('실행 시간은 0..60초입니다.')
        if not isinstance(channel,str)or not re.fullmatch(r'[a-zA-Z0-9_-]{1,40}',channel):raise ValueError('잘못된 CAN 채널입니다.')
        inputs=live_inputs(perception_dir,calibration)
        output=ROOT/'runs'/('showonce-live-'+uuid.uuid4().hex[:12])
        argv=['-P','-m', 'showonce_studio.showonce_policy_runtime',mode,'--policy-id',policy_id,'--seconds',str(seconds),'--channel',channel,
            '--calib',str(calibration or '<현재 보정.json>'),'--perception-dir',str(perception_dir or '<실시간 인식 폴더>'),'--out',str(output)]
        result={'schema':'showonce_execution_plan_v1','policy_id':policy_id,'mode':mode,'runtime_profile':selected['contract']['runtime_profile'],
            'command':shlex.join([str(worker_python('policy')),*argv]),'argv':[str(worker_python('policy')),*argv],
            'live_inputs':inputs,'hardware_motion_requested':False,'hardware_commands_sent':False,
            'hardware_preflight_required':['fresh enabled/CAN-mode joint and gripper feedback','matching live calibration','stationary posture A and open gripper','firmware acceleration >=5rad/s²'],
            'scope':'Command preparation only. No SDK connection, mode change, enable or robot motion.'}
        write_json(self.jobs/'plans'/(uuid.uuid4().hex+'.json'),result);return result
