"""Owned SDK/policy subprocesses, persistent job state and explicit execution choice."""
from pathlib import Path
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
from .config import ROOT,runtime_env,worker_python
from .showonce_stages import read_json,write_json,recording_path
from .showonce_deployment_api import live_inputs
from .robot_sdk_support import MODES,SELECTABLE

ACTIVE={'running','stopping'}

def process_birth(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    except (OSError,IndexError):return None

class RobotActions:
    def __init__(self,recordings,library,jobs=None,sdk_command=None,policy_command=None,environment=None,offline=False,calibration=None):
        self.recordings=Path(recordings);self.library=library;self.jobs=Path(jobs or ROOT/'runs/showonce-robot');self.jobs.mkdir(parents=True,exist_ok=True)
        self.sdk_command=sdk_command or [str(worker_python('robot')),'-P','-u','-m','showonce_studio.showonce_robot']
        self.policy_command=policy_command or [str(worker_python('policy')),'-P','-u','-m','showonce_studio.showonce_policy_runtime']
        self.environment=environment or runtime_env;self.offline=offline;self.lock=threading.RLock();self.processes={}
        self.calibration=Path(calibration).expanduser().resolve()if calibration else None
    def folder(self,job_id):
        if not isinstance(job_id,str)or not re.fullmatch(r'[a-f0-9]{32}',job_id):raise ValueError('잘못된 작업 ID입니다.')
        return self.jobs/job_id
    def owned(self,job):
        pid=job['pid']
        try:return process_birth(pid)==job.get('process_birth')and str(self.folder(job['id'])).encode()in Path(f'/proc/{pid}/cmdline').read_bytes()
        except OSError:return False
    def _job(self,job_id):
        folder=self.folder(job_id);meta=read_json(folder/'job.json')
        if not meta:raise ValueError('작업 기록이 없습니다.')
        report=read_json(folder/'report.json');progress=read_json(folder/'progress.json',{})
        process=self.processes.get(job_id);returncode=process.poll()if process else None;alive=self.owned(meta)
        if report and not alive:status='cancelled'if report.get('stopped_by_request')else 'complete'if report.get('success',report.get('status')=='passed')else 'error'
        elif alive:status='stopping'if meta.get('stop_requested_s')else 'running'
        elif meta.get('stop_requested_s'):status='cancelled'
        else:status='error'
        result={**meta,'status':status,'alive':alive,'returncode':returncode,'elapsed_s':round((report.get('completed_s',time.time())if report else meta.get('ended_s',time.time()))-meta['started_s'],2),'progress':progress}
        clock=read_json(folder/'run-clock.json',{})
        if clock:result.update(control_elapsed_s=max(0.,time.time()-clock['started_s']),remaining_s=max(0.,clock['seconds']-(time.time()-clock['started_s'])))
        if (folder/'watchdog.json').is_file():result.update(status='error',error='프로세스 응답 제한 시간을 초과해 종료했습니다. SDK 요청 결과와 로그를 확인하세요.')
        if report:result['report']={k:v for k,v in report.items()if k not in ('samples',)}
        if meta['kind']=='policy':result['policy_input']=read_json(folder/'policy-input.json',{})
        if not alive and not report:result['error']='프로세스가 종료됐습니다. 로그를 확인하세요.'
        return result
    def job(self,job_id):
        with self.lock:return self._job(job_id)
    def trajectory(self,job_id,after=0):
        job=self.job(job_id);offset=int(after)
        if offset<0:raise ValueError('잘못된 궤적 위치입니다.')
        result={'job_id':job_id,'policy_id':job.get('policy_id'),'mode':job['mode'],
            'samples':[],'next_offset':offset,'alive':job['alive']}
        if job['kind']!='policy'or job['mode']not in ('run','shadow'):return result
        path=self.folder(job_id)/'policy-inputs.jsonl'
        if not path.is_file():return result
        with path.open('rb')as stream:
            if offset>path.stat().st_size:raise ValueError('궤적 커서가 파일 범위를 벗어났습니다.')
            stream.seek(offset)
            for _ in range(256):
                before=stream.tell();line=stream.readline(16384)
                if not line.endswith(b'\n'):
                    stream.seek(before);break
                value=json.loads(line)
                if value.get('input_kind')=='live':result['samples'].append(value)
            result['next_offset']=stream.tell()
        return result
    def state(self):
        values=[]
        for p in self.jobs.glob('*/job.json'):
            try:values.append(self.job(p.parent.name))
            except (ValueError,OSError):continue
        values.sort(key=lambda j:(j['status']in ACTIVE,j['started_s']),reverse=True)
        return {'jobs':values[:20],'active_jobs':[j for j in values if j['status']in ACTIVE],
            'modes':[{'value':value,'label':label,'selectable':value in SELECTABLE}for value,label in MODES.items()],
            'default_execution':'command','offline':self.offline,'calibration':str(self.calibration)if self.calibration else None}
    def logs(self,job_id):
        self.job(job_id);path=self.folder(job_id)/'console.log'
        if not path.exists():return {'text':''}
        with path.open('rb')as stream:stream.seek(max(0,path.stat().st_size-16384));value=stream.read().decode('utf-8',errors='replace')
        return {'text':value}
    def display_command(self,base,argv):
        prefix=['python','-m',base[base.index('-m')+1]]if '-m'in base else ['python',Path(base[-1]).name]
        args=argv[len(base):];visible=[];index=0
        while index<len(args):
            item=args[index]
            if item in ('--out','--job-token'):
                index+=2;continue
            if item=='--channel'and args[index+1]=='can_piper':
                index+=2;continue
            visible.append(item);index+=1
        return shlex.join(prefix+visible)
    def plan(self,body):
        kind=body.get('kind');channel=body.get('channel','can_piper')
        if not isinstance(channel,str)or not re.fullmatch(r'[A-Za-z0-9_-]{1,40}',channel):raise ValueError('잘못된 CAN 채널입니다.')
        job_id=uuid.uuid4().hex;folder=self.folder(job_id);live=None;seconds=float(body.get('seconds',60. if kind=='sdk'and body.get('action')=='home'else 2.))
        limit=600 if kind=='sdk'and body.get('action')=='manual'else 60
        if not math.isfinite(seconds)or not 0<seconds<=limit:raise ValueError(f'실행 시간은 0초 초과, 최대 {limit}초입니다.')
        if kind=='sdk':
            action=body.get('action')
            if action not in ('home','read','enable','disable','stop','reset','mode','manual','jog','rotate','gripper','unwind'):raise ValueError('지원하지 않는 SDK 명령입니다.')
            command=[*self.sdk_command,action,'--channel',channel,'--out',str(folder),'--job-token',job_id]
            if action=='mode':
                value=body.get('mode_value')
                if type(value)is not int or value not in SELECTABLE:raise ValueError('SDK로 선택 가능한 모드는 0, 1, 3, 4, 7입니다.')
                command+=['--value',str(value)]
            if action in ('home','manual','jog','rotate','unwind'):
                command+=['--seconds',str(seconds)]
                calibration=body.get('calibration')or self.calibration
                if calibration:command+=['--calib',str(calibration)]
                if action=='home'and body.get('policy_id'):
                    self.library.get(body['policy_id']);command+=['--policy-id',body['policy_id']]
            if action in ('jog','rotate'):
                parameters=manual_parameters(body,action)
                command+=['--axis',parameters['axis'],'--direction',str(parameters['direction'])]
                if action=='jog':command+=['--speed-mm-s',str(parameters['speed_mm_s'])]
            if action=='gripper':
                parameters=manual_parameters(body,action)
                command+=['--width-mm',str(parameters['width_mm']),'--force',str(parameters['force'])]
            if action=='manual':command+=['--leased']
            if action=='unwind'and manual_parameters(body,action)['plan_only']:command+=['--plan-only']
            mode=action;hardware=action!='read'and not(action=='unwind'and body.get('plan_only'))
        elif kind=='policy':
            policy_id=body.get('policy_id');policy=self.library.get(policy_id);mode=body.get('mode')
            if mode not in ('validate','mock','replay','shadow','run'):raise ValueError('지원하지 않는 정책 실행입니다.')
            command=[*self.policy_command,mode,'--policy-id',policy_id,'--seconds',str(seconds),'--out',str(folder),'--channel',channel,'--job-token',job_id]
            if mode=='replay':command+=['--recording',str(recording_path(self.recordings,body.get('recording_id'))[0])]
            if mode in ('shadow','run'):
                live=live_inputs(body.get('perception_dir'),body.get('calibration'))
                command+=['--calib',str(body.get('calibration')or '<현재 보정.json>'),'--perception-dir',str(body.get('perception_dir')or '<실시간 인식 폴더>')]
            hardware=mode=='run'
        else:raise ValueError('SDK 또는 policy 종류를 선택하세요.')
        ready=live is None or live['inputs_ready']
        return {'id':job_id,'kind':kind,'mode':mode,'channel':channel,'seconds':seconds,'argv':command,'command':shlex.join(command),
            'command_ready':ready,'display_command':self.display_command(self.sdk_command if kind=='sdk'else self.policy_command,command)if ready else None,
            'execution_metadata':{'working_directory':str(ROOT),'output_directory':str(folder),'job_token':job_id},'live_inputs':live,'hardware_motion_requested':hardware,'process_started':False,**({'policy_name':policy.get('name',policy_id),'policy_id':policy_id}if kind=='policy'else {})}
    def perform(self,body):
        execution=body.get('execution','command')
        if execution not in ('command','direct'):raise ValueError('명령어 보기 또는 바로 실행을 선택하세요.')
        plan=self.plan(body)
        if execution=='command':return {'execution':'command',**plan}
        if self.offline and (plan['kind']=='sdk'or plan['mode']in ('run','shadow')):raise ValueError('오프라인 예제에서는 실제 SDK/로봇 연결을 실행하지 않습니다.')
        if plan['live_inputs']and not plan['live_inputs']['inputs_ready']:raise ValueError(' '.join(plan['live_inputs']['reasons']))
        with self.lock:
            active=self.state()['active_jobs'];same=[j for j in active if j['channel']==plan['channel']and ((j['kind']=='sdk'and j['mode']!='read')or j['mode']in ('shadow','run'))]
            if plan['kind']=='policy'and any(j['kind']=='policy'and j['channel']==plan['channel']for j in active):raise ValueError('정책 작업이 이미 실행 중입니다. 중단 또는 완료 후 실행하세요.')
            if plan['kind']=='sdk'and plan['mode']=='read'and any(j['kind']=='sdk'and j['mode']=='read'and j['channel']==plan['channel']for j in active):raise ValueError('상태 조회가 진행 중입니다.')
            priority=plan['kind']=='sdk'and plan['mode']in ('stop','disable')
            if priority:
                for job in same:self._halt(job)
            elif same and ((plan['kind']=='sdk'and plan['mode']!='read')or plan['mode']in ('shadow','run')):raise ValueError('같은 채널의 작업이 실행 중입니다. 정책 중단 후 실행하세요.')
            folder=self.folder(plan['id']);folder.mkdir()
            if plan['mode']=='manual':write_json(folder/'manual-lease.json',{'expires_monotonic':time.monotonic()+2.})
            meta={k:plan[k]for k in ('id','kind','mode','channel','seconds','argv','command','hardware_motion_requested')};meta.update(started_s=time.time(),policy_id=body.get('policy_id'),policy_name=plan.get('policy_name'))
            with (folder/'console.log').open('w')as log:
                process=subprocess.Popen(plan['argv'],cwd=ROOT,env=dict(self.environment(),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            self.processes[plan['id']]=process;meta.update(pid=process.pid,process_birth=process_birth(process.pid));write_json(folder/'job.json',meta)
            threading.Thread(target=self._watch,args=(plan['id'],),daemon=True).start()
            return {'execution':'direct','process_started':True,'job':self._job(plan['id'])}
    def _watch(self,job_id):
        process=self.processes[job_id];meta=read_json(self.folder(job_id)/'job.json')
        try:process.wait(timeout=meta['seconds']+20 if meta['mode']in ('home','manual','jog','rotate','unwind')else 15 if meta['kind']=='sdk'else meta['seconds']+150)
        except subprocess.TimeoutExpired:
            with self.lock:
                job=self._job(job_id)
                if self.owned(job):
                    try:os.killpg(job['pid'],signal.SIGKILL)
                    except ProcessLookupError:pass
                    process.wait()
                    write_json(self.folder(job_id)/'watchdog.json',{'reason':'worker_timeout','at_s':time.time()})
        finally:
            with self.lock:
                value=read_json(self.folder(job_id)/'job.json');value['ended_s']=time.time();write_json(self.folder(job_id)/'job.json',value)
    def _halt(self,job):
        if not self.owned(job):return
        folder=self.folder(job['id']);(folder/'halt.now').touch();meta=read_json(folder/'job.json');meta['stop_requested_s']=time.time();meta['stop_kind']='sdk_priority';write_json(folder/'job.json',meta)
        try:os.killpg(job['pid'],signal.SIGINT)
        except ProcessLookupError:return
        deadline=time.monotonic()+.3
        while self.owned(job)and time.monotonic()<deadline:time.sleep(.01)
        if self.owned(job):
            try:os.killpg(job['pid'],signal.SIGKILL)
            except ProcessLookupError:pass
    def interrupt(self,job_id):
        with self.lock:
            job=self._job(job_id)
            if job['kind']!='policy'and job['mode']not in ('home','manual','jog','rotate','unwind'):raise ValueError('정책·Home·수동 TCP 작업의 중단 버튼을 사용하세요.')
            if not self.owned(job):return job
            meta=read_json(self.folder(job_id)/'job.json');meta['stop_requested_s']=time.time();meta['stop_kind']='sigint';write_json(self.folder(job_id)/'job.json',meta)
            try:os.killpg(job['pid'],signal.SIGINT)
            except ProcessLookupError:pass
            return self._job(job_id)

    def manual_input(self,body):
        with self.lock:
            job=self._job(body.get('id'))
            if job['mode']!='manual'or not job['alive']:raise ValueError('수동 제어 연결을 먼저 시작하세요.')
            folder=self.folder(job['id']);action=body.get('action')
            if action=='ping':
                # A released/expired connection cannot be brought back by delayed requests.
                old=read_json(folder/'manual-lease.json',{})
                if time.monotonic()>old.get('expires_monotonic',0):raise ValueError('수동 제어 연결이 만료되었습니다. 다시 연결하세요.')
                write_json(folder/'manual-lease.json',{'expires_monotonic':time.monotonic()+2.})
                return {'accepted':True}
            token=body.get('operation_id')
            if not isinstance(token,str)or not re.fullmatch(r'[a-f0-9]{32}',token):raise ValueError('잘못된 수동 조작 ID입니다.')
            previous=read_json(folder/'manual-input.json',{})
            released=read_json(folder/'manual-released.json',[])
            if action=='release':
                if token not in released:write_json(folder/'manual-released.json',(released+[token])[-128:])
                if previous.get('id')==token:
                    write_json(folder/'manual-input.json',dict(previous,released=True))
                return {'accepted':True}
            params=manual_parameters(body,action)
            if token in released:return {'accepted':False,'released':True}
            same=previous.get('id')==token
            if same and previous.get('released'):return {'accepted':False,'released':True}
            if same and any(previous.get(key)!=value for key,value in params.items()):raise ValueError('조작 중 방향·속도 변경은 새 버튼 입력으로 시작하세요.')
            progress=job['progress'];age=max(time.time()-progress.get('updated_s',0),time.time()-progress.get('feedback_s',progress.get('updated_s',0)))
            if age>1. or progress.get('phase')not in ('ready','moving'):raise ValueError('최신 수동 상태를 기다리세요.')
            if action=='unwind'and progress.get('unwind_position_fixed')is not True:
                raise ValueError('기존 수동 연결은 TCP 이동 방식입니다. 연결 종료 후 다시 연결해야 위치 고정 팔 풀기를 사용할 수 있습니다.')
            if action=='jog'and params['speed_mm_s']>progress.get('jog_speed_limit_mm_s',20):
                raise ValueError('현재 수동 연결은 이전 속도 상한을 사용합니다. 수동 연결 종료 후 다시 연결하세요.')
            if not same and progress.get('active_operation'):raise ValueError('진행 중인 수동 조작을 먼저 마치세요.')
            if action in ('jog','rotate'):
                key=params['axis']+('+'if params['direction']>0 else '-')
                if not progress.get('available',{}).get(key,{}).get('allowed'):raise ValueError(progress.get('available',{}).get(key,{}).get('reason')or '현재 자세에서 허용되지 않는 방향입니다.')
            write_json(folder/'manual-input.json',dict(id=token,action=action,expires_monotonic=time.monotonic()+.45,released=False,**params))
            return {'accepted':True}

def manual_parameters(body,action):
    def number(key,low,high,default=None):
        value=body.get(key,default)
        if isinstance(value,bool)or not isinstance(value,(int,float))or not math.isfinite(value)or not low<=value<=high:raise ValueError(f'{key}: {low}..{high} 사이 숫자가 필요합니다.')
        return float(value)
    if action=='unwind':
        value=body.get('plan_only',False)
        if type(value)is not bool:raise ValueError('plan_only는 true/false여야 합니다.')
        if body.get('keep_tcp_position',True)is not True:raise ValueError('팔 풀기는 TCP 위치 고정으로만 실행합니다.')
        return {'plan_only':value,'keep_tcp_position':True}
    if action in ('jog','rotate'):
        axis=body.get('axis');axes=('x','y','z')if action=='jog'else ('roll','pitch','yaw')
        if axis not in axes or type(body.get('direction'))is not int or body['direction']not in (-1,1):raise ValueError('올바른 이동 축과 방향을 선택하세요.')
        value=dict(axis=axis,direction=body['direction'])
        if action=='jog':value['speed_mm_s']=number('speed_mm_s',1,50,10.)
        return value
    if action=='gripper':return dict(width_mm=number('width_mm',0,100),force=number('force',.5,3,.5))
    raise ValueError('지원하지 않는 수동 입력입니다.')
