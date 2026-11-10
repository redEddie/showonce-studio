from showonce_studio.config import worker_python
"""Job launcher and checked file access for simulation assets; no heavyweight imports."""
from pathlib import Path
import re
import subprocess
import threading
import urllib.parse
from showonce_studio.config import ROOT,runtime_env
from showonce_studio.showonce_stages import read_json,recording_path
from showonce_studio.showonce_sim_assets import directory,checkpoint_status


class SimulationAssetsManager:
    def __init__(self,root):self.root=Path(root);self.lock=threading.Lock();self.process=None

    def state(self,recording_id):
        recording,_=recording_path(self.root,recording_id);folder=directory(recording)
        inputs=read_json(folder/'inputs.json')
        return {'recording_id':recording_id,'checkpoints':checkpoint_status(),
            'runtime':read_json(ROOT/'release/sam3d-runtime.json',{'verified':False}),
            'job':read_json(folder/'job.json',{'status':'idle'}),'manifest':read_json(folder/'manifest.json'),
            'inputs_ready':inputs is not None,
            'keyframes':[{ 'id':o['id'],'name':o['name'],'frames':[e['index'] for e in o['keyframes']]} for o in (inputs or {}).get('objects',[])]}

    def start(self,recording_id):
        recording,_=recording_path(self.root,recording_id)
        with self.lock:
            current=self.state(recording_id)
            if current['manifest'] and current['manifest'].get('complete'):return current
            if not current['runtime'].get('verified'):raise ValueError('SAM 3D 실행 환경 설치·검증 중입니다.')
            if self.process is not None and self.process.poll() is None:raise ValueError('SAM 3D 에셋 작업이 이미 실행 중입니다.')
            # Reuse jobs started before a web restart instead of launching duplicates.
            job=current['job'];pid=job.get('pid')
            if pid and Path(f'/proc/{pid}/cmdline').is_file():
                command=Path(f'/proc/{pid}/cmdline').read_bytes()
                if b'showonce_sim_assets.py' in command and b'run' in command:
                    raise ValueError('SAM 3D 에셋 작업이 이미 실행 중입니다.')
            with (ROOT/'runs/showonce-sim-assets.log').open('a') as log:
                self.process=subprocess.Popen([str(worker_python('sam3d')),'-u','-m', 'showonce_studio.showonce_sim_assets',
                    'run','--recording',str(recording.resolve())],cwd=ROOT,
                    env=dict(runtime_env(),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',ATTN_BACKEND='sdpa',SPARSE_ATTN_BACKEND='sdpa',
                        TRITON_PTXAS_BLACKWELL_PATH='/usr/local/cuda/bin/ptxas'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            return self.state(recording_id)

    def artifact(self,recording_id,name):
        recording,_=recording_path(self.root,recording_id)
        allowed=name in ('manifest.json','inputs.json','scene.json','scene.usd','scene.xml','physics-report.json','simulation-assets.zip') or re.fullmatch(
            r'assets/[a-z0-9_]{1,64}/(mesh.glb|mesh.ply|mesh.obj|measured.ply|points.json|thumbnail.png|asset.json)',name) or re.fullmatch(
            r'objects/[a-z0-9_]{1,64}/(obj.usd|colliders/part_[0-9]{3}.obj)',name)
        if not allowed:raise ValueError('잘못된 시뮬레이션 에셋 경로입니다.')
        root=directory(recording).resolve();path=(root/name).resolve()
        if root not in path.parents:raise ValueError('에셋 폴더 밖의 파일입니다.')
        return path
