from showonce_studio.config import worker_python
"""Lightweight FoundationPose job and read-only artifact adapter."""
import json
from pathlib import Path
import re
import subprocess
import threading
from showonce_studio.config import ROOT,runtime_env
from showonce_studio.showonce_stages import recording_path,read_json
from showonce_studio.showonce_pose import directory


class PoseManager:
    def __init__(self,root):self.root=Path(root);self.lock=threading.Lock();self.process=None

    def state(self,recording_id):
        recording,_=recording_path(self.root,recording_id);folder=directory(recording)
        manifest=read_json(folder/'manifest.json')
        return {'recording_id':recording_id,'job':read_json(folder/'job.json',{'status':'idle'}),
            'manifest':manifest,'tracking_ready':bool(manifest and manifest.get('complete')and(folder/'tracking.json').is_file()),
            'references_ready':(recording/'geometry/sam3d_assets_v1/manifest.json').is_file(),
            'runtime_ready':(worker_python('foundationpose')).is_file()}

    def start(self,recording_id):
        recording,_=recording_path(self.root,recording_id)
        with self.lock:
            current=self.state(recording_id)
            if current['tracking_ready']:return current
            if not current['runtime_ready']:raise ValueError('FoundationPose 실행 환경이 없습니다.')
            if not current['references_ready']:raise ValueError('참조 에셋을 먼저 준비하세요.')
            if self.process is not None and self.process.poll()is None:raise ValueError('FoundationPose 추적 작업이 진행 중입니다.')
            pid=current['job'].get('pid');cmd=Path(f'/proc/{pid}/cmdline')if pid else None
            if cmd and cmd.is_file()and b'showonce_pose.py'in cmd.read_bytes():return current
            with (ROOT/'runs/showonce-foundationpose.log').open('a')as log:
                self.process=subprocess.Popen([str(worker_python('foundationpose')),'-u','-m', 'showonce_studio.showonce_pose',
                    '--recording',str(recording)],cwd=ROOT,env=dict(runtime_env(),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',
                    PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            return self.state(recording_id)

    def artifact(self,recording_id,name):
        recording,_=recording_path(self.root,recording_id)
        if name not in ('manifest.json','tracking.json')and not re.fullmatch(r'overlays/[a-z0-9_]{1,64}\.json',name):
            raise ValueError('잘못된 포즈 파일 경로입니다.')
        root=directory(recording).resolve();path=(root/name).resolve()
        if root not in path.parents:raise ValueError('포즈 폴더 밖의 파일입니다.')
        return path
