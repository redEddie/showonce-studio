from showonce_studio.config import service_port
from showonce_studio.config import worker_python
"""Lightweight web-to-analysis-node adapter and checked artifact paths."""
import json
from pathlib import Path
import re
import subprocess
import threading
import time
import urllib.request
import urllib.error
from showonce_studio.config import ROOT, runtime_env
from showonce_studio.showonce_stages import recording_path, read_json
from showonce_studio.showonce_geometry import directory


class GeometryManager:
    def __init__(self,root,port=None):
        self.root=Path(root);self.port=port or service_port('geometry',8769);self.lock=threading.Lock()

    def request(self,path,body=None):
        request=urllib.request.Request(f'http://127.0.0.1:{self.port}'+path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(request,timeout=3) as response:return json.load(response)
        except urllib.error.HTTPError as error:raise ValueError(json.load(error).get('error',str(error)))

    def state(self,recording_id):
        recording,_=recording_path(self.root,recording_id)
        try:return self.request('/api/geometry/state?id='+recording_id)
        except (urllib.error.URLError,TimeoutError):
            root=directory(recording)
            return {'recording_id':recording_id,'model_retained':False,'node_alive':False,
                    'job':read_json(root/'job.json',{'status':'idle'}),'manifest':read_json(root/'manifest.json'),
                    'tracking_ready':(root/'tracking.json').is_file(),
                    'partial_tracking_ready':(root/'tracking.partial.json').is_file()}

    def ensure_node(self):
        with self.lock:
            try:health=self.request('/health')
            except (urllib.error.URLError,TimeoutError):
                python=worker_python('geometry')
                if not python.is_file():raise ValueError('docs/SAM-WORKER.md에 따라 SAM 분석 환경을 설치하고 python.geometry 경로를 설정하세요.')
                with (ROOT/'runs/showonce-geometry-node.log').open('a') as log:
                    subprocess.Popen([str(python),'-u','-m', 'showonce_studio.showonce_geometry_node',
                                      '--out',str(self.root),'--port',str(self.port)],
                                     env=dict(runtime_env(),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4'),
                                     stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                for _ in range(30):
                    try:health=self.request('/health');break
                    except (urllib.error.URLError,TimeoutError):time.sleep(.1)
                else:raise ValueError('형상 분석 노드를 시작하지 못했습니다.')
            if Path(health['root']).resolve()!=self.root.resolve():raise ValueError('분석 노드의 데이터 경로가 다릅니다.')
            return health

    def start(self,recording_id):
        recording_path(self.root,recording_id)
        self.ensure_node()
        return self.request('/api/geometry/start',{'recording_id':recording_id})

    def artifact(self,recording_id,name):
        recording_path(self.root,recording_id)
        allowed=(name in ('manifest.json','tracking.json','tracking.partial.json') or
                 re.fullmatch(r'assets/[a-z0-9_]{1,64}/(mesh.glb|mesh.ply|measured.ply|points.json|overlay_points.json|thumbnail.png|asset.json|diagnostics.json)',name))
        if not allowed:raise ValueError('잘못된 분석 파일 경로입니다.')
        root=directory(self.root/recording_id).resolve();path=(root/name).resolve()
        if root not in path.parents:raise ValueError('분석 폴더 밖의 파일입니다.')
        return path
