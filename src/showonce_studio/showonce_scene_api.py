from showonce_studio.config import service_port
"""Local web proxy to an independent MuJoCo process."""
import json
from pathlib import Path
import subprocess
import threading
import time
import urllib.request
import urllib.error
from showonce_studio.config import ROOT,PYTHON,runtime_env
from showonce_studio.showonce_stages import recording_path


class SceneManager:
    def __init__(self,root,port=None):self.root=Path(root);self.port=port or service_port('scene',8770);self.lock=threading.Lock()

    def request(self,path,body=None):
        request=urllib.request.Request(f'http://127.0.0.1:{self.port}'+path,
            data=json.dumps(body).encode() if body is not None else None,headers={'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(request,timeout=5) as response:return json.load(response)
        except urllib.error.HTTPError as error:raise ValueError(json.load(error).get('error',str(error)))

    def state(self,recording_id):
        recording_path(self.root,recording_id)
        try:return self.request('/state?id='+recording_id)
        except (urllib.error.URLError,TimeoutError):return {'loaded':False,'node_alive':False,'recording_id':recording_id}

    def command(self,recording_id,action):
        recording_path(self.root,recording_id)
        if action not in ('load','play','pause','reset','drop'):raise ValueError('잘못된 시뮬레이션 명령입니다.')
        with self.lock:
            try:health=self.request('/health')
            except (urllib.error.URLError,TimeoutError):
                with (ROOT/'runs/showonce-scene-node.log').open('a') as log:
                    subprocess.Popen([str(PYTHON),'-u','-m', 'showonce_studio.showonce_scene_node','--out',str(self.root.resolve()),'--port',str(self.port)],
                        cwd=ROOT,env=dict(runtime_env(),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                for _ in range(50):
                    try:health=self.request('/health');break
                    except (urllib.error.URLError,TimeoutError):time.sleep(.1)
                else:raise ValueError('MuJoCo 장면 노드를 시작하지 못했습니다.')
            if Path(health['root']).resolve()!=self.root.resolve():raise ValueError('MuJoCo 노드 저장 경로가 다릅니다.')
            return self.request('/command',{'recording_id':recording_id,'action':action})
