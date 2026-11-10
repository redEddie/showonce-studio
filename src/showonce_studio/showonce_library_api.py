from showonce_studio.config import worker_python
"""Asset selection and asynchronous reference-analysis pipeline, no GPU imports."""
from pathlib import Path
import subprocess
import threading
import time
from showonce_studio.config import ROOT,runtime_env
from showonce_studio.showonce_stages import read_json,recording_path,write_json
from showonce_studio.showonce_asset_library import AssetLibrary,selection,bundle
from showonce_studio.showonce_pipeline import directory

class LibraryManager:
    def __init__(self,recordings,library=None):self.root=Path(recordings);self.library=library or AssetLibrary();self.lock=threading.Lock();self.process=None

    def state(self,recording_id):
        recording,_=recording_path(self.root,recording_id);selected=selection(recording)
        manifest=read_json(bundle(recording)/'manifest.json',{});legacy=bool(manifest.get('complete')and manifest.get('source')!='asset_library')
        result=read_json(directory(recording)/'result.json');pose=read_json(recording/'geometry/foundationpose_v1/job.json',{})
        return {'recording_id':recording_id,'selection':selected,'legacy':legacy,'result':result,
            'job':read_json(directory(recording)/'job.json',{'status':'idle'}),
            'mask_job':read_json(recording/'geometry/rgbd_surface_v1/mask_job.json',{}),'pose_job':pose,
            'relation_job':read_json(recording/'relations/scene_graph_v1/job.json',{}),
            'analysis_complete':legacy or bool(result and result.get('complete'))}

    def analyze(self,recording_id,mapping):
        recording,_=recording_path(self.root,recording_id)
        with self.lock:
            current=self.state(recording_id)
            if current['legacy']:raise ValueError('이 영상의 기존 분석은 보존합니다. 새 영상에서 에셋을 선택하세요.')
            if self.process is not None and self.process.poll()is None:raise ValueError('다른 영상의 분석이 진행 중입니다.')
            self.library.bind(recording,mapping)
            if current['analysis_complete']:return self.state(recording_id)
            pid=current['job'].get('pid');command=Path(f'/proc/{pid}/cmdline')if pid else None
            if command and command.is_file()and b'showonce_pipeline.py'in command.read_bytes():return current
            folder=directory(recording);folder.mkdir(parents=True,exist_ok=True)
            write_json(folder/'job.json',{'status':'queued','message':'선택한 에셋으로 분석을 준비합니다.','updated_s':time.time()})
            with (folder/'pipeline.log').open('a')as log:
                self.process=subprocess.Popen([str(worker_python('sam3d')),'-u','-m', 'showonce_studio.showonce_pipeline','--recording',str(recording)],
                    cwd=ROOT,env=dict(runtime_env(),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            return self.state(recording_id)
