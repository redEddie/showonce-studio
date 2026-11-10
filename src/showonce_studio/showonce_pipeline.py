from showonce_studio.config import worker_python
"""Reuse selected reference assets: SAM masks -> FoundationPose -> relations."""
from pathlib import Path
import argparse
import fcntl
import os
import subprocess
import time
from showonce_studio.config import ROOT,runtime_env
from showonce_studio.showonce_asset_library import selection,sha
from showonce_studio.showonce_geometry_api import GeometryManager
from showonce_studio.showonce_stages import read_json,write_json

def directory(recording):return Path(recording)/'analysis/reference_pipeline_v1'

def wait_masks(manager,recording_id):
    manager.request('/api/geometry/masks/start',{'recording_id':recording_id})
    while True:
        value=manager.request('/api/geometry/masks/state?id='+recording_id)
        if value.get('tracking_ready'):return
        if value['job'].get('status')=='error':raise ValueError(value['job'].get('error'))
        time.sleep(.5)

def run(recording,progress):
    recording=Path(recording).resolve();binding=selection(recording)
    if not binding:raise ValueError('먼저 보관함 에셋을 선택하세요.')
    manager=GeometryManager(recording.parent);manager.ensure_node()
    progress('masks',message='선택한 물체의 영상 마스크 추적 중')
    wait_masks(manager,recording.name)
    progress('pose',message='참조 에셋으로 위치·자세를 추적합니다.')
    env=dict(runtime_env(),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    existing=read_json(recording/'geometry/foundationpose_v1/tracking.json',{})
    fresh=existing.get('complete')and existing.get('asset_selection_sha256')==binding['selection_sha256']and existing.get('source_frames_sha256')==sha(recording/'frames.jsonl')and existing.get('calibration_sha256')==sha(recording/'calibration.json')
    if not fresh:
        with (directory(recording)/'pose.log').open('a')as log:
            subprocess.run([str(worker_python('foundationpose')),'-u','-m', 'showonce_studio.showonce_pose','--recording',str(recording)],
                cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    progress('relations',message='손과 물체의 작업 관계를 추출합니다.')
    existing=read_json(recording/'relations/scene_graph_v1/manifest.json',{})
    if not existing.get('complete')or existing.get('sources',{}).get('poses_sha256')!=sha(recording/'geometry/foundationpose_v1/tracking.json'):
        with (directory(recording)/'relations.log').open('a')as log:
            subprocess.run([str(worker_python('sam3d')),'-u','-m', 'showonce_studio.showonce_relations','--recording',str(recording)],
                cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    poses=read_json(recording/'geometry/foundationpose_v1/manifest.json');graph=read_json(recording/'relations/scene_graph_v1/manifest.json')
    if not poses.get('complete')or not graph.get('complete'):raise ValueError('분석 완료 상태를 확인하지 못했습니다.')
    result={'schema':'showonce_reference_pipeline_v1','complete':True,'recording_id':recording.name,'completed_s':time.time(),
        'asset_selection_sha256':binding['selection_sha256'],'assets':binding['assets'],
        'pose_summary':poses['summary'],'mesh_reconstruction_performed':False,'robot_commands_sent':False}
    write_json(directory(recording)/'result.json',result);progress('complete',message='포즈 추적·관계 추출 완료');return result

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--recording',type=Path,required=True);args=parser.parse_args()
    folder=directory(args.recording);folder.mkdir(parents=True,exist_ok=True);started=time.time()
    def progress(status,**values):write_json(folder/'job.json',{'status':status,'pid':os.getpid(),'started_s':started,'updated_s':time.time(),**values})
    with (folder/'run.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(args.recording,progress)
        except Exception as error:progress('error',error=f'{type(error).__name__}: {error}',message='분석 기록에서 실패 단계를 확인하세요.');raise

if __name__=='__main__':main()
