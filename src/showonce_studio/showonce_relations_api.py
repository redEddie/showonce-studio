from showonce_studio.config import worker_python
"""Read-only graph/stage review and a separate relation extraction worker."""
from pathlib import Path
import subprocess
import threading
from showonce_studio.config import ROOT,runtime_env
from showonce_studio.showonce_stages import read_json,recording_path
from showonce_studio.showonce_relations import directory,sha
from showonce_studio.showonce_task_template import display_annotation


def stage_comparison(graph,task):
    mappings={'grasp_start':('contact','접촉 추정 시작'),'transfer_start':('above_table','들림 감지'),
        'release_start':('aligned','그릇과 XY 정렬'),'retreat_start':('grasped','파지 추정 종료')}
    values=[]
    for key,(relation,label)in mappings.items():
        boundary=task.get('boundaries',{}).get(key,{});vlm=boundary.get('time_s')
        to='false'if key=='retreat_start'else 'true'
        events=[e for e in graph['events']if e['relation_id']==relation and e['to']==to and
            (e['from']=='true'if to=='false'else e['time_s']>.3)]
        nearest=min(events,key=lambda e:abs(e['time_s']-vlm))if events and vlm is not None else None
        values.append({'boundary':key,'confirmed_time_s':vlm,'relation_event':label,'event_time_s':nearest['time_s']if nearest else None,
            'delta_s':nearest['time_s']-vlm if nearest else None,'definition_matches_exactly':False})
    return values


class RelationsManager:
    def __init__(self,root):self.root=Path(root);self.lock=threading.Lock();self.process=None

    def state(self,recording_id):
        recording,_=recording_path(self.root,recording_id);folder=directory(recording);graph=read_json(folder/'manifest.json')
        task=read_json(recording/'task_definition.json',{})
        return {'recording_id':recording_id,'job':read_json(folder/'job.json',{'status':'idle'}),
            'hand_job':read_json(folder/'hand_job.json',{'status':'idle'}),
            'graph_ready':bool(graph and graph.get('complete')),'created_s':graph.get('created_s')if graph else None,
            'confirmed_stage_revision':task.get('revision')if task.get('review_status')=='confirmed'else None,
            'prerequisites_ready':(recording/'geometry/foundationpose_v1/tracking.json').is_file()}

    def graph(self,recording_id):
        recording,_=recording_path(self.root,recording_id);graph=read_json(directory(recording)/'scene_graph.json')
        if not graph or not graph.get('complete'):raise ValueError('씬 그래프 분석 결과를 준비 중입니다.')
        sources=graph['sources']
        if sha(recording/'geometry/foundationpose_v1/tracking.json')!=sources['poses_sha256']:
            raise ValueError('물체 포즈가 변경되어 관계 분석을 다시 해야 합니다.')
        task=display_annotation(read_json(recording/'task_definition.json',{}))
        confirmed=task.get('review_status')=='confirmed'
        return {**graph,'stages':task.get('stages',[])if confirmed else [],
            'confirmed_stage_revision':task.get('revision')if confirmed else None,
            'stage_comparison':stage_comparison(graph,task)if confirmed else [],
            'stage_alignment_note':'스테이지 경계와 관계 사건은 정의가 다릅니다. 관계 추출에 스테이지 시각을 사용하지 않았습니다.'}

    def start(self,recording_id):
        recording,_=recording_path(self.root,recording_id)
        with self.lock:
            current=self.state(recording_id)
            if current['graph_ready']:return current
            if not current['prerequisites_ready']:raise ValueError('FoundationPose 추적 결과가 필요합니다.')
            if self.process is not None and self.process.poll()is None:raise ValueError('관계 분석이 진행 중입니다.')
            pid=current['job'].get('pid');command=Path(f'/proc/{pid}/cmdline')if pid else None
            if command and command.is_file()and b'showonce_relations.py'in command.read_bytes():return current
            with (ROOT/'runs/showonce-relations.log').open('a')as log:
                self.process=subprocess.Popen([str(worker_python('sam3d')),'-u','-m', 'showonce_studio.showonce_relations',
                    '--recording',str(recording)],cwd=ROOT,env=dict(runtime_env(),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1'),
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            return self.state(recording_id)
