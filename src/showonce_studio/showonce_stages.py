from showonce_studio.config import worker_python
"""Versioned stage annotations and a separate, local VLM inference process."""
from copy import deepcopy
import datetime as dt
import fcntl
import json
import math
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from showonce_studio.config import ROOT, runtime_env
from showonce_studio.showonce_models import DEFAULT_MODEL, MODELS, model_spec, model_status, result_directory
from showonce_studio.showonce_task_template import TEMPLATE_ID, STAGES, BOUNDARIES, template_spec, validate_template, display_annotation


def read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def recording_path(root, recording_id):
    if not isinstance(recording_id, str) or not re.fullmatch(r'\d{8}-\d{6}-\d{6}', recording_id):
        raise ValueError('올바른 시연 ID가 필요합니다.')
    path = Path(root).resolve()/recording_id
    if path.resolve().parent != Path(root).resolve():
        raise ValueError('잘못된 시연 경로입니다.')
    manifest = read_json(path/'manifest.json')
    if not manifest or not manifest.get('complete'):
        raise ValueError('저장이 완료된 시연만 분석할 수 있습니다.')
    return path, manifest


def source_frames(path):
    rows = [json.loads(s) for s in (path/'frames.jsonl').read_text().splitlines() if s.strip()]
    if len(rows) < 2:
        raise ValueError('영상 프레임이 부족합니다.')
    zero = rows[0]['color_device_timestamp_ms']
    times = [(r['color_device_timestamp_ms']-zero)/1000 for r in rows]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('원본 프레임 시각이 증가하지 않습니다.')
    return rows, times


def validate_annotation(value, duration, times):
    """Validate the complete temporal partition; never invent missing stages."""
    if not isinstance(value, dict):
        raise ValueError('단계 주석은 JSON 객체여야 합니다.')
    result = deepcopy(value)
    objects = result.get('objects')
    if not isinstance(objects, list) or not 1 <= len(objects) <= 30:
        raise ValueError('객체 목록이 필요합니다.')
    object_ids = set()
    for obj in objects:
        if not isinstance(obj, dict) or not isinstance(obj.get('id'), str) or not obj['id']:
            raise ValueError('객체 ID가 필요합니다.')
        if obj['id'] in object_ids or not isinstance(obj.get('name'), str):
            raise ValueError('객체 ID가 중복되었거나 이름이 없습니다.')
        object_ids.add(obj['id'])
    stages = result.get('stages')
    if not isinstance(stages, list) or not 1 <= len(stages) <= 40:
        raise ValueError('단계는 1~40개여야 합니다.')
    stage_ids = set()
    previous = 0.
    for i, stage in enumerate(stages):
        if not isinstance(stage, dict):
            raise ValueError('잘못된 단계 형식입니다.')
        for key in ('label', 'description', 'goal'):
            if not isinstance(stage.get(key), str) or not stage[key].strip() or len(stage[key]) > 2000:
                raise ValueError(f'{i+1}단계의 {key} 내용을 확인하세요.')
        if not isinstance(stage.get('id'), str) or not stage['id'] or stage['id'] in stage_ids:
            raise ValueError('단계 ID는 중복될 수 없습니다.')
        stage_ids.add(stage['id'])
        for key in ('start_s', 'end_s'):
            n = stage.get(key)
            if isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n):
                raise ValueError('단계 시각은 유한한 숫자여야 합니다.')
        start, end = float(stage['start_s']), float(stage['end_s'])
        if start < -.001 or end > duration+.001 or end-start < .03:
            raise ValueError(f'{i+1}단계의 시작/종료 시각이 영상 범위를 벗어났습니다.')
        if abs(start-previous) > .001:
            raise ValueError('단계 사이에 빈 구간이나 겹치는 구간이 있습니다.')
        ids = stage.get('object_ids')
        if not isinstance(ids, list) or not ids or any(oid not in object_ids for oid in ids):
            raise ValueError('단계의 객체 ID를 확인하세요.')
        stage['start_frame'] = min(range(len(times)), key=lambda k: abs(times[k]-start))
        stage['end_frame'] = min(range(len(times)), key=lambda k: abs(times[k]-end))
        previous = end
    if abs(previous-duration) > .001:
        raise ValueError('마지막 단계는 영상 끝까지 포함해야 합니다.')
    result['duration_s'] = duration
    return result


class StageManager:
    def __init__(self, root):
        self.root = Path(root)
        self.lock = threading.RLock()
        self.process = None
        self.log = None

    def state(self, recording_id, model_id=DEFAULT_MODEL):
        path, manifest = recording_path(self.root, recording_id)
        directory = result_directory(path, model_id)
        job = read_json(directory/'job.json', {'status': 'idle'})
        if job['status'] not in ('idle', 'complete', 'error') and job.get('pid'):
            try:
                os.kill(job['pid'], 0)
            except ProcessLookupError:
                job = dict(job, status='error', error='분석 프로세스가 중단되었습니다. 로그를 확인하세요.')
        models = []
        for key in MODELS:
            status = model_status(key)
            result = read_json(result_directory(path, key)/'automatic.json')
            model_job = read_json(result_directory(path, key)/'job.json', {})
            source = result.get('source', {}) if result else {}
            status.update(has_result=bool(result), stage_count=len(result['stages']) if result else None,
                          analysis_status=model_job.get('status', 'idle'),
                          inference_seconds=source.get('inference_seconds_after_load'))
            models.append(status)
        return {'recording_id': recording_id, 'model_id': model_id, 'model': model_status(model_id),
                'models': models, 'task_name': manifest.get('task_name'),
                'template': template_spec(), 'comparison': self.comparison(recording_id),
                'job': job, 'automatic': display_annotation(read_json(directory/'automatic.json')),
                'diagnostics': {'raw_response': (directory/'raw_response.txt').read_text()[:30000],
                                'frame_observations': read_json(directory/'frame_observations.json')}
                               if job['status'] == 'error' and (directory/'raw_response.txt').is_file() else None,
                'annotation': display_annotation(read_json(directory/'draft.json'))}

    def start(self, recording_id, model_id=DEFAULT_MODEL):
        with self.lock:
            path, _ = recording_path(self.root, recording_id)
            current = self.state(recording_id, model_id)
            if current['automatic']:
                return current
            if current['job']['status'] not in ('idle', 'error'):
                return current
            if self.process is not None and self.process.poll() is None:
                raise ValueError('다른 시연을 분석하고 있습니다. 완료 후 다시 시도하세요.')
            if not current['model']['installed']:
                raise ValueError('모델 설치가 필요합니다. '+current['model']['setup_command'])
            python = worker_python('kanana' if model_id=='kanana-v-3b' else 'vlm')
            if not python.is_file():
                raise ValueError('VLM 실행 환경이 없습니다. docs/WORKERS.md에 따라 설치하고 python.vlm 또는 python.kanana 경로를 설정하세요.')
            with (ROOT/'runs/.showonce-vlm.lock').open('a') as gpu_lock:
                try:
                    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise ValueError('다른 모델의 VLM 분석이 진행 중입니다. 완료 후 다시 시도하세요.')
            directory = result_directory(path, model_id)
            directory.mkdir(parents=True, exist_ok=True)
            if self.log:
                self.log.close()
            self.log = (directory/'inference.log').open('a')
            write_json(directory/'job.json', {'status': 'queued', 'started_s': time.time()})
            self.process = subprocess.Popen([str(python), '-u', '-m', 'showonce_studio.showonce_segment',
                '--recording', str(path), '--model-id', model_id], env=runtime_env(), stdout=self.log,
                stderr=subprocess.STDOUT, start_new_session=True)
            return self.state(recording_id, model_id)

    def save(self, recording_id, annotation, expected_revision, confirm=False, model_id=DEFAULT_MODEL):
        with self.lock:
            path, _ = recording_path(self.root, recording_id)
            directory = result_directory(path, model_id)
            current = read_json(directory/'draft.json')
            if not current:
                raise ValueError('먼저 자동 스테이지 분할을 실행하세요.')
            if expected_revision != current['revision']:
                raise ValueError('다른 화면에서 수정됐습니다. 새로 불러온 뒤 다시 저장하세요.')
            if not isinstance(annotation, dict):
                raise ValueError('단계 주석은 JSON 객체여야 합니다.')
            if annotation.get('model_id', model_id) != model_id or annotation.get('recording_id', recording_id) != recording_id:
                raise ValueError('현재 선택한 시연·모델과 초안의 출처가 다릅니다.')
            _, times = source_frames(path)
            value = validate_annotation(annotation, times[-1], times)
            if current.get('template_id') == TEMPLATE_ID:
                # Accept the previous display label from an already-open tab.
                for stage in value['stages']:
                    if stage.get('code') == 'grasp' and stage.get('label') == 'Grasp · 파지·들기':
                        stage['label'] = 'Grasp · 파지, 들기'
                validate_template(value)
            # Provenance is server-owned; a browser cannot relabel an edited
            # result as untouched model output or replace the source model.
            value = dict(current, model_id=model_id, stages=value['stages'], objects=value['objects'],
                         revision=current['revision']+1, review_status='confirmed' if confirm else 'draft',
                         updated_s=time.time(), confirmed_s=time.time() if confirm else None)
            if value.get('template_id') == TEMPLATE_ID:
                proposals = read_json(directory/'automatic.json', {}).get('boundaries', {})
                value['boundaries'] = {b['key']: {'time_s': value['stages'][i]['end_s'],
                    'source_frame_index': value['stages'][i]['end_frame'], 'reviewed': True,
                    'model_proposal': proposals.get(b['key'])} for i,b in enumerate(BOUNDARIES)}
            history = directory/'history'/f"review-{value['revision']:04d}.json"
            write_json(history, value)
            write_json(directory/'draft.json', value)
            if confirm:
                # Keep each model's confirmed definition. Only explicit human
                # confirmation selects the recording's active task definition.
                old = read_json(path/'task_definition.json')
                if old:
                    old_dir = result_directory(path, old.get('model_id', DEFAULT_MODEL)) if old.get('template_id') == TEMPLATE_ID else path/'stages/legacy-task-definitions'
                    if not (old_dir/'task_definition.json').exists():
                        write_json(old_dir/'task_definition.json', old)
                task = dict(value, schema='showonce_task_definition_v1',
                    source='human_confirmed_vlm_stages', training_ready=False,
                    geometry_status='not_reconstructed',
                    note='Semantic stage definition only; 3D goals/rewards/controller compatibility require subsequent verification.')
                write_json(directory/'task_definition.json', task)
                write_json(path/'task_definition.json', task)
                if value.get('template_id') == TEMPLATE_ID:
                    reference = dict(recording_id=recording_id, template_id=TEMPLATE_ID,
                                     model_id=model_id, revision=value['revision'], confirmed_s=value['confirmed_s'],
                                     protocol_id=value.get('source', {}).get('protocol_id'),
                                     boundaries={b['key']: value['stages'][i]['end_s'] for i,b in enumerate(BOUNDARIES)})
                    write_json(path/'stages/templates'/TEMPLATE_ID/'references'/f"{time.time_ns()}-{model_id}.json", reference)
                    write_json(path/'stages/templates'/TEMPLATE_ID/'reference.json', reference)
            else:
                # Keep the confirmed file as history, but make staleness explicit.
                old = read_json(path/'task_definition.json')
                if old and old.get('model_id', DEFAULT_MODEL) == model_id:
                    write_json(path/'task_definition.json', dict(old, superseded_by_draft_revision=value['revision']))
            return self.state(recording_id, model_id)

    def export(self, recording_id, model_id=DEFAULT_MODEL):
        with self.lock:
            path, _ = recording_path(self.root, recording_id)
            directory = result_directory(path, model_id)
            value = read_json(directory/'task_definition.json')
            if value is None and model_id == DEFAULT_MODEL:
                legacy = read_json(path/'task_definition.json')
                if legacy and legacy.get('model_id', DEFAULT_MODEL) == model_id and legacy.get('template_id') == TEMPLATE_ID:
                    value = legacy
            draft = read_json(directory/'draft.json', {})
            if not value or draft.get('review_status') != 'confirmed' or value.get('revision') != draft.get('revision'):
                raise ValueError('현재 모델의 초안을 확정한 뒤 내보내세요.')
            return display_annotation(value)

    def comparison(self, recording_id):
        path, _ = recording_path(self.root, recording_id)
        reference = read_json(path/'stages/templates'/TEMPLATE_ID/'reference.json')
        results = {}
        for key in MODELS:
            directory = result_directory(path, key)
            value = read_json(directory/'automatic.json') or read_json(directory/'assessment.json')
            if value and value.get('template_id') == TEMPLATE_ID:
                results[key] = value
        protocols = {v.get('source', {}).get('protocol_id') for v in results.values()}
        protocols.discard(None)
        same_protocol = len(protocols) == 1 and all(v.get('source', {}).get('protocol_id') for v in results.values())
        rows = []
        errors = {key: [] for key in results}
        for boundary in BOUNDARIES:
            expected = reference['boundaries'].get(boundary['key']) if reference else None
            values = {}
            for key, value in results.items():
                event = value.get('boundaries', {}).get(boundary['key'], {})
                actual = event.get('time_s')
                comparable = same_protocol and reference and reference.get('protocol_id') in protocols
                error = actual-expected if actual is not None and expected is not None and comparable else None
                if error is not None:
                    errors[key].append(abs(error))
                values[key] = dict(event, signed_error_s=error)
            available = [v['time_s'] for v in values.values() if v.get('time_s') is not None]
            rows.append(dict(boundary, values=values, reference_s=expected,
                             spread_s=max(available)-min(available) if len(available)>1 and same_protocol else None))
        return {'template_id': TEMPLATE_ID, 'same_protocol': bool(same_protocol),
                'reference': reference, 'rows': rows,
                'mae_s': {key: sum(e)/4 if len(e)==4 else None for key,e in errors.items()},
                'sampling_max_gap_s': max((v.get('source', {}).get('sampling_max_gap_s', 0.) for v in results.values()), default=0.)}
