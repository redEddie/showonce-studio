from showonce_studio.config import worker_python
from showonce_studio.config import models_root
"""Pinned, local vision-model catalog. No model downloads during web requests."""
import argparse
import json
from pathlib import Path
import time
from showonce_studio.config import ROOT
from showonce_studio.showonce_task_template import TEMPLATE_ID

DEFAULT_MODEL = 'qwen35-9b'
MODELS = {
    DEFAULT_MODEL: {
        'id': DEFAULT_MODEL, 'name': 'Qwen3.5 · 9B', 'provider': 'Alibaba',
        'repo': 'Qwen/Qwen3.5-9B', 'revision': 'c202236235762e1c871ad0ccb60c8ee5ba337b9a',
        'folder': 'Qwen3.5-9B', 'environment': '.venv-vlm', 'adapter': 'qwen35',
        'license': 'Apache 2.0', 'license_url': 'https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/LICENSE',
        'license_note': 'Apache 2.0 · 재배포 시 라이선스·고지 유지', 'attribution': '',
    },
    'kanana-v-3b': {
        'id': 'kanana-v-3b', 'name': 'Kanana 1.5 V · 3.67B', 'provider': 'Kakao · 한국',
        'repo': 'kakaocorp/kanana-1.5-v-3b-instruct', 'revision': '2e00ef13ccec2e99459a8eada18a1bfd05bff44b',
        'folder': 'Kanana-1.5-V-3B', 'environment': '.venv-vlm-kanana', 'adapter': 'kanana',
        'license': 'Kanana License',
        'license_url': 'https://huggingface.co/kakaocorp/kanana-1.5-v-3b-instruct/blob/main/LICENSE',
        'license_note': '자체 서비스 운영 허용 조건 있음 · 모델 API 제공·SI/온프레미스·온디바이스 제공은 별도 상업 라이선스 필요. 출력으로 학습한 모델 배포 시 명명 조건 있음.',
        'attribution': 'Powered by Kanana',
    },
    'gemma3-4b': {
        'id': 'gemma3-4b', 'name': 'Gemma 3 · 4B', 'provider': 'Google',
        'repo': 'google/gemma-3-4b-it', 'revision': '093f9f388b31de276ce2de164bdc2081324b9767',
        'folder': 'Gemma-3-4B-IT', 'environment': '.venv-vlm', 'adapter': 'gemma3',
        'license': 'Gemma Terms', 'license_url': 'https://ai.google.dev/gemma/terms',
        'license_note': 'Gemma 이용 조건 적용 · 공식 저장소 이용 조건 동의와 HF 인증 후 다운로드',
        'attribution': '',
    },
}


def model_spec(model_id=DEFAULT_MODEL):
    if not isinstance(model_id, str) or model_id not in MODELS:
        raise ValueError('지원하지 않는 비전 모델입니다.')
    return dict(MODELS[model_id])


def model_path(model_id):
    return models_root()/model_spec(model_id)['folder']


def result_directory(recording, model_id=DEFAULT_MODEL):
    model_spec(model_id)  # whitelist before constructing a filesystem path
    base = Path(recording)/'stages'
    # A new protocol cannot reuse or relabel earlier free-form model outputs.
    # The legacy stages/ and stages/models/ directories remain untouched.
    return base/'templates'/TEMPLATE_ID/'models'/model_id


def model_status(model_id):
    spec = model_spec(model_id)
    path = model_path(model_id)
    weights = []
    try:
        index = json.loads((path/'model.safetensors.index.json').read_text())
        weights = list(set(index['weight_map'].values()))
    except FileNotFoundError:
        if (path/'model.safetensors').is_file():
            weights = ['model.safetensors']
    installed = (path/'config.json').is_file() and bool(weights) and all(
        (path/name).is_file() and (path/name).stat().st_size > 0 for name in weights)
    runtime_ready = (worker_python('kanana' if model_id=='kanana-v-3b' else 'vlm')).is_file()
    metadata_path = path.parent/(spec['folder']+'-download.json')
    try:
        metadata = json.loads(metadata_path.read_text())
    except FileNotFoundError:
        metadata = {}
    setup_commands = []
    if not runtime_ready:
        setup_commands.append('# docs/WORKERS.md: VLM 환경을 설치하고 python.kanana 또는 python.vlm 경로를 설정하세요.')
    if not installed:
        setup_commands.append(f"python -P -m showonce_studio.showonce_models download {model_id}")
    return dict(spec, installed=installed, runtime_ready=runtime_ready,
                available=installed and runtime_ready,
                inference_tested=bool(metadata.get('inference_tested')),
                setup_command='\n'.join(setup_commands))


def download(model_id):
    import os
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    from huggingface_hub import snapshot_download
    spec = model_spec(model_id)
    dest = model_path(model_id)
    snapshot_download(spec['repo'], revision=spec['revision'], local_dir=dest,
        allow_patterns=['*.json', '*.py', '*.safetensors', '*.model', '*.jinja',
                        'LICENSE', 'LICENSE.txt', 'README.md', 'NOTICE'], max_workers=3)
    metadata_path = dest.parent/(spec['folder']+'-download.json')
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata.update(repo=spec['repo'], revision=spec['revision'], downloaded_s=time.time(), license=spec['license'])
    from showonce_studio.showonce_stages import write_json
    write_json(metadata_path, metadata)
    print(json.dumps(model_status(model_id), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['list', 'download'])
    parser.add_argument('model_id', nargs='?', choices=list(MODELS))
    args = parser.parse_args()
    if args.command == 'download':
        if not args.model_id:
            parser.error('download requires model_id')
        download(args.model_id)
    else:
        print(json.dumps([model_status(m) for m in MODELS], ensure_ascii=False, indent=2))
