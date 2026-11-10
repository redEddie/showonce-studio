"""Import the pinned final inference archive without training or device I/O."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile

SPEC_PATH=Path(__file__).with_name('data')/'show108-24u-final.json'

def release_spec():
    return json.loads(SPEC_PATH.read_text())

def presentation(meta):
    spec=release_spec()
    if any(meta.get(key)!=spec[key] for key in ('checkpoint_sha256','weights_sha256')):
        return {}
    if meta.get('training_config_sha256')!=spec['config_sha256']:
        return {}
    return dict(is_final=True,recommended=True,role='trained',source_release=spec['source_release'],
        architecture='MLP 256 → 64 → LayerNorm → LSTM 128',
        cumulative_training_steps=spec['cumulative_training_steps'],
        checkpoint_step=spec['checkpoint_step'],simulation_placement=spec['simulation_placement'],
        selection_label='최종 24배 · 실기 성능 미검증')

def verify_archive(archive,spec):
    archive=Path(archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=spec['archive']['sha256']:
        raise ValueError('최종 24배 추론 아카이브 SHA256이 다릅니다.')
    expected=spec['archive']['member_sha256'];seen=set()
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar:
            path=PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0]!=spec['archive']['root']:
                raise ValueError('아카이브 경로가 올바르지 않습니다.')
            if member.isdir():continue
            if not member.isfile():raise ValueError('아카이브 링크·특수 파일은 허용하지 않습니다.')
            name=str(PurePosixPath(*path.parts[1:]))
            if name not in expected or name in seen:raise ValueError('아카이브 파일 목록이 다릅니다.')
            if hashlib.sha256(tar.extractfile(member).read()).hexdigest()!=expected[name]:
                raise ValueError('아카이브 내부 파일 SHA256이 다릅니다: '+name)
            seen.add(name)
    if seen!=set(expected):raise ValueError('아카이브 파일이 누락됐습니다.')

def install(archive,library=None,artifacts=None):
    from .config import ROOT
    from .showonce_policies import PolicyLibrary,sha
    from .showonce_policy_runtime import validate
    spec=release_spec();verify_archive(archive,spec)
    library=library or PolicyLibrary()
    artifacts=Path(artifacts) if artifacts is not None else ROOT/'artifacts'
    artifacts.mkdir(parents=True,exist_ok=True)
    target=artifacts/spec['key']
    expected={name.removeprefix('policy/'):digest for name,digest in spec['archive']['member_sha256'].items() if name.startswith('policy/')}
    if target.exists():
        if any(not(target/name).is_file() or sha(target/name)!=digest for name,digest in expected.items()):
            raise ValueError('기존 최종 모델 폴더 내용이 다릅니다. 덮어쓰지 않았습니다.')
    else:
        temporary=Path(tempfile.mkdtemp(prefix='.final24-',dir=artifacts))
        try:
            with tarfile.open(archive,'r:gz') as tar:
                for name in expected:
                    data=tar.extractfile(spec['archive']['root']+'/policy/'+name).read()
                    (temporary/name).write_bytes(data)
            # Includes raw-input inference, recurrent state and declared contract.
            validate(temporary)
            temporary.rename(target)
        finally:
            if temporary.exists():shutil.rmtree(temporary)
    result=library.register(target,spec['name'])
    return dict(policy=result,validation=validate(library.folder(result['id'])),
                hardware_commands_sent=False,source_release=spec['source_release'])

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archive',type=Path,required=True,help='Verified final24 inference.tar.gz downloaded from GitHub Release')
    args=ap.parse_args()
    print(json.dumps(install(args.archive),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
