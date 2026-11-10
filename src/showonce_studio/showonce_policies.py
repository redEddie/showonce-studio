"""Immutable policy packages and explicit deployment contracts; no device I/O."""
from pathlib import Path
import hashlib
import json
import math
import re
import shutil
import tempfile
import time
from showonce_studio.config import ROOT
from showonce_studio.showonce_stages import read_json,write_json

from piper_runtime.contracts import PROFILE,ACTION_CONTRACT,layout,contract
from piper_runtime.resources import MODEL_XML,GEOMETRY
POLICIES=ROOT/'datasets/policy_library'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class PolicyLibrary:
    def __init__(self,root=POLICIES):self.root=Path(root)
    def folder(self,policy_id):
        if not isinstance(policy_id,str)or not re.fullmatch(r'[a-f0-9]{20}',policy_id):raise ValueError('잘못된 정책 ID입니다.')
        p=(self.root/policy_id).resolve()
        if self.root.resolve()not in p.parents:raise ValueError('정책 보관함 밖의 경로입니다.')
        return p
    def get(self,policy_id,verify=True):
        folder=self.folder(policy_id);value=read_json(folder/'manifest.json')
        if not value:raise ValueError('등록되지 않은 정책입니다.')
        if verify:
            for name,digest in value['files'].items():
                path=(folder/name).resolve()
                if folder not in path.parents or sha(path)!=digest:raise ValueError('등록한 정책 파일이 변경되었습니다.')
        return value
    def catalog(self):
        result=[]
        for path in sorted(self.root.glob('*/manifest.json')):
            item=read_json(path);meta=read_json(path.parent/'policy.json',{});env=meta.get('environment',{})
            item['training_conditions']={
                'observation_randomization':env.get('obs_randomization'),
                'spawn_ranges_m':{role:{axis:env.get(role+'_'+axis)for axis in ('x','y')}for role in ('block','bowl')},
                'spawn_max_radius_m':env.get('spawn_max_radius'),
                'spawn_min_distance_m':env.get('spawn_min_distance'),
                'checkpoint_step':meta.get('checkpoint_step',meta.get('training_step')),
                'parent_training_steps':meta.get('parent_training_steps',0),
                'cumulative_training_steps':meta.get('cumulative_training_steps',meta.get('training_step')),
                'episode_duration_s':env.get('episode_length',200)*env.get('ctrl_dt',.1)}
            from .policy_release import presentation
            item.update(presentation(meta))
            result.append(item)
        return result
    def register(self,source,name,model_xml=MODEL_XML):
        source=Path(source).expanduser().resolve();model_xml=Path(model_xml).expanduser().resolve()
        for item in ('policy.json','weights.npz','golden.npz'):
            if not(source/item).is_file():raise ValueError('내보낸 policy.json·weights.npz·golden.npz가 있는 폴더를 지정하세요.')
        meta=read_json(source/'policy.json');resolved=contract(meta)
        if sha(source/'weights.npz')!=meta['weights_sha256']:raise ValueError('정책 가중치 체크섬이 맞지 않습니다.')
        if sha(model_xml)!=meta['kinematics']['piper_xml_sha256']:raise ValueError('정책의 로봇 모델 XML과 다릅니다.')
        if sha(model_xml)!=sha(MODEL_XML):raise ValueError('현재 검증된 Piper 모델·간격 검사와 다른 XML에는 별도 어댑터가 필요합니다.')
        if not isinstance(name,str)or not name.strip()or len(name)>150:raise ValueError('정책 이름을 입력하세요.')
        names=['policy.json','weights.npz','golden.npz']
        names += [key for key in ('training-config.json','export-validation.json','controller-validation.json','observation-validation.json','recorded-golden.npz')if(source/key).is_file()]
        digests={key:sha(source/key)for key in names};digests['piper.xml']=sha(model_xml)
        policy_id=hashlib.sha256(json.dumps(digests,sort_keys=True).encode()).hexdigest()[:20]
        target=self.folder(policy_id)
        if target.exists():return self.get(policy_id)
        self.root.mkdir(parents=True,exist_ok=True);temp=Path(tempfile.mkdtemp(prefix='.policy-',dir=self.root))
        try:
            for key in names:shutil.copy2(source/key,temp/key)
            shutil.copy2(model_xml,temp/'piper.xml')
            value={'schema':'showonce_policy_package_v1','id':policy_id,'name':name.strip(),'created_s':time.time(),
                'source_dir':str(source),'source_checkpoint':meta.get('source_checkpoint'),'training_step':meta.get('cumulative_training_steps',meta.get('training_step')),
                'files':digests,'contract':resolved,'validation':'pending','hardware_validated':False,
                'training_task':meta['environment'].get('task'),'format':'piper_policy_runtime_npz_v1'}
            value.update(actor=meta['actor'],architecture=meta.get('display',{}).get('architecture',meta['actor'].upper()),
                recommended=meta.get('display',{}).get('recommended',False),role=meta.get('display',{}).get('role','legacy'),
                source_release=meta.get('source_release'),runtime_profile_id=meta.get('runtime_profile_id','studio-legacy-serialized-v1'))
            from .policy_release import presentation
            value.update(presentation(meta))
            write_json(temp/'manifest.json',value);temp.rename(target);return value
        except Exception:
            if temp.exists():shutil.rmtree(temp)
            raise
