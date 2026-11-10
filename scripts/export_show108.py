"""Export the selected SHOW-108 actor using its pinned original Flax loader.

Run in a separate CPU JAX/Flax/Brax environment, never the live robot env.
This script reads the original learner/config without modifying either one.
"""
import argparse
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

os.environ['JAX_PLATFORMS']='cpu'
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

RELEASES={
    'final24':dict(commit='939bdb623a6bd641152388429396b8ad74e1e70a',
        learner_sha256='84e064bf50d34c30b346f08c109a387b0d4b1ca58fdeb30b93db638380d44306',
        config_sha256='dd6f8eb88c7e6425008a244bc73944f5d8578e657e9803d4011b682bc3b26139',
        step=33554432,cumulative=201326592,parent=167772160,
        training_source='d377cae0afdc87a389b09071eae6bcde72465e86',
        tag='show108-mlp-lstm-24u-final-20261109',
        title='SHOW-108 · 최종 24배 · 관측 DR · 확장 스폰'),
    '16u':dict(commit='85fcf98b56b830cc4c16bcf6522b152d3ab6d3db',
        learner_sha256='6e52e2f5fd63c29ea5e3e6ae9676488acbe2ba5bf76823aeeb4a60cb9aa734de',
        config_sha256='f5ecd0b55466e8b42c1c81cfc54826fec50469f70ef2ed670eb50ad3c91e0852',
        step=134217728,cumulative=134217728,parent=0,
        training_source='49267d34714545266fbdf85c77540307ca5c2f51',
        tag='show108-mlp-lstm-16u-20261101',title='SHOW-108 · MLP + LSTM · 16배'),
    'obsdr-wide':dict(commit='9b006cf9e2b8bb63b0b0ada98c5368f0013e1dbf',
        learner_sha256='19438a198e31345d04e4234073028c1e0d9f20b48b90f9cb917af48b55f09c2c',
        config_sha256='0a083727cb5f14a2a4bf923ce53e318942f1c5ffe5443da9b96f2fd5dadfab2a',
        step=16777216,cumulative=134217728,parent=117440512,
        training_source='0f5aa042416f49c7a8839396179e230a0e853242',
        tag='show108-mlp-lstm-obsdr-wide-14plus2u-20261101',
        title='SHOW-108 · 관측 DR · 넓은 범위 · 14+2배'),
}

def verify_release(source,run,release):
    spec=RELEASES[release]
    commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if commit!=spec['commit']:raise ValueError('Use the pinned source revision for this SHOW-108 release')
    checkpoint=run/f"checkpoints/{spec['step']:012d}"
    if digest(checkpoint/'learner.msgpack')!=spec['learner_sha256']:raise ValueError('Learner differs from the verified release')
    if digest(run/'config.json')!=spec['config_sha256']:raise ValueError('Training configuration differs from the verified release')
    return spec,commit,checkpoint

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-repo',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--release',choices=RELEASES,default='final24')
    args=p.parse_args()
    source=args.source_repo.resolve();run=args.run.resolve();output=args.output.resolve()
    spec,commit,checkpoint=verify_release(source,run,args.release)
    cfg=json.loads((run/'config.json').read_text());env=cfg['resolved_environment']
    if cfg['actor']!='mlp_lstm':raise ValueError('MLP encoder -> LAST LSTM required')
    sys.path.insert(0,str(source/'src'))
    import jax
    import jax.numpy as jp
    import numpy as np
    from piper_sim_lab.recurrent import load_actor_checkpoint,zero_carry
    from piper_runtime.contracts import ACTION_CONTRACT,SHOW108_PROFILE_ID,contract,observation_layout
    from piper_runtime.resources import MODEL_XML
    from piper_runtime.identity import identity
    from piper_policy_runtime.check import check
    sizes={'state':(49,),'privileged':(66,)}
    actor,params,normalizer=load_actor_checkpoint(run,checkpoint,sizes,7)
    weights={}
    def flatten(node,prefix=''):
        for key,value in node.items():
            name=f'{prefix}/{key}'if prefix else key
            if isinstance(value,Mapping):flatten(value,name)
            else:weights[name]=np.asarray(value,dtype=np.float32)
    flatten(params)
    weights['normalizer/mean']=np.asarray(normalizer.mean['state'],np.float32)
    weights['normalizer/std']=np.asarray(normalizer.std['state'],np.float32)
    for name,flax_path in [('input_kernel','i{gate}/kernel'),('hidden_kernel','h{gate}/kernel'),('bias','h{gate}/bias')]:
        weights['runtime/lstm_'+name]=np.concatenate([weights['params/lstm/'+flax_path.format(gate=gate)]for gate in 'ifgo'],axis=-1)
    xml=source/'src/piper_sim_lab/assets/agilex_piper_v100/piper.xml'
    if digest(xml)!=digest(MODEL_XML):raise ValueError('Robot XML differs from runtime')
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'weights.npz',**weights)
    meta={'schema_version':1,'actor':'mlp_lstm','recurrent_width':128,'encoder_sizes':[256,64],'head_sizes':[],
        'observation_size':49,'action_size':7,'observation_layout':observation_layout(env),
        'action_contract':ACTION_CONTRACT,'runtime_profile_id':SHOW108_PROFILE_ID,'environment':env,
        'training_step':spec['cumulative'],'cumulative_training_steps':spec['cumulative'],
        'checkpoint_step':spec['step'],'parent_training_steps':spec['parent'],
        'checkpoint_format':'matched-recurrent-msgpack',
        'source_checkpoint':str(checkpoint),'checkpoint_sha256':digest(checkpoint/'learner.msgpack'),
        'training_config_sha256':digest(run/'config.json'),'code_commit':commit,
        'training_source_commit':spec['training_source'],
        'source_release':'https://github.com/redEddie/piper-sim-lab/releases/tag/'+spec['tag'],
        'display':{'title':spec['title'],'architecture':'MLP 256 → 64 → LayerNorm → LSTM 128',
            'recommended':True,'role':'trained'},
        'kinematics':{'robot_model':'piper_v100','tcp_local_position_m':[0.,0.,.1385],
            'piper_xml_sha256':digest(xml),'observation_frame':'base_link','base_height_m':.015},
        'runtime_identity':identity(),'weights_sha256':digest(output/'weights.npz')}
    contract(meta)
    (output/'policy.json').write_text(json.dumps(meta,indent=2)+'\n')
    shutil.copy2(xml,output/'piper.xml');shutil.copy2(run/'config.json',output/'training-config.json')
    rng=np.random.default_rng(108901);length,batch=256,4
    obs=(weights['normalizer/mean']+weights['normalizer/std']*rng.normal(0,.7,(length,batch,49))).astype(np.float32)
    reset=np.zeros((length,batch),bool);reset[0]=True;reset[25,1]=True;reset[80,2]=True;reset[150]=True
    def step(carry,item):
        observation,mask=item
        carry,logits=actor.step(normalizer,params,carry,{'state':observation},mask)
        return carry,(actor.dist.mode(logits),logits,carry[0],carry[1])
    _,values=jax.jit(lambda:jax.lax.scan(step,zero_carry(batch,128),(jp.asarray(obs),jp.asarray(reset))))()
    arrays={name:np.asarray(value)for name,value in zip(('actions','logits','memory_c','memory_h'),values)}
    np.savez_compressed(output/'golden.npz',observations=obs,resets=reset,**arrays)
    report=check(output)
    (output/'export-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'output':str(output),'validation':report},indent=2))

if __name__=='__main__':main()
