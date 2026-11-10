"""Compare the CPU native planner with the release's original JAX equations.

Only fk_floor is loaded by AST from hierarchy_env to avoid importing a GPU
physics environment. The original smooth_plan, tracking_qp, hierarchy and
controllers modules execute unchanged. No MuJoCo steps, SDK or CAN access.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import types
os.environ['JAX_PLATFORMS']='cpu'
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-repo',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--golden',type=Path,required=True);args=p.parse_args()
    sys.path.insert(0,str(args.source_repo/'src'))
    import jax
    import jax.numpy as jp
    import numpy as np
    from piper_policy_runtime.kinematics import PiperKinematics,_matrix
    from piper_policy_runtime.native103 import Planner103
    from piper_runtime.resources import MODEL_XML
    # This function is taken verbatim from the reference source, not reimplemented.
    src=args.source_repo/'src/piper_sim_lab/control_lab';tree=ast.parse((src/'hierarchy_env.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='fk_floor')
    shim=types.ModuleType('piper_sim_lab.control_lab.hierarchy_env');shim.__dict__.update(jp=jp,_matrix=_matrix)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(src/'hierarchy_env.py'),'exec'),shim.__dict__)
    sys.modules[shim.__name__]=shim
    from piper_sim_lab.control_lab.controllers import Settings
    from piper_sim_lab.control_lab.tracking_qp import TrackingSettings
    from piper_sim_lab.control_lab.smooth_plan import checked_plan
    model=PiperKinematics(MODEL_XML);model.tcp_local_position=np.array([0.,0.,.1385])
    settings=Settings(dt=.005,policy_dt=.1,linear_scale=.1,angular_scale=.5,damping=.05,
        velocity=1.,lag=.15,acceleration=5.,jerk=100.,qp_jerk=True,
        floor_offset_base=-.015,floor_normal_base=(0.,0.,1.))
    tracking=TrackingSettings()
    native_settings=np.array([.005,1.,5.,100.,.15,.02,.02,.05,100.,.001,-.015,0.,0.,1.,2e-5,.01])
    planner=Planner103(model,native_settings)
    ref=jax.jit(lambda q,qd,st,target,old,hold:checked_plan(model,q,qd,st,target,old,hold,
        jp.asarray(model.lower),jp.asarray(model.upper),settings,tracking,20,64))
    rng=np.random.default_rng(108);rows=[];max_error=0.;matches=0
    q0=np.array([0.,.523,-.45,0.,0.,0.]);state=np.r_[q0,np.zeros(12)];old=np.zeros(6)
    for i in range(32):
        q=state[:6].copy();qd=state[6:12].copy()
        if i==12:q[4]-=.155  # soft tracking recovery
        if i==22:q[4]-=.3    # rejected packet: previous checked stop is retained
        target=rng.uniform(-.8,.8,6);hold=(0,4,10,0)[i%4]
        original,ok=ref(*[jp.asarray(v,dtype=jp.float32)for v in (q,qd,state,target,old)],jp.asarray(hold))
        original=np.asarray(original);ok=bool(ok)
        cpu,accepted,_=planner.plan(q,qd,state,target,old,hold)
        error=float(np.max(np.abs(cpu-original)));max_error=max(error,max_error)
        if accepted!=ok:raise AssertionError(f'Planner acceptance differs at case {i}: CPU={accepted}, JAX={ok}')
        # Acceleration is a finite difference amplified by 1/dt; FP32 and FP64 differ slightly.
        if error>5e-3:raise AssertionError(f'Planner parity differs at {i}: {error}')
        matches+=1;rows.append((q,qd,state.copy(),target,old.copy(),hold,original,ok))
        if accepted:state=cpu[19].copy();old=target.copy()
    args.golden.parent.mkdir(parents=True,exist_ok=True)
    values={key:np.asarray([row[i]for row in rows])for i,key in enumerate(('q','qdot','state','target','old_target','hold','plans','accepted'))}
    np.savez_compressed(args.golden,settings=native_settings,**values)
    report={'status':'passed','reference':'SHOW-108 release original JAX checked_plan, FP32',
        'cases':len(rows),'matching_acceptance_decisions':matches,'accepted':sum(bool(row[-1])for row in rows),
        'max_reference_position_abs_error_rad':float(max(np.max(np.abs(planner.plan(*row[:5],row[5])[0][:,:6]-row[6][:,:6]))for row in rows)),
        'max_plan_state_abs_error':max_error,'state_tolerance':.005,
        'source_sha256':{name:hashlib.sha256((src/name).read_bytes()).hexdigest()for name in ('smooth_plan.py','tracking_qp.py','hierarchy_env.py','hierarchy.py','controllers.py')},
        'hardware_commands_sent':False,'scope':'reference planning equations only; no physics or real robot validation'}
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
