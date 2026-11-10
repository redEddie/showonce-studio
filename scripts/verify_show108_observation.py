"""Check all 49 deployed inputs against the release's original observation code.

MuJoCo CPU forward kinematics supplies independent robot transforms. Original
_get_obs/_robot_features/_block_pose/_bowl_pos are loaded without a MJX env.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
os.environ['JAX_PLATFORMS']='cpu'
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'

def main():
    p=argparse.ArgumentParser();p.add_argument('--source-repo',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    import jax
    import jax.numpy as jp
    import mujoco
    import numpy as np
    from piper_policy_runtime.kinematics import PiperKinematics
    from piper_policy_runtime.show100 import assemble_observation
    src=args.source_repo/'src/piper_sim_lab/pick_place.py';tree=ast.parse(src.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='PiperPickPlace')
    nodes=[n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name in ('_get_obs','_robot_features','_block_pose','_bowl_pos')]
    helper=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='_quat_to_rotvec')
    class NoAnnotations(ast.NodeTransformer):
        def visit_FunctionDef(self,node):
            node.returns=None
            for arg in [*node.args.args,*node.args.kwonlyargs]:arg.annotation=None
            return node
    module=NoAnnotations().visit(ast.Module(body=[helper,*nodes],type_ignores=[]))
    scope={'jp':jp,'jax':jax,'np':np,'N_STAGES':4};exec(compile(module,str(src),'exec'),scope)
    Reference=type('OriginalObservation',(),{n.name:scope[n.name]for n in nodes})
    xml=args.source_repo/'src/piper_sim_lab/assets/agilex_piper_v100/piper.xml'
    mj=mujoco.MjModel.from_xml_path(str(xml));data=mujoco.MjData(mj)
    kin=PiperKinematics(xml);kin.tcp_local_position=np.array([0.,0.,.1385])
    env=dict(stage_obs=False,size_obs=True,gripper_goal_obs=True,gripper_max=.05,
        ctrl_dt=.1,bowl_pose_delay_max=.3)
    original=Reference();original._config=SimpleNamespace(**env,obs_randomization=False,bowl_free=True)
    original._qpos_ids=np.arange(6);original._qvel_ids=np.arange(6);original._arm_ctrl_ids=np.arange(6)
    original._finger_qpos_ids=[6,7];original._gripper_ctrl_id=6;original._tcp_id=0
    original._link6_id=0;original._block_id=1;original._bowl_id=2;original._block_qvel_adr=8
    original._base_offset=jp.array([0.,0.,.015]);rng=np.random.default_rng(10849);max_error=0.
    link=mujoco.mj_name2id(mj,mujoco.mjtObj.mjOBJ_BODY,'link6')
    for _ in range(64):
        q=rng.uniform(kin.lower,kin.upper);qd=rng.uniform(-.8,.8,6);reference=q+rng.uniform(-.12,.12,6)
        width=rng.uniform(0,.1);goal=rng.choice([0.,.1]);half=rng.uniform(.02,.03)
        block=rng.uniform([.25,-.2,0.],[.35,.2,.1]);bowl=rng.uniform([.25,-.2,-.015],[.35,.2,-.015])
        quat=rng.normal(size=4);quat/=np.linalg.norm(quat);previous=rng.uniform(-1.,1.,7)
        data.qpos[:6]=q;mujoco.mj_forward(mj,data)
        tcp=data.xpos[link]+data.xmat[link].reshape(3,3)@kin.tcp_local_position
        fake=SimpleNamespace(qpos=jp.array(np.r_[q,width/2,-width/2]),qvel=jp.array(np.r_[qd,np.zeros(5)]),
            ctrl=jp.array(np.r_[reference,width/2]),site_xpos=jp.array([tcp+np.array([0.,0.,.015])]),
            xquat=jp.array([data.xquat[link],quat,[1.,0.,0.,0.]]),
            xpos=jp.array([[0.,0.,0.],block+[0.,0.,.015],bowl+[0.,0.,.015]]))
        original._config.block_half=half
        percep=dict(robot_delay=jp.array(0.),obj_delay=jp.array(0.),bowl_offset=jp.zeros(3),size_scale=jp.array(1.),
            block_hist=jp.tile(original._block_pose(fake),(3,1)),robot_prev=original._robot_features(fake),
            bowl_hist=jp.tile(original._bowl_pos(fake),(4,1)),bowl_delay=jp.array(0.))
        flags={name:False for name in ('pre_grasp','grasp','pre_inside','inside')}
        expected=original._get_obs(fake,jp.array(previous),flags,percep,jax.random.key(0),gripper_goal=jp.array(goal/2))[0]['state']
        sensors=dict(frame='piper_base_link',units='SI',q_rad=q,qvel_rad_s=qd,gripper_full_width_m=width,
            block_pos_m=block,block_quat_wxyz=quat,bowl_pos_m=bowl,block_half_size_m=half)
        actual=assemble_observation(sensors,kin,env,joint_reference_rad=reference,
            gripper_final_goal_width_m=goal,previous_raw_action=previous)
        error=float(np.max(np.abs(actual-np.asarray(expected))));max_error=max(error,max_error)
        if error>5e-6:raise AssertionError(f'Observation mismatch: {error}; field {np.argmax(np.abs(actual-np.asarray(expected)))}')
    report=dict(status='passed',cases=64,observation_size=49,max_abs_error=max_error,
        reference='original _get_obs with CPU MuJoCo FK, zero observation delay/noise',source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
        hardware_commands_sent=False,scope='input arithmetic and coordinate conventions; not camera calibration accuracy')
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
