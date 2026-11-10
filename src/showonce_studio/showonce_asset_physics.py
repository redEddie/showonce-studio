"""Headless MuJoCo import, settling and container-opening checks for exported assets."""
from pathlib import Path
import copy
import xml.etree.ElementTree as ET
import numpy as np


def scene_xml(root,scene,*,drop_test=False):
    doc=ET.Element('mujoco',model='showonce_reconstructed_scene')
    ET.SubElement(doc,'compiler',angle='radian',autolimits='true')
    ET.SubElement(doc,'option',timestep='.002',gravity='0 0 -9.81',integrator='implicitfast',iterations='100',cone='elliptic')
    defaults=ET.SubElement(doc,'default');ET.SubElement(defaults,'geom',solref='.005 1',solimp='.95 .99 .001',condim='4')
    assets=ET.SubElement(doc,'asset');world=ET.SubElement(doc,'worldbody')
    ET.SubElement(world,'geom',name='table',type='plane',size='1 1 .05',friction='.8 .005 .0001')
    bowl=next((obj for obj in scene['objects'] if obj['id']=='bowl'),None)
    for obj in scene['objects']:
        oid=obj['id'];pose=copy.deepcopy(obj['pose_world']);position=pose['position']
        if drop_test and oid=='green_block':
            if bowl is None:raise ValueError('Container drop check needs bowl')
            position[:2]=bowl['pose_world']['position'][:2]
            position[2]=bowl['pose_world']['position'][2]+max(bowl['extents'])/2+.12
        body=ET.SubElement(world,'body',name=oid,pos=' '.join(map(str,position)),quat=' '.join(map(str,pose['quat_wxyz'])))
        if not (drop_test and oid=='bowl'):ET.SubElement(body,'freejoint',name=oid+'_free')
        ET.SubElement(assets,'mesh',name=oid+'_visual',file=obj['mesh'])
        ET.SubElement(body,'geom',name=oid+'_visual',type='mesh',mesh=oid+'_visual',contype='0',conaffinity='0',mass='0',group='1')
        for j,path in enumerate(obj['colliders']):
            name=oid+f'_collider_{j:03d}'
            ET.SubElement(assets,'mesh',name=name,file=path)
            ET.SubElement(body,'geom',name=name,type='mesh',mesh=name,density='500',friction='.8 .005 .0001',group='3')
    return doc


def validate(root,scene):
    import mujoco
    import trimesh
    root=Path(root);doc=scene_xml(root,scene)
    # MuJoCo accepts OBJ/STL/MSH visual meshes, not PLY. Keep a separate OBJ reference.
    for mesh in doc.findall('./asset/mesh'):
        path=Path(mesh.attrib['file'])
        if path.suffix=='.ply':
            destination=path.with_suffix('.obj')
            if not (root/destination).exists():trimesh.load(root/path,force='mesh').export(root/destination)
            mesh.set('file',str(destination))
    ET.ElementTree(doc).write(root/'scene.xml',encoding='unicode')
    model=mujoco.MjModel.from_xml_path(str((root/'scene.xml').resolve()));data=mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    initial={obj['id']:data.xpos[model.body(obj['id']).id].copy() for obj in scene['objects']}
    history={oid:[] for oid in initial}
    for i in range(1500):
        mujoco.mj_step(model,data)
        if i>=1300:
            for oid in initial:
                body=model.body(oid).id;history[oid].append((data.xpos[body].copy(),data.xquat[body].copy()))
    def span(samples):
        positions=np.array([p for p,q in samples]);quats=np.array([q for p,q in samples])
        return float(np.linalg.norm(np.ptp(positions,axis=0))),float((2*np.arccos(np.clip(np.abs(quats@quats[0]),0,1))).max())
    settling=[]
    for oid,start in initial.items():
        end=data.xpos[model.body(oid).id].copy();speed=float(np.linalg.norm(data.cvel[model.body(oid).id]))
        position_span,rotation_span=span(history[oid])
        settling.append({'id':oid,'displacement_m':float(np.linalg.norm(end-start)),'end_position_m':end.tolist(),'body_speed_norm':speed,
                         'position_span_m':position_span,'rotation_span_rad':rotation_span,
                         'passed':bool(np.linalg.norm(end-start)<.02 and position_span<.002 and rotation_span<.05)})
    result={'schema':'showonce_physics_checks_v1','backend':'MuJoCo '+mujoco.__version__,'dt_s':.002,'settling_seconds':3.,
            'settling':settling,'parameters_are_assumptions':True,'isaac_physx_executed':False}
    if {'bowl','green_block'}<=set(initial):
        drop=scene_xml(root,scene,drop_test=True)
        for mesh in drop.findall('./asset/mesh'):
            if mesh.attrib['file'].endswith('.ply'):mesh.set('file',mesh.attrib['file'][:-4]+'.obj')
        ET.ElementTree(drop).write(root/'container-check.xml',encoding='unicode')
        model=mujoco.MjModel.from_xml_path(str((root/'container-check.xml').resolve()));data=mujoco.MjData(model)
        cube_id=model.body('green_block').id;bowl_id=model.body('bowl').id
        contacts=0;history=[]
        for i in range(2000):
            mujoco.mj_step(model,data)
            if i>=1800:
                history.append((data.xpos[cube_id].copy(),data.xquat[cube_id].copy()))
                for contact in data.contact:
                    bodies={int(model.geom_bodyid[contact.geom1]),int(model.geom_bodyid[contact.geom2])}
                    if bodies=={cube_id,bowl_id}:contacts+=1;break
        bowl=next(obj for obj in scene['objects'] if obj['id']=='bowl')
        mesh=trimesh.load(root/bowl['mesh'],force='mesh',process=False)
        from scipy.spatial.transform import Rotation
        q=bowl['pose_world']['quat_wxyz'];rotation=Rotation.from_quat([*q[1:],q[0]]).as_matrix()
        vertices=np.asarray(mesh.vertices)@rotation.T+np.asarray(bowl['pose_world']['position'])
        top=float(vertices[:,2].max());floor=float(vertices[:,2].min());cube=data.xpos[cube_id].copy();bcenter=data.xpos[bowl_id]
        xy=float(np.linalg.norm(cube[:2]-bcenter[:2]));speed=float(np.linalg.norm(data.cvel[cube_id]))
        position_span,rotation_span=span(history)
        passed=bool(contacts>=160 and floor<cube[2]<top-.008 and xy<max(bowl['extents'][:2])*.3 and position_span<.002 and rotation_span<.05)
        result['container_drop']={'passed':passed,'bowl_fixed_for_test':True,'seconds':4.,'cube_position_m':cube.tolist(),
            'bowl_top_m':top,'bowl_bottom_m':floor,'final_200_steps_with_cube_bowl_contact':contacts,'cube_speed_norm':speed,
            'position_span_m':position_span,'rotation_span_rad':rotation_span}
    result['passed']=all(item['passed'] for item in settling) and result.get('container_drop',{}).get('passed',False)
    return result
