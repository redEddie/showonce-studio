"""Versioned reference assets; recording-specific poses never enter the library."""
from pathlib import Path
import copy
import fcntl
import hashlib
import json
import re
import shutil
import tempfile
import time
from showonce_studio.config import ROOT
from showonce_studio.showonce_stages import read_json,write_json,recording_path

LIBRARY=ROOT/'datasets/asset_library'
ROLES=('green_block','bowl')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def bundle(recording):return Path(recording)/'geometry/sam3d_assets_v1'
def selection(recording):return read_json(Path(recording)/'asset_selection.json')


class AssetLibrary:
    def __init__(self,root=LIBRARY):self.root=Path(root)

    def folder(self,asset_id,version):
        if not isinstance(asset_id,str)or not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,63}',asset_id):raise ValueError('잘못된 에셋 ID입니다.')
        if not isinstance(version,str)or not re.fullmatch(r'v[1-9][0-9]*',version):raise ValueError('잘못된 에셋 버전입니다.')
        path=(self.root/asset_id/version).resolve()
        if self.root.resolve()not in path.parents:raise ValueError('보관함 밖의 경로입니다.')
        return path

    def get(self,asset_id,version,verify=False):
        folder=self.folder(asset_id,version);manifest=read_json(folder/'manifest.json')
        if not manifest:raise ValueError('등록되지 않은 에셋입니다.')
        if verify:
            for name,digest in manifest['files'].items():
                if sha(self.artifact(asset_id,version,name))!=digest:raise ValueError('에셋 내용이 등록된 버전과 다릅니다.')
        return manifest

    def catalog(self):
        return [read_json(p)for p in sorted(self.root.glob('*/v*/manifest.json'))]

    def artifact(self,asset_id,version,name):
        if name not in ('manifest.json','mesh.ply','mesh.obj','mesh.glb','thumbnail.png','points.json','object.usd')and not re.fullmatch(r'colliders/part_[0-9]{3}\.obj',name):
            raise ValueError('잘못된 에셋 파일입니다.')
        folder=self.folder(asset_id,version);path=(folder/name).resolve()
        if folder not in path.parents:raise ValueError('에셋 폴더 밖의 경로입니다.')
        return path

    def register(self,recording,role,asset_id):
        recording=Path(recording);source=bundle(recording);manifest=read_json(source/'manifest.json')
        asset=next(a for a in manifest['assets']if a['id']==role)
        self.folder(asset_id,'v1');self.root.mkdir(parents=True,exist_ok=True)
        files={'mesh.ply':source/asset['mesh_ply'],'mesh.obj':source/'assets'/role/'mesh.obj','thumbnail.png':source/asset['thumbnail']}
        preview=recording/'geometry/optimized_assets_v1/web_preview/assets'/role/'mesh.glb'
        files['mesh.glb']=preview if preview.is_file()else source/asset['mesh']
        for i,path in enumerate(asset['collision_meshes']):files[f'colliders/part_{i:03d}.obj']=source/path
        usd=source/'objects'/role/'obj.usd'
        if usd.is_file():files['object.usd']=usd
        digests={name:sha(path)for name,path in files.items()}
        content=hashlib.sha256(json.dumps(digests,sort_keys=True).encode()).hexdigest()
        with (self.root/'.register.lock').open('a')as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            versions=[m for m in self.catalog()if m['asset_id']==asset_id]
            for existing in versions:
                if existing['content_sha256']==content:return existing
            version='v'+str(max((int(v['version'][1:])for v in versions),default=0)+1)
            target=self.folder(asset_id,version);target.parent.mkdir(parents=True,exist_ok=True)
            temp=Path(tempfile.mkdtemp(prefix='.register-',dir=target.parent))
            try:
                for name,path in files.items():
                    destination=temp/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination)
                write_json(temp/'points.json',{'positions':[],'source':'A reference asset; no measurements from the current recording.'})
                digests['points.json']=sha(temp/'points.json')
                static={key:copy.deepcopy(asset[key])for key in ('name','color','prompt','extent_m','units','frame','watertight','generated_hidden_surfaces')if key in asset}
                static.update(id=role,variant='library',library_asset_id=asset_id,library_version=version,mesh='mesh.glb',mesh_ply='mesh.ply',thumbnail='thumbnail.png',
                    point_cloud='points.json',measured_points=0,fusion_frame_indices=[],generation_frames=[],total_frames=0,
                    vertices=asset['vertices'],triangles=asset['triangles'],reference_triangles=asset['triangles'],collider_parts=asset['collider_parts'],
                    reference_frame_index=None,reference_time_s=None,median_diagnostic_surface_distance_m=None,source='등록된 참조 에셋',physics_validated=asset.get('physics_validated',False))
                result={'schema':'showonce_library_asset_v1','asset_id':asset_id,'version':version,'role':role,'content_sha256':content,
                    'reference_mesh_sha256':digests['mesh.ply'],'created_s':time.time(),'asset':static,'files':digests,
                    'provenance':{'recording_id':recording.name,'source_manifest_sha256':sha(source/'manifest.json'),
                        'source_model':manifest.get('model'),'source_revisions':manifest.get('source_revisions'),'weights_revision':manifest.get('weights_revision'),
                        'dimensions_corrected':False,'pose_is_recording_specific':True}}
                write_json(temp/'manifest.json',result);temp.rename(target);return result
            except Exception:shutil.rmtree(temp);raise

    def bind(self,recording,mapping):
        recording=Path(recording)
        if not isinstance(mapping,dict)or set(mapping)!=set(ROLES):raise ValueError('옮길 물체와 목표 그릇을 선택하세요.')
        selected={}
        for role in ROLES:
            item=mapping[role]
            if not isinstance(item,dict):raise ValueError('잘못된 에셋 선택입니다.')
            meta=self.get(item.get('asset_id'),item.get('version'),verify=True)
            if meta['role']!=role:raise ValueError('선택한 에셋의 역할이 다릅니다.')
            selected[role]={'asset_id':meta['asset_id'],'version':meta['version'],'reference_mesh_sha256':meta['reference_mesh_sha256']}
        digest=hashlib.sha256(json.dumps(selected,sort_keys=True).encode()).hexdigest();existing=selection(recording)
        if existing and existing['selection_sha256']==digest:return existing
        if (bundle(recording)/'manifest.json').exists()or (recording/'geometry/foundationpose_v1/tracking.json').exists():
            raise ValueError('분석한 영상의 에셋은 보존합니다. 새 영상에서 에셋을 선택하세요.')
        target=bundle(recording);target.parent.mkdir(parents=True,exist_ok=True)
        temp=Path(tempfile.mkdtemp(prefix='.references-',dir=target.parent));assets=[];objects=[]
        try:
            for role,ref in selected.items():
                meta=self.get(ref['asset_id'],ref['version']);origin=self.folder(ref['asset_id'],ref['version'])
                asset=copy.deepcopy(meta['asset']);asset.update(variant='sam3d_assets_v1',source='에셋 보관함',reference_frame_index=0,
                    reference_time_s=None,pose_base_at_reference=None,reference_triangles=meta['asset']['reference_triangles'])
                for field,name in [('mesh','mesh.glb'),('mesh_ply','mesh.ply'),('thumbnail','thumbnail.png'),('point_cloud','points.json')]:
                    destination=temp/'assets'/role/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(origin/name,destination)
                    asset[field]=str(destination.relative_to(temp))
                shutil.copy2(origin/'mesh.obj',temp/'assets'/role/'mesh.obj')
                colliders=[]
                for name in sorted(meta['files']):
                    if not name.startswith('colliders/'):continue
                    destination=temp/'objects'/role/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(origin/name,destination)
                    colliders.append(str(destination.relative_to(temp)))
                asset['collision_meshes']=colliders;assets.append(asset)
                objects.append({'id':role,'label':asset['name'],'mesh':asset['mesh_ply'],'extents':asset['extent_m'],'colliders':colliders,'pose_world':None})
            write_json(temp/'manifest.json',{'schema':'showonce_reference_assets_v1','version':'sam3d_assets_v1','recording_id':recording.name,
                'complete':True,'replay_ready':False,'simulation_ready':False,'physics_validated':False,'source':'asset_library',
                'created_s':time.time(),'assets':assets,'asset_selection_sha256':digest,'table_height_base_m':read_json(recording/'calibration.json')['table_check']['height_m'],
                'note':'Reference meshes copied from immutable library versions; no new mesh reconstruction.'})
            write_json(temp/'scene.json',{'objects':objects,'pose_source':'awaiting_new_recording_tracking'})
            temp.rename(target)
            value={'schema':'showonce_asset_selection_v1','recording_id':recording.name,'selection_sha256':digest,'assets':selected,'created_s':time.time()}
            write_json(recording/'asset_selection.json',value);return value
        except Exception:
            if temp.exists():shutil.rmtree(temp)
            raise


def object_specs(recording):
    selected=selection(recording)
    if not selected:return None
    manifest=read_json(bundle(recording)/'manifest.json')
    return [{key:a[key]for key in ('id','name','prompt','color')}for a in manifest['assets']]


def finalize_replay(recording,tracking):
    """Current RGB-D measurement and current poses only; never source clip poses."""
    recording=Path(recording);manifest=read_json(bundle(recording)/'manifest.json')
    if not manifest or manifest.get('source')!='asset_library':return
    import cv2
    import numpy as np
    from scipy.spatial.transform import Rotation
    camera=read_json(recording/'manifest.json');intr=camera['camera']['intrinsics'];scale=camera['depth']['meters_per_unit']
    T=np.asarray(read_json(recording/'calibration.json')['T_base_cam']);table=float(manifest['table_height_base_m'])
    scene=read_json(bundle(recording)/'scene.json');rows,_=source_frames_local(recording)
    for asset,obj in zip(manifest['assets'],scene['objects']):
        frame=next((f for f in tracking['frames']if f['objects'][asset['id']]['pose_base']is not None),None)
        if frame is None:raise ValueError(asset['name']+'의 유효한 포즈를 찾지 못했습니다. 오버레이와 영상을 확인하세요.')
        index=frame['index'];pose=np.asarray(frame['objects'][asset['id']]['pose_base']);world=pose.copy();world[2,3]-=table
        q=Rotation.from_matrix(world[:3,:3]).as_quat();obj['pose_world']={'position':world[:3,3].tolist(),'quat_wxyz':[float(q[3]),*q[:3].tolist()]}
        asset.update(reference_frame_index=index,reference_time_s=frame['time_s'],pose_base_at_reference=pose.tolist(),total_frames=len(rows))
        depth=cv2.imread(str(recording/rows[index]['depth']),-1).astype(np.float32)*scale
        mask=cv2.imread(str(recording/'geometry/rgbd_surface_v1/masks'/asset['id']/f'{index:06d}.png'),0)>0
        y,x=np.nonzero(mask&(depth>.1)&(depth<2));step=max(1,len(x)//2500);x=x[::step];y=y[::step];z=depth[y,x]
        points=np.column_stack(((x-intr['cx'])*z/intr['fx'],(y-intr['cy'])*z/intr['fy'],z))@T[:3,:3].T+T[:3,3]
        local=(points-pose[:3,3])@pose[:3,:3];asset['measured_points']=len(local)
        write_json(bundle(recording)/asset['point_cloud'],{'positions':local.reshape(-1).tolist(),'source_recording_id':recording.name,'frame_index':index})
        write_json(bundle(recording)/'assets'/asset['id']/'asset.json',asset)
    manifest.update(replay_ready=True,created_s=time.time());scene['pose_source']='FoundationPose on this recording; no source clip poses or table snap'
    write_json(bundle(recording)/'scene.json',scene);write_json(bundle(recording)/'manifest.json',manifest)

def source_frames_local(recording):
    from showonce_studio.showonce_stages import source_frames
    return source_frames(recording)
