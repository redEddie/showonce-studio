"""SAM 3D adapter for recorded RGB-D; no camera or robot connections.

The real2sim reconstruction/coordinate conversion and USD writer are used from
the pinned gibeom25 checkout. Generated geometry remains separate from scans.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
import zipfile
from showonce_studio.showonce_stages import read_json, write_json, source_frames
from showonce_studio.showonce_geometry import directory as measured_directory, transform

from showonce_studio.config import ROOT,THIRD_PARTY_ROOT,PACKAGE_ROOT,models_root
VERSION='sam3d_assets_v1'
SAM_ROOT=THIRD_PARTY_ROOT/'sam-3d-objects'
REAL2SIM=THIRD_PARTY_ROOT/'real2sim'
WEIGHTS=Path(os.environ.get('SHOWONCE_SAM3D_WEIGHTS',str(models_root()/'SAM-3D-Objects/checkpoints'))).expanduser()
REQUIRED=('pipeline.yaml','ss_generator.ckpt','slat_generator.ckpt','ss_decoder.ckpt',
          'slat_decoder_mesh.ckpt','slat_decoder_gs.ckpt')


def directory(recording):return Path(recording)/'geometry'/VERSION


def checkpoint_status():
    missing=[name for name in REQUIRED if not (WEIGHTS/name).is_file() or (WEIGHTS/name).stat().st_size<10]
    return {'ready':not missing,'missing':missing,'repository':'facebook/sam-3d-objects',
            'access_url':'https://huggingface.co/facebook/sam-3d-objects'}


def download_checkpoints():
    from huggingface_hub import HfApi,snapshot_download
    info=HfApi().model_info('facebook/sam-3d-objects')
    snapshot_download('facebook/sam-3d-objects',revision=info.sha,local_dir=str(WEIGHTS.parent),
                      allow_patterns=['checkpoints/*','LICENSE','README.md'],max_workers=2)
    write_json(WEIGHTS.parent/'download.json',{'repository':'facebook/sam-3d-objects','revision':info.sha,'finished_s':time.time()})


def source_hash(recording):
    return hashlib.sha256((recording/'frames.jsonl').read_bytes()+(recording/'calibration.json').read_bytes()).hexdigest()


def prepare(recording):
    """Select pre-contact keyframes; preserve actual RGB, aligned metric depth and SAM masks."""
    import cv2
    import numpy as np
    import open3d as o3d
    recording=Path(recording);root=directory(recording)
    measured=read_json(measured_directory(recording)/'tracking.json')
    if not measured or not measured.get('complete'):raise ValueError('먼저 영상 객체 추적을 완료하세요.')
    manifest=read_json(recording/'manifest.json');rows,times=source_frames(recording)
    task=read_json(recording/'task_definition.json',{})
    end=task.get('boundaries',{}).get('grasp_start',{}).get('time_s',min(1.4,times[-1]))-.15
    if end<=0:raise ValueError('파지 전 관측 구간이 부족합니다.')
    k=manifest['camera']['intrinsics'];scale=manifest['depth']['meters_per_unit'];objects=[]
    for spec in measured['objects']:
        oid=spec['id'];ranked=[]
        for i,row in enumerate(rows):
            observation=measured['frames'][i]['objects'][oid]
            if times[i]>end or observation['status']!='observed':continue
            rgb=cv2.imread(str(recording/row['rgb']))
            depth=cv2.imread(str(recording/row['depth']),-1).astype(np.float32)*scale
            mask=cv2.imread(str(measured_directory(recording)/'masks'/oid/f'{i:06d}.png'),0)>0
            clean=cv2.erode(mask.astype(np.uint8),np.ones((5,5),np.uint8))>0
            valid=clean&(depth>.15)&(depth<1.8)
            if valid.sum()<80:continue
            sharp=float(cv2.Laplacian(cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY),cv2.CV_32F)[clean].var())
            score=float(valid.sum()*min(1.,sharp/80.))
            ranked.append((score,i))
        selected=[]
        for score,i in sorted(ranked,reverse=True):
            if all(abs(times[i]-times[j])>=.3 for _,j in selected):selected.append((score,i))
            if len(selected)==3:break
        if len(selected)<2:raise ValueError(spec['name']+'에 서로 다른 시각의 유효 RGB-D가 부족합니다.')
        frames=[]
        for score,i in selected:
            row=rows[i];dest=root/'inputs'/oid/f'{i:06d}';dest.mkdir(parents=True,exist_ok=True)
            rgb=cv2.imread(str(recording/row['rgb']));depth=cv2.imread(str(recording/row['depth']),-1).astype(np.float32)*scale
            mask=cv2.imread(str(measured_directory(recording)/'masks'/oid/f'{i:06d}.png'),0)
            yy,xx=np.indices(depth.shape);pointmap=np.stack(((xx-k['cx'])*depth/k['fx'],(yy-k['cy'])*depth/k['fy'],depth),-1).astype(np.float32)
            pointmap[depth<=0]=np.nan
            valid=(cv2.erode(mask,np.ones((5,5),np.uint8))>0)&(depth>.15)&(depth<1.8)
            if oid=='green_block':
                hsv=cv2.cvtColor(rgb,cv2.COLOR_BGR2HSV);valid&=(hsv[:,:,0]>=30)&(hsv[:,:,0]<=100)&(hsv[:,:,1]>=45)
            points=pointmap[valid];colors=cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB)[valid]/255.
            cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points));cloud.colors=o3d.utility.Vector3dVector(colors)
            cloud,_=cloud.remove_statistical_outlier(nb_neighbors=16,std_ratio=1.8)
            o3d.io.write_point_cloud(str(dest/'measured_camera.ply'),cloud)
            cv2.imwrite(str(dest/'rgb.png'),rgb);cv2.imwrite(str(dest/'mask.png'),mask)
            np.save(dest/'depth.npy',depth);np.save(dest/'pointmap_camera.npy',pointmap)
            write_json(dest/'intrinsics.json',{key:k[key] for key in ('fx','fy','cx','cy','width','height')})
            frames.append({'index':i,'time_s':times[i],'rank_score':score,'points':len(cloud.points),'directory':str(dest.relative_to(root)),
                'rgb_sha256':hashlib.sha256((dest/'rgb.png').read_bytes()).hexdigest(),
                'depth_sha256':hashlib.sha256((dest/'depth.npy').read_bytes()).hexdigest(),
                'mask_sha256':hashlib.sha256((dest/'mask.png').read_bytes()).hexdigest()})
        objects.append(dict(spec,keyframes=frames))
    inputs={'schema':'showonce_sam3d_inputs_v1','recording_id':recording.name,'source_hash':source_hash(recording),
        'source_frames':len(rows),'objects':objects,'pointmap_convention':'OpenCV x right, y down, z forward; meters',
        'intrinsics':k,'depth_scale_m':scale,'selection':'pre-contact visible object, valid depth, image sharpness, >=0.3s separation',
        'selection_end_s':end,'confirmed_stages_modified':False}
    write_json(root/'inputs.json',inputs)
    return inputs


def load_reconstructor():
    """Use Kim's inference/output conversion; avoid optional notebook/Gradio render dependencies."""
    os.environ.setdefault('LIDRA_SKIP_INIT','true')
    os.environ.setdefault('ATTN_BACKEND','sdpa');os.environ.setdefault('SPARSE_ATTN_BACKEND','sdpa')
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    sys.path[:0]=[str(REAL2SIM),str(SAM_ROOT),str(PACKAGE_ROOT/'_vendor/kaolin-check-tensor')]
    from real2sim.reconstruction import Sam3DReconstructor
    from sam3d_objects.pipeline.inference_pipeline_pointmap import camera_to_pytorch3d_camera
    from pytorch3d.transforms import Transform3d
    config=OmegaConf.load(WEIGHTS/'pipeline.yaml')
    config.rendering_engine='pytorch3d';config.compile_model=False;config.workspace_dir=str(WEIGHTS)
    # D435i provides the metric pointmap. Do not download/run monocular MoGe weights.
    config.depth_model=None
    reconstructor=Sam3DReconstructor.__new__(Sam3DReconstructor)
    pipeline=instantiate(config)
    class VertexColorPipeline:
        def run(self,*args,**kwargs):
            # Upstream's optional 1000-view hole filling drops vertex colors and
            # may alter the container opening. Keep the model's actual mesh.
            kwargs['with_mesh_postprocess']=False
            return pipeline.run(*args,**kwargs)
    reconstructor.pipeline=VertexColorPipeline()
    reconstructor.layout_postprocess=False  # visible-surface RGB-D ICP below
    rotation=camera_to_pytorch3d_camera(device='cuda').rotation
    reconstructor._cv_to_p3d=Transform3d().rotate(rotation).to('cuda')
    return reconstructor


def mesh_distance(mesh,points):
    import numpy as np
    import open3d as o3d
    scene=o3d.t.geometry.RaycastingScene()
    legacy=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),o3d.utility.Vector3iVector(mesh.faces))
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy))
    return scene.compute_distance(o3d.core.Tensor(np.asarray(points,dtype=np.float32))).numpy()


def export_physics(root,assets,table_height):
    """Write actual convex parts, USD and MuJoCo; no single hull across the bowl cavity."""
    import numpy as np
    import trimesh
    import coacd
    from scipy.spatial.transform import Rotation
    sys.path.insert(0,str(REAL2SIM))
    from real2sim.usd_export import export_usd_scene
    from pxr import Usd,UsdGeom,UsdPhysics,Gf,Vt
    objects=[]
    for asset in assets:
        oid=asset['id'];folder=root/'objects'/oid;folder.mkdir(parents=True,exist_ok=True)
        mesh=trimesh.load(root/asset['mesh_ply'],force='mesh',process=False)
        if not mesh.is_watertight:raise ValueError(oid+' SAM 메시가 닫힌 표면이 아닙니다. 물리 에셋으로 완료 처리하지 않습니다.')
        parts=coacd.run_coacd(coacd.Mesh(np.asarray(mesh.vertices),np.asarray(mesh.faces)),threshold=.03,
            max_convex_hull=64,resolution=2000,mcts_iterations=100,seed=42)
        collision=folder/'colliders';collision.mkdir(exist_ok=True);paths=[]
        for j,(v,f) in enumerate(parts):
            part=trimesh.Trimesh(v,f,process=False);path=collision/f'part_{j:03d}.obj';part.export(path);paths.append(str(path.relative_to(root)))
        T=np.asarray(asset['pose_base_at_reference']).copy();T[2,3]-=table_height
        raw_pose=T.copy()
        bottom=float(transform(mesh.vertices,T)[:,2].min())
        # These selected frames are before grasp, with both objects supported
        # by the table. Apply Kim's support-plane initialization only to the
        # simulation scene, preserving scan alignment and source poses for review.
        correction=.001-bottom
        if abs(correction)>.03:raise ValueError(oid+' 테이블 지지면 보정이 30mm를 넘습니다. 배치를 검토해야 합니다.')
        T[2,3]+=correction
        asset['simulation_initialization_dz_m']=correction
        quat=Rotation.from_matrix(T[:3,:3]).as_quat();q=[float(quat[3]),*map(float,quat[:3])]
        objects.append({'id':oid,'label':asset['name'],'mesh':asset['mesh_ply'],'extents':asset['extent_m'],
            'pose_world':{'position':T[:3,3].tolist(),'quat_wxyz':q},'colliders':paths,
            'raw_pose_world_matrix':raw_pose.tolist(),'support_plane_initialization_dz_m':correction})
        asset.update(collider_parts=len(paths),collision_meshes=paths)
    scene={'objects':objects,'camera':{'intrinsics':read_json(root/'inputs.json')['intrinsics'],
        'T_world_cam':(np.diag([1.,1.,1.,1.])).tolist()},'physics_assumptions':{'density_kg_m3':500.,'friction':.8,'measured':False}}
    # Camera extrinsic is installed by the caller from the recording calibration.
    calibration=read_json(root.parent.parent/'calibration.json')
    T=np.asarray(calibration['T_base_cam']);T[2,3]-=table_height;scene['camera']['T_world_cam']=T.tolist()
    write_json(root/'scene.json',scene);export_usd_scene(root)
    for obj in objects:
        stage=Usd.Stage.Open(str(root/'objects'/obj['id']/'obj.usd'))
        stage.RemovePrim('/Object/collision')
        for j,path in enumerate(obj['colliders']):
            mesh=trimesh.load(root/path,force='mesh',process=False)
            prim=UsdGeom.Mesh.Define(stage,f'/Object/collision_{j:03d}')
            prim.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(mesh.vertices,np.float32)))
            prim.CreateFaceVertexCountsAttr(Vt.IntArray([3]*len(mesh.faces)))
            prim.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(mesh.faces,np.int32).ravel()))
            prim.CreatePurposeAttr(UsdGeom.Tokens.guide)
            UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(prim.GetPrim()).CreateApproximationAttr('convexHull')
            from real2sim.usd_export import _add_physics_material
            from pxr import UsdShade
            UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(_add_physics_material(stage,'/Object/Looks/physics',.8),UsdShade.Tokens.weakerThanDescendants,'physics')
        stage.GetRootLayer().Save()
    return scene


def reconstruct(recording,progress):
    import numpy as np
    import open3d as o3d
    import trimesh
    from PIL import Image
    sys.path.insert(0,str(REAL2SIM))
    from real2sim import geometry as geo
    from real2sim.camera import Intrinsics
    from real2sim.reconstruction import clean_mesh
    root=directory(recording);inputs=read_json(root/'inputs.json') or prepare(recording)
    if inputs['source_hash']!=source_hash(recording):raise ValueError('원본 영상·보정이 변경되었습니다. 입력을 다시 준비하세요.')
    if not checkpoint_status()['ready']:raise ValueError('SAM 3D Objects 접근 승인·가중치 다운로드가 필요합니다.')
    progress('loading','SAM 3D Objects 로딩 중');reconstructor=load_reconstructor()
    calib=read_json(recording/'calibration.json');T_base_cam=np.asarray(calib['T_base_cam']);assets=[]
    for spec in inputs['objects']:
        oid=spec['id'];candidates=[];held_out=spec['keyframes'][-1]
        validation=np.asarray(o3d.io.read_point_cloud(str(root/held_out['directory']/'measured_camera.ply')).points)
        for entry in spec['keyframes'][:-1]:
            i=entry['index'];folder=root/entry['directory'];progress('generating',f"{spec['name']} · 원본 {i}프레임 형상 복원")
            rgb=np.asarray(Image.open(folder/'rgb.png').convert('RGB'));mask=np.asarray(Image.open(folder/'mask.png'))>0
            rec=reconstructor(rgb,mask,np.load(folder/'pointmap_camera.npy'),seed=42)
            # Preserve any anisotropic scale residual in vertices, not the rigid pose.
            local=transform(rec.vertices_cam,np.linalg.inv(rec.T_cam_local))
            raw=trimesh.Trimesh(local,rec.faces,vertex_colors=rec.vertex_colors,process=False)
            mesh=clean_mesh(raw.copy())
            if rec.vertex_colors is not None:
                from scipy.spatial import cKDTree
                nearest=cKDTree(raw.vertices).query(mesh.vertices)[1]
                mesh.visual.vertex_colors=rec.vertex_colors[nearest]
            obs=np.asarray(o3d.io.read_point_cloud(str(folder/'measured_camera.ply')).points)
            cam_mesh=mesh.copy();cam_mesh.apply_transform(rec.T_cam_local)
            before=float(np.median(mesh_distance(cam_mesh,obs)))
            visible=geo.visible_surface(cam_mesh.vertices,cam_mesh.faces,Intrinsics(**read_json(folder/'intrinsics.json')))
            adjustment,fit,rmse=geo.icp_refine(visible,obs)
            angle=np.arccos(np.clip((np.trace(adjustment[:3,:3])-1)/2,-1,1))
            adjusted=cam_mesh.copy();adjusted.apply_transform(adjustment)
            after=float(np.median(mesh_distance(adjusted,obs)))
            accepted=fit>=.5 and np.linalg.norm(adjustment[:3,3])<=.025 and angle<=np.deg2rad(15) and after<=before
            pose=(adjustment@rec.T_cam_local) if accepted else rec.T_cam_local
            final=mesh.copy();final.apply_transform(pose)
            distance=float(np.median(mesh_distance(final,validation)))
            dest=root/'candidates'/oid/f'{i:06d}';dest.mkdir(parents=True,exist_ok=True);mesh.export(dest/'mesh.glb')
            candidates.append((distance,mesh,pose,entry,{'fitness':fit,'rmse_m':rmse,'accepted':bool(accepted),
                'before_median_m':before,'after_median_m':after,'held_out_median_m':distance}))
        distance,mesh,pose,entry,metrics=min(candidates,key=lambda item:item[0])
        if distance>.015:raise ValueError(spec['name']+' 복원이 별도 프레임의 깊이와 15mm 이상 어긋납니다. 결과를 검토해야 합니다.')
        center=mesh.bounds.mean(0);mesh.apply_translation(-center);pose[:3,3]+=pose[:3,:3]@center
        folder=root/'assets'/oid;folder.mkdir(parents=True,exist_ok=True)
        mesh.export(folder/'mesh.glb');mesh.export(folder/'mesh.ply');mesh.export(folder/'mesh.obj')
        obs_cloud=o3d.io.read_point_cloud(str(root/entry['directory']/'measured_camera.ply'));obs_cloud.transform(np.linalg.inv(pose))
        o3d.io.write_point_cloud(str(folder/'measured.ply'),obs_cloud)
        write_json(folder/'points.json',{'positions':np.asarray(obs_cloud.points).round(6).ravel().tolist(),'units':'m','frame':'object_local'})
        from shutil import copy2
        copy2(measured_directory(recording)/'assets'/oid/'thumbnail.png',folder/'thumbnail.png')
        asset={**{k:spec[k] for k in ('id','name','color','prompt')},'variant':VERSION,'source':'SAM 3D Objects + RGB-D metric pointmap + gated visible-surface ICP',
            'mesh':f'assets/{oid}/mesh.glb','mesh_ply':f'assets/{oid}/mesh.ply','point_cloud':f'assets/{oid}/points.json',
            'thumbnail':f'assets/{oid}/thumbnail.png','measured_ply':f'assets/{oid}/measured.ply','extent_m':mesh.extents.tolist(),
            'units':'m','frame':'object_local','reference_frame_index':entry['index'],'reference_time_s':entry['time_s'],
            'pose_base_at_reference':(T_base_cam@pose).tolist(),'measured_points':len(obs_cloud.points),'vertices':len(mesh.vertices),'triangles':len(mesh.faces),
            'watertight':bool(mesh.is_watertight),'model_mesh_postprocess':False,'collider_parts':0,
            'generated_hidden_surfaces':True,'generation_frames':[e['index'] for e in spec['keyframes'][:-1]],
            'validation_frame_index':held_out['index'],'median_diagnostic_surface_distance_m':distance,'registration':metrics,
            'total_frames':inputs['source_frames'],'fusion_frame_indices':[],'completeness':'model_completed_geometry','physics_validated':False}
        assets.append(asset)
    return finalize(recording,assets,inputs['source_hash'],progress)


def finalize(recording,assets,input_hash,progress):
    root=directory(recording);calib=read_json(recording/'calibration.json')
    progress('exporting','충돌 형상·USD·시뮬레이터 검증 준비 중')
    table=float(calib.get('table_check',{}).get('height_m',-.015))
    try:
        scene=export_physics(root,assets,table)
        progress('validating','MuJoCo에서 정착·그릇 내부 낙하 검사 중')
        from showonce_studio.showonce_asset_physics import validate
        physics=validate(root,scene)
    except Exception as error:
        physics={'passed':False,'error':f'{type(error).__name__}: {error}','isaac_physx_executed':False}
    write_json(root/'physics-report.json',physics)
    for asset in assets:asset['physics_validated']=bool(physics['passed'])
    result={'schema':'showonce_simulation_assets_v1','complete':True,'version':VERSION,'recording_id':recording.name,
        'created_s':time.time(),'assets':assets,'source_hash':input_hash,'table_height_base_m':table,
        'physics_parameters_measured':False,'physics_validated':bool(physics['passed']),'simulation_ready':bool(physics['passed']),
        'physics_report':'physics-report.json','model':'facebook/sam-3d-objects',
        'weights_revision':read_json(WEIGHTS.parent/'download.json',{}).get('revision'),
        'source_revisions':read_json(ROOT/'release/sam3d-runtime.json',{}).get('source_revisions'),
        'note':'뒤쪽 형상은 모델 추정입니다. 깊이 정합 수치는 실물 정답 정확도가 아닙니다. 물리 파라미터는 설정값입니다.'}
    for asset in assets:write_json(root/'assets'/asset['id']/'asset.json',asset)
    write_json(root/'manifest.json',result)
    with zipfile.ZipFile(root/'simulation-assets.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob('*')):
            if path.is_file() and (path.name in ('scene.usd','scene.xml','container-check.xml','physics-report.json','scene.json','manifest.json','asset.json') or 'objects' in path.relative_to(root).parts or (path.name.startswith('mesh.') and 'assets' in path.relative_to(root).parts)):
                archive.write(path,path.relative_to(root))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','run','export'])
    parser.add_argument('--recording',type=Path,required=True);args=parser.parse_args();root=directory(args.recording)
    def progress(status,message):write_json(root/'job.json',{'status':status,'message':message,'pid':os.getpid(),'updated_s':time.time()});print(message,flush=True)
    try:
        if args.command=='export':
            manifest=read_json(root/'manifest.json')
            if not manifest or manifest['source_hash']!=source_hash(args.recording):raise ValueError('재사용할 SAM 3D 복원 결과가 없습니다.')
            finalize(args.recording,manifest['assets'],manifest['source_hash'],progress)
            progress('complete','기존 SAM 3D 메시의 물리 파일 재생성 완료');return
        inputs=prepare(args.recording)
        if args.command=='prepare':
            progress('prepared','RGB-D 키프레임 입력 준비 완료');return
        if not checkpoint_status()['ready']:
            progress('downloading','승인된 계정으로 SAM 3D Objects 가중치 다운로드 중')
            try:download_checkpoints()
            except Exception as error:
                from huggingface_hub.errors import GatedRepoError
                if isinstance(error,GatedRepoError):
                    progress('awaiting_approval','Hugging Face의 SAM 3D Objects 접근 승인이 필요합니다. 승인 후 다시 실행하세요.');return
                raise
        import fcntl
        with (ROOT/'runs/.showonce-vlm.lock').open('a') as lock:
            progress('waiting_for_gpu','GPU 분석 대기');fcntl.flock(lock,fcntl.LOCK_EX)
            reconstruct(args.recording,progress)
        progress('complete','SAM 3D 형상·시뮬레이션 파일 생성 완료')
    except Exception as error:
        progress('error',f'{type(error).__name__}: {error}');raise


if __name__=='__main__':main()
