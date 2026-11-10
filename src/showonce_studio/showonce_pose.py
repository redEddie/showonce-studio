"""Reference-mesh FoundationPose on saved RGB-D; no camera or robot access."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os
import sys
import time

from showonce_studio.showonce_stages import read_json,write_json,source_frames

from showonce_studio.config import ROOT,THIRD_PARTY_ROOT
VERSION='foundationpose_v1'

def directory(recording):return Path(recording)/'geometry'/VERSION
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def pose_valid(pose):
    import numpy as np
    p=np.asarray(pose)
    return bool(p.shape==(4,4) and np.isfinite(p).all() and
        np.allclose(p[3],[0,0,0,1],atol=1e-5) and np.allclose(p[:3,:3].T@p[:3,:3],np.eye(3),atol=2e-3) and
        abs(np.linalg.det(p[:3,:3])-1)<2e-3 and p[2,3]>.05)

def alignment_quality(render_depth,depth,mask):
    """Observed depth/silhouette evidence; not a calibrated pose probability."""
    import numpy as np
    rendered=render_depth>.05;mask=mask.astype(bool);valid=mask&rendered&(depth>.05)
    intersection=int((mask&rendered).sum());union=int((mask|rendered).sum())
    error=np.abs(depth[valid]-render_depth[valid])
    result={'mask_iou':intersection/max(1,union),'mask_coverage':intersection/max(1,int(mask.sum())),
        'render_coverage':intersection/max(1,int(rendered.sum())),'depth_pixels':int(valid.sum()),
        'median_depth_residual_m':float(np.median(error)) if len(error)else None,
        'occluded_render_fraction':float((rendered&(depth>.05)&(depth<render_depth-.012)).sum()/max(1,rendered.sum()))}
    result['accepted']=bool(result['mask_coverage']>=.35 and result['mask_iou']>=.15 and
        result['depth_pixels']>=40 and result['median_depth_residual_m']<=.020)
    return result


class PoseEngine:
    def __init__(self):self.scorer=self.refiner=self.context=None

    def load(self):
        if self.scorer is not None:return
        sys.path.insert(0,str(THIRD_PARTY_ROOT/'FoundationPose'))
        import torch
        from estimater import ScorePredictor,PoseRefinePredictor,dr
        torch.set_num_threads(4)
        self.scorer=ScorePredictor();self.refiner=PoseRefinePredictor();self.context=dr.RasterizeCudaContext()

    def run(self,recording,progress,limit=None):
        import cv2
        import numpy as np
        import trimesh
        recording=Path(recording);out=directory(recording)
        measured=recording/'geometry/rgbd_surface_v1';asset_root=recording/'geometry/sam3d_assets_v1'
        source=read_json(measured/'tracking.json');assets=read_json(asset_root/'manifest.json')
        if not source or not source.get('complete') or not assets or not assets.get('complete'):
            raise ValueError('완료된 SAM 영상 마스크와 참조 에셋이 필요합니다.')
        recording_manifest=read_json(recording/'manifest.json');calib=read_json(recording/'calibration.json')
        intr=recording_manifest['camera']['intrinsics'];scale=recording_manifest['depth']['meters_per_unit']
        K=np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1]],np.float32)
        T_base_cam=np.asarray(calib['T_base_cam'],float)
        rows,times=source_frames(recording)
        if limit:rows,times=rows[:limit],times[:limit]
        if len(source['frames'])<len(rows):raise ValueError('SAM masks and recording timestamps differ')
        progress('loading_model',message='FoundationPose 모델을 준비합니다.')
        self.load()
        import torch
        from estimater import FoundationPose,nvdiffrast_render
        frames=[{'index':i,'time_s':float(t),'objects':{}}for i,t in enumerate(times)]
        references=[];started=time.time()
        for asset in assets['assets']:
            oid=asset['id'];model=trimesh.load(asset_root/asset['mesh_ply'],force='mesh',process=False)
            # Keep the complete reference model, its meters and its original
            # local origin. Never feed display LOD or collision shapes to FP.
            estimator=FoundationPose(model_pts=model.vertices,model_normals=model.vertex_normals,mesh=model,
                scorer=self.scorer,refiner=self.refiner,glctx=self.context,debug=0,debug_dir=str(out/'debug'/oid))
            lod_path=recording/'geometry/optimized_assets_v1/web_preview/assets'/oid/'mesh.glb'
            lod=trimesh.load(lod_path,force='mesh',process=False) if lod_path.is_file()else model.simplify_quadric_decimation(face_count=1000)
            overlay={'vertices':np.asarray(lod.vertices).round(7).tolist(),'edges':np.asarray(lod.edges_unique).tolist(),
                'bounds_m':model.bounds.tolist(),'units':'m','frame':'reference_asset_local'}
            write_json(out/'overlays'/f'{oid}.json',overlay)
            references.append({'id':oid,'name':asset['name'],'reference_mesh':asset['mesh_ply'],
                'reference_mesh_sha256':sha(asset_root/asset['mesh_ply']),'reference_triangles':len(model.faces),
                'overlay':f'overlays/{oid}.json','mesh_frame':'sam3d_asset_local','units':'m','scale_changed':False,
                'symmetry_note':'Appearance/geometry symmetry can make orientation ambiguous; no unobserved exact orientation claimed.'})
            misses=0;last_register=-999;last_good=None
            for i,row in enumerate(rows):
                item=copy.deepcopy(source['frames'][i]['objects'][oid])
                # Old reconstruction poses use a DIFFERENT object frame.
                for key in ('pose_base','pose_status','pose_method','fitness','rmse_m'):item.pop(key,None)
                item.update(pose_base=None,pose_cam=None,pose_status='unresolved',pose_method='foundationpose',mesh_frame='sam3d_asset_local')
                rgb=cv2.cvtColor(cv2.imread(str(recording/row['rgb'])),cv2.COLOR_BGR2RGB)
                depth=cv2.imread(str(recording/row['depth']),cv2.IMREAD_UNCHANGED).astype(np.float32)*scale
                mask_image=cv2.imread(str(measured/'masks'/oid/f'{i:06d}.png'),cv2.IMREAD_GRAYSCALE)
                if mask_image is None:raise ValueError(f'Missing saved SAM mask: {oid}/{i}')
                mask=mask_image>0;valid=mask&(depth>.05)&(depth<2)
                previous=estimator.pose_last.clone() if estimator.pose_last is not None else None
                tick=time.perf_counter()
                if valid.sum()>=50:
                    registration=estimator.pose_last is None or (misses>=3 and i-last_register>=15)
                    mode='registration' if registration else 'tracking'
                    if registration:
                        candidate=estimator.register(K=K,rgb=rgb,depth=depth,ob_mask=mask,iteration=5);last_register=i
                    else:candidate=estimator.track_one(rgb=rgb,depth=depth,K=K,iteration=2)
                    candidate=np.asarray(candidate,float).reshape(4,4)
                    quality={'accepted':False,'reason':'invalid_transform'}
                    if pose_valid(candidate):
                        # The renderer consumes the CENTERED estimator mesh.
                        centered=candidate@np.linalg.inv(estimator.get_tf_to_centered_mesh().cpu().numpy())
                        _,rendered,_=nvdiffrast_render(K=K,H=depth.shape[0],W=depth.shape[1],
                            ob_in_cams=torch.tensor(centered[None],dtype=torch.float32,device='cuda'),
                            glctx=self.context,mesh_tensors=estimator.mesh_tensors,extra={})
                        quality=alignment_quality(rendered[0].cpu().numpy(),depth,mask)
                    item.update(candidate_pose_cam=candidate.tolist()if np.isfinite(candidate).all()else None,quality=quality,inference_mode=mode)
                    if quality['accepted']:
                        item.update(pose_cam=candidate.tolist(),pose_base=(T_base_cam@candidate).tolist(),pose_status='estimated')
                        last_good=i;misses=0
                    else:
                        estimator.pose_last=previous;misses+=1
                else:
                    item.update(quality={'accepted':False,'reason':'insufficient_mask_depth'},inference_mode='skipped');misses+=1
                item['inference_ms']=(time.perf_counter()-tick)*1000
                item['last_accepted_frame']=last_good;frames[i]['objects'][oid]=item
                if i%10==0 or i==len(rows)-1:
                    progress('tracking',message=asset['name']+' FoundationPose 추적 중',object_id=oid,frame=i+1,total=len(rows))
                    write_json(out/'tracking.partial.json',{'schema':'showonce_reference_pose_tracks_v1','complete':False,'frames':frames,'objects':source['objects']})
            del estimator
        summary={}
        for spec in source['objects']:
            observations=[f['objects'][spec['id']]for f in frames];accepted=[o for o in observations if o['pose_base'] is not None]
            summary[spec['id']]={'frames':len(observations),'accepted':len(accepted),'unresolved':len(observations)-len(accepted),
                'median_mask_iou':float(np.median([o['quality']['mask_iou']for o in accepted]))if accepted else None,
                'median_depth_residual_mm':float(np.median([o['quality']['median_depth_residual_m']for o in accepted])*1000)if accepted else None}
        result={'schema':'showonce_reference_pose_tracks_v1','recording_id':recording.name,'frames':frames,
            'objects':source['objects'],'complete':limit is None or limit>=recording_manifest['frames_written'],
            'pose_backend':'foundationpose','mesh_frame':'sam3d_asset_local','reference_assets':references,
            'source':'FoundationPose register + track_one on recorded RGB-D; saved SAM3 masks for initialization and validation',
            'source_frames_sha256':sha(recording/'frames.jsonl'),'calibration_sha256':sha(recording/'calibration.json'),
            'source_assets_sha256':sha(asset_root/'manifest.json'),'source_masks_tracking_sha256':sha(measured/'tracking.json'),
            'intrinsics':intr,'T_cam_base':np.linalg.inv(T_base_cam).tolist(),
            'asset_selection_sha256':assets.get('asset_selection_sha256'),
            'raw_poses_preserved':True,'trajectory_smoothing_applied':False,'ground_truth_accuracy_verified':False,
            'parameters':{'registration_refine_iterations':5,'tracking_refine_iterations':2,
                'mask_coverage_min':.35,'mask_iou_min':.15,'valid_depth_pixels_min':40,'median_depth_residual_max_m':.020,
                'quality_is_calibrated_probability':False,'world_frame':'robot_base','units':'m'},
            'summary':summary,'elapsed_s':time.time()-started,'created_s':time.time()}
        if result['complete']:
            from showonce_studio.showonce_asset_library import finalize_replay
            finalize_replay(recording,result)
        write_json(out/('tracking.json'if result['complete']else 'probe-tracking.json'),result)
        write_json(out/'manifest.json',{k:v for k,v in result.items()if k not in ('frames','objects')})
        progress('complete'if result['complete']else 'probe_complete',message='FoundationPose 추적 결과 저장 완료',summary=summary)
        return result


def main():
    import fcntl
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--recording',type=Path,required=True);p.add_argument('--limit',type=int)
    args=p.parse_args();started=time.time()
    def progress(status,**values):
        job={'status':status,'pid':os.getpid(),'started_s':started,'updated_s':time.time(),**values}
        write_json(directory(args.recording)/'job.json',job);print(json.dumps(job,ensure_ascii=False),flush=True)
    try:
        with (ROOT/'runs/.showonce-vlm.lock').open('a')as gpu:
            while True:
                try:fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:
                    progress('waiting_for_gpu',message='다른 영상 분석이 끝나기를 기다립니다.');time.sleep(.5)
            PoseEngine().run(args.recording,progress,args.limit)
    except Exception as error:
        progress('error',error=f'{type(error).__name__}: {error}');raise

if __name__=='__main__':main()
