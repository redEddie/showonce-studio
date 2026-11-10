from showonce_studio.config import sam_checkpoint
"""Offline, measured RGB-D object tracking and multi-frame surface fusion.

No camera, CAN, robot SDK, VLM boundary edits or generated hidden geometry.
"""
from pathlib import Path
import hashlib
import json
import os
import threading
import time
from showonce_studio.showonce_stages import read_json, write_json, source_frames

VERSION = 'rgbd_surface_v1'
OBJECTS = ({'id': 'green_block', 'name': '초록 큐브', 'prompt': 'green cube', 'color': '#4fae7a'},
           {'id': 'bowl', 'name': '그릇', 'prompt': 'bowl', 'color': '#e5a452'})


def directory(recording):
    return Path(recording)/'geometry'/VERSION


def transform(points, matrix):
    return points @ matrix[:3,:3].T + matrix[:3,3]


def rigid_alignment(source,target):
    import numpy as np
    a=source.mean(axis=0);b=target.mean(axis=0)
    u,_,vt=np.linalg.svd((source-a).T@(target-b));rotation=vt.T@u.T
    if np.linalg.det(rotation)<0:vt[-1]*=-1;rotation=vt.T@u.T
    matrix=np.eye(4);matrix[:3,:3]=rotation;matrix[:3,3]=b-rotation@a
    return matrix


def rgbd_motion(previous_rgb,rgb,previous_depth,depth,previous_mask,mask,intrinsics,T_base_cam):
    """Estimate measured inter-frame rigid motion; never use VLM phase rules."""
    import cv2
    import numpy as np
    common=previous_mask&mask
    union=previous_mask|mask
    if common.sum()>=30:
        py,px=np.nonzero(previous_mask);cy,cx=np.nonzero(mask)
        if (common.sum()/max(1,union.sum())>.88 and
            np.hypot(px.mean()-cx.mean(),py.mean()-cy.mean())<.9 and
            np.median(np.abs(depth[common]-previous_depth[common]))<.004):
            return np.eye(4),'stationary_mask_and_depth_observation'
    flow=cv2.calcOpticalFlowFarneback(cv2.cvtColor(previous_rgb,cv2.COLOR_RGB2GRAY),
        cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),None,.5,3,21,3,5,1.2,0)
    y,x=np.nonzero(previous_mask);take=np.arange(0,len(x),max(1,len(x)//1200));x,y=x[take],y[take]
    if len(x)<30:return None,'insufficient_correspondences'
    dx=flow[y,x,0];dy=flow[y,x,1];xx=np.rint(x+dx).astype(int);yy=np.rint(y+dy).astype(int)
    valid=(xx>=0)&(xx<depth.shape[1])&(yy>=0)&(yy<depth.shape[0])
    x,y,xx,yy,dx,dy=(v[valid] for v in (x,y,xx,yy,dx,dy))
    valid=mask[yy,xx]&(previous_depth[y,x]>0)&(depth[yy,xx]>0)
    x,y,xx,yy,dx,dy=(v[valid] for v in (x,y,xx,yy,dx,dy))
    if len(x)<30:return None,'insufficient_correspondences'
    def cloud(u,v,z):return transform(np.column_stack(((u-intrinsics['cx'])*z/intrinsics['fx'],
        (v-intrinsics['cy'])*z/intrinsics['fy'],z)),T_base_cam)
    source=cloud(x,y,previous_depth[y,x]);target=cloud(xx,yy,depth[yy,xx])
    displacement=np.median(target-source,axis=0)
    if np.median(np.hypot(dx,dy))<.35 and np.linalg.norm(displacement)<.003:
        return np.eye(4),'stationary_rgbd_observation'
    rng=np.random.default_rng(42);best=None;best_count=0
    for _ in range(48):
        indices=rng.choice(len(source),3,replace=False);estimate=rigid_alignment(source[indices],target[indices])
        inliers=np.linalg.norm(transform(source,estimate)-target,axis=1)<.006
        if inliers.sum()>best_count:best,best_count=inliers,int(inliers.sum())
    if best_count<max(30,len(source)*.4):return None,'rigid_motion_unresolved'
    estimate=rigid_alignment(source[best],target[best])
    angle=np.arccos(np.clip((np.trace(estimate[:3,:3])-1)/2,-1,1))
    if angle>np.deg2rad(15) or np.linalg.norm(displacement)>.06:return None,'rigid_motion_jump_rejected'
    return estimate,'rgbd_optical_flow_rigid_fit'


class GeometryEngine:
    def __init__(self):
        self.predictor = None
        self.model_lock = threading.RLock()

    def load(self):
        with self.model_lock:
            self._load()

    def _load(self):
        if self.predictor is None:
            # The bundled older ptxas cannot compile Thor sm_110a. Override the
            # compiler only in this isolated process, retaining PyTorch CUDA libs.
            os.environ.setdefault('TRITON_PTXAS_BLACKWELL_PATH', '/usr/local/cuda/bin/ptxas')
            import torch
            from sam3.model.sam3_video_predictor import Sam3VideoPredictor
            torch.set_num_threads(4)
            self.predictor = Sam3VideoPredictor(
                checkpoint_path=str(sam_checkpoint()), compile=False)

    def track(self, recording, progress):
        import cv2
        import numpy as np
        root = directory(recording)
        from showonce_studio.showonce_asset_library import object_specs,selection
        specs=object_specs(recording)or OBJECTS
        binding=selection(recording);selection_hash=binding['selection_sha256']if binding else None
        rows, times = source_frames(recording)
        intrinsics=read_json(recording/'manifest.json')['camera']['intrinsics']
        cache = read_json(root/'tracking.json')
        if cache and cache.get('complete') and len(cache['frames']) == len(rows)and cache.get('asset_selection_sha256')==selection_hash:
            return cache
        progress('loading_model', message='SAM 영상 추적 모델을 준비합니다.')
        self.load()
        frames = [{'index': i, 'time_s': t, 'objects': {}} for i,t in enumerate(times)]
        predictor = self.predictor
        for spec in specs:
            progress('tracking', message=spec['name']+' 영상 추적 중', object_id=spec['id'], frame=0, total=len(rows))
            session = predictor.handle_request({'type': 'start_session', 'resource_path': str(recording/'rgb'),
                                                'offload_video_to_cpu': True})['session_id']
            try:
                actual = predictor._all_inference_states[session]['state']['num_frames']
                if actual != len(rows):
                    raise ValueError('SAM decoded frame count differs from RGB-D timestamps')
                initial = predictor.handle_request({'type': 'add_prompt', 'session_id': session,
                    'frame_index': 0, 'text': spec['prompt']})['outputs']
                if len(initial['out_obj_ids']) == 0:
                    raise ValueError(spec['name']+' 초기 프레임 검출 실패')
                rank = np.asarray(initial['out_probs']) * np.sqrt(np.maximum(1, initial['out_binary_masks'].sum(axis=(1,2))))
                tracked_id = int(initial['out_obj_ids'][int(np.argmax(rank))])
                for response in predictor.handle_stream_request({'type': 'propagate_in_video', 'session_id': session,
                    'propagation_direction': 'forward', 'start_frame_index': 0}):
                    i, out = response['frame_index'], response['outputs']
                    found = np.flatnonzero(np.asarray(out['out_obj_ids']) == tracked_id)
                    mask = np.zeros((intrinsics['height'],intrinsics['width']), np.uint8)
                    score = None
                    if len(found):
                        index = int(found[0]);mask = out['out_binary_masks'][index].astype(np.uint8)
                        score = float(out['out_probs'][index])
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours = [cv2.approxPolyDP(c, 1.2, True).reshape(-1,2).tolist()
                                for c in contours if cv2.contourArea(c) >= 8]
                    ys,xs = np.nonzero(mask)
                    frames[i]['objects'][spec['id']] = {'status': 'observed' if len(xs)>=20 else 'not_visible',
                        'score': score, 'sam_track_id': tracked_id, 'area_px': int(len(xs)), 'contours': contours,
                        'bbox': [int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)] if len(xs) else None}
                    path = root/'masks'/spec['id']/f'{i:06d}.png';path.parent.mkdir(parents=True,exist_ok=True)
                    cv2.imwrite(str(path), mask*255)
                    if i%10==0:
                        progress('tracking', message=spec['name']+' 영상 추적 중', object_id=spec['id'], frame=i+1,total=len(rows))
                        write_json(root/'tracking.partial.json', {'frames': frames, 'complete': False, 'objects': specs})
            finally:
                predictor.handle_request({'type': 'close_session', 'session_id': session})
        result = {'schema': 'showonce_object_tracks_v1', 'recording_id': recording.name, 'frames': frames,
                  'objects': specs, 'complete': True, 'source': 'SAM3 video tracker on source lossless RGB PNGs',
                  'asset_selection_sha256':selection_hash,
                  'source_frames_sha256': hashlib.sha256((recording/'frames.jsonl').read_bytes()).hexdigest()}
        write_json(root/'tracking.json', result)
        return result

    def reconstruct(self, recording, tracking, progress):
        import cv2
        import numpy as np
        import open3d as o3d
        import trimesh
        root=directory(recording);manifest=read_json(recording/'manifest.json');calib=read_json(recording/'calibration.json')
        camera=manifest['camera'];intr=camera['intrinsics'];scale=manifest['depth']['meters_per_unit']
        rows,times=source_frames(recording);T_base_cam=np.asarray(calib['T_base_cam'],float);T_cam_base=np.linalg.inv(T_base_cam)
        intrinsic=o3d.camera.PinholeCameraIntrinsic(intr['width'],intr['height'],intr['fx'],intr['fy'],intr['cx'],intr['cy'])
        table=float(calib.get('table_check',{}).get('height_m',-.015))
        y,x=np.indices((intr['height'],intr['width']))
        assets=[]
        for spec in OBJECTS:
            oid=spec['id'];progress('reconstructing',message=spec['name']+' 다중 프레임 표면 정합',object_id=oid)
            clouds=[];measurements=[];clean_masks=[]
            for i,row in enumerate(rows):
                rgb=cv2.cvtColor(cv2.imread(str(recording/row['rgb'])),cv2.COLOR_BGR2RGB)
                depth=cv2.imread(str(recording/row['depth']),cv2.IMREAD_UNCHANGED).astype(np.float32)*scale
                mask=cv2.imread(str(root/'masks'/oid/f'{i:06d}.png'),cv2.IMREAD_GRAYSCALE)>0
                mask=cv2.erode(mask.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
                if oid=='green_block':
                    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
                    mask &= (hsv[:,:,0]>=30)&(hsv[:,:,0]<=100)&(hsv[:,:,1]>=45)
                # Skin-colored pixels cannot add measured object geometry.
                colors=rgb.astype(float)
                skin=(colors[:,:,0]>colors[:,:,1]*1.14)&(colors[:,:,1]>colors[:,:,2]*1.12)&(colors[:,:,0]>90)
                valid=mask&~skin&(depth>.15)&(depth<1.8)
                z=depth[valid];u=x[valid];v=y[valid]
                pc=np.column_stack(((u-intr['cx'])*z/intr['fx'],(v-intr['cy'])*z/intr['fy'],z))
                pb=transform(pc,T_base_cam);color=rgb[valid]/255.
                if len(pb):
                    good=(pb[:,2]>table+.002)&(np.abs(pb-np.median(pb,axis=0)).max(axis=1)<.22)
                    pb,color=pb[good],color[good]
                    valid_indices=np.flatnonzero(valid);good_indices=valid_indices[good]
                    valid[:]=False;valid.flat[good_indices]=True
                cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pb));cloud.colors=o3d.utility.Vector3dVector(color)
                if len(pb)>30:
                    cloud,keep=cloud.remove_statistical_outlier(nb_neighbors=16,std_ratio=1.8)
                    keep_indices=np.flatnonzero(valid)[keep];valid[:]=False;valid.flat[keep_indices]=True
                clouds.append(cloud);clean_masks.append(valid)
                measurements.append({'points':len(cloud.points),'depth_fraction':float(valid.sum()/max(1,mask.sum()))})
            candidates=[i for i in range(len(rows)) if len(clouds[i].points)>=60]
            if not candidates:raise ValueError(spec['name']+' 유효 깊이 점이 부족합니다.')
            # A well-observed early frame establishes a fixed object coordinate
            # frame. No spawn coordinates, nominal cube dimensions or table snap.
            early=[i for i in candidates if times[i]<1.4] or candidates
            ref_index=max(early,key=lambda i:len(clouds[i].points))
            origin=np.median(np.asarray(clouds[ref_index].points),axis=0)
            ref=o3d.geometry.PointCloud(clouds[ref_index]);ref.translate(-origin)
            ref=ref.voxel_down_sample(.002)
            previous=np.eye(4);previous[:3,3]=origin;previous_center=origin.copy()
            poses=[];accepted=[];registration=o3d.pipelines.registration
            previous_rgb=previous_depth=None;previous_valid=False
            for i,cloud in enumerate(clouds):
                pose=None;fitness=None;rmse=None
                rgb=cv2.cvtColor(cv2.imread(str(recording/rows[i]['rgb'])),cv2.COLOR_BGR2RGB)
                depth=cv2.imread(str(recording/rows[i]['depth']),cv2.IMREAD_UNCHANGED).astype(np.float32)*scale
                method='icp_reacquisition'
                if len(cloud.points)>=60:
                    center=np.median(np.asarray(cloud.points),axis=0)
                    guess=previous.copy();guess[:3,3]+=center-previous_center
                    motion=None
                    if i>0 and previous_valid:
                        motion,method=rgbd_motion(previous_rgb,rgb,previous_depth,depth,
                            clean_masks[i-1],clean_masks[i],intr,T_base_cam)
                    if motion is not None:guess=motion@previous
                    fit=registration.registration_icp(ref,cloud,.014,guess,
                        registration.TransformationEstimationPointToPoint(),registration.ICPConvergenceCriteria(max_iteration=35))
                    if motion is not None:
                        correction=np.linalg.inv(guess)@fit.transformation
                        correction_angle=np.arccos(np.clip((np.trace(correction[:3,:3])-1)/2,-1,1))
                        # Weak or partial depth must not rotate a visibly static
                        # object, or override a measured small inter-frame motion.
                        if method.startswith('stationary_') or correction_angle>np.deg2rad(2) or np.linalg.norm(correction[:3,3])>.003:
                            candidate=guess
                        else:candidate=fit.transformation
                    else:candidate=fit.transformation
                    evaluation=registration.evaluate_registration(ref,cloud,.012,candidate)
                    delta=np.linalg.inv(previous)@candidate
                    angle=np.arccos(np.clip((np.trace(delta[:3,:3])-1)/2,-1,1))
                    fitness=float(evaluation.fitness);rmse=float(evaluation.inlier_rmse)
                    if fitness>=.55 and rmse<=.006 and np.linalg.norm(delta[:3,3])<.09 and angle<np.deg2rad(25):
                        pose=candidate.copy();previous=pose;previous_center=center
                        accepted.append(i)
                poses.append(pose)
                previous_rgb,previous_depth,previous_valid=rgb,depth,pose is not None
                item=tracking['frames'][i]['objects'][oid]
                item.update(depth_points=measurements[i]['points'],valid_depth_fraction=measurements[i]['depth_fraction'],
                            pose_status='registered' if pose is not None else 'unresolved',
                            pose_base=pose.tolist() if pose is not None else None,pose_method=method,fitness=fitness,rmse_m=rmse)
                if i%20==0:progress('registering',message=spec['name']+' 프레임별 깊이 정합',frame=i,total=len(rows),object_id=oid)
            # Disjoint frame roles: reconstruction inputs and diagnostic frames.
            fusion=[i for i in accepted if i%3==0 and
                    tracking['frames'][i]['objects'][oid]['fitness']>=.75 and
                    tracking['frames'][i]['objects'][oid]['rmse_m']<=.0045 and
                    (oid!='bowl' or tracking['frames'][i]['objects'][oid]['area_px']>=.85*tracking['frames'][ref_index]['objects'][oid]['area_px'])]
            if len(fusion)<3:raise ValueError(spec['name']+' 정합된 다중 프레임이 부족합니다.')
            # A bounded object-local volume also avoids a ScalableTSDFVolume
            # failure in this aarch64 Open3D wheel (empty even for a plane).
            length=float(np.clip(np.max(ref.get_max_bound()-ref.get_min_bound())*1.8+.04,.10,.40))
            resolution=min(256,int(np.ceil(length/.002)))
            volume=o3d.pipelines.integration.UniformTSDFVolume(length=length,resolution=resolution,sdf_trunc=.008,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
                origin=ref.get_axis_aligned_bounding_box().get_center()-length/2)
            local_points=[];local_colors=[]
            for n,i in enumerate(fusion):
                rgb=cv2.cvtColor(cv2.imread(str(recording/rows[i]['rgb'])),cv2.COLOR_BGR2RGB)
                depth=cv2.imread(str(recording/rows[i]['depth']),cv2.IMREAD_UNCHANGED).astype(np.float32)*scale
                depth[~clean_masks[i]]=0
                image=o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(rgb),o3d.geometry.Image(depth),
                    depth_scale=1.,depth_trunc=1.8,convert_rgb_to_intensity=False)
                volume.integrate(image,intrinsic,T_cam_base@poses[i])
                local_points.append(transform(np.asarray(clouds[i].points),np.linalg.inv(poses[i])))
                local_colors.append(np.asarray(clouds[i].colors))
                if n%8==0:progress('fusing',message=spec['name']+' 측정 표면 융합',frame=n+1,total=len(fusion),object_id=oid)
            mesh=volume.extract_triangle_mesh();mesh.remove_degenerate_triangles();mesh.remove_duplicated_triangles()
            if len(mesh.triangles)<10:raise ValueError(spec['name']+' 메시 표면을 추출하지 못했습니다.')
            labels,counts,areas=mesh.cluster_connected_triangles();labels=np.asarray(labels);counts=np.asarray(counts)
            mesh.remove_triangles_by_mask(counts[labels]<max(8,int(counts.max()*.015)))
            mesh.remove_unreferenced_vertices();mesh.compute_vertex_normals()
            points=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(local_points)))
            points.colors=o3d.utility.Vector3dVector(np.concatenate(local_colors));points=points.voxel_down_sample(.0015)
            points,_=points.remove_statistical_outlier(nb_neighbors=16,std_ratio=2.)
            center=mesh.get_axis_aligned_bounding_box().get_center()
            mesh.translate(-center);points.translate(-center)
            for i,pose in enumerate(poses):
                if pose is not None:
                    pose[:3,3]+=pose[:3,:3]@center
                    tracking['frames'][i]['objects'][oid]['pose_base']=pose.tolist()
            adir=root/'assets'/oid;adir.mkdir(parents=True,exist_ok=True)
            o3d.io.write_triangle_mesh(str(adir/'mesh.ply'),mesh)
            o3d.io.write_point_cloud(str(adir/'measured.ply'),points)
            tri=trimesh.Trimesh(vertices=np.asarray(mesh.vertices),faces=np.asarray(mesh.triangles),
                vertex_colors=np.uint8(np.clip(np.asarray(mesh.vertex_colors)*255,0,255)),process=False)
            tri.export(adir/'mesh.glb')
            overlay=np.asarray(mesh.vertices)
            if len(overlay)>700:overlay=overlay[np.linspace(0,len(overlay)-1,700,dtype=int)]
            write_json(adir/'overlay_points.json',{'positions':np.round(overlay,6).ravel().tolist()})
            image=cv2.imread(str(recording/rows[ref_index]['rgb']))
            box=tracking['frames'][ref_index]['objects'][oid]['bbox']
            if box:
                x0,y0,x1,y1=box;pad=8;x0=max(0,x0-pad);y0=max(0,y0-pad);x1=min(intr['width'],x1+pad);y1=min(intr['height'],y1+pad)
                crop=image[y0:y1,x0:x1];crop_mask=cv2.imread(str(root/'masks'/oid/f'{ref_index:06d}.png'),0)[y0:y1,x0:x1]>0
                crop[~crop_mask]=(240,245,242)
                ratio=min(160/crop.shape[1],128/crop.shape[0]);crop=cv2.resize(crop,(max(1,int(crop.shape[1]*ratio)),max(1,int(crop.shape[0]*ratio))))
                thumbnail=np.full((144,180,3),(240,245,242),np.uint8);h,w=crop.shape[:2]
                thumbnail[(144-h)//2:(144-h)//2+h,(180-w)//2:(180-w)//2+w]=crop
                cv2.imwrite(str(adir/'thumbnail.png'),thumbnail)
            pp=np.asarray(points.points);cc=np.asarray(points.colors)
            if len(pp)>20000:
                indices=np.linspace(0,len(pp)-1,20000,dtype=int);pp,cc=pp[indices],cc[indices]
            write_json(adir/'points.json',{'positions':np.round(pp,6).ravel().tolist(),
                                         'colors':np.round(cc,3).ravel().tolist(),'frame':'object_local','units':'m'})
            diagnostics=[];distances=[]
            vertex_cloud=o3d.geometry.PointCloud(mesh.vertices)
            for i in accepted:
                if i in fusion:continue
                local=o3d.geometry.PointCloud(clouds[i]);local.transform(np.linalg.inv(poses[i]))
                dist=np.asarray(local.compute_point_cloud_distance(vertex_cloud));distances.extend(dist.tolist())
                diagnostics.append({'frame_index':i,'median_surface_distance_m':float(np.median(dist))})
            write_json(adir/'diagnostics.json',diagnostics)
            asset=dict(spec,frame='object_local',units='m',origin_base_at_reference=poses[ref_index][:3,3].tolist(),reference_frame_index=ref_index,
                mesh=f'assets/{oid}/mesh.glb',point_cloud=f'assets/{oid}/points.json',
                thumbnail=f'assets/{oid}/thumbnail.png',overlay_points=f'assets/{oid}/overlay_points.json',
                origin_definition='reconstructed mesh axis-aligned bounding-box center',
                mesh_ply=f'assets/{oid}/mesh.ply',measured_ply=f'assets/{oid}/measured.ply',
                vertices=len(mesh.vertices),triangles=len(mesh.triangles),measured_points=len(points.points),
                extent_m=(np.asarray(mesh.get_max_bound())-np.asarray(mesh.get_min_bound())).tolist(),
                fusion_frame_indices=fusion,accepted_pose_frames=len(accepted),total_frames=len(rows),
                diagnostic_frame_count=len(diagnostics),median_diagnostic_surface_distance_m=float(np.median(distances)) if distances else None,
                reconstruction='RGB-D motion tracking + gated ICP + TSDF fusion of measured surfaces',
                fusion_volume={'type':'UniformTSDFVolume','length_m':length,'resolution':resolution,'voxel_m':length/resolution},
                completeness='observed_surfaces_only',watertight=bool(tri.is_watertight),
                note='가려진 뒷면은 생성하지 않았습니다. 표면 정합 오차는 실물 정답 형상 오차가 아닙니다.')
            write_json(adir/'asset.json',asset);assets.append(asset)
        write_json(root/'tracking.json',tracking)
        result={'schema':'showonce_rgbd_assets_v1','recording_id':recording.name,'version':VERSION,
                'complete':True,'created_s':time.time(),'assets':assets,'frame_count':len(rows),'duration_s':times[-1],
                'tracking':'tracking.json','coordinate_frame':'piper_base_link',
                'calibration_sha256':hashlib.sha256((recording/'calibration.json').read_bytes()).hexdigest(),
                'intrinsics':intr,'T_cam_base':T_cam_base.tolist(),
                'camera_placement_verified_by_operator':manifest['calibration'].get('camera_placement_verified_by_operator',False),
                'depth_scale_m':scale,'model':'SAM3 video tracking + RGB-D optical flow + gated rigid ICP + measured TSDF fusion',
                'ground_truth_accuracy_verified':False}
        write_json(root/'manifest.json',result)
        return result
