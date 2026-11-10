"""Observed RGB-D relations, independent of confirmed VLM stage timings."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time
from showonce_studio.showonce_stages import read_json,write_json,source_frames

from showonce_studio.config import ROOT
VERSION='scene_graph_v1'
def directory(recording):return Path(recording)/'relations'/VERSION
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def track_hands(recording,engine,progress):
    """Called inside the persistent SAM node, reusing its loaded predictor."""
    import cv2
    import numpy as np
    out=directory(recording);rows,times=source_frames(recording)
    digest=sha(recording/'frames.jsonl');cached=read_json(out/'hand_tracking.json')
    if cached and cached.get('complete')and cached.get('source_frames_sha256')==digest:return cached
    progress('loading_model',message='SAM 손 추적 모델을 준비합니다.');engine.load()
    predictor=engine.predictor
    session=predictor.handle_request({'type':'start_session','resource_path':str(recording/'rgb'),'offload_video_to_cpu':True})['session_id']
    frames=[{'index':i,'time_s':float(t),'hands':[]}for i,t in enumerate(times)]
    try:
        initial=predictor.handle_request({'type':'add_prompt','session_id':session,'frame_index':0,'text':'human hand'})['outputs']
        if not len(initial['out_obj_ids']):raise ValueError('초기 프레임에서 사람 손을 찾지 못했습니다.')
        for response in predictor.handle_stream_request({'type':'propagate_in_video','session_id':session,'propagation_direction':'forward','start_frame_index':0}):
            i=response['frame_index'];value=response['outputs']
            for j,track_id in enumerate(value['out_obj_ids']):
                mask=value['out_binary_masks'][j].astype(np.uint8)
                if mask.sum()<40:continue
                track_id=int(track_id);contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                contours=[cv2.approxPolyDP(c,1.2,True).reshape(-1,2).tolist()for c in contours if cv2.contourArea(c)>=10]
                y,x=np.nonzero(mask);path=out/'hand_masks'/str(track_id)/f'{i:06d}.png';path.parent.mkdir(parents=True,exist_ok=True)
                cv2.imwrite(str(path),mask*255)
                frames[i]['hands'].append({'track_id':track_id,'score':float(value['out_probs'][j]),'contours':contours,
                    'area_px':int(mask.sum()),'bbox':[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)]})
            if i%10==0:progress('tracking_hands',message='SAM 사람 손 추적 중',frame=i+1,total=len(rows))
    finally:predictor.handle_request({'type':'close_session','session_id':session})
    result={'schema':'showonce_hand_tracks_v1','complete':True,'recording_id':recording.name,'frames':frames,
        'source':'SAM3 video predictor, human hand prompt, all returned track IDs','source_frames_sha256':digest,'created_s':time.time()}
    write_json(out/'hand_tracking.json',result);return result


class TemporalPredicate:
    """Three-valued debounce. Missing evidence is unknown, never false."""
    def __init__(self,dwell_s=.12):self.value='unknown';self.pending=None;self.since=None;self.dwell_s=dwell_s
    def update(self,time_s,proposal):
        if proposal=='unknown':self.pending=self.since=None;self.value='unknown';return self.value
        if self.value==proposal:self.pending=self.since=None;return self.value
        if self.pending!=proposal:self.pending=proposal;self.since=time_s
        if time_s-self.since>=self.dwell_s-1e-8:self.value=proposal;self.pending=self.since=None
        return self.value


def threshold(value,previous,on,off,less=True):
    if value is None:return 'unknown'
    limit=off if previous=='true'else on
    return 'true'if(value<=limit if less else value>=limit)else 'false'


class VelocityWindow:
    """Timestamp-aware linear velocity over 180ms, keeping raw poses intact."""
    def __init__(self):self.points=[]
    def update(self,time_s,point):
        import numpy as np
        if point is None:self.points=[];return None
        self.points=[(t,p)for t,p in self.points if time_s-t<=.18]
        self.points.append((time_s,np.asarray(point)))
        if len(self.points)<3 or time_s-self.points[0][0]<.06:return None
        ts=np.asarray([t for t,_ in self.points]);ts-=ts.mean();points=np.asarray([p for _,p in self.points])
        return (ts[:,None]*points).sum(0)/np.dot(ts,ts)


def cavity_polygon(asset_root):
    """Free cavity boundary from convex-part union, not enclosing bowl AABB."""
    import numpy as np
    import trimesh
    from scipy.spatial import ConvexHull
    objects=read_json(asset_root/'scene.json')['objects'];bowl=next(o for o in objects if o['id']=='bowl')
    bounds=trimesh.load(asset_root/bowl['mesh'],force='mesh',process=False).bounds
    z=float(bounds[:,2].mean());angles=np.linspace(0,2*np.pi,180,endpoint=False)
    directions=np.column_stack((np.cos(angles),np.sin(angles)));radii=np.full(len(angles),np.inf)
    for path in bowl['colliders']:
        eq=ConvexHull(trimesh.load(asset_root/path,force='mesh',process=False).vertices).equations
        for i,d in enumerate(directions):
            slope=eq[:,:2]@d;constant=eq[:,2]*z+eq[:,3]
            if np.any((np.abs(slope)<1e-12)&(constant>1e-9)):continue
            lower=max(0.,float(np.max(-constant[slope< -1e-12]/slope[slope< -1e-12],initial=-np.inf)))
            upper=float(np.min(-constant[slope>1e-12]/slope[slope>1e-12],initial=np.inf))
            if lower<=upper and upper>=0:radii[i]=min(radii[i],lower)
    if not np.isfinite(radii).all()or np.min(radii)<.01:raise ValueError('참조 그릇의 내부 단면을 정의할 수 없습니다.')
    return (directions*radii[:,None]).astype(np.float32),bounds


def pose_metrics(cube,bowl,cube_vertices,cavity,bowl_bounds,table_z):
    import cv2
    import numpy as np
    if cube is None or bowl is None:return None
    cube=np.asarray(cube);bowl=np.asarray(bowl);points=cube_vertices@cube[:3,:3].T+cube[:3,3]
    local=(points-bowl[:3,3])@bowl[:3,:3]
    center=(np.linalg.inv(bowl)@cube)[:3,3]
    hull=cv2.convexHull(local[:,:2].astype(np.float32)).reshape(-1,2)
    # Vertices on the projected footprint must lie in the cavity section.
    fraction=float(np.mean([cv2.pointPolygonTest(cavity,(float(x),float(y)),False)>=0 for x,y in hull]))
    return {'cube_bottom_table_clearance_m':float(points[:,2].min()-table_z),
        'cube_center_bowl_local_m':center.tolist(),'cube_footprint_in_cavity_fraction':fraction,
        'bowl_rim_local_z_m':float(bowl_bounds[1,2]),'bowl_floor_local_z_m':float(bowl_bounds[0,2]),
        'cube_center_relative_rim_m':float(center[2]-bowl_bounds[1,2]),
        'cube_xy_target_offset_m':float(np.linalg.norm(center[:2])),
        'bowl_bottom_table_clearance_m':float((np.array([[x,y,z]for x in bowl_bounds[:,0]for y in bowl_bounds[:,1]for z in bowl_bounds[:,2]])@bowl[:3,:3].T+bowl[:3,3])[:,2].min()-table_z)}


RELATIONS=(
    ('near','hand','green_block','가까움'),
    ('contact','hand','green_block','접촉 추정'),
    ('grasped','hand','green_block','파지 추정'),
    ('near_bowl','hand','bowl','가까움'),
    ('contact_bowl','hand','bowl','접촉 추정'),
    ('above_table','green_block','table','들림'),
    ('on_table','green_block','table','테이블 지지 추정'),
    ('aligned','green_block','bowl','그릇과 XY 정렬'),
    ('inside','green_block','bowl','용기 내부 추정'),
    ('stable_in','green_block','bowl','안착 추정'),
    ('bowl_on_table','bowl','table','테이블 지지 추정'),
)


def extract(recording,progress):
    import cv2
    import numpy as np
    import trimesh
    from scipy.spatial import cKDTree
    recording=Path(recording);out=directory(recording);rows,times=source_frames(recording)
    poses=read_json(recording/'geometry/foundationpose_v1/tracking.json');hands=read_json(out/'hand_tracking.json')
    if not poses or not poses.get('complete')or not hands or not hands.get('complete'):raise ValueError('FoundationPose와 손 추적 결과가 필요합니다.')
    if poses.get('calibration_sha256')!=sha(recording/'calibration.json')or hands.get('source_frames_sha256')!=sha(recording/'frames.jsonl'):
        raise ValueError('보정 또는 영상이 변경되어 추적을 다시 해야 합니다.')
    if [f['time_s']for f in poses['frames']]!=times or [f['time_s']for f in hands['frames']]!=times:raise ValueError('추적과 RGB-D 프레임 시각이 다릅니다.')
    manifest=read_json(recording/'manifest.json');calib=read_json(recording/'calibration.json')
    intr=manifest['camera']['intrinsics'];scale=manifest['depth']['meters_per_unit'];T=np.asarray(calib['T_base_cam'])
    table_z=float(calib['table_check']['height_m']);asset_root=recording/'geometry/sam3d_assets_v1'
    for reference in poses['reference_assets']:
        if sha(asset_root/reference['reference_mesh'])!=reference['reference_mesh_sha256']:raise ValueError('참조 메시가 변경되어 포즈 추적을 다시 해야 합니다.')
    cube_vertices=np.asarray(trimesh.load(asset_root/'assets/green_block/mesh.ply',force='mesh',process=False).vertices)
    cavity,bowl_bounds=cavity_polygon(asset_root);masks=recording/'geometry/rgbd_surface_v1/masks'
    raw=[];hand_velocity={};cube_velocity=VelocityWindow();relative_velocity=VelocityWindow()
    def cloud(mask,depth):
        y,x=np.nonzero(mask&(depth>.10)&(depth<2));step=max(1,len(x)//1600);x=x[::step];y=y[::step];z=depth[y,x]
        p=np.column_stack(((x-intr['cx'])*z/intr['fx'],(y-intr['cy'])*z/intr['fy'],z))
        p=p@T[:3,:3].T+T[:3,3]
        return p[p[:,2]>table_z-.02]
    for i,row in enumerate(rows):
        depth=cv2.imread(str(recording/row['depth']),cv2.IMREAD_UNCHANGED).astype(np.float32)*scale
        cube_mask=cv2.imread(str(masks/'green_block'/f'{i:06d}.png'),0)>0;bowl_mask=cv2.imread(str(masks/'bowl'/f'{i:06d}.png'),0)>0
        cp=cloud(cv2.erode(cube_mask.astype(np.uint8),np.ones((3,3),np.uint8))>0,depth)
        bp=cloud(cv2.erode(bowl_mask.astype(np.uint8),np.ones((3,3),np.uint8))>0,depth)
        tree=cKDTree(cp)if len(cp)>=30 else None;bowl_tree=cKDTree(bp)if len(bp)>=30 else None
        observations=[];dt=times[i]-times[i-1]if i else None
        cube=poses['frames'][i]['objects']['green_block'].get('pose_base');bowl=poses['frames'][i]['objects']['bowl'].get('pose_base')
        cube_center=np.asarray(cube)[:3,3]if cube is not None else None
        velocity=cube_velocity.update(times[i],cube_center)
        for hand in hands['frames'][i]['hands']:
            mask=cv2.imread(str(out/'hand_masks'/str(hand['track_id'])/f'{i:06d}.png'),0)>0
            # Mutually exclusive visible pixels prevent object pixels from
            # manufacturing zero-distance hand contact.
            hp=cloud(mask&~cube_mask&~bowl_mask,depth)
            observation={**hand,'depth_points':len(hp),'center_base':None,'cube_distance_m':None,'bowl_distance_m':None,'relative_speed_m_s':None}
            if len(hp)>=30:
                center=np.median(hp,axis=0);observation['center_base']=center.tolist()
                if tree is not None:observation['cube_distance_m']=float(np.quantile(tree.query(hp)[0],.02))
                if bowl_tree is not None:observation['bowl_distance_m']=float(np.quantile(bowl_tree.query(hp)[0],.02))
                hv=hand_velocity.setdefault(hand['track_id'],VelocityWindow()).update(times[i],center)
                if velocity is not None and hv is not None:observation['relative_speed_m_s']=float(np.linalg.norm(hv-velocity))
            observations.append(observation)
        valid=[h for h in observations if h['cube_distance_m']is not None]
        selected=min(valid,key=lambda h:h['cube_distance_m'])if valid else None
        geometry=pose_metrics(cube,bowl,cube_vertices,cavity,bowl_bounds,table_z)
        if geometry:
            geometry['cube_observed_lower_table_m']=float(np.quantile(cp[:,2],.02)-table_z)if len(cp)>=30 else None
            geometry['bowl_observed_lower_table_m']=float(np.quantile(bp[:,2],.02)-table_z)if len(bp)>=30 else None
        relative=np.asarray(geometry['cube_center_bowl_local_m'])if geometry else None
        rv=relative_velocity.update(times[i],relative);relative_speed=float(np.linalg.norm(rv))if rv is not None else None
        raw.append({'index':i,'time_s':times[i],'hands':observations,'selected_hand_id':selected['track_id']if selected else None,
            'hand_cube_distance_m':selected['cube_distance_m']if selected else None,
            'hand_bowl_distance_m':min((h['bowl_distance_m']for h in observations if h['bowl_distance_m']is not None),default=None),
            'hand_cube_relative_speed_m_s':selected['relative_speed_m_s']if selected else None,
            'cube_speed_m_s':float(np.linalg.norm(velocity))if velocity is not None else None,
            'cube_bowl_relative_speed_m_s':relative_speed,'geometry':geometry})
        if i%30==0:progress('measuring',message='손·객체 사이의 관측 거리와 상대 운동 계산 중',frame=i+1,total=len(rows))
    return assemble(recording,raw,poses,hands,table_z,cavity,progress)


def assemble(recording,observations,poses,hands,table_z,cavity,progress):
    states={key:TemporalPredicate(.20 if key=='stable_in'else .10)for key,_,_,_ in RELATIONS}
    frames=[];events=[]
    for row in observations:
        t=row['time_s'];g=row['geometry'];distance=row['hand_cube_distance_m'];speed=row['hand_cube_relative_speed_m_s']
        previous={key:state.value for key,state in states.items()};proposals={key:'unknown'for key in states}
        proposals['near']=threshold(distance,previous['near'],.08,.11)
        proposals['contact']=threshold(distance,previous['contact'],.012,.022)
        proposals['near_bowl']=threshold(row.get('hand_bowl_distance_m'),previous['near_bowl'],.08,.11)
        proposals['contact_bowl']=threshold(row.get('hand_bowl_distance_m'),previous['contact_bowl'],.012,.022)
        if distance is not None:
            if distance>.045:proposals['grasped']='false'
            elif speed is not None and row['cube_speed_m_s']is not None:
                if distance<=.025 and speed<=.20 and row['cube_speed_m_s']>.025:proposals['grasped']='true'
                elif previous['grasped']=='true'and distance<=.035 and speed<.30:proposals['grasped']='true'
                elif distance>.035 or speed>.30:proposals['grasped']='false'
        if g:
            proposals['above_table']=threshold(g['cube_bottom_table_clearance_m'],previous['above_table'],.025,.018,less=False)
            aligned=g['cube_footprint_in_cavity_fraction']>=.80
            proposals['aligned']='true'if aligned else 'false'
            inside=aligned and g['cube_center_relative_rim_m']<-.008 and g['cube_center_bowl_local_m'][2]>g['bowl_floor_local_z_m']-.015
            proposals['inside']='true'if inside else 'false'
            observed_cube=g['cube_observed_lower_table_m'];observed_bowl=g['bowl_observed_lower_table_m']
            if inside or g['cube_bottom_table_clearance_m']>.025:proposals['on_table']='false'
            elif observed_cube is not None and row['cube_speed_m_s']is not None and abs(observed_cube)<=.018 and row['cube_speed_m_s']<.05:
                proposals['on_table']='true'
            proposals['bowl_on_table']='true'if observed_bowl is not None and abs(observed_bowl)<=.018 else 'unknown'
            if not inside:proposals['stable_in']='false'
            elif row['cube_bowl_relative_speed_m_s']is not None:
                if distance is None:proposals['stable_in']='unknown'
                else:proposals['stable_in']='true'if row['cube_bowl_relative_speed_m_s']<.035 and distance>.04 else 'false'
        relations=[]
        for key,source,target,label in RELATIONS:
            value=states[key].update(t,proposals[key])
            evidence={'proposed':proposals[key],'hand_track_id':row['selected_hand_id']}
            if key in ('near_bowl','contact_bowl'):evidence.update(surface_distance_m=row.get('hand_bowl_distance_m'))
            elif key in ('near','contact','grasped'):evidence.update(surface_distance_m=distance,relative_speed_m_s=speed,cube_speed_m_s=row['cube_speed_m_s'])
            else:evidence.update(g or {});evidence['relative_speed_m_s']=row['cube_bowl_relative_speed_m_s']
            relations.append({'id':key,'source':source,'target':target,'label':label,'state':value,'evidence':evidence,'interpretation':'vision_estimate'})
            if value!=previous[key]:events.append({'index':len(events),'frame_index':row['index'],'time_s':t,'relation_id':key,
                'source':source,'target':target,'label':label,'from':previous[key],'to':value,
                'kind':'uncertain'if value=='unknown'else 'birth'if value=='true'else 'death'if previous[key]=='true'else 'established_false'})
        frames.append({'index':row['index'],'time_s':t,'relations':relations,'hands':row['hands'],'selected_hand_id':row['selected_hand_id']})
    result={'schema':'showonce_scene_graph_v1','recording_id':recording.name,'complete':True,'created_s':time.time(),
        'nodes':[{'id':'hand','label':'사람 손','kind':'observed_hands'},{'id':'green_block','label':'큐브','kind':'object'},
                 {'id':'bowl','label':'그릇','kind':'container'},{'id':'table','label':'테이블','kind':'support_plane'}],
        'relation_specs':[{'id':k,'source':s,'target':d,'label':l}for k,s,d,l in RELATIONS],
        'frames':frames,'events':events,'duration_s':frames[-1]['time_s'],
        'parameters':{'near_on_m':.08,'near_off_m':.11,'contact_on_m':.012,'contact_off_m':.022,
            'grasp_relative_speed_max_m_s':.20,'surface_distance_quantile':.02,'minimum_dwell_s':.10,'settled_dwell_s':.20,
            'table_uncertainty_m':.018,'inside_footprint_fraction_min':.80,'table_height_base_m':table_z,'velocity_window_s':.18,
            'support_evidence':'Observed surface 2nd percentile near table and low velocity; not reconstructed bottom alone.'},
        'cavity_cross_section':{'frame':'bowl_local','polygon_xy_m':cavity.tolist()},
        'sources':{'pose_backend':'foundationpose','poses_sha256':sha(recording/'geometry/foundationpose_v1/tracking.json'),
            'hand_tracking_sha256':sha(directory(recording)/'hand_tracking.json'),'calibration_sha256':sha(recording/'calibration.json')},
        'limitations':['Contact and grasp are geometric/motion estimates, not measured forces.',
            'Unknown visibility/depth remains unknown; it is not a release event.',
            'Inside denotes center below rim and footprint inside the measured reference cavity, not exact volume containment.',
            'Grasp does not establish force closure. StableIn is a visual settling/separation estimate.',
            'Symmetry, reconstructed size errors and pose bias remain. No physical collision validation or reward generation.'],
        'stage_timings_used_for_inference':False,'ground_truth_contact_verified':False,'robot_commands_sent':False}
    write_json(directory(recording)/'observations.json',{'frames':observations})
    write_json(directory(recording)/'scene_graph.json',result)
    write_json(directory(recording)/'manifest.json',{key:result[key]for key in ('schema','recording_id','complete','created_s','duration_s','sources')})
    progress('complete',message='영상 관계 씬 그래프 추출 완료',events=len(events));return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--recording',type=Path,required=True);p.add_argument('--skip-hands',action='store_true')
    args=p.parse_args();recording=args.recording.resolve();started=time.time()
    def progress(status,**values):
        job={'status':status,'pid':os.getpid(),'started_s':started,'updated_s':time.time(),**values};write_json(directory(recording)/'job.json',job);print(json.dumps(job,ensure_ascii=False),flush=True)
    try:
        if not args.skip_hands:
            from showonce_studio.showonce_geometry_api import GeometryManager
            manager=GeometryManager(recording.parent);manager.ensure_node()
            manager.request('/api/geometry/hands/start',{'recording_id':recording.name})
            progress('tracking_hands',message='SAM 노드에서 손 관측을 준비합니다.')
            while True:
                hand_state=manager.request('/api/geometry/hands/state?id='+recording.name)
                if hand_state.get('tracking_ready'):break
                if hand_state['job'].get('status')=='error':raise ValueError(hand_state['job'].get('error'))
                time.sleep(.5)
        extract(recording,progress)
    except Exception as error:progress('error',error=f'{type(error).__name__}: {error}');raise

if __name__=='__main__':main()
