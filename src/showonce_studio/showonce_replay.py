"""Read-only registration of SAM assets to timestamped measured object tracks."""
from pathlib import Path
import hashlib
import numpy as np
from showonce_studio.showonce_stages import recording_path,read_json
from showonce_studio.showonce_geometry import directory as measured_directory
from showonce_studio.showonce_sim_assets import directory as asset_directory
from showonce_studio.showonce_task_template import display_annotation


def build_replay(recording,backend='measured'):
    if backend not in ('measured','foundationpose'):raise ValueError('Unknown pose backend')
    recording=Path(recording);tracking_path=(recording/'geometry/foundationpose_v1' if backend=='foundationpose'else measured_directory(recording))/'tracking.json'
    manifest_path=asset_directory(recording)/'manifest.json'
    tracking=read_json(tracking_path);manifest=read_json(manifest_path)
    if not tracking or not tracking.get('complete') or not manifest or not manifest.get('complete'):
        raise ValueError('영상 추적과 SAM 3D 에셋 생성이 필요합니다.')
    frames=tracking['frames'];table=float(manifest['table_height_base_m']);registrations=[]
    if not frames or any(b['time_s']<=a['time_s'] for a,b in zip(frames,frames[1:])):
        raise ValueError('원본 프레임 시각이 올바르지 않습니다.')
    for asset in manifest['assets']:
        oid=asset['id'];ref=asset['reference_frame_index']
        if backend=='foundationpose':
            if tracking.get('mesh_frame')!='sam3d_asset_local':raise ValueError('Reference asset pose frame mismatch')
            reference=next((r for r in tracking.get('reference_assets',[])if r['id']==oid),None)
            if reference is None or hashlib.sha256((asset_directory(recording)/reference['reference_mesh']).read_bytes()).hexdigest()!=reference['reference_mesh_sha256']:
                raise ValueError('Reference mesh changed; retrack this recording')
            local_from_asset=np.eye(4)
        else:
            measured=frames[ref]['objects'][oid].get('pose_base')
            if measured is None:raise ValueError(asset['name']+' 기준 프레임 포즈가 없습니다.')
            local_from_asset=np.linalg.inv(np.asarray(measured,float))@np.asarray(asset['pose_base_at_reference'],float)
        registrations.append({'id':oid,'name':asset['name'],'reference_frame_index':ref,
            'tracking_local_from_asset':local_from_asset.tolist(),
            'scene_support_offset_m':0. if backend=='foundationpose'else float(asset.get('simulation_initialization_dz_m',0.))})
    replay=[]
    for frame in frames:
        objects={}
        for registration in registrations:
            oid=registration['id'];source=frame['objects'][oid];measured=source.get('pose_base');pose=None
            if measured is not None:
                matrix=np.asarray(measured,float)@np.asarray(registration['tracking_local_from_asset'])
                matrix[2,3]+=-table+registration['scene_support_offset_m']
                pose=matrix.tolist()
            objects[oid]={'pose_world':pose,'pose_status':'estimated' if pose is not None else 'unresolved',
                          'visible':source.get('status')=='observed','fitness':source.get('fitness')}
        replay.append({'index':frame['index'],'time_s':frame['time_s'],'objects':objects})
    task=display_annotation(read_json(recording/'task_definition.json',{}))
    stages=task.get('stages',[]) if task.get('review_status')=='confirmed' else []
    return {'schema':'showonce_video_replay_v1','recording_id':recording.name,'frames':replay,'objects':registrations,
        'duration_s':frames[-1]['time_s'],'coordinate_frame':'robot base X/Y, table surface Z=0','units':'m','pose_backend':backend,
        'stages':stages,'confirmed_stage_revision':task.get('revision') if stages else None,
        'policy':{'short_gap_max_s':.2,'long_gap':'show last estimated pose translucent and mark age; no long-gap interpolation',
                  'physics_used':False,'support_offset':'none; reference asset poses directly measured'if backend=='foundationpose'else 'same constant per-asset scene initialization offset; not re-snapped per frame'},
        'source_hash':hashlib.sha256(tracking_path.read_bytes()+manifest_path.read_bytes()).hexdigest()}


class ReplayManager:
    def __init__(self,root):self.root=Path(root);self.cache={}
    def get(self,recording_id,backend='measured'):
        if backend not in ('measured','foundationpose'):raise ValueError('Unknown pose backend')
        recording,_=recording_path(self.root,recording_id)
        paths=[(recording/'geometry/foundationpose_v1'if backend=='foundationpose'else measured_directory(recording))/'tracking.json',asset_directory(recording)/'manifest.json',recording/'task_definition.json']
        key=tuple((p.stat().st_mtime_ns,p.stat().st_size) if p.exists() else None for p in paths)
        cache_id=(recording_id,backend);cached=self.cache.get(cache_id)
        if cached is None or cached[0]!=key:
            value=build_replay(recording,backend);self.cache[cache_id]=(key,value)
        return self.cache[cache_id][1]
