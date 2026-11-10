import * as THREE from 'three';

export function prepareReplay(data){
  return {...data,tracks:new Map(data.objects.map(spec=>[spec.id,data.frames.map(frame=>{
    const pose=frame.objects[spec.id]?.pose_world;
    if(!pose)return null;
    const matrix=new THREE.Matrix4().set(...pose.flat()),position=new THREE.Vector3(),quaternion=new THREE.Quaternion(),scale=new THREE.Vector3();
    matrix.decompose(position,quaternion,scale);quaternion.normalize();
    return {index:frame.index,time_s:frame.time_s,position,quaternion};
  })]))};
}
export function smoothReplay(replay,tau_s=.1){
  if(!(tau_s>0))throw Error('EMA time constant must be positive');
  const tracks=new Map();
  for(const [id,source]of replay.tracks){
    const result=source.map(pose=>pose?{...pose,position:pose.position.clone(),quaternion:pose.quaternion.clone()}:null);
    const segments=[];let segment=[];
    for(let i=0;i<source.length;i++)if(source[i]){
      if(segment.length&&source[i].time_s-source[segment.at(-1)].time_s>replay.policy.short_gap_max_s){segments.push(segment);segment=[];}
      segment.push(i);
    }
    if(segment.length)segments.push(segment);
    for(const indices of segments){
      function pass(order){const output=new Map();let previous=null;
        for(const i of order){const raw=source[i],pose={...raw,position:raw.position.clone(),quaternion:raw.quaternion.clone()};
          if(previous){const alpha=1-Math.exp(-Math.abs(raw.time_s-previous.time_s)/tau_s);
            pose.position.copy(previous.position).lerp(raw.position,alpha);
            pose.quaternion.copy(previous.quaternion).slerp(raw.quaternion,alpha).normalize();}
          output.set(i,pose);previous=pose;
        }return output;}
      const forward=pass(indices),backward=pass([...indices].reverse());
      for(const i of indices){result[i].position.copy(forward.get(i).position).lerp(backward.get(i).position,.5);
        result[i].quaternion.copy(forward.get(i).quaternion).slerp(backward.get(i).quaternion,.5).normalize();}
    }
    tracks.set(id,result);
  }
  return {...replay,tracks,smoothing:{method:'symmetric timestamp-aware EMA + quaternion SLERP',tau_s,offline:true}};
}
export function nearestIndex(frames,time){
  let lo=0,hi=frames.length-1;while(lo<hi){const mid=(lo+hi)>>1;if(frames[mid].time_s<time)lo=mid+1;else hi=mid;}
  if(lo>0&&time-frames[lo-1].time_s<frames[lo].time_s-time)lo--;return lo;
}
function blend(left,right,time,status){
  const fraction=Math.max(0,Math.min(1,(time-left.time_s)/(right.time_s-left.time_s)));
  return {position:left.position.clone().lerp(right.position,fraction),quaternion:left.quaternion.clone().slerp(right.quaternion,fraction),
    status,source_index:left.index,age_s:0};
}
export function samplePose(replay,id,time){
  const track=replay.tracks.get(id);if(!track)return null;
  const index=nearestIndex(replay.frames,time),exact=track[index];
  if(exact){
    if(time>exact.time_s&&track[index+1]&&track[index+1].time_s-exact.time_s<=replay.policy.short_gap_max_s)
      return blend(exact,track[index+1],time,'estimated');
    if(time<exact.time_s&&track[index-1]&&exact.time_s-track[index-1].time_s<=replay.policy.short_gap_max_s)
      return blend(track[index-1],exact,time,'estimated');
    return {...exact,position:exact.position.clone(),quaternion:exact.quaternion.clone(),status:'estimated',source_index:exact.index,age_s:0};
  }
  let left=index-1,right=index+1;while(left>=0&&!track[left])left--;while(right<track.length&&!track[right])right++;
  if(left>=0&&right<track.length&&track[right].time_s-track[left].time_s<=replay.policy.short_gap_max_s)
    return blend(track[left],track[right],time,'interpolated');
  if(left>=0)return {...track[left],position:track[left].position.clone(),quaternion:track[left].quaternion.clone(),
    status:'held',source_index:track[left].index,age_s:Math.max(0,time-track[left].time_s)};
  return {status:'unresolved',position:null,quaternion:null,source_index:null,age_s:null};
}
