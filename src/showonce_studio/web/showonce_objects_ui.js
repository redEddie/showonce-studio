/* Recording selection for object/relationship inspection, independent of VLM review. */
const el=id=>document.getElementById(id),video=el('video');
let recording='',epoch=0,initialTime=Number(new URLSearchParams(location.search).get('t'));
async function json(url){const response=await fetch(url,{cache:'no-store'}),value=await response.json();if(!response.ok)throw Error(value.error||'시연을 불러오지 못했습니다.');return value;}
function links(){const query='id='+encodeURIComponent(recording),time=Number.isFinite(video.currentTime)?video.currentTime.toFixed(3):'0';
  for(const id of ['stages-link','stage-editor-link'])el(id).href='/stages?'+query+'&t='+time;
  el('policies-link').href='/policies?'+query;el('objects-link').href='/objects?'+query;}
async function stageStatus(){if(!recording)return;const generation=epoch;
  try{const value=await json('/api/relations/state?id='+encodeURIComponent(recording));if(generation!==epoch)return;
    const badge=el('stage-review-status');badge.textContent=value.confirmed_stage_revision?'스테이지 확정 · 버전 '+value.confirmed_stage_revision:'스테이지 미확정';badge.classList.toggle('warn',!value.confirmed_stage_revision);
  }catch(error){if(generation===epoch)el('stage-review-status').textContent='스테이지 상태 확인 필요';}}
function choose(id){if(id===recording)return;recording=id;epoch++;el('recording').value=id;el('error').textContent='';
  history.replaceState({},'','/objects?id='+encodeURIComponent(id));video.src='/recordings/'+encodeURIComponent(id)+'/rgb.mp4';links();
  window.dispatchEvent(new CustomEvent('showonce-recording',{detail:{id}}));stageStatus();}
el('recording').onchange=()=>{initialTime=0;choose(el('recording').value);};
video.addEventListener('timeupdate',()=>{el('time').textContent=video.currentTime.toFixed(2)+'s';links();});
video.addEventListener('loadedmetadata',()=>{if(Number.isFinite(initialTime)&&initialTime>0&&initialTime<video.duration)video.currentTime=initialTime;initialTime=0;});
document.addEventListener('DOMContentLoaded',()=>{if(recording)window.dispatchEvent(new CustomEvent('showonce-recording',{detail:{id:recording}}));});
(async()=>{try{const value=await json('/api/stages/recordings'),select=el('recording');select.replaceChildren();
  for(const item of value.recordings){const option=document.createElement('option');option.value=item.id;option.textContent=(item.task_name||'시연')+' · '+item.id+' · '+Number(item.duration_s).toFixed(2)+'초';select.append(option);}
  const id=new URLSearchParams(location.search).get('id');if(value.recordings.length)choose(value.recordings.some(r=>r.id===id)?id:value.recordings[0].id);
  else{select.disabled=true;el('error').textContent='먼저 수집 화면에서 시연을 촬영하세요.';el('stage-review-status').textContent='시연 없음';}
}catch(error){el('error').textContent=error.message;}})();
setInterval(()=>{if(!document.hidden)stageStatus();},4000);
