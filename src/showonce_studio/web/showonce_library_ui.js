/* Reuse versioned assets for each newly recorded demonstration. */
const el=id=>document.getElementById(id);let recording='',catalog=[],epoch=0,lastSelection=null;
async function json(url,body){const response=await fetch(url,{cache:'no-store',method:body?'POST':'GET',headers:body?{'Content-Type':'application/json'}:{},body:body?JSON.stringify(body):undefined});
  const value=await response.json();if(!response.ok)throw Error(value.error||'에셋을 불러오지 못했습니다.');return value;}
const optionValue=asset=>asset.asset_id+'@'+asset.version;
function choose(role){const [asset_id,version]=el('library-'+role).value.split('@');return {asset_id,version};}
async function loadCatalog(){const value=await json('/api/library/catalog');catalog=value.assets;
  for(const role of ['green_block','bowl']){const select=el('library-'+role);select.replaceChildren();
    for(const item of catalog.filter(a=>a.role===role)){const option=document.createElement('option');option.value=optionValue(item);option.textContent=item.asset.name+' · '+item.version;select.append(option);}}
  const cards=el('library-cards');cards.replaceChildren();
  for(const item of catalog){const card=document.createElement('button');card.className='asset-card';
    const image=document.createElement('img');image.src='/library/'+item.asset_id+'/'+item.version+'/thumbnail.png';image.alt=item.asset.name;
    const text=document.createElement('div'),title=document.createElement('strong'),meta=document.createElement('small');title.textContent=item.asset.name+' · '+item.version;
    meta.textContent='등록된 참조 형상 · 마우스로 돌려보기';text.append(title,meta);card.append(image,text);
    card.onclick=()=>window.dispatchEvent(new CustomEvent('showonce-open-library-asset',{detail:{asset:item.asset}}));cards.append(card);}
  el('library-count').textContent=catalog.length+'개 등록';}
function reset(id){if(id===recording)return;recording=id;epoch++;lastSelection=null;poll();}
window.addEventListener('showonce-recording',event=>reset(event.detail.id));
async function poll(){if(!recording||!catalog.length)return;const serial=epoch,id=recording;
  try{const state=await json('/api/library/state?id='+encodeURIComponent(id));if(serial!==epoch)return;
    const busy=!['idle','error','complete'].includes(state.job.status),locked=state.legacy||!!state.selection||busy;
    el('pose-start').hidden=!!state.selection;el('relation-start').hidden=!!state.selection;
    for(const role of ['green_block','bowl'])el('library-'+role).disabled=locked;
    if(state.selection&&state.selection.selection_sha256!==lastSelection){for(const role of ['green_block','bowl'])el('library-'+role).value=optionValue(state.selection.assets[role]);lastSelection=state.selection.selection_sha256;}
    el('library-analyze').disabled=busy||state.analysis_complete;
    el('library-analyze').textContent=state.analysis_complete?'분석 완료':busy?'분석 중…':state.job.status==='error'?'이어서 분석':'선택한 에셋으로 분석';
    const nested=state.job.status==='masks'?state.mask_job:state.job.status==='pose'?state.pose_job:state.relation_job;
    el('library-progress').textContent=state.legacy?'현재 영상의 기존 분석을 사용합니다. 새 영상을 선택하면 보관함 에셋으로 분석할 수 있습니다.':
      state.analysis_complete?'포즈·관계 추출 완료 · 영상과 3D 장면에서 결과를 확인하세요.':state.job.status==='error'?
      (nested?.error||state.pose_job?.error||state.relation_job?.error||state.job.error):busy?
      (state.job.message||'분석 중')+(nested?.total?` · ${nested.frame||0}/${nested.total}`:''):
      '등록한 메시를 재사용해 마스크 → 포즈 → 관계를 추출합니다. 스테이지 분할·확정은 별도 스테이지 페이지에서 진행할 수 있습니다.';
  }catch(error){if(serial===epoch)el('library-progress').textContent=error.message;}}
el('library-analyze').onclick=async()=>{if(!recording)return;el('library-analyze').disabled=true;
  try{await json('/api/library/analyze',{recording_id:recording,selection:{green_block:choose('green_block'),bowl:choose('bowl')}});await poll();}
  catch(error){el('library-progress').textContent=error.message;el('library-analyze').disabled=false;}};
loadCatalog().then(()=>{const id=new URLSearchParams(location.search).get('id');if(id)reset(id);poll();}).catch(error=>el('library-progress').textContent=error.message);
setInterval(()=>{if(!document.hidden)poll();},1800);
