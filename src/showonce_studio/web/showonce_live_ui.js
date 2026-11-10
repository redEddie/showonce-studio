/* Live perception and the actual policy input are separate sources. Read-only. */
(()=>{
 const $=id=>document.getElementById(id);let packet=null,policyJob=null,reading=false,closed=false;
 const sources={vision_update:'실측 갱신',stationary_memory:'저장 위치',held_tcp:'파지 추정 · TCP',released_prediction:'놓은 뒤 낙하 추정',released_rest:'놓인 위치 추정'};
 const xyz=value=>Array.isArray(value)&&value.length===3?value.map(v=>(Number(v)*1000).toFixed(1)).join(', ')+' mm':'—';
 const age=value=>!Number.isFinite(value)?'없음':value>=86400?(value/86400).toFixed(1)+'일':value>=3600?(value/3600).toFixed(1)+'시간':value>=60?(value/60).toFixed(1)+'분':value<1?(Math.max(0,value)*1000).toFixed(0)+'ms':value.toFixed(1)+'초';
 function inputView(){
  const box=$('policy-input-values');box.replaceChildren();const input=policyJob?.policy_input;
  if(!input||input.input_kind!=='live'){$('policy-input-status').textContent='shadow 또는 run을 실행하면 정책 입력 위치와 관측·추정 출처를 표시합니다. 모의 입력은 여기에 실측처럼 표시하지 않습니다.';return;}
  const elapsed=Date.now()/1000-input.updated_s,active=policyJob.alive&&elapsed<1.;
  $('policy-input-status').textContent=(active?'현재 실행':'이전 실행 기록')+' · '+(policyJob.policy_name||'선택한 정책')+' · '+new Date(input.updated_s*1000).toLocaleTimeString()+(policyJob.report?.error?' · '+policyJob.report.error:'');
  const estimate=input.state_estimation||{};
  for(const [name,key,sourceKey,oid]of [['큐브','block_pos_m','cube_source','green_block'],['그릇','bowl_pos_m','bowl_source','bowl']]){
   const line=document.createElement('div');line.className='policy-input-row';line.textContent=name+' 입력 XYZ: '+xyz(input[key])+' · '+(sources[estimate[sourceKey]]||'실측 입력')+' · 마지막 관측 '+age(estimate.raw_observation_age_s?.[oid]);box.append(line);
  }
 }
 function render(){if(!packet)return;
  $('perception-connect-command').textContent='입력: Studio 촬영 노드 RGB-D 공유\n출력: '+(packet.folder||'인식 준비 중')+'\n진행: '+(packet.phase||'연결 확인 중');
  $('perception-badge').textContent=packet.connected?'인식 노드 연결됨':packet.phase==='SAM_model_loading'?'SAM 모델 로딩 중':packet.phase==='starting'?'인식 시작 중':'인식 미연결';$('live-input-status').dataset.liveMonitor='true';$('live-input-status').textContent=packet.message;$('perception-message').textContent=packet.message;
  $('perception-frame-label').textContent={capture_only:'촬영 원본 · SAM 인식 미연결',capture_offline:'촬영 카메라 연결 끊김',perception_overlay:'실시간 인식 오버레이',pending:'인식 오버레이 준비 중'}[packet.preview_kind]||'영상 연결 확인';
  $('perception-source').textContent=(packet.folder?'인식 출력: '+packet.folder+'\n':'')+(packet.calibration?'보정: '+packet.calibration:'');
  const box=$('perception-objects');box.replaceChildren();const rows=packet.objects?.length?packet.objects:[{id:'green_block',label:'큐브',status:'missing'},{id:'bowl',label:'그릇',status:'missing'}];
  for(const object of rows){const cell=document.createElement('div');cell.className='perception-object '+object.status;const name=document.createElement('strong');name.textContent=object.label;const state=document.createElement('span');state.className='object-state';state.textContent={fresh:'최신 관측',delayed:'관측 지연',missing:'현재 관측 없음'}[object.status];
   const text=document.createElement('p');text.className='hint';text.textContent='XYZ: '+xyz(object.status==='missing'?null:object.position_base_m)+'\n관측 나이: '+age(object.observation_age_s);text.style.whiteSpace='pre-line';cell.append(name,state,text);box.append(cell);
  }
  $('perception-diagnostics').textContent=JSON.stringify({phase:packet.phase,sample_age_s:packet.sample_age_s,sam:packet.sam,objects:packet.objects,error:packet.error},null,2);
  window.dispatchEvent(new CustomEvent('showonce:perception',{detail:packet}));
  inputView();
 }
 async function poll(){if(reading||document.hidden)return;reading=true;try{
  const active=policyJob?.alive?policyJob.policy_input?.perception_dir:null;
  const folder=active||($('perception-dir').dataset.edited==='true'?$('perception-dir').value:'');const query=folder?'?dir='+encodeURIComponent(folder):'';
  const response=await fetch('/api/deployment/perception'+query,{cache:'no-store',signal:AbortSignal.timeout(2000)});const value=await response.json();if(!response.ok)throw Error(value.error||'인식 상태를 읽지 못했습니다.');packet=value;
  if(packet.preview_kind==='capture_only'){
   try{const captureResponse=await fetch('/api/capture/state',{cache:'no-store',signal:AbortSignal.timeout(1200)}),capture=await captureResponse.json();
    const camera=capture.camera,elapsed=Date.now()/1000-Number(camera?.last_frame_host_s);
    if(!captureResponse.ok||camera?.status!=='ready'||!Number.isFinite(elapsed)||elapsed<-.01||elapsed>1.5)throw Error('촬영 카메라의 최신 RGB-D 프레임이 없습니다.');
   }catch(error){packet.preview_url=null;packet.preview_kind='capture_offline';packet.message+=' 촬영 카메라의 최신 프레임도 확인되지 않습니다.';}
  }
  if(packet.connected&&$('perception-dir').dataset.edited!=='true'){$('perception-dir').value=packet.folder;if(!$('calibration').dataset.edited)$('calibration').value=packet.calibration||$('calibration').value;}
  render();
 }catch(error){packet={connected:false,preview_kind:'capture_only',preview_url:'/capture.jpg',message:error.message,objects:[]};render();}finally{reading=false;}}
 const image=$('perception-frame'),empty=$('perception-image-empty');let imageTimer,lastImage=0;
 function refreshImage(){clearTimeout(imageTimer);if(closed)return;if(document.hidden||!packet?.preview_url){imageTimer=setTimeout(refreshImage,400);if(!packet?.preview_url){image.hidden=true;empty.hidden=false;}return;}
  image.src=packet.preview_url+(packet.preview_url.includes('?')?'&':'?')+'t='+Date.now();
 }
 image.onload=()=>{lastImage=Date.now();image.hidden=false;empty.hidden=true;imageTimer=setTimeout(refreshImage,125);};
 image.onerror=()=>{image.hidden=true;empty.hidden=false;empty.textContent='카메라 영상 연결 대기 중';imageTimer=setTimeout(refreshImage,800);};
 setInterval(()=>{if(lastImage&&Date.now()-lastImage>2000&&!document.hidden){image.hidden=true;empty.hidden=false;empty.textContent='영상 갱신이 끊겼습니다. 연결을 확인하고 있습니다.';refreshImage();}},2000);
 window.addEventListener('showonce:robot-state',event=>{const id=$('policy').value;policyJob=event.detail.jobs.find(job=>job.policy_id===id&&['shadow','run'].includes(job.mode))||null;inputView();});
 $('policy').addEventListener('change',()=>{policyJob=null;inputView();});$('perception-dir').addEventListener('input',()=>{$('perception-dir').dataset.edited='true';});$('perception-dir').addEventListener('change',poll);
 document.addEventListener('visibilitychange',()=>{if(!document.hidden){poll();refreshImage();}});window.addEventListener('pagehide',()=>{closed=true;clearTimeout(imageTimer);});
 window.addEventListener('pageshow',()=>{closed=false;poll();refreshImage();});
 $('perception-retry').onclick=async()=>{$('perception-retry').disabled=true;try{const response=await fetch('/api/deployment/live-service',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'}),value=await response.json();if(!response.ok||!value.configured)throw Error(value.error||'실시간 인식 환경 구성이 필요합니다.');$('perception-retry-status').textContent=value.alive?'인식 노드 연결 중 · 기존 모델이 실행 중이면 재로딩하지 않습니다.':'인식 시작 로그를 확인하세요.';await poll();}catch(error){$('perception-retry-status').textContent=error.message;}finally{$('perception-retry').disabled=false;}};
 poll();setInterval(poll,500);refreshImage();
})();
