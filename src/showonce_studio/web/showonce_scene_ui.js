import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {OBJLoader} from 'three/addons/loaders/OBJLoader.js';
import {prepareReplay,samplePose,nearestIndex,smoothReplay} from '/showonce_replay_math.js';

const el=id=>document.getElementById(id);
let recording='',generation=0,view=null,state=null,busy=false,loading=false,retryAt=0,visible=false;
let replay=null,rawReplay=null,smoothedReplay=null,replayTime=0,lastFrameEvent='';
const video=el('video'),physicalMode=()=>el('scene-physics').open;
const assetURL=path=>'/simulation/'+encodeURIComponent(recording)+'/'+path;
async function json(url,body){const response=await fetch(url,{cache:'no-store',method:body?'POST':'GET',
  headers:body?{'Content-Type':'application/json'}:{},body:body?JSON.stringify(body):undefined});
  const result=await response.json();if(!response.ok)throw Error(result.error||'시뮬레이션 연결을 확인하세요.');return result;}
function disposeTree(tree){tree.traverse(object=>{object.geometry?.dispose();
  for(const material of Array.isArray(object.material)?object.material:[object.material])material?.dispose();});}
function dispose(){generation++;state=null;replay=rawReplay=smoothedReplay=null;loading=false;busy=false;lastFrameEvent='';
  if(!view)return;cancelAnimationFrame(view.animation);view.controls.dispose();disposeTree(view.scene);
  view.renderer.dispose();view.renderer.domElement.remove();view=null;el('scene-labels').replaceChildren();}
function select(id){if(id===recording)return;dispose();recording=id;el('scene-physics').open=false;retryAt=0;updateButtons();loadScene();}
window.addEventListener('showonce-recording',event=>select(event.detail.id));
function updateButtons(){for(const id of ['scene-play','scene-reset','scene-drop'])el(id).disabled=busy||!state?.loaded;
  el('scene-load').disabled=loading||busy;el('scene-play').textContent=state?.running?'물리 일시정지':'물리 재생';
  el('scene-video-play').disabled=!replay||physicalMode();el('scene-video-start').disabled=!replay||physicalMode();
  el('scene-smoothing').disabled=!replay||physicalMode();
  el('scene-video-play').textContent=video.paused?'영상 재생':'영상 일시정지';}
function openAsset(id){window.dispatchEvent(new CustomEvent('showonce-open-asset',{detail:{id,variant:'sam3d_assets_v1'}}));}

async function loadScene(){if(!recording||loading)return;loading=true;const serial=++generation,id=recording;
  el('scene-status').textContent='영상과 연결된 복원 장면을 준비합니다.';updateButtons();
  try{const info=await json('/api/simulation/state?id='+encodeURIComponent(id));if(serial!==generation)return;
    el('scene-physics').hidden=info.manifest?.source==='asset_library'&&!info.manifest.simulation_ready;
    if(info.manifest?.source==='asset_library'&&!info.manifest.replay_ready){el('scene-status').textContent='선택한 에셋으로 이 영상의 포즈 추적을 진행하세요.';retryAt=Date.now()+4000;return;}
    if(!info.manifest?.complete){el('scene-status').textContent='SAM 3D 에셋이 생성되면 영상에 연결할 수 있습니다.';retryAt=Date.now()+5000;return;}
    const backend=el('pose-backend')?.value||'measured';
    const [layout,replayData]=await Promise.all([json(assetURL('scene.json')),json('/api/scene/replay?id='+encodeURIComponent(id)+'&backend='+backend)]);
    if(serial!==generation)return;
    rawReplay=prepareReplay(replayData);smoothedReplay=smoothReplay(rawReplay,.1);
    replay=el('scene-smoothing').checked?smoothedReplay:rawReplay;
    if(view){cancelAnimationFrame(view.animation);view.controls.dispose();disposeTree(view.scene);view.renderer.dispose();view.renderer.domElement.remove();}
    const scene=new THREE.Scene();scene.background=new THREE.Color('#eaf0ed');
    const camera=new THREE.PerspectiveCamera(42,1,.005,20);camera.up.set(0,0,1);
    const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.outputColorSpace=THREE.SRGBColorSpace;
    el('scene-viewer').append(renderer.domElement);
    const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.minDistance=.12;controls.maxDistance=3;
    scene.add(new THREE.HemisphereLight(0xffffff,0x84917e,2.8));const light=new THREE.DirectionalLight(0xffffff,2.3);light.position.set(.3,-.4,1);scene.add(light);
    const bounds=new THREE.Box3().setFromPoints([new THREE.Vector3(0,0,0),...layout.objects.map(o=>new THREE.Vector3(...o.pose_world.position))]);
    bounds.expandByScalar(.15);const center=bounds.getCenter(new THREE.Vector3());center.z=0;
    center.x=Math.round(center.x/.05)*.05;center.y=Math.round(center.y/.05)*.05;
    const span=Math.max(.8,Math.ceil(Math.max(bounds.max.x-bounds.min.x,bounds.max.y-bounds.min.y)/.1)*.1);
    const table=new THREE.Mesh(new THREE.BoxGeometry(span,span,.025),new THREE.MeshStandardMaterial({color:0xdfebe1,roughness:1}));
    table.position.set(center.x,center.y,-.0125);scene.add(table);
    const minor=new THREE.GridHelper(span,Math.round(span/.01),0xa4bbaa,0xc1d4c5),major=new THREE.GridHelper(span,Math.round(span/.05),0x819e8b,0x93b09a);
    for(const grid of [minor,major]){grid.rotation.x=Math.PI/2;grid.position.set(center.x,center.y,.0002);scene.add(grid);}
    const origin=new THREE.AxesHelper(.08);origin.position.z=.001;scene.add(origin);
    view={scene,camera,renderer,controls,center,radius:span*.7,objects:new Map(),specs:info.manifest.assets,layout,labels:new Map(),animation:null,dirty:true,collidersLoaded:false,collidersLoading:false};
    controls.addEventListener('change',()=>{if(view)view.dirty=true});
    el('scene-labels').replaceChildren();el('scene-objects').replaceChildren();
    await Promise.all(info.manifest.assets.map(async asset=>{
      const gltf=await new GLTFLoader().loadAsync(assetURL(asset.mesh));if(serial!==generation){disposeTree(gltf.scene);return;}
      const group=new THREE.Group();group.userData.assetId=asset.id;
      gltf.scene.traverse(object=>{if(object.isMesh){object.material.side=THREE.DoubleSide;object.material.transparent=false;object.material.opacity=1;}});
      gltf.scene.name='asset-mesh';group.add(gltf.scene);scene.add(group);view.objects.set(asset.id,group);
      const label=document.createElement('button');label.className='scene-object-label';label.textContent=asset.name;label.onclick=()=>openAsset(asset.id);
      el('scene-labels').append(label);view.labels.set(asset.id,{element:label,height:asset.extent_m[2]/2+.018});
      const button=document.createElement('button');button.className='secondary';button.textContent=asset.name+' 상세 ↗';button.onclick=()=>openAsset(asset.id);el('scene-objects').append(button);
    }));if(serial!==generation)return;
    let down=null;renderer.domElement.addEventListener('pointerdown',event=>{down=[event.clientX,event.clientY]});
    renderer.domElement.addEventListener('pointerup',event=>{if(!down||Math.hypot(event.clientX-down[0],event.clientY-down[1])>4)return;
      const r=renderer.domElement.getBoundingClientRect(),mouse=new THREE.Vector2((event.clientX-r.left)/r.width*2-1,-(event.clientY-r.top)/r.height*2+1),ray=new THREE.Raycaster();
      ray.setFromCamera(mouse,camera);const hits=ray.intersectObjects([...view.objects.values()].filter(group=>group.visible),true);if(!hits.length)return;
      let object=hits[0].object;while(object&&!object.userData.assetId)object=object.parent;if(object)openAsset(object.userData.assetId);});
    resize();setView('iso');if(physicalMode())await command('load');else syncVideo(video.currentTime);
    const animate=()=>{if(!view||serial!==generation)return;controls.update();
      if(visible&&!document.hidden&&view.dirty){updateLabels();renderer.render(scene,camera);view.dirty=false;}
      view.animation=requestAnimationFrame(animate);};animate();
    if(el('scene-collisions').checked)loadColliders();
    window.dispatchEvent(new CustomEvent('showonce-scene-loaded',{detail:{id,backend:'timestamped RGB-D pose replay',objects:[...view.objects.keys()]}}));
  }catch(error){if(serial===generation){el('scene-status').textContent=error.message;retryAt=Date.now()+5000;}}
  finally{if(serial===generation){loading=false;updateButtons();}}}

function applyState(value){state=value;updateButtons();if(!physicalMode())return;
  if(!value.loaded){el('scene-status').textContent='물리 검증 장면을 불러오세요.';return;}
  if(view)for(const [id,pose]of Object.entries(value.objects)){
    const group=view.objects.get(id);if(!group)continue;group.visible=true;appearance(group,'estimated');group.position.set(...pose.position);
    const q=pose.quat_wxyz;group.quaternion.set(q[1],q[2],q[3],q[0]);view.dirty=true;
    const label=view.labels.get(id);label.element.textContent=view.specs.find(asset=>asset.id===id).name;label.element.classList.remove('pose-uncertain');
  }
  el('scene-engine').textContent=value.backend;
  el('scene-time').textContent=value.time_s.toFixed(2)+'s';
  el('scene-status').textContent=(value.scenario==='drop'?'그릇 고정 · 큐브 낙하 테스트':'초기 배치 · 테이블 위 큐브와 그릇')+
    ' · '+(value.running?'물리 시뮬레이션 실행 중':'일시정지')+' · 접촉 '+value.contacts+'개'+(value.pause_reason?' · '+value.pause_reason:'');
  window.dispatchEvent(new CustomEvent('showonce-scene-physics-frame',{detail:value}));
}
function appearance(group,status){if(group.userData.poseStatus===status)return;group.userData.poseStatus=status;
  const mesh=group.getObjectByName('asset-mesh');if(!mesh)return;
  mesh.traverse(object=>{if(object.isMesh)for(const material of Array.isArray(object.material)?object.material:[object.material]){
    material.transparent=status!=='estimated';material.opacity=status==='held'?.3:status==='interpolated'?.8:1;material.depthWrite=status!=='held';}});}
function syncVideo(time=video.currentTime){updateButtons();if(!view||!replay||physicalMode())return;replayTime=time;
  const statuses=[],objects={};
  for(const spec of replay.objects){const group=view.objects.get(spec.id),pose=samplePose(replay,spec.id,time);if(!group)continue;
    const label=view.labels.get(spec.id);label.element.textContent=spec.name;
    group.visible=!!pose?.position;
    if(!pose?.position){statuses.push(spec.name+': 포즈 없음');objects[spec.id]={status:'unresolved',position:null};continue;}
    group.position.copy(pose.position);group.quaternion.copy(pose.quaternion);appearance(group,pose.status);
    if(pose.status==='held'){label.element.textContent=spec.name+' · 위치 유지';statuses.push(spec.name+': 포즈 누락 · 직전 위치 '+pose.age_s.toFixed(2)+'s 유지');}
    else if(pose.status==='interpolated'){label.element.textContent=spec.name+' · 보간';statuses.push(spec.name+': 짧은 누락 보간');}
    else statuses.push(spec.name+': 추정 포즈');
    label.element.classList.toggle('pose-uncertain',pose.status!=='estimated');
    objects[spec.id]={status:pose.status,position:group.position.toArray(),quat_xyzw:group.quaternion.toArray(),source_index:pose.source_index,age_s:pose.age_s};
  }
  const frame=replay.frames[nearestIndex(replay.frames,time)],stage=replay.stages.find(s=>time>=s.start_s&&(time<s.end_s||time>=replay.duration_s&&s===replay.stages.at(-1)));
  el('scene-engine').textContent='영상 시간 연동';el('scene-time').textContent=time.toFixed(2)+'s';
  el('scene-status').textContent=(replay.pose_backend==='foundationpose'?'FoundationPose · ':'RGB-D · ')+'원본 프레임 '+frame.index+(replay.smoothing?' · EMA 100ms':' · 원본 포즈')+' · '+statuses.join(' / ');
  el('scene-stage').textContent=stage?'확정 스테이지: '+stage.label+' · '+stage.start_s.toFixed(2)+'–'+stage.end_s.toFixed(2)+'s':'확정 스테이지 없음';
  view.dirty=true;el('scene-viewer').dataset.mode='video';el('scene-viewer').dataset.sourceFrame=String(frame.index);
  const key=String(time);if(key!==lastFrameEvent){lastFrameEvent=key;
    window.dispatchEvent(new CustomEvent('showonce-replay-frame',{detail:{recording_id:recording,time_s:time,frame_index:frame.index,stage:stage?.code,smoothing:!!replay.smoothing,objects}}));}
}
el('scene-smoothing').onchange=()=>{if(!rawReplay)return;replay=el('scene-smoothing').checked?smoothedReplay:rawReplay;lastFrameEvent='';syncVideo(replayTime);};
window.addEventListener('showonce-pose-backend',event=>{if(event.detail.id===recording){dispose();loadScene();}});
window.addEventListener('showonce-pose-ready',event=>{if(event.detail.id===recording&&el('pose-backend').value==='foundationpose'){dispose();loadScene();}});
async function command(action){if(busy||!recording)return;busy=true;updateButtons();const serial=generation;
  try{const value=await json('/api/scene/command',{recording_id:recording,action});if(serial===generation)applyState(value);}
  catch(error){if(serial===generation)el('scene-status').textContent=error.message;}
  finally{if(serial===generation){busy=false;updateButtons();}}}
el('scene-load').onclick=()=>loadScene();el('scene-play').onclick=()=>command(state?.running?'pause':'play');
el('scene-reset').onclick=()=>command('reset');el('scene-drop').onclick=()=>command('drop');
el('scene-video-play').onclick=()=>video.paused?video.play().catch(error=>{el('scene-status').textContent=error.message;}):video.pause();
el('scene-video-start').onclick=()=>{video.pause();video.currentTime=0;syncVideo(0);};
el('scene-physics').addEventListener('toggle',async()=>{
  window.dispatchEvent(new CustomEvent('showonce-scene-mode',{detail:{recording_id:recording,mode:physicalMode()?'physics':'video'}}));
  const serial=generation;
  if(physicalMode()){video.pause();el('scene-stage').textContent='별도 물리 검증 · 영상 재생과 독립';
    el('scene-viewer').dataset.mode='physics';await command('load');}
  else {if(state?.running)await command('pause');if(serial!==generation)return;
    el('scene-collisions').checked=false;for(const group of view?.objects.values()||[]){const collision=group.getObjectByName('collisions');if(collision)collision.visible=false;}
    syncVideo(video.currentTime);}
  updateButtons();});
for(const event of ['loadedmetadata','timeupdate','seeking','seeked','play','pause'])video.addEventListener(event,()=>syncVideo(video.currentTime));
if('requestVideoFrameCallback'in video){const tick=(now,meta)=>{syncVideo(meta.mediaTime);video.requestVideoFrameCallback(tick)};video.requestVideoFrameCallback(tick);}

async function loadColliders(){if(!view||view.collidersLoading||view.collidersLoaded)return;
  const serial=generation;view.collidersLoading=true;
  try{await Promise.all(view.layout.objects.map(async object=>{
    const group=view.objects.get(object.id);if(!group)return;
    const collision=new THREE.Group();collision.name='collisions';
    await Promise.all(object.colliders.map(async path=>{
      const obj=await new OBJLoader().loadAsync(assetURL(path));
      obj.traverse(part=>{if(part.isMesh){part.material.dispose();part.material=new THREE.MeshBasicMaterial({color:0xd68b28,transparent:true,opacity:.32,wireframe:true,depthTest:false});}});
      collision.add(obj);
    }));if(serial!==generation){disposeTree(collision);return;}group.add(collision);
    collision.visible=el('scene-collisions').checked;
  }));if(serial===generation){view.collidersLoaded=true;view.dirty=true;
    window.dispatchEvent(new CustomEvent('showonce-scene-collisions-loaded',{detail:{id:recording,parts:view.layout.objects.reduce((n,o)=>n+o.colliders.length,0)}}));}}
  catch(error){if(serial===generation)el('scene-status').textContent='충돌 형상: '+error.message;}
  finally{if(serial===generation&&view)view.collidersLoading=false;}}
el('scene-collisions').onchange=()=>{if(!view)return;loadColliders();for(const group of view.objects.values()){
  const collisions=group.getObjectByName('collisions');if(collisions)collisions.visible=el('scene-collisions').checked;}view.dirty=true;};
function updateLabels(){if(!view)return;const r=el('scene-viewer').getBoundingClientRect(),placed=[];
  for(const [id,label]of view.labels){const group=view.objects.get(id),p=group.position.clone();p.z+=label.height;p.project(view.camera);
    label.element.hidden=!group.visible||Math.abs(p.x)>1||Math.abs(p.y)>1||p.z>1||(physicalMode()&&state?.scenario==='drop'&&id==='bowl');
    if(label.element.hidden)continue;
    const width=label.element.offsetWidth,height=label.element.offsetHeight;
    const x=Math.max(width/2+4,Math.min(r.width-width/2-4,(p.x+1)*r.width/2));let y=(-p.y+1)*r.height/2;
    for(const previous of placed)if(x+width/2>previous.left&&x-width/2<previous.right&&y>previous.top-4&&y-height<previous.bottom+4)y=previous.top-6;
    y=Math.max(height+4,y);placed.push({left:x-width/2,right:x+width/2,top:y-height,bottom:y});
    label.element.style.left=x+'px';label.element.style.top=y+'px';}}
function resize(){if(!view)return;const r=el('scene-viewer').getBoundingClientRect();view.renderer.setSize(r.width,r.height);view.camera.aspect=r.width/r.height;view.camera.updateProjectionMatrix();view.dirty=true;}
function setView(mode){if(!view)return;const {camera,controls,center,radius}=view;
  const direction=mode==='top'?new THREE.Vector3(0,-.0001,radius*1.6):new THREE.Vector3(radius,-radius,radius*.9);
  camera.position.copy(center).add(direction);controls.target.copy(center).add(new THREE.Vector3(0,0,.025));camera.lookAt(controls.target);controls.update();view.dirty=true;}
for(const button of document.querySelectorAll('[data-scene-view]'))button.onclick=()=>setView(button.dataset.sceneView);
new ResizeObserver(resize).observe(el('scene-viewer'));
new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;if(view)view.dirty=true;},{rootMargin:'100px'}).observe(el('scene-viewer'));
window.addEventListener('beforeunload',dispose);
async function tick(){try{if(recording&&!document.hidden&&!loading&&!busy){
    if(!view&&Date.now()>retryAt)await loadScene();
    else if(view&&physicalMode()){const serial=generation,value=await json('/api/scene/state?id='+encodeURIComponent(recording));if(serial===generation)applyState(value);}
  }}catch(error){el('scene-status').textContent=error.message;}
  setTimeout(tick,physicalMode()&&state?.running?50:600);}
const initial=new URLSearchParams(location.search).get('id');if(initial)select(initial);tick();
