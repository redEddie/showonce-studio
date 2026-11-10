import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';

const el=id=>document.getElementById(id), video=el('video'), overlay=el('object-overlay');
let recording='',epoch=0,packet=null,tracks=null,artifactStamp=null,assets=[],assetNodes=new Map(),projectionPoints=new Map();
let overlayFrame=null,viewer=null,viewerGeneration=0;
let simulationPacket=null,simulationStamp=null;
let posePacket=null,poseTracks=null,poseStamp=null,poseGeometry=new Map();
const poseURL=name=>'/poses/'+encodeURIComponent(recording)+'/'+name;
const activeTracks=()=>el('pose-backend').value==='foundationpose'?poseTracks:tracks;
const phaseNames={idle:'객체 추적·형상 복원을 시작하세요.',queued:'분석 대기 중',waiting_for_gpu:'GPU 분석 대기 중',
  loading_model:'SAM 영상 추적 모델 준비 중',tracking:'객체 마스크 추적 중',reconstructing:'깊이 표면 준비 중',
  registering:'프레임별 포즈 정합 중',fusing:'여러 프레임의 측정 표면을 합칩니다.',complete:'추적·측정 표면 복원 완료',error:'분석 중 오류가 발생했습니다.'};
const assetURL=(name,asset=null)=>asset?.variant==='library'?'/library/'+encodeURIComponent(asset.library_asset_id)+'/'+encodeURIComponent(asset.library_version)+'/'+name:
  (asset?.variant==='sam3d_assets_v1'?'/simulation/':'/geometry/')+encodeURIComponent(recording)+'/'+name;
const displayedAssets=()=>el('simulation-assets').checked&&simulationPacket?.manifest?.complete?simulationPacket.manifest.assets:assets;
async function json(url,body){const response=await fetch(url,{cache:'no-store',method:body?'POST':'GET',
  headers:body?{'Content-Type':'application/json'}:{},body:body?JSON.stringify(body):undefined});
  if(!response.ok){let error;try{error=await response.json()}catch{}throw Error(error?.error||'분석 결과를 불러오지 못했습니다.');}
  return response.json();}
function reset(id){if(id===recording)return;recording=id;epoch++;packet=tracks=null;artifactStamp=null;assets=[];
  simulationPacket=null;simulationStamp=null;
  posePacket=poseTracks=poseStamp=null;poseGeometry.clear();
  projectionPoints.clear();assetNodes.clear();overlayFrame=null;renderOverlay();renderAssets();
  if(el('asset-dialog').open)el('asset-dialog').close();poll();}
window.addEventListener('showonce-recording',event=>reset(event.detail.id));
window.addEventListener('showonce-open-asset',event=>{const list=event.detail.variant==='sam3d_assets_v1'?simulationPacket?.manifest?.assets||[]:displayedAssets();const asset=list.find(value=>value.id===event.detail.id);if(asset)openAsset(asset);});
window.addEventListener('showonce-open-library-asset',event=>openAsset(event.detail.asset));

async function poll(){if(!recording)return;const serial=epoch,id=recording;
  try{const [value,sim,pose]=await Promise.all([json('/api/geometry/state?id='+encodeURIComponent(id)),json('/api/simulation/state?id='+encodeURIComponent(id)),json('/api/pose/state?id='+encodeURIComponent(id))]);if(serial!==epoch)return;packet=value;posePacket=pose;
    const poseBusy=!['idle','error','complete','probe_complete'].includes(pose.job?.status||'idle');
    el('pose-start').disabled=poseBusy||pose.tracking_ready||!pose.references_ready;
    el('pose-start').textContent=pose.tracking_ready?'FoundationPose 추적 완료':poseBusy?'FoundationPose 추적 중…':'FoundationPose 추적';
    el('pose-status').textContent=pose.job?.error||poseBusy?(pose.job?.error||pose.job?.message||'FoundationPose 준비 중')+(pose.job?.total?` · ${pose.job.frame||0}/${pose.job.total}`:''):
      pose.tracking_ready?'FoundationPose · 참조 메시 기반 포즈 · SAM 마스크와 메시 투영을 비교하세요.':'참조 에셋을 선택해 FoundationPose 추적을 실행하세요.';
    const nextPoseStamp=pose.tracking_ready?pose.manifest.created_s:null;
    if(nextPoseStamp&&nextPoseStamp!==poseStamp){
      const data=await json(poseURL('tracking.json')),geometry=new Map();
      await Promise.all(data.reference_assets.map(async asset=>geometry.set(asset.id,await json(poseURL(asset.overlay)))));
      if(serial!==epoch)return;poseTracks=data;poseGeometry=geometry;poseStamp=nextPoseStamp;renderOverlay();
      window.dispatchEvent(new CustomEvent('showonce-pose-ready',{detail:{id,backend:'foundationpose',stamp:poseStamp}}));
    }
    const previousSimulationStamp=simulationStamp;simulationPacket=sim;simulationStamp=sim.manifest?.created_s||null;
    const simStatus=sim.job?.status||'idle',simBusy=['loading','generating','exporting','validating','waiting_for_gpu','downloading'].includes(simStatus);
    el('simulation-start').disabled=simBusy||!sim.runtime?.verified||!!sim.manifest?.complete;
    const reused=sim.manifest?.source==='asset_library';
    el('simulation-start').hidden=true;
    el('simulation-start').textContent=sim.manifest?.complete?'SAM 3D 에셋 생성 완료':simBusy?'SAM 3D 처리 중…':'SAM 3D 에셋 생성';
    el('simulation-status').textContent=reused?'보관함의 고정된 에셋을 사용합니다. 새 영상에서 위치·자세만 추적합니다.':sim.manifest?.complete?
      (sim.manifest.physics_validated?'SAM 3D 형상 · 물리 검증 완료. ':'SAM 3D 형상 생성 · 물리 검증 확인 필요. ')+'측정 점군과 비교하고 시뮬레이션 파일을 다운로드하세요.':
      simBusy||['error','awaiting_approval'].includes(simStatus)?sim.job.message:
      !sim.runtime?.verified?'RGB-D 입력 준비 '+(sim.inputs_ready?'완료':'전')+' · Thor용 SAM 3D 실행 환경 준비 중':
      !sim.checkpoints.ready?'RGB-D 입력 준비 완료 · SAM 3D 가중치 다운로드가 필요합니다. 승인 후 생성 버튼을 누르세요.':
      'RGB-D 입력과 실행 환경 준비 완료 · SAM 3D 에셋을 생성할 수 있습니다.';
    el('simulation-download').hidden=!sim.manifest?.simulation_ready;
    el('simulation-download').href='/simulation/'+encodeURIComponent(id)+'/simulation-assets.zip';
    if(previousSimulationStamp!==simulationStamp)renderAssets();
    const job=value.job||{},busy=!['idle','error','complete'].includes(job.status);
    el('geometry-start').disabled=busy||!!value.manifest?.complete;
    el('geometry-start').textContent=value.manifest?.complete?'객체 분석 완료':busy?'분석 중…':'객체 추적 · 형상 복원';
    el('geometry-status').textContent=(job.error||phaseNames[job.status]||job.status)+
      (busy&&job.total?` · ${job.frame||0}/${job.total}`:'')+(value.manifest?.complete?' · 영상 속 객체 또는 아래 에셋을 클릭해 3D로 확인하세요.':'');
    if(value.tracking_ready&&!tracks){const data=await json(assetURL('tracking.json'));if(serial!==epoch)return;tracks=data;renderOverlay();}
    const stamp=value.manifest?.created_s;
    if(stamp&&stamp!==artifactStamp){
      const data=await json(assetURL('tracking.json'));if(serial!==epoch)return;tracks=data;assets=value.manifest.assets;
      projectionPoints.clear();await Promise.all(assets.map(async asset=>{
        if(asset.overlay_points){const p=await json(assetURL(asset.overlay_points));if(serial===epoch)projectionPoints.set(asset.id,p.positions);}
      }));if(serial!==epoch)return;artifactStamp=stamp;renderAssets();renderOverlay();
    }
  }catch(error){if(serial===epoch)el('geometry-status').textContent=error.message;}}
el('simulation-assets').onchange=()=>renderAssets();
el('pose-backend').onchange=()=>{renderOverlay();window.dispatchEvent(new CustomEvent('showonce-pose-backend',{detail:{id:recording,backend:el('pose-backend').value}}));};
el('pose-start').onclick=async()=>{if(!recording)return;el('pose-start').disabled=true;
  try{await json('/api/pose/start',{recording_id:recording});await poll();}
  catch(error){el('pose-status').textContent=error.message;el('pose-start').disabled=false;}};
el('simulation-start').onclick=async()=>{if(!recording)return;el('simulation-start').disabled=true;
  try{await json('/api/simulation/start',{recording_id:recording});await poll();}
  catch(error){el('simulation-status').textContent=error.message;el('simulation-start').disabled=false;}};
el('geometry-start').onclick=async()=>{if(!recording)return;el('geometry-start').disabled=true;
  try{await json('/api/geometry/start',{recording_id:recording});await poll();}
  catch(error){el('geometry-status').textContent=error.message;el('geometry-start').disabled=false;}};

function renderAssets(){const box=el('asset-cards');box.replaceChildren();assetNodes.clear();
  const shown=displayedAssets();el('assets-count').textContent=shown.length?shown.length+'개 객체':'분석 전';
  if(!shown.length){const empty=document.createElement('div');empty.className='hint';empty.textContent='추적·복원이 완료되면 객체별 형상이 여기에 표시됩니다.';box.append(empty);return;}
  for(const asset of shown){const button=document.createElement('button');button.className='asset-card';button.dataset.asset=asset.id;
    const image=document.createElement('img');image.src=assetURL(asset.thumbnail,asset);image.alt=asset.name+' 촬영 영상';
    const text=document.createElement('div'),name=document.createElement('strong'),meta=document.createElement('small'),open=document.createElement('span');
    name.textContent=asset.name;meta.textContent=asset.variant==='sam3d_assets_v1'?
      (asset.library_asset_id?`보관함 에셋 · ${asset.library_version} · 현재 영상 측정 점군 비교`:`SAM 3D 완성 형상 · ${asset.collider_parts}개 충돌 조각 · ${asset.physics_validated?'물리 검증 통과':'물리 검증 확인 필요'}`):
      `측정 표면 · ${asset.fusion_frame_indices.length}개 프레임 융합 · 전체 형상 복원 전`;
    open.className='asset-open';open.textContent='3D로 돌려보기 · 측정 점군 비교 ↗';text.append(name,meta,open);button.append(image,text);
    button.onclick=()=>openAsset(asset);box.append(button);assetNodes.set(asset.id,button);}}

function nearestFrame(time){const data=activeTracks();if(!data?.frames?.length)return null;let lo=0,hi=data.frames.length-1;
  while(lo<hi){const mid=(lo+hi)>>1;if(data.frames[mid].time_s<time)lo=mid+1;else hi=mid;}
  if(lo>0&&time-data.frames[lo-1].time_s<data.frames[lo].time_s-time)lo--;return data.frames[lo];}
function apply(point,m){return [m[0][0]*point[0]+m[0][1]*point[1]+m[0][2]*point[2]+m[0][3],
  m[1][0]*point[0]+m[1][1]*point[1]+m[1][2]*point[2]+m[1][3],m[2][0]*point[0]+m[2][1]*point[1]+m[2][2]*point[2]+m[2][3]];}
function project(point,pose){const manifest=el('pose-backend').value==='foundationpose'&&posePacket?.manifest?.intrinsics?posePacket.manifest:packet?.manifest;if(!manifest)return null;
  const camera=apply(apply(point,pose),manifest.T_cam_base),k=manifest.intrinsics;if(camera[2]<=0)return null;
  return [k.fx*camera[0]/camera[2]+k.cx,k.fy*camera[1]/camera[2]+k.cy];}
function renderOverlay(time=video.currentTime){const context=overlay.getContext('2d');overlay.width=video.videoWidth||848;overlay.height=video.videoHeight||480;
  context.clearRect(0,0,overlay.width,overlay.height);const frame=nearestFrame(time);overlayFrame=frame;if(!frame)return;
  for(const spec of activeTracks().objects){const object=frame.objects[spec.id];if(!object||object.status==='not_visible')continue;
    if(el('overlay-masks').checked){context.strokeStyle=spec.color;context.fillStyle=spec.color+'33';context.lineWidth=2;
      for(const contour of object.contours){if(contour.length<3)continue;context.beginPath();context.moveTo(...contour[0]);
        for(const point of contour.slice(1))context.lineTo(...point);context.closePath();context.fill();context.stroke();}}
    const pose=object.pose_base;
    const reference=el('pose-backend').value==='foundationpose'?poseGeometry.get(spec.id):null;
    if(pose&&reference&&el('overlay-mesh').checked){
      const points=reference.vertices.map(point=>project(point,pose));context.strokeStyle=spec.color+'a8';context.lineWidth=.65;context.beginPath();
      for(const [a,b]of reference.edges){if(!points[a]||!points[b])continue;context.moveTo(...points[a]);context.lineTo(...points[b]);}context.stroke();
    }
    if(pose&&el('overlay-points').checked){const positions=projectionPoints.get(spec.id)||[];context.fillStyle=spec.color+'c0';
      const points=reference?reference.vertices:Array.from({length:positions.length/3},(_,i)=>positions.slice(i*3,i*3+3));
      for(const point of points){const p=project(point,pose);if(p)context.fillRect(p[0]-.7,p[1]-.7,1.4,1.4);}}
    if(pose&&el('overlay-axes').checked){const origin=project([0,0,0],pose);
      if(origin)for(const [endpoint,color,label] of [[[.025,0,0],'#ed6258','X'],[[0,.025,0],'#45b875','Y'],[[0,0,.025],'#5488ea','Z']]){
        const p=project(endpoint,pose);if(!p)continue;context.strokeStyle=context.fillStyle=color;context.lineWidth=2;
        context.beginPath();context.moveTo(...origin);context.lineTo(...p);context.stroke();context.font='bold 11px system-ui';context.fillText(label,p[0]+3,p[1]-2);}}
    if(object.bbox){const [x,y]=object.bbox,label=spec.name+(object.pose_status==='unresolved'?' · 포즈 미확정':reference?' · FP':'')+
      (reference&&object.quality?.mask_iou!=null?' · IoU '+object.quality.mask_iou.toFixed(2):'');
      context.font='bold 12px system-ui';const width=context.measureText(label).width+12;
      context.fillStyle='#16382de8';context.fillRect(x,Math.max(0,y-22),width,20);context.fillStyle='white';context.fillText(label,x+6,Math.max(14,y-7));}}
  }
for(const id of ['overlay-masks','overlay-axes','overlay-points','overlay-mesh'])el(id).onchange=()=>renderOverlay();
for(const event of ['loadedmetadata','seeked','timeupdate','pause'])video.addEventListener(event,()=>renderOverlay());
if('requestVideoFrameCallback'in video){const tick=(now,meta)=>{renderOverlay(meta.mediaTime);video.requestVideoFrameCallback(tick)};video.requestVideoFrameCallback(tick);}
el('video-stack').addEventListener('click',event=>{if(!overlayFrame||!assets.length)return;const r=video.getBoundingClientRect();
  if(event.clientY>r.bottom-40)return;const x=(event.clientX-r.left)/r.width*overlay.width,y=(event.clientY-r.top)/r.height*overlay.height;
  const candidates=displayedAssets().filter(asset=>{const box=overlayFrame.objects[asset.id]?.bbox;return box&&x>=box[0]&&x<=box[2]&&y>=box[1]&&y<=box[3];})
    .sort((a,b)=>overlayFrame.objects[a.id].area_px-overlayFrame.objects[b.id].area_px);
  if(candidates.length){event.preventDefault();event.stopImmediatePropagation();openAsset(candidates[0]);}},true);

function disposeViewer(){viewerGeneration++;if(!viewer)return;cancelAnimationFrame(viewer.animation);viewer.controls.dispose();
  viewer.scene.traverse(object=>{object.geometry?.dispose();const materials=Array.isArray(object.material)?object.material:[object.material];for(const material of materials)material?.dispose();});
  viewer.renderer.dispose();viewer.renderer.domElement.remove();viewer=null;}
el('asset-close').onclick=()=>el('asset-dialog').close();el('asset-dialog').addEventListener('close',disposeViewer);
el('asset-dialog').addEventListener('click',event=>{if(event.target===el('asset-dialog')){const r=el('asset-dialog').getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)el('asset-dialog').close();}});

async function openAsset(asset){if(!asset)return;disposeViewer();const generation=viewerGeneration,id=recording;
  el('asset-title').textContent=asset.name+' · 3D 에셋 검토';el('asset-viewer-error').textContent='메시와 측정 점군을 불러오는 중…';
  el('asset-kind').textContent=asset.library_asset_id?'보관함 에셋 · '+asset.library_version:asset.variant==='sam3d_assets_v1'?'SAM 3D 완성 형상 · 측정 점군 비교':'측정 표면 · 전체 형상 복원 전';
  el('asset-detail-note').textContent='드래그: 회전 · 휠: 확대/축소 · 오른쪽 드래그: 이동. 격자는 10mm, 큰 격자는 50mm입니다. '+
    (asset.library_asset_id?(asset.variant==='library'?'고정된 참조 형상입니다. 영상에 적용하면 해당 영상의 측정 점군과 비교할 수 있습니다. ':'고정된 참조 형상과 이 영상에서 측정한 점군을 비교합니다. 형상을 다시 생성하거나 이전 영상의 포즈를 사용하지 않습니다. '):asset.variant==='sam3d_assets_v1'?'보이지 않는 면은 모델이 추정한 형상입니다. 청록색 점군은 선택한 원본 RGB-D 프레임에서 측정했습니다. 시뮬레이션 초기 배치는 테이블 지지면에 맞춰 높이를 별도로 보정했습니다. 물리 속성은 설정값이며 실측값이 아닙니다. ':'관측한 표면들을 합친 결과입니다. 가려졌거나 보이지 않은 뒷면은 생성하지 않았습니다. ')+
    '표면 정합 수치는 실물 정답 형상의 오차와 구별합니다.';
  if(!el('asset-dialog').open)el('asset-dialog').showModal();
  const details=el('asset-details');details.replaceChildren();
  for(const [label,value]of [['복원 범위 (X × Y × Z)',asset.extent_m.map(x=>(x*1000).toFixed(1)).join(' × ')+' mm'],
    [asset.library_asset_id?'참조 에셋 버전':asset.variant==='sam3d_assets_v1'?'형상 후보 생성 프레임':'융합에 사용한 프레임',asset.library_asset_id?asset.library_asset_id+' · '+asset.library_version:asset.variant==='sam3d_assets_v1'?asset.generation_frames.join(', '):asset.fusion_frame_indices.length+' / '+asset.total_frames],['측정 점 수',asset.variant==='library'?'영상 적용 후 측정':asset.measured_points.toLocaleString()],
    ['별도 프레임의 표면 정합',asset.median_diagnostic_surface_distance_m!=null?(asset.median_diagnostic_surface_distance_m*1000).toFixed(2)+' mm':'측정 없음']]){
    const cell=document.createElement('div'),strong=document.createElement('strong');cell.textContent=label;strong.textContent=value;cell.append(strong);details.append(cell);}
  try{const scene=new THREE.Scene();scene.background=new THREE.Color('#eaf0ed');
    const camera=new THREE.PerspectiveCamera(40,1,.0002,10);camera.up.set(0,0,1);
    const renderer=new THREE.WebGLRenderer({antialias:true,alpha:false});renderer.setPixelRatio(Math.min(devicePixelRatio,2));
    renderer.outputColorSpace=THREE.SRGBColorSpace;el('asset-viewer').append(renderer.domElement);
    const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.minDistance=.01;controls.maxDistance=2;
    scene.add(new THREE.HemisphereLight(0xffffff,0x7a8b7d,2.6));const light=new THREE.DirectionalLight(0xffffff,2.2);light.position.set(.3,-.2,.6);scene.add(light);
    const extent=Math.max(...asset.extent_m),gridSize=Math.max(.15,Math.ceil(extent/.05)*.05+.10);
    const grid=new THREE.GridHelper(gridSize,Math.round(gridSize/.01),0xa2b9a9,0xcddbd1);grid.rotation.x=Math.PI/2;
    const major=new THREE.GridHelper(gridSize,Math.round(gridSize/.05),0x8da998,0x9ab7a5);major.rotation.x=Math.PI/2;
    const baseZ=-asset.extent_m[2]/2-.002;grid.position.z=major.position.z=baseZ;scene.add(grid,major,new THREE.AxesHelper(.03));
    viewer={scene,camera,renderer,controls,radius:extent,mesh:null,points:null,animation:null};
    resizeViewer();setView('iso');
    const [gltf,data]=await Promise.all([new GLTFLoader().loadAsync(assetURL(asset.mesh,asset)),json(assetURL(asset.point_cloud,asset))]);
    if(generation!==viewerGeneration||id!==recording){
      gltf.scene.traverse(object=>{object.geometry?.dispose();const materials=Array.isArray(object.material)?object.material:[object.material];for(const material of materials)material?.dispose();});return;}
    viewer.mesh=gltf.scene;scene.add(gltf.scene);
    gltf.scene.traverse(object=>{if(object.isMesh){object.material.transparent=true;object.material.opacity=.72;object.material.side=THREE.DoubleSide;}});
    const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(data.positions,3));
    viewer.points=new THREE.Points(geometry,new THREE.PointsMaterial({color:0x249fbe,size:.0010,sizeAttenuation:true}));scene.add(viewer.points);
    el('viewer-mesh').checked=el('viewer-points').checked=true;el('viewer-wire').checked=false;
    el('asset-viewer-error').textContent='';
    const animate=()=>{if(!viewer||generation!==viewerGeneration)return;controls.update();renderer.render(scene,camera);viewer.animation=requestAnimationFrame(animate);};animate();
    window.dispatchEvent(new CustomEvent('showonce-asset-loaded',{detail:{id:asset.id}}));
  }catch(error){if(generation===viewerGeneration&&id===recording)el('asset-viewer-error').textContent='3D 미리보기를 열지 못했습니다: '+error.message;}}

function resizeViewer(){if(!viewer)return;const r=el('asset-viewer').getBoundingClientRect();viewer.renderer.setSize(r.width,r.height);viewer.camera.aspect=r.width/r.height;viewer.camera.updateProjectionMatrix();}
function setView(mode){if(!viewer)return;const {camera,controls}=viewer,d=Math.max(viewer.radius*2.5,.13);
  const positions={iso:[d,-d,d*.8],front:[0,-d,0.0001],side:[d,0,.0001],top:[.0001,-.0001,d],fit:[d,-d,d*.8]};
  camera.position.set(...positions[mode]);controls.target.set(0,0,0);camera.lookAt(0,0,0);controls.update();controls.saveState();}
for(const button of document.querySelectorAll('[data-asset-view]'))button.onclick=()=>setView(button.dataset.assetView);
for(const id of ['viewer-mesh','viewer-points','viewer-wire'])el(id).onchange=()=>{
  if(!viewer)return;if(viewer.mesh){viewer.mesh.visible=el('viewer-mesh').checked;viewer.mesh.traverse(object=>{if(object.isMesh)object.material.wireframe=el('viewer-wire').checked;});}
  if(viewer.points)viewer.points.visible=el('viewer-points').checked;};
window.addEventListener('resize',resizeViewer);
setInterval(()=>{const id=new URLSearchParams(location.search).get('id');if(id&&id!==recording)reset(id);else poll();},1800);
const initial=new URLSearchParams(location.search).get('id');if(initial)reset(initial);
