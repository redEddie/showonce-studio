/* Observed relation graph + confirmed stages on the source video clock. */
const el=id=>document.getElementById(id),video=el('video');
let recording='',epoch=0,graph=null,stamp=null,lastIndex=-1,lastSignature='',selectedRelation='grasped',mode='video';
const names={hand:'손',green_block:'큐브',bowl:'그릇',table:'테이블'};
const colors={true:'#308566',false:'#dce3e8',unknown:'#d5a54b'},stateNames={true:'성립 추정',false:'비성립',unknown:'불확실'};
const chart=echarts.init(el('relation-graph'),null,{renderer:'canvas'});
let timeline=null;
const steps={
  approach:{label:'접근',goal:'손이 큐브에 닿음',ids:['contact'],expected:['true']},
  grasp:{label:'파지, 들기',goal:'큐브를 잡아 테이블에서 들어 올림',ids:['grasped','above_table'],expected:['true','true']},
  transfer:{label:'운반',goal:'큐브가 그릇 위에 도달함',ids:['aligned'],expected:['true']},
  release:{label:'개방, 내려놓기',goal:'큐브가 그릇 안에 남고 손이 놓음',ids:['inside','grasped'],expected:['true','false']},
  retreat:{label:'후퇴',goal:'손이 멀어지고 큐브가 안착함',ids:['near','stable_in'],expected:['false','true']}
};
function stageAt(time){return graph?.stages.find((stage,i)=>time>=stage.start_s&&(time<stage.end_s||i===graph.stages.length-1));}
async function json(url,body){const response=await fetch(url,{cache:'no-store',method:body?'POST':'GET',headers:body?{'Content-Type':'application/json'}:{},body:body?JSON.stringify(body):undefined});
  const data=await response.json();if(!response.ok)throw Error(data.error||'관계 분석을 불러오지 못했습니다.');return data;}
function nearest(time){if(!graph?.frames?.length)return -1;let lo=0,hi=graph.frames.length-1;
  while(lo<hi){const mid=(lo+hi)>>1;if(graph.frames[mid].time_s<time)lo=mid+1;else hi=mid;}
  return lo>0&&time-graph.frames[lo-1].time_s<graph.frames[lo].time_s-time?lo-1:lo;}
function seek(time){video.pause();video.currentTime=time;sync(time);}
function reset(id){if(id===recording)return;recording=id;epoch++;graph=null;stamp=null;mode='video';lastIndex=-1;lastSignature='';chart.clear();timeline?.clear();
  el('relation-events').replaceChildren();el('relation-stages').replaceChildren();el('relation-details').replaceChildren();poll();}
window.addEventListener('showonce-recording',event=>reset(event.detail.id));
async function poll(){if(!recording)return;const serial=epoch,id=recording;
  try{const value=await json('/api/relations/state?id='+encodeURIComponent(id));if(serial!==epoch)return;
    const busy=!['idle','error','complete'].includes(value.job?.status||'idle');
    el('relation-start').disabled=busy||value.graph_ready||!value.prerequisites_ready;
    el('relation-start').textContent=value.graph_ready?'관계 추출 완료':busy?'관계 추출 중…':'관계 추출';
    el('relation-status').textContent=value.job?.error||
      (busy?(value.hand_job?.status==='tracking_hands'?`사람 손 추적 · ${value.hand_job.frame}/${value.hand_job.total}`:value.job?.message||'관계 분석 중'):
       value.graph_ready?'손·객체의 관측 거리와 상대 운동으로 추정한 관계입니다. 그래프와 시간축은 위 영상에 맞춰집니다.':'FoundationPose 추적 후 관계를 추출하세요.');
    if(value.graph_ready){const next=value.created_s+':'+value.confirmed_stage_revision;
      if(next!==stamp){const data=await json('/api/relations/graph?id='+encodeURIComponent(id));if(serial!==epoch)return;
        graph=data;stamp=next;lastIndex=-1;lastSignature='';buildTimeline();buildEvents();buildStages();sync(video.currentTime);}
    }
  }catch(error){if(serial===epoch)el('relation-status').textContent=error.message;}}
el('relation-start').onclick=async()=>{if(!recording)return;el('relation-start').disabled=true;
  try{await json('/api/relations/start',{recording_id:recording});await poll();}catch(error){el('relation-status').textContent=error.message;el('relation-start').disabled=false;}};
el('relation-download').onclick=()=>{if(!graph)return;const link=document.createElement('a'),url=URL.createObjectURL(new Blob([JSON.stringify(graph,null,2)],{type:'application/json'}));
  link.href=url;link.download='scene-graph-'+recording+'.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};

function renderGraph(links,physical=false){
  const nodes=[['hand',0,0,physical?'#c7cdd1':'#8a76b5'],['green_block',280,0,'#4a9d73'],['bowl',280,200,'#ce934f'],['table',0,200,'#647d91']];
  chart.setOption({animation:false,tooltip:{show:false},series:[{type:'graph',layout:'none',roam:false,symbolSize:53,
    label:{show:true,color:'#fff',fontSize:12},edgeSymbol:['none','arrow'],edgeSymbolSize:8,
    data:nodes.map(([id,x,y,color])=>({id,name:names[id],x,y,itemStyle:{color},symbol:id==='table'?'roundRect':'circle'})),
    links:links.map(link=>({...link,lineStyle:{color:link.goal?'#9daea4':physical?'#5486a4':colors.true,type:link.goal?'dashed':'solid',width:link.goal?1.5:2.5},
      label:{show:true,formatter:p=>p.data.value,fontSize:11,color:'#496454',backgroundColor:'#f7faf8',padding:3}})),
    left:32,right:32,top:32,bottom:32}]});
  el('relation-graph').dataset.edges=JSON.stringify(links.map(({source,target,value,goal})=>({source,target,label:value,goal:!!goal})));
}
function drawGraph(frame){const state=Object.fromEntries(frame.relations.map(r=>[r.id,r.state])),links=[];
  function link(source,target,value,id,goal=false){links.push({source,target,value,relationIds:[id],goal});}
  if(state.grasped==='true')link('hand','green_block','잡고 있음','grasped');
  else if(state.contact==='true')link('hand','green_block','닿음','contact');
  else if(state.near==='true')link('hand','green_block','접근 중','near');
  if(state.stable_in==='true')link('green_block','bowl','안착','stable_in');
  else if(state.inside==='true')link('green_block','bowl','그릇 안','inside');
  else if(state.aligned==='true')link('green_block','bowl','그릇에 정렬','aligned');
  else if(state.grasped==='true')link('green_block','bowl','이동 목표','aligned',true);
  if(state.on_table==='true')link('green_block','table','놓여 있음','on_table');
  if(state.bowl_on_table==='true')link('bowl','table','놓여 있음','bowl_on_table');
  renderGraph(links);
}

function abstractState(frame,time){const states=Object.fromEntries(frame.relations.map(r=>[r.id,r.state])),stage=stageAt(time),step=steps[stage?.code];
  const index=graph.stages.indexOf(stage);
  el('relation-current-stage').textContent=step?`${index+1}. ${step.label}`:'작업 상태';
  const text=states.stable_in==='true'?'큐브가 그릇 안에 안착한 상태입니다.':states.inside==='true'&&states.grasped==='false'?'큐브가 그릇 안에 놓였습니다.':
    states.grasped==='true'&&states.above_table==='true'?'손이 큐브를 들고 이동하고 있습니다.':states.grasped==='true'?'손이 큐브를 잡고 있습니다.':
    states.contact==='true'?'손이 큐브에 닿아 있습니다.':states.near==='true'?'손이 큐브에 접근하고 있습니다.':states.on_table==='true'?'큐브가 테이블 위에 놓여 있습니다.':'관측을 확인하고 있습니다.';
  el('relation-current-state').textContent=text;
  el('relation-next-label').textContent=stage?.code==='retreat'?'완료 조건':'다음 단계로 넘어가는 조건';
  el('relation-next-condition').textContent=step?.goal||'확정 스테이지를 확인합니다.';
  const met=step?.ids.every((id,i)=>states[id]===step.expected[i]),unknown=step?.ids.some(id=>states[id]==='unknown');
  el('relation-transition-state').textContent=met?'조건이 관측됐습니다.':unknown?'관측을 더 확인하고 있습니다.':'조건을 기다리고 있습니다.';
  el('relation-transition-state').classList.toggle('met',!!met);
  for(const button of el('relation-stages').children){const i=graph.stages.findIndex(s=>s.code===button.dataset.code);button.classList.toggle('active',i===index);button.classList.toggle('done',i<index);}
}

function buildTimeline(){if(!graph||!el('relation-advanced').open)return;
  if(!timeline){timeline=echarts.init(el('relation-timeline'),null,{renderer:'canvas'});timeline.on('click',params=>{if(params.value)seek(params.value[1]);});}
  const specs=graph.relation_specs,data=[];
  (graph.stages||[]).forEach((stage,i)=>data.push([0,stage.start_s,stage.end_s,3,stage.label,i]));
  specs.forEach((spec,row)=>{let begin=0,state=graph.frames[0].relations.find(r=>r.id===spec.id).state;
    for(let i=1;i<=graph.frames.length;i++){
      const next=i<graph.frames.length?graph.frames[i].relations.find(r=>r.id===spec.id).state:null;
      if(next!==state){data.push([row+1,graph.frames[begin].time_s,i<graph.frames.length?graph.frames[i].time_s:graph.duration_s,
        state==='true'?1:state==='false'?0:2,spec.label]);begin=i;state=next;}
    }});
  const labels=['확정 스테이지',...specs.map(r=>names[r.source]+' → '+names[r.target]+' · '+r.label)];labels[0]='확정 스테이지';
  timeline.setOption({animation:false,grid:{left:215,right:28,top:24,bottom:32},
    tooltip:{formatter:p=>`${p.value[4]}<br>${p.value[1].toFixed(2)}–${p.value[2].toFixed(2)}초 · ${p.value[3]===3?'사용자 확정':stateNames[p.value[3]===1?'true':p.value[3]===0?'false':'unknown']}`},
    xAxis:{type:'value',min:0,max:graph.duration_s,axisLabel:{formatter:v=>v.toFixed(1)+'s'},splitLine:{lineStyle:{color:'#edf1f3'}}},
    yAxis:{type:'category',data:labels,inverse:true,axisTick:{show:false},axisLine:{show:false},axisLabel:{fontSize:11,color:'#4a6274'}},
    series:[{type:'custom',data,encode:{x:[1,2],y:0},renderItem:(params,api)=>{
      const row=api.value(0),left=api.coord([api.value(1),row]),right=api.coord([api.value(2),row]),height=api.size([0,1])[1]*.64;
      const rect=echarts.graphic.clipRectByRect({x:left[0],y:left[1]-height/2,width:Math.max(1,right[0]-left[0]),height},params.coordSys);
      if(!rect)return;
      if(api.value(3)===3){const color=['#7d97a8','#547b94','#3e8f77','#b18b55','#8a7b9c'][api.value(5)%5];
        return {type:'group',children:[{type:'rect',shape:rect,style:{fill:color,stroke:'#fff',lineWidth:1}},
          {type:'text',style:{text:String(api.value(4)).split('·').at(-1).trim(),x:rect.x+rect.width/2,y:rect.y+rect.height/2,
            fill:'#fff',font:'11px sans-serif',align:'center',verticalAlign:'middle'}}]};}
      return {type:'rect',shape:rect,style:{fill:['#e3e9ed','#4a9d79','#e4bc70'][api.value(3)]}};}}]});
}
chart.on('click',params=>{if(mode==='video'&&params.dataType==='edge'&&params.data.relationIds?.length){selectedRelation=params.data.relationIds[0];el('relation-advanced').open=true;lastIndex=-1;sync(video.currentTime);}});

function buildStages(){const box=el('relation-stages');box.replaceChildren();
  for(const [i,stage]of(graph.stages||[]).entries()){const button=document.createElement('button');button.className='relation-stage';button.textContent=(i+1)+'. '+(steps[stage.code]?.label||stage.label);
    button.dataset.code=stage.code;button.title=`${stage.start_s.toFixed(2)}–${stage.end_s.toFixed(2)}초`;button.onclick=()=>{if(mode==='video')seek(stage.start_s);};box.append(button);}
  el('relation-revision').textContent=graph.confirmed_stage_revision?'사용자 확정 · v'+graph.confirmed_stage_revision:'확정 스테이지 없음';
  const comparison=el('relation-comparison');comparison.replaceChildren();
  for(const item of graph.stage_comparison||[]){const line=document.createElement('div');line.className='relation-compare-row';
    line.textContent=item.relation_event+' · 확정 경계 '+(item.confirmed_time_s==null?'—':item.confirmed_time_s.toFixed(2)+'s')+' → '+
      (item.event_time_s==null?'관계 사건 미검출':item.event_time_s.toFixed(2)+'s ('+(item.delta_s>=0?'+':'')+item.delta_s.toFixed(2)+'s)');comparison.append(line);}}

function buildEvents(){const box=el('relation-events');box.replaceChildren();
  for(const event of graph.events.filter(e=>['birth','death','uncertain'].includes(e.kind))){const button=document.createElement('button');
    button.className='relation-event '+event.kind;button.dataset.time=event.time_s;button.textContent=event.time_s.toFixed(2)+'s · '+names[event.source]+' → '+names[event.target]+' · '+event.label+' '+(event.to==='true'?'시작':event.to==='false'?'종료':'불확실');
    button.onclick=()=>{selectedRelation=event.relation_id;seek(event.time_s);};box.append(button);}}

function details(frame){const box=el('relation-details');box.replaceChildren();
  for(const relation of frame.relations){const button=document.createElement('button');button.className='relation-row '+relation.state+(relation.id===selectedRelation?' selected':'');
    const name=document.createElement('span'),value=document.createElement('b');name.textContent=names[relation.source]+' → '+names[relation.target]+' · '+relation.label;value.textContent=stateNames[relation.state];
    button.append(name,value);button.onclick=()=>{selectedRelation=relation.id;lastIndex=-1;sync(video.currentTime);};box.append(button);}
  const selected=frame.relations.find(r=>r.id===selectedRelation),e=selected?.evidence||{},parts=[];
  if(e.surface_distance_m!=null)parts.push('손–'+names[selected.target]+' 표면 거리 '+(e.surface_distance_m*1000).toFixed(1)+'mm');
  if(e.relative_speed_m_s!=null)parts.push('상대속도 '+(e.relative_speed_m_s*100).toFixed(1)+'cm/s');
  if(selected?.id==='on_table'&&e.cube_observed_lower_table_m!=null)parts.push('관측된 큐브 하단 표면–테이블 '+(e.cube_observed_lower_table_m*1000).toFixed(1)+'mm');
  else if(selected?.id==='bowl_on_table'&&e.bowl_observed_lower_table_m!=null)parts.push('관측된 그릇 하단 표면–테이블 '+(e.bowl_observed_lower_table_m*1000).toFixed(1)+'mm');
  else if(selected?.id==='above_table'&&e.cube_bottom_table_clearance_m!=null)parts.push('추정 큐브 하단–테이블 '+(e.cube_bottom_table_clearance_m*1000).toFixed(1)+'mm');
  if(['aligned','inside','stable_in'].includes(selected?.id)&&e.cube_footprint_in_cavity_fraction!=null)parts.push('그릇 내부 단면에 속한 투영 꼭짓점 '+Math.round(e.cube_footprint_in_cavity_fraction*100)+'%');
  el('relation-evidence').textContent=(selected?.label||'관계')+' · '+(parts.join(' / ')||'유효한 깊이·손 관측이 부족합니다.')+' · 순간 변화에는 최소 100ms의 지속 조건을 적용합니다.';
}

function handOverlay(frame){const canvas=el('relation-overlay'),ctx=canvas.getContext('2d');canvas.width=video.videoWidth||848;canvas.height=video.videoHeight||480;
  if(!el('relation-show-hands').checked)return;
  for(const hand of frame?.hands||[]){ctx.strokeStyle=hand.track_id===frame.selected_hand_id?'#9769dd':'#9b92ae';ctx.fillStyle='#9769dd18';ctx.lineWidth=2;
    for(const contour of hand.contours){if(contour.length<3)continue;ctx.beginPath();ctx.moveTo(...contour[0]);for(const p of contour.slice(1))ctx.lineTo(...p);ctx.closePath();ctx.fill();ctx.stroke();}
    if(hand.bbox){ctx.fillStyle='#61438c';ctx.font='bold 12px system-ui';ctx.fillText('손 '+hand.track_id+(hand.depth_points<30?' · 깊이 불확실':''),hand.bbox[0],Math.max(15,hand.bbox[1]-6));}}
}
function sync(time=video.currentTime){if(!graph||mode!=='video')return;const index=nearest(time);if(index<0)return;const frame=graph.frames[index];
  el('relation-time').textContent=time.toFixed(2)+'s · 프레임 '+frame.index;
  if(index!==lastIndex){const signature=frame.relations.map(r=>r.state).join(':');if(signature!==lastSignature){drawGraph(frame);lastSignature=signature;}
    if(el('relation-advanced').open)details(frame);handOverlay(frame);abstractState(frame,time);lastIndex=index;
    for(const button of el('relation-events').children)button.classList.toggle('current',Math.abs(Number(button.dataset.time)-frame.time_s)<.08);
  }
  if(timeline&&el('relation-advanced').open){const x=timeline.convertToPixel({xAxisIndex:0},time);if(Number.isFinite(x))timeline.setOption({graphic:[{id:'playhead',type:'line',silent:true,shape:{x1:x,x2:x,y1:18,y2:timeline.getHeight()-30},style:{stroke:'#24475e',lineWidth:1.5}}]});}
}
el('relation-advanced').addEventListener('toggle',()=>{if(el('relation-advanced').open){buildTimeline();timeline?.resize();lastIndex=-1;sync();}});
window.addEventListener('showonce-scene-mode',event=>{
  if(event.detail.recording_id!==recording)return;mode=event.detail.mode;lastIndex=-1;lastSignature='';
  el('relation-stages').classList.toggle('physics',mode==='physics');el('relation-advanced').hidden=mode==='physics';
  el('relation-source').textContent=mode==='physics'?'물리 시뮬레이션':'영상 관측';
  el('relation-summary-note').textContent=mode==='physics'?'MuJoCo 접촉 결과 · 사람 손은 이 물리 장면에 없습니다.':'실선: 관측한 관계 추정 · 점선: 이동 목표';
  if(mode==='video')sync();else{
    chart.clear();el('relation-current-stage').textContent='물리 장면을 준비합니다.';
    for(const button of el('relation-stages').children){button.classList.remove('active','done');}
  }
});
window.addEventListener('showonce-scene-physics-frame',event=>{
  const value=event.detail;if(value.recording_id!==recording||mode!=='physics')return;
  const links=(value.contact_pairs||[]).map(pair=>{const a=pair.a==='table'?pair.b:pair.a,b=pair.a==='table'?'table':pair.b;
    return {source:a,target:b,value:b==='table'?'테이블 접촉':'그릇 접촉',relationIds:[]};});
  renderGraph(links,true);el('relation-source').textContent='물리 시뮬레이션';
  el('relation-time').textContent=value.time_s.toFixed(2)+'s';
  el('relation-current-stage').textContent=value.scenario==='drop'?'큐브 낙하':'초기 배치';
  el('relation-current-state').textContent=links.some(link=>[link.source,link.target].includes('bowl')&&[link.source,link.target].includes('green_block'))?'큐브와 그릇이 접촉하고 있습니다.':value.running?'물체의 움직임과 접촉을 확인합니다.':'물리 시뮬레이션이 멈춰 있습니다.';
  el('relation-next-label').textContent='현재 확인하는 관계';el('relation-next-condition').textContent='큐브·그릇·테이블 사이의 접촉';
  el('relation-transition-state').classList.remove('met');el('relation-transition-state').textContent='실제 MuJoCo 상태에서 갱신합니다.';
});
el('relation-show-hands').onchange=()=>{lastIndex=-1;sync();};
for(const event of ['loadedmetadata','timeupdate','seeked','seeking','pause'])video.addEventListener(event,()=>sync());
if('requestVideoFrameCallback'in video){const tick=(now,meta)=>{sync(meta.mediaTime);video.requestVideoFrameCallback(tick);};video.requestVideoFrameCallback(tick);}
new ResizeObserver(()=>{chart.resize();if(el('relation-advanced').open)timeline?.resize();lastIndex=-1;sync();}).observe(el('relations-panel'));
window.addEventListener('beforeunload',()=>{chart.dispose();timeline?.dispose();});
setInterval(()=>{if(!document.hidden)poll();},2200);
