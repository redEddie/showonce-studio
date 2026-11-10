/* ECharts base-frame XY: screen up=+X, screen left=+Y. No robot requests. */
(()=>{
 const $=id=>document.getElementById(id),element=$('live-xy-chart');
 if(!element||!window.echarts)return;
 const chart=echarts.init(element,null,{renderer:'svg'}),colors={cube:'#329365',bowl:'#d58b37',tcp:'#4d80ad'};
 const tracks={cube:[],bowl:[],policyCube:[],policyBowl:[],tcp:[]},dots={},seen={};
 let folder='',episode='',job=null,offset=0,pulling=false,policyCount=0,error='';
 const point=(value,stamp,source)=>Array.isArray(value)&&value.length===3&&value.every(Number.isFinite)?
  {value:[value[1]*1000,value[0]*1000],stamp,source}:null;
 function append(key,p){
  if(!p)return;const rows=tracks[key],last=rows.findLast(row=>row?.value?.every(Number.isFinite));
  if(last&&Math.hypot(p.value[0]-last.value[0],p.value[1]-last.value[1])<.5)return;
  if(last&&p.source==='camera'&&p.stamp-last.stamp>1.)rows.push({value:[null,null]});
  rows.push(p);if(rows.length>3000)rows.splice(0,rows.length-3000);
 }
 function extent(){
  const points=[{value:[0,0]},...Object.values(tracks).flat(),...Object.values(dots)].filter(p=>p?.value?.every(Number.isFinite));
  const ys=points.map(p=>p.value[0]),xs=points.map(p=>p.value[1]);
  let minX=Math.floor((Math.min(...xs)-25)/50)*50,maxX=Math.ceil((Math.max(...xs)+25)/50)*50;
  let minY=Math.floor((Math.min(...ys)-25)/50)*50,maxY=Math.ceil((Math.max(...ys)+25)/50)*50;
  const span=Math.max(300,maxX-minX,maxY-minY);
  minX=Math.floor(((minX+maxX)/2-span/2)/50)*50;maxX=minX+span;
  minY=Math.floor(((minY+maxY)/2-span/2)/50)*50;maxY=minY+span;
  return {minX,maxX,minY,maxY};
 }
 function render(){
  const bounds=extent(),width=element.clientWidth,height=element.clientHeight;
  const side=Math.max(100,Math.min(width-90,height-115));
  const axis={type:'value',interval:50,axisLabel:{fontSize:10,color:'#6e817a'},nameTextStyle:{fontSize:11,color:'#577064'},
   axisLine:{show:true,lineStyle:{color:'#cbd8d0'}},axisTick:{show:false},splitLine:{show:true,lineStyle:{color:'#d3ded7'}},
   minorTick:{show:true,splitNumber:5},minorSplitLine:{show:true,lineStyle:{color:'#edf2ee'}},nameLocation:'middle'};
  const series=[{id:'base',name:'베이스',type:'scatter',symbol:'rect',symbolSize:11,data:[{value:[0,0]}],
   itemStyle:{color:'#526b7a'},label:{show:true,formatter:'(0, 0)',position:'bottom',fontSize:10,color:'#526b7a'},
   markLine:{silent:true,symbol:['none','none'],label:{show:false},lineStyle:{color:'#9bb0a3',type:'solid'},data:[{xAxis:0},{yAxis:0}]}}];
  for(const [key,name,color,dashed,symbol,size]of [
   ['cube','큐브 관측',colors.cube,false,'circle',10],['bowl','그릇 관측',colors.bowl,false,'emptyCircle',19],
   ['policyCube','큐브 정책입력',colors.cube,true,'diamond',8],['policyBowl','그릇 정책입력',colors.bowl,true,'emptyCircle',11],
   ['tcp','TCP (피드백 FK)',colors.tcp,true,'circle',7]]){
   series.push({id:key+'-path',name,type:'line',showSymbol:false,connectNulls:false,data:tracks[key],
    lineStyle:{color,width:dashed?1.7:2,type:dashed?'dashed':'solid',opacity:.75},itemStyle:{color}});
   series.push({id:key+'-point',name,type:'scatter',symbol,symbolSize:size,data:dots[key]?[dots[key]]:[],itemStyle:{color,borderWidth:2},z:5});
  }
  chart.setOption({animation:false,aria:{enabled:true},legend:{top:2,left:'center',textStyle:{fontSize:10,color:'#5c7466'},itemGap:10,
    data:['큐브 관측','그릇 관측','큐브 정책입력','그릇 정책입력','TCP (피드백 FK)','베이스']},
   grid:{left:55+(Math.max(0,width-90-side))/2,top:60+(Math.max(0,height-115-side))/2,width:side,height:side},
   xAxis:{...axis,min:bounds.minY,max:bounds.maxY,inverse:true,name:'Y (mm) · + ←',nameGap:28},
   yAxis:{...axis,min:bounds.minX,max:bounds.maxX,name:'X (mm) · + ↑',nameGap:38},
   tooltip:{trigger:'item',formatter:p=>{const v=p.value;if(!Array.isArray(v))return '';return p.seriesName+'<br>X '+Number(v[1]).toFixed(1)+' mm · Y '+Number(v[0]).toFixed(1)+' mm';}},series},false);
  $('xy-status').textContent=error||(!episode?'카메라 관측 궤적 · 정책을 실행하면 정책 입력과 TCP 경로도 표시합니다.':
   (job?.alive?'실행 중':'이전 실행 기록')+' · '+(job?.mode||'')+' · 정책 입력 '+policyCount+'개 · 실선: 카메라 / 점선: 정책 입력·TCP');
 }
 window.addEventListener('showonce:perception',event=>{
  const packet=event.detail;
  if(folder&&packet.folder&&folder!==packet.folder){tracks.cube=[];tracks.bowl=[];delete dots.cube;delete dots.bowl;seen.cube=seen.bowl=null;}
  folder=packet.folder||folder;
  for(const [oid,key]of [['green_block','cube'],['bowl','bowl']]){
   const obj=packet.objects?.find(o=>o.id===oid);
   if(!packet.connected||!obj||obj.status==='missing'){delete dots[key];continue;}
   const stamp=packet.now_s-obj.observation_age_s,p=point(obj.position_base_m,stamp,'camera');dots[key]=p;
   if(Number.isFinite(stamp)&&stamp!==seen[key]){append(key,p);seen[key]=stamp;}
  }
  render();
 });
 function clearPolicy(){for(const key of ['policyCube','policyBowl','tcp']){tracks[key]=[];delete dots[key];}offset=0;policyCount=0;error='';}
 async function pull(){
  if(pulling||!episode||document.hidden)return;pulling=true;const id=episode;
  try{
   const response=await fetch('/api/robot/trajectory?id='+encodeURIComponent(id)+'&after='+offset,{cache:'no-store',signal:AbortSignal.timeout(2500)});
   const value=await response.json();if(id!==episode)return;if(!response.ok)throw Error(value.error||'실행 궤적을 읽지 못했습니다.');
   for(const sample of value.samples){
    if(sample.input_kind!=='live')continue;
    for(const [key,field]of [['policyCube','block_pos_m'],['policyBowl','bowl_pos_m'],['tcp','tcp_pos_m']]){
     const p=point(sample[field],sample.updated_s,'policy');dots[key]=p;append(key,p);
    }
    policyCount++;
   }
   offset=value.next_offset;error='';render();
  }catch(reason){if(id===episode){error='실행 궤적: '+reason.message;render();}}finally{pulling=false;}
 }
 window.addEventListener('showonce:robot-state',event=>{
  const current=event.detail.jobs.find(j=>j.policy_id===$('policy').value&&['run','shadow'].includes(j.mode));
  if(!current)return;
  if(current.id!==episode){episode=current.id;clearPolicy();}
  job=current;pull();render();
 });
 $('policy').addEventListener('change',()=>{episode='';job=null;clearPolicy();render();});
 $('xy-clear').onclick=()=>{for(const key of Object.keys(tracks))tracks[key]=[];render();};
 new ResizeObserver(()=>{chart.resize();render();}).observe(element);
 render();
})();
