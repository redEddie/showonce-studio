/* Robot actions are explicit. Opening this page only reads stored process metadata. */
(()=>{
 const el=id=>document.getElementById(id);let jobs=[],selected='',interruptTarget='',busy=false,reading=false,generation=0;
 const names={home:'시작 자세 A로 이동',read:'상태 조회',enable:'Enable',disable:'Disable',stop:'SDK Stop',reset:'Reset',mode:'Mode',validate:'추론 검증',mock:'정책 mock',replay:'영상 입력 검사',shadow:'정책 shadow',run:'실제 정책 실행',manual:'수동 TCP 연결',jog:'TCP 이동',rotate:'TCP 90° 회전',gripper:'그리퍼 조절',unwind:'팔 풀기'};
 const phases={homing:'시작 자세 A로 이동 중',validating:'정책 검증',connecting:'연결 중',preflight:'시작 조건 확인',running:'제어 중',braking:'감속 종료 중',finished:'종료',ready:'수동 조작 준비',moving:'수동 이동 중',planning:'방향·경로 검사 중'};
 const states={running:'실행 중',stopping:'중단 요청됨',cancelled:'중단됨',complete:'완료',error:'오류'};
 const execution=()=>document.querySelector('input[name="execution"]:checked').value;
 async function request(path,body){const response=await fetch(path,{cache:'no-store',method:body?'POST':'GET',headers:body?{'Content-Type':'application/json'}:{},body:body?JSON.stringify(body):undefined});const value=await response.json();if(!response.ok)throw Error(value.error||'실행 요청을 처리하지 못했습니다.');return value;}
 function updateChoice(){const direct=execution()==='direct';el('execution-choice').classList.toggle('execution-direct',direct);el('execution-note').textContent=direct?'버튼을 누르면 별도 프로세스로 바로 실행합니다.':'명령어를 표시합니다. 프로세스를 시작하지 않습니다.';
  for(const button of document.querySelectorAll('[data-sdk]'))button.textContent=names[button.dataset.sdk]+(direct?' 실행':' 명령 보기');
  for(const [id,label]of [['validate','검증'],['mock','모의 실행'],['replay','영상 입력 검사']])el(id).textContent=label+(direct?'':' 명령 보기');
  el('sdk-mode-apply').textContent=direct?'Mode 실행':'Mode 명령 보기';const mode=el('live-mode').value,seconds=Number(el('seconds').value),label={run:'실제 로봇 실행 (run)',shadow:'관측·추론 확인 (shadow)',mock:'모의 실행 (mock)'}[mode];el('policy-launch').textContent=direct?label+' · '+seconds+'초':mode+' 명령 생성 · 로봇 동작 없음';el('policy-launch').classList.toggle('real-motion',direct&&mode==='run');el('plan').hidden=!direct;
  el('prepare-real-run').disabled=direct&&mode==='run';el('prepare-real-run').textContent=direct&&mode==='run'?'실제 실행 방식 선택됨':'실제 로봇 실행 준비';el('run-help').textContent=direct&&mode==='run'?'아래 실제 로봇 실행 (run)을 누르면 시작 조건 확인 후 팔·그리퍼를 제어합니다.':direct&&mode==='shadow'?'현재 shadow: 실측 입력으로 정책을 계산하지만 팔 명령을 보내지 않습니다.':direct?'현재 mock: 가짜 로봇으로 계산하며 실제 팔은 움직이지 않습니다.':'현재 명령어 보기: 명령만 만들고 팔은 움직이지 않습니다.';el('manual-connect').textContent=direct?'수동 연결 · 현재값 읽기':'현재 상태 명령 보기';el('manual-gripper-apply').textContent=direct?'그리퍼 적용':'그리퍼 명령 보기';el('manual-unwind').textContent=direct?'팔 풀기 실행':'팔 풀기 명령 보기';el('manual-unwind-plan').textContent=direct?'경로만 검사':'경로 검사 명령 보기';if(!direct)releaseJog();refreshManual();}
 for(const input of document.querySelectorAll('input[name="execution"]'))input.addEventListener('change',updateChoice);
 function summary(job){const report=job.report;if(job.kind!=='sdk'||!report?.state)return;const state=report.state,arm=state.get_arm_status?.msg,grip=state.get_gripper_status?.msg,power=state.get_joints_enable_status_list||[];
  el('sdk-state').textContent=`${new Date(report.completed_s*1000).toLocaleString()} 기록 · ${report.feedback_fresh===true?'명령 후 관측':'최신 팔 피드백 미확인'}\n팔 모터 ${power.length===6&&power.every(v=>v!=null)?power.filter(v=>v===true).length+'/6 활성화':'미확인'} · 그리퍼 ${grip?((grip.status_code&64)?'활성':'비활성'):'미수신'}\n제어 모드 ${arm?.ctrl_mode??'미수신'} · 오류 ${arm?.err_code!=null?'0x'+arm.err_code.toString(16).padStart(4,'0'):'미수신'}${grip?.value!=null?' · 폭 '+(grip.value*1000).toFixed(1)+'mm':''}`;}
 async function showJob(job){if(!job)return;selected=job.id;el('task-title').textContent=(job.policy_name?job.policy_name+' · ':'')+names[job.mode]+' · '+states[job.status];
  const phase=phases[job.progress?.phase]||'',remaining=job.remaining_s!=null&&job.progress?.phase==='running'?` · 제어 남은 시간 ${job.remaining_s.toFixed(1)}초`:'',limit=job.kind==='policy'&&['run','shadow','mock'].includes(job.mode)?` · 제어 시간 ${job.seconds}초`:'';
  el('task-meta').textContent=`${phase?phase+' · ':''}경과 ${job.elapsed_s.toFixed(1)}초${limit}${remaining} · PID ${job.pid} · ${job.channel}`;
  const controllable=j=>j.kind==='policy'||['home','manual','jog','rotate','unwind'].includes(j.mode);const activePolicy=controllable(job)&&['running','stopping'].includes(job.status)?job:jobs.find(j=>controllable(j)&&['running','stopping'].includes(j.status));interruptTarget=activePolicy?.id||'';const stop=el('task-interrupt');stop.hidden=!activePolicy;stop.disabled=activePolicy?.status==='stopping';stop.textContent=activePolicy?.status==='stopping'?'중단 처리 중…':activePolicy?.mode==='home'?'자세 A 복귀 중단':activePolicy?.kind==='sdk'?'수동 TCP 중단':'정책 중단 (Ctrl+C)';const localStop=el('live-interrupt');localStop.hidden=stop.hidden;localStop.disabled=stop.disabled;localStop.textContent=stop.textContent;el('home-interrupt').hidden=activePolicy?.mode!=='home';el('home-interrupt').disabled=stop.disabled;
  const metrics=el('task-metrics');metrics.replaceChildren();if(job.report){for(const [label,key,unit]of [['참조 속도','maximum_reference_velocity_rad_s','rad/s'],['참조 가속도','maximum_reference_acceleration_rad_s2','rad/s²'],['참조 jerk','maximum_reference_jerk_rad_s3','rad/s³']]){if(job.report[key]!=null){const cell=document.createElement('div');cell.className='fact';cell.textContent=label;const value=document.createElement('strong');value.textContent=job.report[key].toFixed(2)+' '+unit;cell.append(value);metrics.append(cell);}}}
  if(job.mode==='validate'&&job.report?.parity){const cell=document.createElement('div');cell.className='fact';cell.textContent='원본 대비 행동 오차';const value=document.createElement('strong');value.textContent=job.report.parity.max_action_abs_error.toExponential(2);cell.append(value);metrics.append(cell);}el('task-error').textContent=job.report?.error||job.error||'';el('task-result').textContent=job.report?JSON.stringify(job.report,null,2):'';summary(job);
  try{const log=await request('/api/robot/logs?id='+job.id);if(selected===job.id)el('task-console').textContent=log.text;}catch(error){el('task-error').textContent=error.message;}
 }
 async function poll(){if(reading)return;reading=true;const stamp=generation;try{const value=await request('/api/robot/state');if(stamp!==generation)return;jobs=value.jobs;const recentRun=jobs.find(j=>j.policy_id===el('policy').value&&['run','shadow'].includes(j.mode));el('run-result').textContent=recentRun?'최근 '+recentRun.mode+' · '+states[recentRun.status]+(recentRun.report?.error?' · '+recentRun.report.error:''):'';el('run-result').classList.toggle('error-text',recentRun?.status==='error');window.dispatchEvent(new CustomEvent('showonce:robot-state',{detail:value}));const calibration=el('calibration');if(value.calibration&&!calibration.value&&!calibration.dataset.edited)calibration.value=value.calibration;const select=el('sdk-mode');if(!select.options.length){for(const mode of value.modes){const option=document.createElement('option');option.value=mode.value;option.textContent=mode.value+' · '+mode.label+(mode.selectable?'':' (로봇 버튼/상태 표시)');option.disabled=!mode.selectable;select.append(option);}select.value='1';}
  el('task-count').textContent=value.active_jobs.length?value.active_jobs.length+'개 실행 중':'실행 중인 작업 없음';const list=el('task-list');list.replaceChildren();
  for(const job of jobs.slice(0,8)){const button=document.createElement('button');button.className='secondary';button.textContent=names[job.mode]+' · '+states[job.status];button.onclick=()=>{generation++;showJob(job);};list.append(button);}
  if(!selected||!jobs.some(j=>j.id===selected))selected=value.active_jobs[0]?.id||jobs[0]?.id||'';
  if(selected)await showJob(jobs.find(j=>j.id===selected));
  const home=jobs.find(j=>j.mode==='home');el('home-status').textContent=home?'최근 자세 A 복귀 · '+states[home.status]+(home.progress?.phase==='homing'?' · 시작 자세 A로 이동·그리퍼 개방 중':'')+(home.report?.error?' · '+home.report.error:'')+(home.status==='cancelled'?' · 새 목표 전송을 중단했습니다. 마지막 목표까지 움직임이 남을 수 있습니다.':''):'Enable 후 자세 A 복귀를 실행하면 시작 자세 A로 이동하며 그리퍼를 엽니다.';el('home-status').classList.toggle('error-text',home?.status==='error');for(const button of document.querySelectorAll('[data-sdk="home"]'))button.disabled=jobs.some(j=>j.alive&&(j.mode==='home'||(j.kind==='policy'&&['run','shadow'].includes(j.mode))));
  const latest=jobs.find(j=>j.kind==='sdk'&&j.report?.state);if(latest)summary(latest);refreshManual();
 }catch(error){el('task-error').textContent=error.message;}finally{reading=false;}}
 function configKey(){return JSON.stringify(['policy','seconds','live-mode','calibration','perception-dir','channel'].map(id=>el(id).value));}
 async function perform(body,override){if(busy)return;generation++;busy=true;const key=configKey();el('task-error').textContent='';try{const value=await request('/api/robot/perform',{...body,execution:override||execution(),channel:el('channel').value});
   if(value.execution==='command'){
    const prefix=body.kind==='sdk'?(body.action==='home'?'home-command':'sdk-command'):'robot-command',panel=el(prefix+'-panel');panel.hidden=false;panel.open=true;
    el(prefix+'-title').textContent=(value.policy_name?value.policy_name+' · ':'')+names[value.mode]+' · 생성된 명령';
    const ready=value.command_ready!==false&&key===configKey();el(prefix+'-copy').disabled=!ready;
    el(prefix).textContent=ready?(value.display_command||value.command):value.command_ready===false?'실시간 인식 폴더와 현재 보정이 연결되면 실행 명령을 생성할 수 있습니다.':'선택값이 변경되었습니다. 명령을 다시 생성하세요.';
    const meta=value.execution_metadata;el(prefix+'-meta').textContent=ready&&meta?`웹앱 실행용 관리 정보 · 복사 제외\n실행 위치: ${meta.working_directory}\n저장 경로: ${meta.output_directory}\n작업 토큰: ${meta.job_token}`:'';
    if(body.kind==='policy'){el('plan-status').textContent=ready?'아래 명령을 터미널에서 실행할 수 있습니다. 명령 생성만으로 프로세스가 시작되지는 않습니다.':value.live_inputs?.reasons.join(' ')||'현재 선택값으로 다시 생성하세요.';panel.scrollIntoView({block:'nearest',behavior:'smooth'});}
   }else{generation++;selected=value.job.id;if(value.job.mode==='manual')manualId=value.job.id;await showJob(value.job);await poll();}return value;
  }catch(error){el('task-error').textContent=error.message;if(body.kind==='policy')el('plan-status').textContent=error.message;}finally{busy=false;}}
 for(const button of document.querySelectorAll('[data-sdk]'))button.onclick=()=>perform({kind:'sdk',action:button.dataset.sdk,...(button.dataset.sdk==='home'?{seconds:60,calibration:el('calibration').value||null,policy_id:el('policy').value||null}:{})});
 el('sdk-mode-apply').onclick=()=>perform({kind:'sdk',action:'mode',mode_value:Number(el('sdk-mode').value)});
 function policyBody(mode){return {kind:'policy',policy_id:el('policy').value,mode,seconds:Number(el('seconds').value),recording_id:el('recording').value,calibration:el('calibration').value||null,perception_dir:el('perception-dir').value||null};}
 el('policy-launch').onclick=()=>perform(policyBody(el('live-mode').value));el('plan').onclick=()=>perform(policyBody(el('live-mode').value),'command');
 for(const mode of ['validate','mock','replay'])el(mode).onclick=()=>perform(policyBody(mode));
 el('task-interrupt').onclick=async()=>{if(!interruptTarget)return;try{const target=interruptTarget;await request('/api/robot/interrupt',{id:target});selected=target;generation++;await poll();}catch(error){el('task-error').textContent=error.message;}};
 for(const prefix of ['robot-command','sdk-command','home-command']){
  el(prefix+'-copy').onclick=async()=>{if(el(prefix+'-copy').disabled)return;try{await navigator.clipboard.writeText(el(prefix).textContent);}catch(error){el('task-error').textContent='복사 권한을 확인하거나 명령어를 직접 선택해 복사하세요.';}};
  el(prefix+'-close').onclick=()=>{const panel=el(prefix+'-panel');panel.open=false;};
 }
 el('live-interrupt').onclick=()=>el('task-interrupt').click();el('home-interrupt').onclick=()=>el('task-interrupt').click();
 el('prepare-real-run').onclick=()=>{el('live-mode').value='run';document.querySelector('input[name="execution"][value="direct"]').checked=true;el('live-mode').dispatchEvent(new Event('change'));el('policy-launch').scrollIntoView({block:'nearest',behavior:'smooth'});};
 for(const id of ['policy','seconds','live-mode','calibration','perception-dir','channel'])el(id).addEventListener('change',()=>{
  el('robot-command-copy').disabled=true;el('robot-command').textContent='선택값이 변경되었습니다. 명령 생성 버튼을 눌러 갱신하세요.';updateChoice();
 });
 el('seconds').addEventListener('input',updateChoice);

 let manualId='',heldJog=null,pinging=false;
 el('calibration').addEventListener('input',()=>{el('calibration').dataset.edited='true';});
 const operationId=()=>crypto.randomUUID().replaceAll('-','');
 function currentManual(){return jobs.find(j=>j.id===manualId&&j.mode==='manual'&&j.alive);}
 function refreshManual(){
  const direct=execution()==='direct',job=currentManual(),progress=job?.progress||{};
  const fresh=job&&Date.now()/1000-Math.min(progress.updated_s,progress.feedback_s??progress.updated_s)<.7&&['ready','moving'].includes(progress.phase);
  const ready=fresh&&progress.phase==='ready'&&!progress.active_operation;
  const status=el('manual-status');
  if(job){status.textContent=progress.operation_error||(!fresh?'피드백·방향 검사를 기다리는 중…':progress.phase==='moving'?'수동 이동 중 · 버튼을 놓으면 감속 종료합니다.':'수동 조작 준비 · 가능한 방향만 활성화했습니다.');status.classList.toggle('error-text',!!progress.operation_error);}
  else{status.textContent=direct?'수동 연결을 시작하면 현재값과 방향별 도달 가능 여부를 확인합니다.':'명령어 보기 · 방향 검사는 바로 실행의 수동 연결 후 표시됩니다.';status.classList.remove('error-text');}
  const value=progress.pose||jobs.find(j=>j.id===manualId)?.report?.pose;
  if(value){['x','y','z'].forEach((axis,i)=>el('manual-'+axis).textContent=value.xyz_mm[i].toFixed(1));['roll','pitch','yaw'].forEach((axis,i)=>el('manual-'+axis).textContent=value.rpy_deg[i].toFixed(1));}
  if(progress.gripper_width_mm!=null)el('manual-gripper-current').textContent='실측 '+progress.gripper_width_mm.toFixed(1)+'mm';
  for(const button of document.querySelectorAll('[data-jog],[data-rotate]')){
   const axis=button.dataset.jog||button.dataset.rotate,direction=Number(button.dataset.direction),key=axis+(direction>0?'+':'-'),check=progress.available?.[key];
   const holding=heldJog?.button===button;button.disabled=direct&&!(holding?fresh:ready&&check?.allowed);
   button.title=!direct?'복사할 명령어 표시':!fresh?'최신 로봇 피드백·방향 검사 대기':!ready&&!holding?'진행 중인 동작 종료 대기':check?.reason||'현재 자세에서 도달 가능';
   button.classList.toggle('holding',!!holding);
  }
  for(const id of ['manual-unwind','manual-unwind-plan']){el(id).disabled=direct&&(!ready||progress.unwind_position_fixed!==true);el(id).title=direct&&job&&progress.unwind_position_fixed!==true?'수동 연결을 종료하고 다시 연결하면 TCP 위치 고정 방식이 적용됩니다.':'TCP 위치 고정 · 방향 변경 허용';}
  const plan=progress.unwind_plan||jobs.find(j=>j.id===manualId)?.report?.unwind_plan;
  if(plan){el('manual-plan-status').textContent=plan.tcp_fixed?`TCP 위치 고정 · ${plan.segments}구간 · 계획 최대 편차 ${(plan.planned_max_tcp_error_m*1000).toFixed(2)}mm · 관절 여유 ${plan.goal_joint_margin_deg.toFixed(1)}°`:'이전 기록 · TCP가 이동하는 경로입니다. 새 위치 고정 경로를 계산하세요.';el('manual-plan-details').hidden=false;el('manual-plan-json').textContent=JSON.stringify(plan,null,2);}
  else{el('manual-plan-status').textContent='';el('manual-plan-details').hidden=true;el('manual-plan-json').textContent='';}
  el('manual-gripper-apply').disabled=direct&&!ready;el('manual-connect').disabled=!!job;el('manual-disconnect').hidden=!job;
  if(heldJog&&(!fresh||!direct))releaseJog();
 }
 async function manualRequest(body){if(!manualId)throw Error('수동 연결을 먼저 시작하세요.');return request('/api/robot/manual-input',{id:manualId,...body});}
 async function renewJog(held){if(held.pending||held.released)return;held.pending=true;try{await manualRequest({action:'jog',operation_id:held.id,axis:held.axis,direction:held.direction,speed_mm_s:held.speed});}catch(error){el('manual-status').textContent=error.message;releaseJog();}finally{held.pending=false;if(held.released)await sendRelease(held);}}
 async function sendRelease(held){if(held.sentRelease)return;held.sentRelease=true;try{await manualRequest({action:'release',operation_id:held.id});}catch(error){el('manual-status').textContent='입력 해제 확인 대기 · 입력 기한 만료로 감속합니다.';}}
 function releaseJog(){const held=heldJog;if(!held)return;heldJog=null;held.released=true;clearInterval(held.timer);held.button.classList.remove('holding');if(!held.pending)sendRelease(held);}
 function startJog(button){if(button.disabled||heldJog)return;const axis=button.dataset.jog,direction=Number(button.dataset.direction),speed=Number(el('manual-speed').value);
  if(!Number.isFinite(speed)||speed<1||speed>50){el('manual-status').textContent='속도는 1~50mm/s로 입력하세요.';return;}
  if(execution()==='command'){perform({kind:'sdk',action:'jog',axis,direction,speed_mm_s:speed,seconds:2,calibration:el('calibration').value||null});return;}
  const held={id:operationId(),button,axis,direction,speed,released:false,pending:false};heldJog=held;button.classList.add('holding');renewJog(held);held.timer=setInterval(()=>renewJog(held),150);
 }
 el('manual-connect').onclick=async()=>{if(execution()==='command')return perform({kind:'sdk',action:'read'});const value=await perform({kind:'sdk',action:'manual',seconds:600,calibration:el('calibration').value||null});if(value?.job){manualId=value.job.id;refreshManual();}};
 async function disconnectManual(){releaseJog();if(!manualId||!currentManual())return;try{await request('/api/robot/interrupt',{id:manualId});}catch(error){el('manual-status').textContent=error.message;}}
 el('manual-disconnect').onclick=disconnectManual;
 el('manual-panel').addEventListener('toggle',()=>{if(!el('manual-panel').open)disconnectManual();});
 for(const button of document.querySelectorAll('[data-jog]')){
  button.addEventListener('pointerdown',event=>{if(event.button!==0)return;event.preventDefault();button.setPointerCapture(event.pointerId);startJog(button);});
  button.addEventListener('pointerup',releaseJog);button.addEventListener('pointercancel',releaseJog);button.addEventListener('lostpointercapture',releaseJog);
  button.addEventListener('keydown',event=>{if([' ','Enter'].includes(event.key)&&!event.repeat){event.preventDefault();startJog(button);}});button.addEventListener('keyup',releaseJog);
 }
 document.addEventListener('pointerup',releaseJog);window.addEventListener('blur',releaseJog);document.addEventListener('visibilitychange',()=>{if(document.hidden)releaseJog();});
 for(const button of document.querySelectorAll('[data-rotate]'))button.onclick=async()=>{const body={action:'rotate',axis:button.dataset.rotate,direction:Number(button.dataset.direction)};if(execution()==='command')return perform({kind:'sdk',...body,seconds:30,calibration:el('calibration').value||null});try{await manualRequest({...body,operation_id:operationId()});}catch(error){el('manual-status').textContent=error.message;}};
 for(const [id,planOnly]of [['manual-unwind',false],['manual-unwind-plan',true]])el(id).onclick=async()=>{const body={action:'unwind',plan_only:planOnly,keep_tcp_position:true};if(execution()==='command')return perform({kind:'sdk',...body,seconds:60,calibration:el('calibration').value||null});try{await manualRequest({...body,operation_id:operationId()});}catch(error){el('manual-status').textContent=error.message;}};
 el('manual-gripper-apply').onclick=async()=>{const body={action:'gripper',width_mm:Number(el('manual-gripper-width').value),force:Number(el('manual-gripper-force').value)};if(execution()==='command')return perform({kind:'sdk',...body});try{await manualRequest({...body,operation_id:operationId()});}catch(error){el('manual-status').textContent=error.message;}};
 setInterval(async()=>{if(pinging||!manualId||!currentManual()||!el('manual-panel').open||document.hidden||execution()!=='direct')return;pinging=true;try{await manualRequest({action:'ping'});}catch(error){releaseJog();el('manual-status').textContent=error.message;}finally{pinging=false;}},300);
 updateChoice();poll();setInterval(()=>{if(!document.hidden)poll();},700);
})();
