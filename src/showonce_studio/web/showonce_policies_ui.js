const el=id=>document.getElementById(id);let policies=[],currentJob=null,records=[];
async function api(path,body){const response=await fetch(path,{cache:'no-store',method:body?'POST':'GET',headers:body?{'Content-Type':'application/json'}:{},body:body?JSON.stringify(body):undefined});
  const value=await response.json();if(!response.ok)throw Error(value.error||'정책 실행 준비 요청을 처리하지 못했습니다.');return value;}
function fact(parent,label,value){const item=document.createElement('div');item.className='fact';const name=document.createElement('span'),strong=document.createElement('strong');name.textContent=label;strong.textContent=value;item.append(name,strong);parent.append(item);}
function selected(){return policies.find(p=>p.id===el('policy').value);}
let liveInputs=[];
function rememberPolicy(){const policy=selected();if(!policy)return;localStorage.setItem('showonce.selectedPolicy',policy.id);const url=new URL(location.href);url.searchParams.set('policy',policy.id);history.replaceState(null,'',url);}
function showPolicy(){const policy=selected(),box=el('policy-facts');box.replaceChildren();el('policy-checks').replaceChildren();if(!policy){el('policy-summary').textContent='등록한 정책이 없습니다. 내보낸 정책 패키지를 먼저 등록하세요.';return;}
  const final=policies.find(p=>p.is_final);el('select-final-policy').hidden=!final||policy.is_final;
  el('policy-summary').textContent=(policy.is_final?'최종 24배 · ':policy.recommended?'선정 완료 모델 · ':'이전 모델 · ')+(policy.architecture||policy.actor||'기존 정책')+' · 원본 가중치와 LSTM 상태를 보존합니다.';
  const conditions=policy.training_conditions;
  const duration=conditions?.episode_duration_s;
  if(Number.isFinite(duration)&&policy.contract.arm_gripper_coordinator===false){el('seconds').max=Math.min(60,duration);if(Number(el('seconds').value)>duration)el('seconds').value=duration;}else el('seconds').max=60;
  if(conditions){const ranges=conditions.spawn_ranges_m?.block,mm=v=>Math.round(v*1000),range=v=>Array.isArray(v)?v.map(mm).join('~'):'—';
    el('policy-summary').textContent+=' 학습 관측 DR '+(typeof conditions.observation_randomization!=='boolean'?'정보 없음':conditions.observation_randomization?'켜짐':'꺼짐')+' · 큐브 스폰 X '+range(ranges?.x)+'mm / Y '+range(ranges?.y)+'mm.';
    if(conditions.parent_training_steps)el('policy-summary').textContent+=' 부모 '+(conditions.parent_training_steps/8388608).toLocaleString()+'배 + 추가 '+(conditions.checkpoint_step/8388608).toLocaleString()+'배의 가중치·정규화 통계를 함께 적용합니다.';
  }
  fact(box,'누적 학습량',Number(conditions?.cumulative_training_steps??policy.cumulative_training_steps??policy.training_step).toLocaleString()+' 전이');fact(box,'입력 → 출력',policy.contract.observation_size+' 관측 → 7 행동');
  if(policy.simulation_placement){const p=policy.simulation_placement;fact(box,'시뮬레이션 · 접근→안착',p.success+'/'+p.episodes+' · '+(100*p.success/p.episodes).toFixed(1)+'%');}
  fact(box,'제어기',policy.contract.arm_gripper_coordinator===false?'SHOW-108 · 팔/개폐 독립':'기존 순차 개폐');fact(box,'주기','정책 10Hz · 제어 200Hz');
  const statuses=[['추론',policy.validation],['QP 모의 실행',policy.latest_jobs?.mock?.status==='complete'?'passed':'pending'],['영상 입력',policy.latest_jobs?.replay?.status==='complete'?(policy.latest_jobs.replay.report?.asset_size_matches_training===false?'warning':'passed'):'pending'],['실기',policy.hardware_validated?'passed':'pending']];
  for(const [name,state]of statuses){const badge=document.createElement('span');badge.className='policy-check '+state;badge.textContent=name+' · '+({passed:'통과',failed:'실패',warning:'치수 확인',pending:name==='실기'?'미검증':'검사 전'}[state]||'검사 전');el('policy-checks').append(badge);}
  const changes=Object.entries(policy.contract.training_runtime_limit_differences||{}),note=el('policy-differences');note.hidden=!changes.length&&!policy.contract.deployment_overrides;
  const labels={max_joint_vel:'관절 속도',control_acceleration:'가속도',control_jerk:'jerk'};
  note.textContent=changes.length?'학습과 다른 실행 제한: '+changes.map(([k,v])=>labels[k]+' '+(v.training??'명시 없음')+' → '+v.runtime).join(', '):'팔 속도·가속도·jerk는 학습과 동일합니다. 실기는 TCP 하한을 테이블 +10mm로 적용하며, 그리퍼 force는 0.5입니다.';
  if(policy.latest_jobs?.replay?.report?.asset_size_matches_training===false){note.hidden=false;note.textContent+=' 기록 영상의 복원 메시 치수가 학습 큐브와 다릅니다. 영상 입력 형식 검사는 통과했으며, 실기 인식은 50mm 큐브 기준입니다.';}
  const details=el('policy-details');details.replaceChildren();const text=document.createElement('p');text.textContent='정책 ID: '+policy.id+' · '+(policy.runtime_profile_id||policy.contract.adapter_id);details.append(text);
  if(conditions){const info=document.createElement('p');info.textContent='학습 초기 배치 범위(m): '+JSON.stringify(conditions.spawn_ranges_m)+' · 최대 반경 '+conditions.spawn_max_radius_m+'m · 물체 간 최소 '+conditions.spawn_min_distance_m+'m. 범위 전체의 도달 가능성을 보장하지 않으며 실행 시 QP·관절·테이블 검사를 적용합니다.';details.append(info);}
  if(policy.source_release){const link=document.createElement('a');link.href=policy.source_release;link.target='_blank';link.rel='noopener';link.textContent='원본 체크포인트 · 평가 결과 · 비교 영상';details.append(link);}
  const hint=document.createElement('p');hint.textContent='검사 결과는 선택한 정책별로 저장됩니다. 모의 실행은 물리 시뮬레이션이나 실기 집기 성공 평가가 아닙니다.';details.append(hint);
  const latest=Object.values(policy.latest_jobs||{}).filter(j=>['validate','mock','replay'].includes(j.mode)).sort((a,b)=>b.started_s-a.started_s)[0];if(latest){currentJob=latest;showJob(latest);}
}
async function catalog(){const before=el('policy').value,urlId=new URLSearchParams(location.search).get('policy'),saved=localStorage.getItem('showonce.selectedPolicy');const value=await api('/api/deployment/catalog');policies=value.policies;liveInputs=value.live_inputs||[];el('policy').replaceChildren();
  for(const [label,items]of [['최종 배포 기준',policies.filter(p=>p.is_final)],['이전 학습 모델',policies.filter(p=>p.role==='trained'&&!p.is_final)],['기존 정책 · 연결 검사',policies.filter(p=>p.role!=='trained'&&!p.is_final)]]){if(!items.length)continue;const group=document.createElement('optgroup');group.label=label;for(const policy of items){const option=document.createElement('option');option.value=policy.id;option.textContent=(policy.recommended?'추천 · ':'')+policy.name;group.append(option);}el('policy').append(group);}
  const choice=[before,urlId,saved,policies.find(p=>p.recommended)?.id,policies[0]?.id].find(id=>policies.some(p=>p.id===id));if(choice)el('policy').value=choice;
  el('policy-count').textContent=policies.length+'개 등록';showPolicy();
  if(liveInputs.length&&el('perception-dir').dataset.edited!=='true'){el('perception-dir').value=liveInputs[0].perception_dir;if(!el('calibration').dataset.edited)el('calibration').value=liveInputs[0].calibration;}
  if(!el('live-input-status').dataset.liveMonitor)el('live-input-status').textContent=liveInputs.length?'실시간 인식 노드 감지됨 · 실행 시 현재 보정과 로봇 피드백을 다시 확인합니다.':'실시간 인식 노드 준비 상태를 확인하고 있습니다. 추론 검증과 모의 실행도 사용할 수 있습니다.';
}
el('policy').onchange=()=>{currentJob=null;rememberPolicy();el('job-status').textContent='선택한 정책의 검사를 시작하세요.';el('job-metrics').replaceChildren();el('job-json').textContent='';el('stop').hidden=true;showPolicy();};
el('select-final-policy').onclick=()=>{const final=policies.find(p=>p.is_final);if(final){el('policy').value=final.id;el('policy').onchange();}};
el('register').onclick=async()=>{el('register').disabled=true;el('register-error').textContent='';
  try{const value=await api('/api/deployment/register',{source_dir:el('policy-source').value,name:el('policy-name').value});await catalog();el('policy').value=value.id;rememberPolicy();showPolicy();}
  catch(error){el('register-error').textContent=error.message;}finally{el('register').disabled=false;}};
setInterval(()=>{if(!document.hidden)catalog().catch(()=>{});},5000);
async function readiness(){const id=el('recording').value;if(!id)return;const value=await api('/api/deployment/readiness?id='+encodeURIComponent(id)),box=el('readiness');box.replaceChildren();
  for(const item of value.items){const row=document.createElement('div');row.className='readiness-row';const text=document.createElement('div'),state=document.createElement('b');text.textContent=item.name;
    if(item.detail){const note=document.createElement('small');note.textContent=item.detail;text.append(note);}state.textContent={ready:'준비됨',external:'별도 학습 프로젝트',pending:'확인 필요'}[item.state];state.className=item.state;row.append(text,state);box.append(row);}
  el('stages-link').href='/stages?id='+encodeURIComponent(id);el('objects-link').href='/objects?id='+encodeURIComponent(id);}
el('recording').onchange=()=>readiness().catch(error=>el('job-status').textContent=error.message);
function showJob(job){const report=job.report;el('job-report').classList.toggle('error',job.status==='error');
  el('job-status').textContent=job.status==='running'?({validate:'정책 추론 검증 중…',mock:'가짜 로봇으로 정책과 QP 계산 중…',replay:'기록 영상의 관측 변환 확인 중…'}[job.mode]):job.status==='error'?(report?.error||job.error||'검사 결과를 확인하세요.'):
    {validate:'패키지·정책 추론 검증 통과',mock:report?.stopped_by_request?'모의 실행을 중단하고 참조를 감속했습니다.':'정책 + QP 모의 실행 완료',replay:'기록 영상 입력 변환 검사 완료'}[job.mode];
  el('stop').hidden=job.status!=='running'||job.mode!=='mock';const box=el('job-metrics');box.replaceChildren();
  if(report){if(job.mode==='validate'&&report.parity){fact(box,'최대 행동 오차',report.parity.max_action_abs_error.toExponential(2));fact(box,'정책 CPU 중앙값',report.parity.policy_only_cpu_latency_ms.p50.toFixed(3)+'ms');}
    if(job.mode==='mock'){fact(box,'최대 참조 속도',report.maximum_reference_velocity_rad_s?.toFixed(3)+'rad/s');fact(box,'최대 참조 가속도',report.maximum_reference_acceleration_rad_s2?.toFixed(3)+'rad/s²');fact(box,'최대 참조 jerk',report.maximum_reference_jerk_rad_s3?.toFixed(1)+'rad/s³');fact(box,'실제 로봇 명령',report.hardware_commands_sent?'있음':'없음');}
    if(job.mode==='replay'){fact(box,'유효 프레임',report.frames);fact(box,'로봇 피드백','가정한 정지 자세');fact(box,'에셋 크기와 학습 크기',report.asset_size_matches_training?'일치':'다름 · 치수 검수 필요');fact(box,'실기/작업 성공 검증','수행하지 않음');}}
  el('job-json').textContent=JSON.stringify(report||job,null,2);
}
async function start(mode){const policy=selected();if(!policy)return;try{currentJob=await api('/api/deployment/start',{policy_id:policy.id,mode,seconds:Number(el('seconds').value),recording_id:el('recording').value});showJob(currentJob);}catch(error){el('job-status').textContent=error.message;}}
for(const mode of ['validate','mock','replay'])el(mode).onclick=()=>start(mode);
el('stop').onclick=async()=>{if(currentJob)showJob(await api('/api/deployment/stop',{policy_id:currentJob.policy_id,job_id:currentJob.id}));};
setInterval(async()=>{if(currentJob?.status==='running'){try{currentJob=await api('/api/deployment/job?policy_id='+currentJob.policy_id+'&job_id='+currentJob.id);showJob(currentJob);}catch(error){el('job-status').textContent=error.message;}}},450);
el('plan').onclick=async()=>{const policy=selected();if(!policy)return;try{const value=await api('/api/deployment/plan',{policy_id:policy.id,mode:el('live-mode').value,seconds:Number(el('seconds').value),calibration:el('calibration').value||null,perception_dir:el('perception-dir').value||null,channel:el('channel').value});
    el('plan-status').textContent=value.live_inputs.inputs_ready?'실시간 입력 파일 확인 완료. 실행 시 로봇 피드백과 시작 자세를 다시 확인합니다.':value.live_inputs.reasons.join(' ');
    el('command').hidden=el('copy-command').hidden=false;el('command').textContent=value.command;
  }catch(error){el('plan-status').textContent=error.message;}};
el('copy-command').onclick=()=>navigator.clipboard.writeText(el('command').textContent);
(async()=>{try{await catalog();const value=await api('/api/stages/recordings');records=value.recordings;
  for(const record of records){const option=document.createElement('option');option.value=record.id;option.textContent=(record.task_name||'시연')+' · '+record.id;el('recording').append(option);}
  const requested=new URLSearchParams(location.search).get('id');if(records.some(r=>r.id===requested))el('recording').value=requested;await readiness();
}catch(error){el('job-status').textContent=error.message;}})();
