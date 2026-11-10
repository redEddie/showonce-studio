> 현재 최종 기준은24배입니다. [최종 배포 안내](FINAL24-INTEGRATION.md)를 먼저 보세요. 아래 내용은 이전 모델의 배포·실기 이력입니다.

# SHOW-108 선정 체크포인트 실행

현재 실기 시험용 추천 정책은 **관측 DR · 넓은 범위 · 14+2배** (`ee9da8b8f493eaa56c3c`)다.
[배포 기록](SHOW108-OBSDR-WIDE-DEPLOYMENT.md)을 따른다. 아래 원본 관측 DR 없는16배 모델과 관련 기록은 비교 기준으로 보존한다.

선정 모델은 **MLP256 → 64 → LayerNorm → 마지막 LSTM128 → 선형14**, 134,217,728전이(16배)다.
기존 `lstm`(LSTM→MLP)과 구분한다. [원본 Release](https://github.com/redEddie/piper-sim-lab/releases/tag/show108-mlp-lstm-16u-20261101)의
source `85fcf98`, training source `49267d3`, learner SHA256
`6e52e2f5fd63c29ea5e3e6ae9676488acbe2ba5bf76823aeeb4a60cb9aa734de`를 기준으로 한다.
원본 learner/config/archive는 그대로 보존하고 별도 portable 패키지를 생성한다.

## Studio에서 사용

1. **04 정책 실행 → SHOW-108 · MLP + LSTM · 16배**를 선택한다. 최초 기본값은 선정 모델이고 이후 선택은 URL/브라우저에 유지된다.
2. **바로 실행 → 추론 검증**, **정책 + QP 모의 실행**을 확인한다. 이 둘은 카메라와 로봇에 연결하지 않는다.
3. 실시간 인식 노드가 실행되면 인식 폴더/보정을 자동으로 채운다. 저장 영상 분석은 실시간 입력을 대신하지 않는다.
4. 먼저 **shadow**로 실측 입력·정책 계산을 확인한다. SDK 피드백을 읽지만 모드·enable·목표 명령을 보내지 않는다.
5. **실제 로봇 실행 준비**를 누르면 직접 실행/run이 선택된다. 준비 버튼 자체로 움직이지 않는다. **Enable → 시작 자세 A로 이동 (Home)** 버튼으로 준비한 뒤, 팔 정지·그리퍼 개방(90mm 이상)을 확인하고 **실제 로봇 실행 (run)**을 누른다.
   Enable/home은 자동 실행하지 않는다. 처음 실행 시간 기본값은2초다. 탭을 닫아도 프로세스는 설정 시간까지 계속된다.
6. **정책 중단 (Ctrl+C)**는 해당 작업에 SIGINT를 보내 이미 검사한 정지 참조로 감속한다.
   SDK Stop은 별도의 감쇠 전자정지다. 오류 시 새 목표 전송을 멈추지만 마지막 SDK 목표는 남을 수 있다.

실시간 인식 패널에서 카메라 관측과 정책이 실제로 사용한 물체 위치·저장/파지 추정 출처를 나란히 확인한다. SAM이 꺼져 있으면 촬영 원본으로 표시한다.
명령어 보기에서는 프로세스를 시작하지 않는다. 복사 명령에서 out/job-token을 제외하고 관리 정보에 따로 표시한다.
정책 명령은 실행 버튼 바로 아래, SDK 명령은 SDK 버튼 아래에 표시한다. 보정·인식 폴더가 미연결이면 자리표시자가 들어간 명령을 복사하지 못하게 한다.
정책 이름을 명령/작업 제목에 표시하며, 검사 기록은 정책별로 구분한다. 기존 수동 제어 패널은 기본 접힘·개선 보류 상태다.

## 실기 어댑터

`piper-runtime==0.5.0`, `show108-mlp-lstm-v1` 프로필을 사용한다.
actor49/action7, FP32 normalizer, tanh 평균 행동, LSTM cell/hidden과 에피소드 리셋을 보존한다.
제어기는 원본 `checked_plan`과 같은 DLS+QP 참조 계획 및320ms 정지 예비 경로다.
그리퍼는3상태 최종 목표를 유지하며100ms마다 전체 폭 참조를 최대20mm 변경한다.
MOVE/BRAKE/GRIP 조정기·닫기 중 팔 고정·과제별 행동 덮어쓰기를 삽입하지 않는다.

| 설정 | 학습 | 실기 |
|---|---|---|
| 관절 속도/가속도/jerk | 1 / 5 / 100 | 동일(rad 단위) |
| TCP | 138.5mm | 동일 |
| 선속도/각속도 행동 스케일 | 0.1m/s / 0.5rad/s | 동일 |
| 주기 | 정책10Hz / 제어200Hz | 동일한 참조 주기, 실제 OS 지연 별도 검사 |
| TCP 테이블 하한 | 0mm | 기존 합의한+10mm |
| 그리퍼 힘 | MuJoCo actuator | SDK force0.5, 힘 센서 대응 미보정 |

관절/TCP/팔 링크 테이블 제약과 피드백/시작 자세 검사는 유지한다. 전체 자기 충돌·주변 장애물 플래너는 아니다.
공간 범위는 이 체크포인트의 학습 초기화 범위(x250–350mm, y±200mm, 반경400mm, 물체 간140mm 이상)를 검사한다.
오래된 순차 제어 프로필은 기존 정책용으로 유지하며 새 모델에 자동 대체하지 않는다.

## Thor 운영 경로

```bash
cd ~/piper_ws
source env.sh
python showonce_policy.py validate --policy-id e62899a3bc91c4c241d6
python showonce_policy.py mock --policy-id e62899a3bc91c4c241d6 --seconds 2
```

정책은 `datasets/policy_library/e62899a3bc91c4c241d6`에 등록했다.
원본은 `releases/show108-mlp-lstm-16u-20261101`, 파생 패키지는 `artifacts/show108-mlp-lstm-16u-v2`다.
입력·제어·추론 검증 보고서는 파생 패키지에 보관한다. 실기 실행은 진행했으며, 자동 파지 성공은 아직 확인하지 못했다.
사용자가 큐브를 쥐어준 뒤 들어 올렸다가 다시 개방한 사례가 있다. 실행 보고서의 `success`는 실행 완료 여부이며 작업 성공률이 아니다.

### 파지 진단과 관측 랜덤화 검토 (2026-11-01)

선정 Release는 `obs_randomization=false`다. 확인한 Release 소스의
`StageRewardPickPlace.__init__`는 true를 거부하므로 설정 변경만으로 켤 수 없다.
학습 저장소에서 이 제한을 개정하고 지연/관측 회귀 검사를 추가해야 한다. 현재 Studio에서 학습 코드나 원본 체크포인트를 변경하지 않는다.

기본 구현은 큐브 위치에 매 정책 스텝 독립 Gaussian 잡음(축별 sigma10mm, 축별 clamp30mm),
임의 3D 축 회전각 균등 ±5°를 적용한다. 그릇 위치 bias(sigma5mm/clamp15mm), 큐브 크기 ±10%,
큐브 지연 0–150ms, 로봇 지연/행동 hold 각각0–50ms는 에피소드마다 한 번 추출한다.
그릇 관측 지연, 가림·누락, 저장 위치 및 파지/낙하 상태 추정의 전환은 이 잡음 모델에 포함되지 않는다.

최근 7회 실행의 초기2초를 물체 정지 구간으로 가정하여 중복 관측 시각을 제외했다.
정책에 입력된 큐브 XY의 축별 표준편차는 약0.04–0.50mm, 그릇은0.02–0.14mm였다.
이는 상태 추정 후 입력의 반복 정밀도이며 절대 좌표 정확도가 아니다. 작은 Z 흔들림도 알려진 크기/테이블 평면 가정의 결과다.
같은 구간에서 마지막 영상 관측 age는 큐브 중앙값71ms/p95 115ms, 그릇141ms/p95 272ms였다.
실행 전체에서는 가림 때문에 수 초 전 영상도 유지한다. 정지 물체의 오래된 관측이나 정상적인 파지 추정에는
그 age만큼의 실제 위치 오차가 생기는 것이 아니다. 큐브를 들어 올렸는데 저장 위치에 남는 경우는 별도의 추정 실패다.

현재 로그에 맞추는 초기 실험안(검증된 최적 범위 아님):

| 관측 요소 | 제안 |
|---|---|
| 큐브 위치의 빠른 흔들림 | XY sigma2–3mm/축별 cap6–9mm부터 시작. Z는 평면·크기 오차와 구분 |
| 카메라 보정 bias | 두 물체에 공통으로 적용하는 에피소드 고정 변환을 별도 모델링. 절대 오차 실측 전 범위 확정 금지 |
| 방향 | 우선 upright 큐브 yaw ±5°. 가림/대칭 모호성에 따른 불연속은 별도 처리 |
| 그릇 위치 | 기존 고정 sigma5mm/cap15mm를 강건성 시험 범위로 유지 가능. 공통 보정 오차와 독립 오차를 구분 |
| 크기 | 현재50mm 실측 물체는 지각 오차 ±2–5%부터. 다양한 물체 실험에서는 ±10% 비교 |
| 큐브 영상 지연 | 0–200ms 후보, 실측 분포로 재평가. 수 초 가림은 지연 상한 증가가 아닌 누락/추정 과정으로 모델링 |
| 로봇/명령 지연 | 0–50ms를 시험 범위로 유지 가능하나 실제 지연을 측정한 수치는 아님. 가변 지연과 그리퍼 응답은 별도 식별 |

관측 좌표를 바꿀 때 block–TCP, block–bowl 등의 파생 입력도 동일한 관측에서 계산하고,
보상·접촉 판정은 참 상태에 둔다. 비교는 무잡음 기준, 기존 기본 DR, 파이프라인을 반영한 DR을
같은 초기 조건/seed로 수행한다. 파지 위치 입력 오류를 큰 독립 잡음으로 대신하지 않는다.

저장 로그의 가상 비교에서 큐브 위치만 TCP로 바꾸고 상대 위치를 일관되게 재계산하자,
한 개방 시점의 출력이 +0.694에서 −0.919로 바뀌었다. 나머지 관측/과거 행동은 저장 기록을 사용했으므로
이는 조건부 추론 비교이며 실제 폐루프 궤적이나 성공을 검증한 결과가 아니다.

별도 진단 도구는 SDK 없이 CAN을 수신하고, Studio의 공유 RGB-D와 원본 시각에 맞춰 계산 TCP를 투영한다.
실제 그리퍼와 표시가 여러 정지 자세에서 일치하는지 비교한다. 같은 보정으로 투영한 점의 일치는
물체 절대 위치를 독립적으로 검증한 것은 아니다. 웹앱/정책의 제어 동작은 바꾸지 않는다.

```bash
python -P -m showonce_studio.tcp_diagnostic --policy-id e62899a3bc91c4c241d6 --calib /path/to/current/T_base_cam_robot.json --seconds 15 --view
```

Thor 호환 진입점은 `python showonce_tcp_check.py`다. 결과는 `runs/showonce-tcp-check/`에 남는다.
실제 정지 자세에서 수신 전용2초/17프레임 수집과 투영을 확인했고 해당 구간 CAN TX 증가량은0이었다.

### 파지 위치 갱신 수정

SHOW-108 실행에 Studio의 `DeploymentSceneState`(`showonce_scene_state_v2`)를 적용한다.
등록한 정책 ID/파일/가중치, 49차원 입력 규약, QP, TCP138.5mm, 힘0.5는 유지한다.
공통 Runtime의 잠긴 파일을 수정하지 않고 배포 관측 어댑터에서 상태 추정만 확장한다.
실행 보고서의 `deployment_state_estimator`에 적용 schema/소스 hash를 별도로 기록한다.

기존의 현재 위치 기반 파지 판정에 더해, 닫기 직전350ms 이내 큐브 근처에 도달한 이력을 기억한다.
닫기 시작 후1초 이내/접근 지점에서120mm 이내에서 실측 폭이50±6mm이고,
직전 그리퍼 참조가 실측보다6mm 이상 작으며,100ms 이상 폭 변화 범위가2mm 이하이면 파지로 추정한다.
폭이50mm를 잠깐 통과하는 빈손 닫기나 명령한 폭에 도착한 뒤의 일시 정지는 이 추가 조건을 만족하지 않는다.
파지 뒤에는 기존 TCP 기반 위치 갱신과 영상의 테이블 투영 무시, 실제 개방 후 낙하/정지 추정을 이어 쓴다.
상태 추정으로부터 별도의 팔 고정·그리퍼 개폐 명령을 추가하지 않는다. 접촉 센서가 없으므로 파지는 추정이며 성공 판정이 아니다.

터미널에 `파지 추정: 폭 … 유지 → 큐브 위치를 TCP로 갱신합니다.`가 출력되고,
04의 큐브 입력 출처는 `파지 추정 · TCP`가 된다. 전환은 `events.jsonl`의 `scene_transition`,
근거는 정책 입력의 `state_estimation.grasp_evidence`, 최종 요약은 `scene_estimation_summary`에 남는다.

기록한52.4mm 폭 유지 사례를 재생하면 이전 추정기는 계속stationary지만 새 추정기는3.49초에held_tcp가 된다.
빈손 닫기 사례는 전환하지 않았다. 테스트는 저장 로그/가짜 피드백이며 새 버전으로 실제 팔을 구동한 검증은 사용자가 진행한다.
이번 수정은 파지 후 입력 갱신 문제에 대한 것이다. 접근 위치 오차나 자동 파지 성공까지 해결됐다는 의미는 아니다.

### 정지 후 닫는 스크립트 비교

Thor의 `python showonce_scripted.py plan` / `run`은 기존 `show100_scripted.py`를 재사용하되,
현재 Studio 인식 폴더와 보정을 자동 선택한다. 등록한 SHOW-108의 모델/XML·설정만 읽으며 actor 추론은 하지 않는다.
현재 실시간 인식과 보정 파일 내용/해시가 일치해야 하며, 큐브·그릇 각각 새로운 관측5개를 수집한다.
카메라 heartbeat만 새롭고 물체 좌표가 오래된 경우에는 시작하지 않는다.

기본 범위는 `approach → descend → close → lift`다. `run`은 각 단계마다 Enter를 받으며 q/Ctrl+C로 중단한다.
물체의 초기 좌표는 수집 후 고정하며 실행 중 이동시키면 안 된다. Enable/Home은 사용자가 먼저 수행한다.
TCP138.5mm, SDK force0.5, 관절 v1/a5/j100, TCP 테이블+10mm 및 기존 경로 검사를 유지한다.
닫기는 기존 SHOW-103 정지/그리퍼 조정기로 수행하며, 이 비교가 RL 실행의 제어기나 정책을 바꾸지는 않는다.

닫기 단계의 TCP 이동량을100ms 주기로 기록하며 샘플 이동이5mm를 넘으면 lift를 진행하지 않는다.
그리퍼 폭만으로 성공을 확정하지 않고 close 완료 후 실제 파지를 확인한 다음 Enter로 lift를 진행한다.
lift에서 종료하므로 자동으로 그릇에 놓거나 그리퍼를 다시 열지 않는다.

```bash
cd ~/piper_ws
source env.sh
python showonce_scripted.py plan
python showonce_scripted.py run
```

`plan`은 가짜 로봇만 사용한다. `run`도 선택한 전체 단계의 모의 도달성 검사를 먼저 수행한다.
파지만 시험할 때는 선택하지 않은 그릇 운반/후퇴 자세의 도달성을 요구하지 않는다.
현재는 외부 스크립트 경로 없이 실행하는 [독립 튜토리얼](../examples/scripted-grasp/README.md)로 분리했다.
`showonce-scripted` 또는 `python -P -m showonce_studio.scripted_grasp`를 사용한다.
`--policy-id`는 선택이며 기본 실행에는 정책 패키지가 필요하지 않다.
결과는 `runs/showonce-scripted/<시각>/summary.json`과 `events.jsonl`에 저장한다.

2026-11-01 검증: 과거 실측 장면을 사용한 CPU/mock 네 단계 경로 통과, 실제 SDK 생성 금지 sentinel 적용,
새 테스트8개 통과. 현재 live 연결도 확인했으나 물체 관측이 expired라 실제 좌표의 plan은 시작하지 않았다.
사용자가 큐브를 그릇 밖에 두고 현재 관측을 확보한 뒤 plan/run을 진행해야 한다. 로봇 동작 명령은 보내지 않았다.

### D435i 공유와 상시 인식

Studio 촬영 노드가 D435i를 한 번만 열고, `/api/capture/rgbd`에서 정렬된 RGB-D·카메라 내부 파라미터·원본 프레임 시각을 제공합니다.
실시간 인식 worker는 이 스트림을 구독합니다. 이전 인식 웹앱에서 확인한 SAM·큐브/그릇 이미지 기하 추정·앵커 보존 코드를 그대로 사용하며 카메라 입력만 구독형으로 바꿉니다.
카메라를 촬영/인식 사이에서 수동 전환하지 않습니다. 기존 `show100_sam_node.py`를 추가로 실행할 필요도 없습니다.

Thor에서는 Studio 시작 시 지속 인식 worker를 자동으로 연결합니다. 웹 종료/새로고침·정책 체크포인트 교체로 SAM을 다시 로드하지 않습니다.
새 장비에서는 `configs/example.toml`의 `[live_vision]`에 검증된 inspector/show33 경로와 Python 환경을 설정합니다. 별도 GPU 환경의 설치는 필요합니다.
해당 설정이 없거나 offline 모드이면 자동으로 카메라·SAM을 시작하지 않습니다.
연결이 끊겼다면 04의 **인식 노드 상태 · 재연결 → 인식 연결 재시도**를 사용합니다. 인식 연결은 로봇 명령을 보내지 않습니다.

현재 인식 프로세스/PID/출력 폴더는 `runs/showonce-live-vision/service.json`, 로그는 같은 폴더의 `console.log`에 있습니다.
`manifest.json`에 `camera_source=studio_rgbd_broker`를 기록합니다. 보정과 카메라 시리얼·RGB-D 정렬·원본 시각·스트림 ID를 확인하며, 카메라가 재시작되면 오래된 프레임을 새 관측으로 사용하지 않습니다.

인식이 연결되면 정책 입력 폴더와 보정이 자동으로 채워집니다. 해당 경로를 확인한 뒤 shadow/run 명령을 생성하세요.
새 카메라 배치라면 그 배치에 맞는 보정을 사용해야 합니다.

## 내보내기와 검증 재현

실행 환경에 JAX/Flax를 추가할 필요가 없다. 별도 CPU Python3.12 환경에 JAX0.10.2,
Flax0.12.8, Brax0.14.2, OSQP1.1.3, Ruckig0.19.4와 공통 Runtime을 설치하여 실행한다.
훈련 source checkout을 수정하거나 학습을 시작하지 않는다.

```bash
python scripts/export_show108.py --source-repo /path/to/pinned/show108-source --run /path/to/extracted/training --output /path/to/new/export
python scripts/verify_show108_controller.py --source-repo /path/to/pinned/show108-source --output /path/to/new/export/controller-validation.json --golden /path/to/control-golden.npz
python scripts/verify_show108_observation.py --source-repo /path/to/pinned/show108-source --output /path/to/new/export/observation-validation.json
```

원본 모델 추론1,024입력, LSTM 상태/전체14출력, 원본 JAX 경로32사례, 원본 관측+MuJoCo FK64사례를 비교했다.
검증은 산술/계약 비교와 가짜 피드백이며 물리 시뮬레이션·실기 성공률 평가가 아니다.
