# 웹앱 없이 실행하는 스크립트 파지 튜토리얼

50mm 초록 큐브를 줄무늬 그릇에 옮기는 기존 7단계 예제입니다.
**Studio 웹 서버·브라우저·VLM·정책 체크포인트가 필요하지 않습니다.**
정책 실행의 필수 단계나 웹앱 버튼으로 사용하지 않습니다.

접근 → 하강 → 닫기 → 들어올리기 → 운반 → 개방 → 후퇴.
기본 `--through lift`는 들어올리기까지만 실행합니다. 이전처럼 전체 작업은
`--through retreat`로 선택하고, 실기 `run`은 단계마다 Enter를 기다립니다.
`--auto`를 명시한 경우에만 단계 확인을 생략합니다.

## 준비

CUDA/RealSense가 준비된 장비에서 Studio/Runtime과 해당 worker를 설치합니다.
제어 환경에는 `piper-runtime[robot]`, `showonce-studio[motion]`이 필요합니다.
SAM은 [별도 NumPy 1.x 환경](../../docs/SAM-WORKER.md)에 두고 각 환경에 Studio/Runtime을 설치합니다.
모델·worker 경로를 지정한 `SHOWONCE_CONFIG`와 현재 신뢰된 보정 파일이 필요합니다.
기존 `home_runner`, `scripted_runner`, SHOW-33/Codex 작업 폴더는 사용하지 않습니다.

아래의 Python 경로와 보정 경로를 자신의 설치에 맞게 바꿉니다.
각 터미널에 같은 `SHOWONCE_CONFIG`를 설정합니다. 카메라를 점유하는 다른 앱은 먼저 종료합니다.

### 터미널 1 — 카메라 노드

```bash
export SHOWONCE_CONFIG=/path/to/local.toml
/path/to/capture/bin/showonce-capture --calib /path/to/current.json
```

capture 기본 포트는 8768입니다. 이 노드는 카메라만 소유하며 웹앱을 시작하지 않습니다.

### 터미널 2 — SAM 실시간 인식 노드

```bash
export SHOWONCE_CONFIG=/path/to/local.toml
/path/to/sam/bin/showonce-live-vision \
  --capture-url http://127.0.0.1:8768 \
  --calib /path/to/current.json \
  --out /path/to/runs/show100-vision-tutorial
```

두 노드는 이후 계획/실행을 반복해도 유지합니다. 각 물체의 실제 관측 시각은 보존하며,
새 heartbeat로 가려진 물체의 관측을 새것처럼 취급하지 않습니다.

### 터미널 3 — 계획 확인 및 단계 실행

```bash
# 제어 환경에서 실행
export SHOWONCE_CONFIG=/path/to/local.toml
showonce-scripted plan \
  --perception-dir /path/to/runs/show100-vision-tutorial \
  --calib /path/to/current.json
```

`plan`은 현재 인식 좌표를 모아 가짜 로봇으로 경로를 계산합니다. SDK를 열거나 CAN 명령을 보내지 않습니다.
물체 접촉·파지 성공을 검증하는 물리 시뮬레이션은 아닙니다.

실기 실행은 팔이 활성화되고 CAN 제어 상태일 때 시작 자세 A에서 진행합니다.

```bash
showonce-robot home --calib /path/to/current.json
showonce-scripted run \
  --perception-dir /path/to/runs/show100-vision-tutorial \
  --calib /path/to/current.json
```

전체 옮기기 예제는 `plan`과 `run`에 각각 `--through retreat`를 추가합니다.
`run`은 먼저 선택한 단계 전체를 모의 계산하고 실제 시작 자세·정지 상태·펌웨어 가속도·
관절 및 테이블 간격을 검사합니다. 닫기 중 TCP가 5mm 넘게 움직이면 다음 들어올리기로 진행하지 않습니다.
기본 force는 0.5, TCP는 138.5mm, TCP/팔 링크 테이블 간격은 10mm입니다.

정책 설정의 시작 자세만 선택적으로 재사용하려면 `--policy-id`를 줍니다. 기본 예제에는 필요하지 않으며
actor 가중치를 읽거나 추론하지 않습니다. `python -m showonce_studio.scripted_grasp`도 같은 진입점입니다.
기존 Thor의 `python showonce_scripted.py plan|run` 호환 명령도 유지합니다.

`q` 또는 Ctrl+C는 새 목표 전송을 중지합니다. 자동 damping/disable은 보내지 않으며
마지막으로 전달한 SDK 목표까지 움직임이 남을 수 있습니다.
결과는 `runs/showonce-scripted/<시각>/summary.json`과 `events.jsonl`에 저장됩니다.

원본 팀 구현의 출처는 `provenance/posture-and-scripted.json`에 기록했습니다.
튜토리얼은 특정 큐브·그릇과 단계별 규칙을 사용하는 예제이며 일반 정책 실행기로 사용하지 않습니다.
