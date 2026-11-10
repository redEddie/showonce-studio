# ShowOnce Studio

Studio: [redEddie/showonce-studio](https://github.com/redEddie/showonce-studio) · 학습 인계본: [redEddie/piper-learning](https://github.com/redEddie/piper-learning) · 공통 실행: [redEddie/piper-runtime](https://github.com/redEddie/piper-runtime)

세 저장소는 GitHub에 공개되어 있습니다. 자체 코드 라이선스는 아직 미정이며, [공개 범위와 조건](docs/PUBLICATION.md)을 확인하세요.

RGB-D 시연 수집 → 스테이지 분할 → 객체 관계 정의 → 정책 실행 준비 웹앱입니다.
학습 인계·통합용 코드는 `piper-learning`, 관측·행동 계약과 제어 커널은 `piper-runtime`에서 관리합니다.
현재 3090에서 개발·학습하는 원본은 별도 [piper-sim-lab](https://github.com/redEddie/piper-sim-lab)이며,
`piper-learning`은 사용자 확정24배 소스를 동기화하고 공통 제어만 Runtime에서 import합니다. 이후 원본의 모든 실험 브랜치를 자동으로 미러하지는 않습니다.

화면을 수정할 때는 [화면·설계 기준](docs/DESIGN.md)을 참고합니다.
네 페이지의 역할, 공통 디자인과 상호작용을 간단히 정리했습니다.

## 설치와 공개 준비

Studio와 Runtime 두 저장소만 있으면 됩니다. 학습 저장소는 기본 설치에 필요하지 않습니다.
[설치 안내](docs/INSTALLATION.md)에 새 Python 환경 → Runtime → Studio → 합성 예제 순서를 정리했습니다.
예제는 `showonce-studio --offline`으로 시작하며 카메라·모델·로봇을 연결하지 않습니다.

- [기능별 필수 worker / 외부 코드 설치](docs/WORKERS.md)
- [Studio 내부 실시간 인식](docs/LIVE-PERCEPTION.md)
- [외부 코드·모델 고지](THIRD_PARTY.md)
- [공개 준비 상태와 남은 항목](docs/PUBLIC-RELEASE.md)
- [라이선스 제안: Apache-2.0, 아직 적용 전](docs/LICENSE-PROPOSAL.md)

## 실제 구성

`configs/local.toml` 또는 `SHOWONCE_CONFIG`로 데이터 경로, 현재 보정, 모델 경로,
각 worker의 Python 경로와 포트를 지정합니다. 환경 변수 `SHOWONCE_HOME`, `SHOWONCE_MODELS`,
`SHOWONCE_CALIB`, `SHOWONCE_*_PYTHON`으로 덮어쓸 수 있습니다.
기존 모델/보정은 읽기 경로로 연결할 수 있으며 복사·재다운로드가 필요하지 않습니다.

D435i 수집은 RealSense 환경, 실제 Piper 연결은 공통 패키지의 robot extra가 필요합니다.
SAM·FoundationPose·SAM 3D는 별도 GPU 실행 환경입니다. `requirements/`와
`docs/source-inventory/`는 검증한 Thor 환경의 버전/수정 내역이며 범용 CUDA 설치 보장은 아닙니다.
모델 가중치와 외부 프로젝트 소스는 포함하지 않습니다. 설치 경로·가중치 준비·접근 권한이 필요한
모델은 각 공식 배포 절차에 따라 준비합니다. Thor/aarch64 환경을 RTX/x86에 복사하지 않습니다.

웹 프로세스는 정책 제어 주기를 담당하지 않습니다. CPU worker를 별도로 실행하며,
실기 명령 준비는 실시간 인식 폴더와 현재 보정을 요구합니다. 저장 영상은 실기 입력으로 사용하지 않습니다.
04 정책 실행에서 **명령어 보기 / 바로 실행**을 선택합니다. 기본값은 명령어 보기이며, 바로 실행에서는 SDK/정책을 별도 프로세스로 시작합니다.
실행 중인 작업·로그·종료 결과와 정책 중단(Ctrl+C) 버튼을 제공합니다. 새로고침 후에도 작업을 복구합니다.
SDK Stop은 감쇠 전자정지이며 정책 감속 중단과 구분합니다. 자세한 동작은 [로봇 제어 안내](docs/ROBOT-CONTROL.md)를 참고하세요.

## 구조와 검사

- `src/showonce_studio/`: API·수집·분석 어댑터·작업/에셋/정책 관리
- `src/showonce_studio/web/`: 네 페이지와 로컬 Three.js/ECharts
- `examples/`: 개인 데이터를 포함하지 않는 합성 예제 생성기
- `configs/`: 머신별 설정 예제 (`local.toml`은 Git 제외)
- `tests/`: GPU·실제 카메라·로봇 없이 실행하는 검사

```bash
python -m pytest -q
```

현재 운영 중인 `piper_ws`와는 별개 작업본입니다. 운영 환경을 이 코드로 자동 전환하지 않습니다.
원본 영상·에셋·체크포인트·캐시·가상환경·토큰은 제외합니다.
프로젝트 라이선스 결정과 GPU 환경의 새 머신 설치 검증은 공개 릴리스 전 남은 항목입니다.
외부 라이선스/고지를 프로젝트 전체의 라이선스로 확대하지 않습니다.

## SHOW-108 완성 모델 실행

현재 최종 배포 기준은 **관측 DR·확장 스폰·MLP+LSTM24배**입니다.
[설치·등록·선택·검증 안내](docs/FINAL24-INTEGRATION.md)를 따릅니다. 2026-11-10 사용자 현장 확인으로 **10회 시도·10회 성공**을 기록했습니다. [실기 결과·마감 기록](docs/FINAL24-REAL-ROBOT-20261110.md)을 참고하세요.
14+2배의 [이전 배포 안내](docs/SHOW108-OBSDR-WIDE-DEPLOYMENT.md)와 검증 기록은 보존합니다.
기존 DR 없는 16배 모델은 [이전 배포 안내](docs/SHOW108-DEPLOYMENT.md)와 함께 보존합니다.

2026-11-01 단일 실기에서 파지·운반·그릇 안 배치를 영상으로 확인했습니다.
[실기 기록](docs/REAL-ROBOT-DEMO-20261101.md)과
[영상·운영 소스 Release](https://github.com/redEddie/showonce-studio/releases/tag/show108-obsdr-wide-real-demo-20261101)를 참고하세요.
운영용 호환 진입점은 [compat/README.md](compat/README.md)에 설명합니다.

카메라 보정 생성도 Studio 내부의 `calibrate-camera`로 실행합니다.
[환경·보정 방식·결과 확인](docs/CAMERA-CALIBRATION.md)을 참고하세요.

시작 자세 A 복귀는 외부 실행기 없이 `showonce-robot home`으로 실행합니다([안내](docs/ROBOT-CONTROL.md)).
정책 없는 7단계 파지는 [웹앱 없는 독립 튜토리얼](examples/scripted-grasp/README.md)이며 기본 실행 흐름에서 요구하지 않습니다.

기존 작업 폴더 없이 설치할 때의 범위와 외부 코드/데이터 항목은 [경로 점검 결과](docs/PATH-AUDIT.md)에 정리했습니다.
