# Thor 운영 호환 진입점

`thor/`는 2026-11-01 실기에 사용한 `~/piper_ws` 진입점의 사본입니다.
패키지 본체는 `src/showonce_studio`, 공통 제어기는 별도 `piper-runtime` 저장소에 있습니다.
이 사본을 이 디렉터리에서 직접 실행하지 않습니다.

기존 Thor 배치는 `~/piper_ws`와 `~/projects/{showonce-studio,piper-runtime}`가 형제 경로이며,
`show_env.py`가 `~/miniforge3/envs/piper-show/bin/python`과 CUDA 환경을 선택합니다.
`showonce_robot.py`, `showonce_policy.py`, `showonce_scripted.py`, `showonce_tcp_check.py`는
작업 디렉터리에 복사하여 사용하던 호환 진입점입니다. 자동 복사·환경 변경은 하지 않습니다.

새 머신은 README의 패키지 설치와 로컬 설정을 사용합니다. 설치한 환경에서는
`showonce-robot`, `showonce-policy`가 같은 패키지 기능의 표준 진입점입니다.
스크립트 파지·TCP 진단은 각각 `python -m showonce_studio.scripted_grasp`,
`python -m showonce_studio.tcp_diagnostic`으로 접근할 수 있습니다.

실제 Thor에서 사용한 평면 배치 코드와 Home/인식/스크립트 파지의 기존 보조 소스는
[실기 기록 Release](https://github.com/redEddie/showonce-studio/releases/tag/show108-obsdr-wide-real-demo-20261101)의
`showonce-thor-operating-sources-20261101.tar.gz`에 보존합니다.
이는 경로와 외부 모델 의존성이 남아 있는 운영 기록이며, 독립 설치 패키지가 아닙니다.
원본 RGB-D 데이터·가중치·보정·인증정보·외부 프로젝트는 포함하지 않습니다.

실시간 인식 이식 후 `thor/showonce_deployment_api.py`는 기존 flat GUI와 패키지의 연결 진입점입니다.
`thor/flat-studio-live.patch`는 로컬 GUI가 자체 보정 사본과 비공개 `configs/studio.local.toml`을 기본으로 사용하게 하는 변경입니다.
이 파일/패치는 운영 디렉터리에 적용하는 호환 자료이며 compat 폴더에서 직접 실행하지 않습니다.
실시간 인식 자체는 Studio 패키지 안에 있습니다.

`thor/calibrate_camera.py`는 설치된 Studio 보정 모듈을 호출합니다. 기존 파일에 교체하여 사용하며,
비공개 `configs/studio.local.toml`을 읽고 `-P`로 flat `showonce_studio.py`와의 이름 충돌을 피합니다.
보정 계산/모델은 모두 패키지에 포함되며 SHOW-33 폴더가 필요하지 않습니다.
`thor/flat-capture-calibration.patch`는 flat capture의 기본 보정 경로도 운영 폴더의
`calib/current.json`으로 바꾸는 호환 변경입니다. 명시적인 `--calib` 인수는 그대로 지원합니다.

자세 A 복귀는 `showonce_robot.py home`의 기존 CLI를 유지하면서 Studio 내부 제어기로 바뀌었습니다.
`home_runner` 설정은 더 이상 읽지 않습니다. `showonce_scripted.py`는 웹앱 없는 튜토리얼 진입점이며
`scripted_runner` 설정도 사용하지 않습니다. `thor/flat-posture-labels.patch`는 운영 GUI의 Home 표시를 자세 A로 통일합니다.

`thor/flat-setup-hints.patch`는 이전 setup shell script/flat 모델 다운로드 파일 대신
현재 worker 설치 문서와 패키지 모듈 명령을 안내하게 하는 운영 호환 변경입니다.
