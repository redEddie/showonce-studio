# 기능별 필수 worker와 외부 코드 준비

웹 프로세스는 UI·작업 관리를 담당하고 카메라, SAM, 포즈 추적, VLM, 정책 실행은 별도 프로세스에서 수행합니다.
정책 교체나 브라우저 새로고침은 SAM 가중치 재로딩 사유가 아닙니다.

worker는 **전체 파이프라인에서 생략해도 되는 부가기능**이라는 뜻이 아닙니다.
아래 기능을 사용하면 해당 worker와 의존성이 필수입니다. CPU 화면 예제는 설치 확인용 첫 단계이며 전체 설치 완료가 아닙니다.

| 수행할 기능 | 필수 worker / 구성 |
|---|---|
| 카메라 보정 생성 | capture + SAM worker + native render-and-compare worker |
| RGB-D 시연 녹화 | capture worker, D435i/librealsense |
| 자동 스테이지 분할 | VLM worker, Qwen 또는 지원하는 다른 모델 하나 |
| 물체·손 마스크 검출/추적 | SAM geometry worker, SAM 가중치 |
| 참조 에셋의 6D 포즈 추적 | FoundationPose worker, 참조 에셋과 scorer/refiner |
| 객체 관계·씬 그래프 추출 | SAM 손/물체 추적 + 포즈 결과 + 관계 추출 worker |
| 실제 정책 실행 | 실시간 인식 worker + 정책/control worker + SDK/CAN + 보정/정책 패키지 |
| 신규 에셋 메시 복원 | SAM 3D/real2sim 복원 worker (기존 에셋 재사용 흐름에서는 사용하지 않음) |

카메라→스테이지 분할→객체 관계 정의→정책 실행까지 사용하려면 각 행의 필수 구성이 모두 필요합니다.
모델 선택지가 있다는 것과 해당 작업 자체가 생략 가능하다는 것은 다릅니다.

## 설정 원칙

먼저 [CPU 설치](INSTALLATION.md)를 완료합니다. 각 worker 환경에도 동일한 Studio/Runtime wheel 또는
checkout을 설치해야 `python -m showonce_studio.…`가 동작합니다. CUDA와 native 라이브러리는 upstream의
지원 조합으로 준비하고, 기존 Thor venv의 `site-packages`를 다른 장비로 복사하지 않습니다.

```toml
[paths]
home = "~/.local/share/showonce-studio"
third_party = "~/.local/share/showonce-studio/third_party"
models = "~/models"
calibration = "~/.local/share/showonce-studio/calib/current.json"
sam_checkpoint = "~/models/sam3/sam3.pt"

[python]
capture = "/path/to/capture/bin/python"
vlm = "/path/to/vlm/bin/python"
geometry = "/path/to/sam/bin/python"
foundationpose = "/path/to/foundationpose/bin/python"
# 기존 에셋의 관계 추출만 수행하면 SAM 환경을 지정할 수 있음.
# 신규 메시 복원도 수행하면 별도 SAM 3D 환경을 지정.
sam3d = "/path/to/sam/bin/python"
robot = "/path/to/control/bin/python"
policy = "/path/to/control/bin/python"

# 실시간 인식은 Studio 내부 코드. 모델·보정·worker Python만 지정.
# [live_vision]
# autostart = true
# python = "/path/to/showonce-studio/.venv-sam/bin/python"
# objects = "/path/to/custom_objects.json"  # 생략하면 패키지 기본 정의
```

위 경로는 예시이며 실제 절대 경로를 사용합니다. `SHOWONCE_*_PYTHON`, `SHOWONCE_THIRD_PARTY`,
`SHOWONCE_MODELS`, `SHOWONCE_CALIB` 환경 변수도 사용할 수 있습니다.

## Capture: D435i RGB-D

capture 환경에 Studio/Runtime, NumPy, OpenCV, PyAV(`av`), `pyrealsense2`가 필요합니다.
librealsense의 [공식 Python 설치/빌드 안내](https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python)를 따릅니다.
지원되는 Python/아키텍처에서만 PyPI wheel을 사용하고, Thor에서는 별도 native build가 필요할 수 있습니다.
기록된 Thor binding은 2.59.1이지만 완전한 native build recipe/리비전은 아직 배포 자료가 아닙니다.

```bash
# 준비한 capture 환경에서 실행. 실제 카메라를 여는 명령입니다.
showonce-capture --calib /path/to/T_base_cam_robot.json
```

한 D435i는 촬영 노드 하나가 소유합니다. SAM은 촬영 노드의 정렬 RGB-D를 구독합니다.
`showonce-studio`를 일반 모드로 실행하면 기존 capture 노드에 연결하거나 없으면 시작합니다.
보정 생성 코드는 Studio 내부로 이식했습니다. `calibrate-camera` 실행과 필수 native 환경은
[CAMERA-CALIBRATION.md](CAMERA-CALIBRATION.md)를 따릅니다.

## VLM: Qwen 우선

VLM 환경에 해당 GPU용 PyTorch/torchvision과 `requirements/requirements-showonce-vlm.txt`를 설치합니다.
`torch` wheel의 CUDA 조합은 [공식 설치 안내](https://pytorch.org/get-started/locally/)를 사용합니다.
Qwen은 `Qwen/Qwen3.5-9B`, 고정 리비전은 `docs/external-dependencies.json`과
`showonce_models.py`에 기록되어 있습니다. 웹앱의 모델 준비 동작으로 별도 다운로드합니다.
Kanana/Gemma는 선택 어댑터이며 각 모델의 조건과 인증이 적용됩니다. HF 토큰은 로컬 로그인으로 관리합니다.

## SAM 3: 저장 영상과 실시간 인식

SAM은 물체·손 추적에 필수입니다. [SAM worker 설치·검증](SAM-WORKER.md)을 따릅니다.
원본 SAM 0.1.0과 NumPy 1.26.4를 별도 환경에서 사용하며 SAM 원본의 의존성 선언을 바꾸지 않습니다.
Studio/control은 NumPy 2.5.4를 유지합니다. OpenCV/SciPy도 각 NumPy 환경에 맞는 버전으로 구분합니다.
공통 Runtime 유틸리티는 두 버전에서 검증한 범위를 허용합니다.

**Thor/Linux aarch64, Python 3.12 새 환경에서 설치·pip check·실제 녹화 RGB 추론 및
NumPy 2 Studio→HTTP→NumPy 1 SAM→파일 연결을 검증했습니다.**
결과는 `sam-worker-validation.json`에 있습니다. 카메라 연결·FoundationPose 전체 설치는 별도입니다.

저장 영상 SAM worker는 `showonce-geometry`이고, 실시간 인식은 Studio 내부의
`showonce-live-vision`입니다. 외부 inspector/SHOW-33 폴더 없이 capture 노드의 RGB-D를 구독합니다.
설정과 보존한 동작은 [LIVE-PERCEPTION.md](LIVE-PERCEPTION.md)에 있습니다.
카메라 보정과 시작 자세 A 복귀도 Studio 내부 코드입니다. 외부 Home 실행기를 사용하지 않습니다.

## FoundationPose: 6D 포즈 추적에 필수인 연구·평가 프로필

현재 adapter가 검증한 소스는 `shingarey/FoundationPose-plus-plus`의
`58aa7154de23ae86d542366858fc6bcca3b747b0`입니다.
`<third_party>/FoundationPose/estimater.py`가 보이도록 checkout의 `FoundationPose/` 하위 폴더를 연결합니다.
scorer/refiner 가중치와 `mycpp`, nvdiffrast 등의 native extension은 upstream에서 준비합니다.
호환 패치는 저장소의 `vendor/patches/FoundationPose-Utils.patch`이며, 적용 전
해당 소스 디렉터리에서 `git apply --check /path/to/patch`로 리비전을 확인합니다.
`requirements/requirements-showonce-foundationpose.txt`는 native build를 대신하지 않습니다.

**FoundationPose 원본 및 그 파생 부분은 비상업 연구·평가 조건입니다.**
우리 코드에 Apache-2.0을 선택하더라도 이 조건은 없어지지 않습니다.
상업 사용 가능한 전체 파이프라인이라고 표시하지 않습니다.
[원본 라이선스](https://github.com/NVlabs/FoundationPose/blob/main/LICENSE)와
[포함 고지](source-inventory/third-party-notices/FOUNDATIONPOSE-LICENSE.txt)를 확인합니다.

## SAM 3D/real2sim: 신규 메시 복원에 필요한 별도 구성

이미 준비한 에셋을 선택하는 흐름과 새 에셋을 복원하는 흐름을 구별합니다.
복원에는 SAM 3D 승인 가중치, manifest에 고정한 native 소스들과
`requirements/requirements-showonce-sam3d.txt`가 필요합니다.
`vendor/patches/`의 PyTorch3D/SAM 3D 패치를 각 upstream checkout에 적용합니다.
Kaolin `check_tensor` 호환 함수와 Apache 고지는 Studio wheel 안에 포함됩니다.

`gibeom25/real2sim`의 확인한 리비전에는 LICENSE가 없습니다. 작성자의 코드 사용·재배포 조건을
확인하기 전에는 이를 공개 기본 설치의 필수 단계로 만들거나 소스/가중치와 묶어 배포하지 않습니다.
기존 로컬 연구용 연결은 보존하며, 복원 경로의 독립 재현성은 미완료로 표시합니다.
SAM 가중치에는 별도의 SAM License가 적용됩니다.

## Robot/policy: 명시적으로 준비한 장비만

control 환경에서 `python -m pip install '../piper-runtime[robot]'`로 SDK 의존성을 준비합니다.
TCP·정규화 통계·모델·공통 Runtime 해시가 일치하는 정책 패키지와 현재 보정, 실시간 인식 폴더가 필요합니다.
실기 명령과 SDK 상태는 [로봇 제어 안내](ROBOT-CONTROL.md)를 따릅니다.
시작 자세 A 복귀는 Studio 내부 제어기를 사용합니다. 제어 환경에 `showonce-studio[motion]`을 설치하면
Ruckig·기존 QP로 동작하며 정책 가중치나 외부 실행기 경로가 필요하지 않습니다.
스크립트 파지는 [독립 실행 튜토리얼](../examples/scripted-grasp/README.md)이며 기본 정책 흐름에서 요구하지 않습니다.

## 관계 추출의 interpreter 설정

현재 `python.sam3d` 설정 키는 관계 추출과 기존 에셋 분석 orchestration에도 재사용됩니다.
기존 에셋으로 진행할 때는 이 키를 SAM worker 환경으로 지정할 수 있습니다. 관계 추출 자체는
SAM 3D 가중치를 사용하지 않습니다. 신규 메시 복원까지 수행하면 이 키에 별도 복원 환경을 지정합니다.

## 검증과 현재 한계

각 worker 환경에서 `python -m pip check`를 실행하고 upstream native import/CUDA 테스트를 수행합니다.
CPU 예제 검증은 모델 추론·카메라·SDK 검증을 대신하지 않습니다.
원본 가중치·실측 보정·시연 영상은 설치 시 자동으로 복사하거나 다운로드하지 않습니다.
SAM 새 환경 검증 결과는 [sam-worker-validation.json](sam-worker-validation.json)에 있습니다.
