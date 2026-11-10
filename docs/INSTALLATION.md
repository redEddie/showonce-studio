# 설치: CPU 확인 후 전체 파이프라인의 필수 worker 구성

Studio와 Runtime은 GitHub에서 공개되어 있습니다. [공개 범위와 라이선스 상태](PUBLICATION.md)를 확인하세요.
Linux / Python 3.12를 기준으로 합니다. 학습 저장소는 Studio 설치에 필요하지 않습니다.
GPU·카메라·로봇 환경을 한 Python 환경으로 합칠 필요가 없습니다.
아래 CPU 예제는 첫 확인 단계입니다. 전체 파이프라인에는 촬영·VLM·SAM·포즈/관계·정책 실행의 필수 worker가 필요합니다.

## 1. 두 저장소와 새 환경

```bash
mkdir -p ~/showonce-work
cd ~/showonce-work
git clone https://github.com/redEddie/piper-runtime.git
git clone https://github.com/redEddie/showonce-studio.git
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install ./piper-runtime
python -m pip install -c ./showonce-studio/requirements/cpu-verified-20261105.txt './showonce-studio[analysis,test]'
python -m pip check
python -m piper_runtime.identity --lock ./showonce-studio/piper-runtime.lock.json
```

Runtime은 PyPI에서 받지 않고 위 형제 checkout에서 설치합니다. Studio는 버전 **0.5.0**과
소스·모델 내용 해시를 함께 검사합니다. 일치하지 않으면 lock을 임의로 덮어쓰지 않고 같은
배포 조합을 사용합니다. Runtime의 C++ CPU planner에는 C++17 컴파일러(`g++` 또는 `clang++`)가 필요합니다.

## 2. 설정과 합성 예제

```bash
cd ~/showonce-work/showonce-studio
cp configs/example.toml configs/local.toml
export SHOWONCE_CONFIG="$PWD/configs/local.toml"
# 기존 촬영 데이터를 건드리지 않는 예제 전용 데이터 위치
export SHOWONCE_HOME="$HOME/.local/share/showonce-studio-example"
python examples/make_demo.py
showonce-studio --offline --port 8765
```

`http://127.0.0.1:8765`에서 수집 → 스테이지 분할 → 객체 관계 정의 → 정책 실행 페이지를 확인합니다.
종료는 Ctrl+C입니다. `--offline`은 카메라·SAM·VLM·실기 제어를 시작하지 않습니다.
예제는 합성 RGB/가짜 깊이/미학습 zero policy이며 실제 모델 추론이나 작업 성공 결과가 아닙니다.
같은 예제를 다시 만들면 기존 파일 보호를 위해 중단합니다. 다른 `SHOWONCE_HOME`을 선택하세요.

wheel 설치에서는 저장소 옆 설정을 자동으로 찾는다고 가정하지 말고 `SHOWONCE_CONFIG`를 지정합니다.
화면의 JavaScript/Three.js/ECharts는 패키지 안에서 제공되며 Node.js나 CDN은 기본 실행에 필요하지 않습니다.

## 3. 전체 파이프라인의 필수 worker 설치

[기능별 필수 worker 안내](WORKERS.md)에 따라 사용할 작업의 필수 구성을 준비합니다.
물체·손 추적의 SAM은 [고정 설치 절차](SAM-WORKER.md)로 **별도의 NumPy 1.26 환경**에 설치합니다.
Studio/control의 `analysis` 프로필과 SAM의 `perception` 프로필을 한 환경에 같이 설치하지 않습니다.

| 필요한 기능 | 추가 환경/자료 |
|---|---|
| D435i 녹화 | capture Python, librealsense/pyrealsense2, OpenCV, PyAV, 실제 카메라 |
| 자동 스테이지 분할 | VLM Python, CUDA PyTorch, Qwen 가중치 |
| 저장 영상 SAM 추적·손/관계 분석 | geometry Python, SAM 3, Open3D, 현재 영상 보정 |
| 참조 메시의 6D 포즈 추적 | FoundationPose Python, CUDA native extensions, scorer/refiner, 참조 에셋 |
| 메시 신규 복원 | SAM 3D Python과 가중치, real2sim 사용 조건 확인 |
| SDK/정책 실기 실행 | robot/policy Python, Runtime robot extra, CAN, 실시간 인식, 현재 보정·정책 패키지 |

`requirements-showonce-*.txt` 파일은 **기존 Thor에서 기록한 추가 의존성**입니다.
새로 검증한 SAM 프로필은 `sam-worker-thor-cu130.txt`와 설치 스크립트로 구별합니다. 새 CUDA/x86 환경에서
그 파일 하나만 설치하면 모든 모델이 준비된다는 뜻이 아닙니다. 가중치 승인·native build·프로필별
`pip check`와 실제 입력 검증을 별도로 진행합니다. 현재 새 환경 검증 범위는
[공개 준비 상태](PUBLIC-RELEASE.md)에 기록합니다.

## 4. 개발 검사

```bash
cd ~/showonce-work/showonce-studio
python -m pytest -q
```

테스트는 저장 fixture와 가짜 SDK를 사용합니다. Open3D 전용 TSDF 검사는 `geometry` extra가 없으면
사유를 표시하고 skip합니다. 전체 geometry 검사는 별도 환경에서 `.[analysis,geometry,test]` 설치 후 실행합니다.
설치 wheel의 네 페이지·합성 예제를 반복 확인하려면 `python scripts/smoke_cpu.py`를 실행합니다.
이 검사는 임시 데이터/임시 포트의 offline 웹만 열고 종료합니다.

CPU 의존성 버전 기록은 `requirements/cpu-verified-20261105.txt`이며 검증 플랫폼의 constraints입니다.
다른 아키텍처에서 동일 버전 wheel 제공 여부는 별도로 확인합니다.

 Runtime 기본 검사에는 GPU/JAX/실제 로봇이 필요하지 않습니다.
공개 전 라이선스 결정은 [라이선스 제안](LICENSE-PROPOSAL.md), 포함·외부 의존성 고지는
[THIRD_PARTY.md](../THIRD_PARTY.md)를 따릅니다.

## 장비 환경의 전제

GPU 기능은 해당 장비에서 CUDA/PyTorch와 RealSense/librealsense가 동작하는 것을 전제로 합니다.
그 위에 Studio/Runtime을 설치하고, 필요한 모델 가중치를 원본 배포처에서 준비한 뒤
`configs/example.toml`의 worker Python·모델·보정 경로를 설정합니다.
SAM은 원본이 요구하는 NumPy 1.x 환경을 따로 쓰고, 앱/제어기의 NumPy 2.x와 HTTP/파일로 연결합니다.

실시간 인식과 카메라 보정 코드는 Studio에 포함되어 SHOW-33 폴더가 필요하지 않습니다.
보정 방식·실행 명령은 [CAMERA-CALIBRATION.md](CAMERA-CALIBRATION.md)에 있습니다.
시작 자세 A 복귀도 Studio 내부 코드입니다. 제어 환경에는 `piper-runtime[robot]`과
`showonce-studio[motion]`을 설치합니다. 외부 Home/스크립트 실행기 경로는 사용하지 않습니다.
스크립트 파지는 [웹앱 없는 튜토리얼](../examples/scripted-grasp/README.md)로만 제공합니다.
새 장비 전체 설치 검증 여부는 CUDA/RealSense 준비 여부와 구분합니다.

기존 폴더와 별도 설치 항목의 구분 및 현재 Thor 환경의 잔여 연결은
[기존 경로 점검](PATH-AUDIT.md)을 참고합니다.
