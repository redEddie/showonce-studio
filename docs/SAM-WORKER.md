# SAM worker: 원본 모델과 NumPy 1.x 환경

SAM은 물체·손 검출/추적에 필수입니다. 원본 모델이 요구하는 NumPy 조건을 변경하지 않고,
Studio·Runtime과 **서로 다른 Python 환경**에서 실행합니다.

| 환경 | 사용 구성 | 역할 |
|---|---|---|
| Studio·정책 실행 | NumPy 2.5.4, OpenCV 5, SciPy 1.18 | 웹 UI·작업 관리·정책/제어 |
| SAM worker | NumPy 1.26.4, OpenCV 4.11.0.86, SciPy 1.15.3 | 원본 SAM 0.1.0의 마스크 검출/추적 |

두 환경에서 같은 Studio 소스를 설치하되 서로 다른 dependency extra를 선택합니다.
SAM 환경에도 필요한 우리 Runtime의 프레임·좌표 유틸리티는 NumPy 1.26.4와 2.5.4 양쪽에서 검사했습니다.
우리 공통 패키지의 불필요한 NumPy 2 전용 제한을 완화했으며, 정책/제어 환경의 NumPy는 그대로 유지합니다.

## 설치

Linux aarch64 / NVIDIA Thor / Python 3.12 / CUDA 13.0 PyTorch로 검증한 절차입니다.
Studio와 Runtime을 형제 디렉터리에 clone한 뒤 실행합니다. 현재 저장소는 GitHub에 공개되어 있습니다. [공개 범위](PUBLICATION.md)를 확인하세요.

```bash
cd /path/to/showonce-studio
python3.12 scripts/setup_sam_worker.py
```

- 별도 `.venv-sam`을 만들며 system-site-packages를 상속하지 않습니다.
- SAM은 `.workspace/sam3-upstream`에 고정 리비전 `7f473cfd983d5eb43040ff608f66bff0d7bdcb53`으로 받습니다.
- **SAM 원본 코드·pyproject·패키지 버전·가중치를 수정하지 않습니다.**
- NumPy 2가 설치된 기존 venv나 수정된 SAM checkout을 지정하면 덮어쓰지 않고 중단합니다.
- `--python`, `--venv`, `--source`, `--runtime`으로 별도 경로를 지정할 수 있습니다.
- `pip check`와 Runtime 내용 해시 검사를 수행합니다. 의존성을 무시한 설치는 하지 않습니다.

전체 worker 버전은 `requirements/sam-worker-thor-cu130.txt`에 있습니다. 이 프로필을
Studio/control 환경에 설치하지 않습니다. 가중치와 개인 영상은 자동 다운로드/배포하지 않습니다.

## 추론 검사

```bash
.venv-sam/bin/python scripts/verify_sam_worker.py \
  --checkpoint /path/to/approved/sam3.pt \
  --rgb-dir /path/to/recording/rgb \
  --report .workspace/sam-worker-check.json
```

원본 SAM 0.1.0과 NumPy 1.x인지 먼저 검사합니다. 실제 Studio GeometryEngine으로 모델을 로드하고
녹화된 PNG의 처음 4개 프레임에서 `green cube`, `bowl`을 검출·추적합니다.
카메라·SDK/CAN은 열지 않습니다. 해당 물체가 없는 영상은 검증 입력으로 적합하지 않습니다.

## Studio와 연결

기존 HTTP 요청 및 JSON/PNG 결과 형식을 그대로 사용합니다. Python 배열이나 모델을 환경 사이에 전달하지 않습니다.
`[python].geometry`에 SAM 환경을 지정합니다. 기존 에셋의 관계 추출만 수행할 때는 현재 재사용되는
`[python].sam3d` 키에도 같은 환경을 지정할 수 있습니다. 신규 메시 복원은 별도 구성입니다.

```toml
[python]
geometry = "/absolute/path/to/showonce-studio/.venv-sam/bin/python"
sam3d = "/absolute/path/to/showonce-studio/.venv-sam/bin/python"
```

NumPy 2의 Studio 환경에서 아래 검사로 실제 HTTP→SAM→파일 경로를 확인할 수 있습니다.

```bash
python scripts/verify_sam_ipc.py \
  --worker-python /absolute/path/to/showonce-studio/.venv-sam/bin/python \
  --checkpoint /path/to/approved/sam3.pt \
  --recording /path/to/complete/recording \
  --report .workspace/sam-ipc-check.json
```

임시 녹화 사본·임시 포트에서 SAM 노드를 실행하고 검사 후 종료합니다. 기존 서비스와 원본 파일은 건드리지 않습니다.
검증 결과는 [sam-worker-validation.json](sam-worker-validation.json)에 있습니다.
실시간 인식에는 현재 보정·capture 노드가 필요합니다. 인식 코드는 Studio에 포함되며 SHOW-33 폴더를 사용하지 않습니다.
[실시간 인식 안내](LIVE-PERCEPTION.md)를 따릅니다.

## Thor의 CUDA 관련 주의점

PyTorch 2.10.0+cu130, torchvision 0.25.0+cu130을 사용합니다. Thor sm_110a를 지원하는
`ptxas`가 필요하며 `/usr/local/cuda/bin/ptxas` 13.2.86으로 검사했습니다.
다른 경로는 `TRITON_PTXAS_BLACKWELL_PATH`로 지정합니다.
검사 스크립트와 Studio worker는 시스템 CUDA 13.2 라이브러리가 PyTorch cu130보다 먼저 로드되지 않도록
정리된 환경에서 실행합니다.

이전부터 확인된 NVIDIA cuSPARSELt 0.8.0 wheel의 잘못된 `sbsa` metadata tag는
`repair_cusparselt_tag.py`로 교정합니다. 원본 SHA256와 ARM64 ELF를 검사하고 라이브러리·라이선스의
모든 바이트가 유지되는지 비교합니다. 이 문제는 SAM/NumPy와 별개이며, SAM에는 패치를 적용하지 않습니다.
`--no-deps`는 교정할 wheel 다운로드에만 사용하며 설치 의존성을 무시하지 않습니다.

이전 NumPy 2용 SAM 패치는 채택하지 않습니다. 과거 실험은 Git 이력에만 남고 현재 설치는 원본 SAM 조건을 따릅니다.
