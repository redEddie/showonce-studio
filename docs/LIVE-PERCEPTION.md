# Studio 내부 실시간 인식

실시간 큐브·그릇 인식은 이제 Studio 패키지에 포함됩니다.
`show33`, `live_shadow.py`, 외부 `inspect_perception.py`를 import하거나 `sys.path`에 추가하지 않습니다.
카메라는 기존 capture 노드 하나가 소유하고, 인식은 그 노드의 RGB-D를 구독합니다.

## 유지한 동작

- 같은 SAM image model, `green cube`/`striped bowl` 프롬프트, confidence threshold 0.4.
- 측정 시점의 RGB-D 프레임을 보관하는 retained-anchor 방식.
- RGB 기반 큐브 추정과 그릇 silhouette/테이블 평면 추정, 두 추정기의 병렬 실행.
- 크기·형상·색상·재인식 유효 시간·경계 관측 확인 기준.
- `image_cube_bowl_v4` revision과 `piper_base_link`/m 단위.
- `tracked_poses.json`, `object_poses.json`, `diagnostics.json`, `latest.jpg/png` 형식과 원래 관측 시각.
- SAM은 별도 프로세스에 유지되며 웹 새로고침/정책 교체로 모델을 다시 로드하지 않음.

기존 실기에서 사용하던 image 경로를 이식했습니다. 과거 depth-only 대체 경로와 로봇 제어/보정 생성 코드는 옮기지 않았습니다.
출처·원본 파일 해시는 [provenance/live-perception.json](../provenance/live-perception.json)에 있습니다.

## 필요한 파일과 환경

1. [원본 SAM/NumPy 1.26 worker 환경](SAM-WORKER.md).
2. 승인된 `sam3.pt`와 현재 카메라 배치의 보정 JSON.
3. 실행 중인 D435i capture 노드. 인식 worker는 USB나 CAN을 직접 열지 않습니다.

보정 JSON은 데이터 파일입니다. 기존 보정을 `SHOWONCE_HOME/calib/current.json` 등 독립 위치에 복사해서
사용하면 됩니다. 보정 내용이나 좌표계는 바꾸지 않습니다. 보정 생성 도구는 아직 별도 작업입니다.
기본 물체 정의는 wheel 내부 `perception/data/objects_task.json`이며 큐브 50mm, 기존 줄무늬 그릇 치수를 보존합니다.
다른 설정은 `[live_vision].objects` 또는 `--objects`로 지정합니다.

```toml
[paths]
home = "~/.local/share/showonce-studio"
calibration = "~/.local/share/showonce-studio/calib/current.json"
sam_checkpoint = "~/models/sam3/sam3.pt"

[live_vision]
autostart = true
python = "/absolute/path/to/showonce-studio/.venv-sam/bin/python"
# objects = "/absolute/path/to/custom_objects.json"
```

`inspector`, `show33`, `SHOWONCE_LIVE_INSPECTOR`, `SHOWONCE_SHOW33`는 실시간 인식에 사용하지 않습니다.
시작 자세 A 복귀도 Studio 내부 제어기이며 외부 Home 실행기를 설정하지 않습니다.

직접 실행하려면 다음과 같이 호출합니다. capture와 calibration의 카메라 serial이 일치해야 합니다.

```bash
SHOWONCE_CONFIG=/absolute/path/to/local.toml .venv-sam/bin/python -m showonce_studio.showonce_live_vision \
  --capture-url http://127.0.0.1:8768 \
  --calib /absolute/path/to/calib/current.json \
  --checkpoint /absolute/path/to/sam3.pt \
  --out /absolute/path/to/new-perception-run
```

출력 폴더는 새 경로를 지정합니다. 종료는 Ctrl+C이며 소유한 인식 스레드·SAM 자식 프로세스를 정리합니다.
기존 capture 노드는 인식 프로그램 종료로 닫지 않습니다.
worker는 자신의 설치 패키지를 사용하며 GUI 환경의 `PYTHONPATH`를 전달받지 않습니다. Python 실행 파일의
venv 심볼릭 링크도 그대로 유지하므로 NumPy 2 앱과 NumPy 1 SAM 환경이 섞이지 않습니다.
이전 외부 인식 프로세스가 이미 실행 중이면 먼저 종료한 후 새 인식기를 시작합니다.

## 검증

[live-perception-validation.json](live-perception-validation.json)에 범위를 기록했습니다.

- 기존 코드와 옮긴 추정기/추적 상태 로직의 AST 비교.
- 과거 저장 RGB-D/SAM 마스크 42개 사례에서 채택·거부 및 좌표 비교.
- SHOW-33 및 외부 인식 모듈 접근을 차단한 상태에서 실제 D435i→SAM→JSON 파일 검사.
- 새 위치 측정의 프레임/시각/단위 유지, SDK/CAN import 없는 기본 경로와 서비스 실행 명령 회귀 검사.

실제 카메라 검사 당시에는 큐브가 검출되지 않았고 그릇 마스크도 형상 기준을 통과하지 않았습니다.
따라서 실시간 두 물체의 완전한 좌표 관측이나 독립 위치 정확도 검증으로 해석하지 않습니다.
저장 관측의 비교는 두 추정기가 같은 입력에 같은 결과를 내는지 검사한 것입니다.

## 아직 별도로 남은 항목

카메라 보정 생성도 Studio 내부로 이식했습니다([보정 안내](CAMERA-CALIBRATION.md)).
FoundationPose 설치 재현 및
프로젝트/팀 기여 코드의 라이선스 결정은 별도입니다. GitHub 저장소는 공개되어 있으며 [공개 범위](PUBLICATION.md)를 확인하세요.
