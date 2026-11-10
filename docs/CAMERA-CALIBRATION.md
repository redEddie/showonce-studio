# Studio 내부 카메라 보정

카메라 보정 생성 코드는 `showonce_studio.calibration`에 있습니다. SHOW-33 checkout이나
예전 Codex 작업 폴더, `pac2026` 하드웨어 감사 폴더를 실행 시 읽지 않습니다.
기존 보정 JSON은 그대로 사용할 수 있으며, 현재 활성 보정을 자동으로 교체하지 않습니다.

## 유지한 보정 방식

1. D435i의 정렬 RGB-D 5프레임을 촬영합니다. 최종 RGB와 깊이 중앙값을 저장합니다.
2. 촬영 시점 이전에 수신한 관절각·관절 속도·그리퍼 폭을 수신 전용 SocketCAN으로 읽습니다.
   데이터가 오래되거나 팔이 움직이는 중이면 등록을 시작하지 않습니다.
3. SAM 3의 `robot arm` 마스크와 관절각을 고정한 로봇 모델을 비교합니다.
4. nvdiffrast/PyTorch3D 기반 렌더링·깊이 정합으로 카메라 자세를 구하고,
   투영한 베이스 영역을 SAM 박스 프롬프트로 보완한 뒤 다시 정합합니다.
5. 로봇 외부의 깊이 점군으로 테이블 높이·기울기·잔차를 독립 검사합니다.

일반적인 FoundationPose 물체 추적과 환경을 공유할 수 있지만 이 보정 계산은
**기존 팀 render-and-compare 최적화기**입니다. FoundationPose의 신경망 scorer/refiner나
그 가중치를 보정 계산에서 호출하지 않습니다. 로봇 이동·enable·disable·CAN 송신도 없습니다.

제조사 URDF의 팔 + Menagerie 그리퍼 조합과 그 메시를 패키지에 포함했습니다.
원본과 바이트가 같은 94개 설정/자산 파일의 출처·해시는
`provenance/camera-calibration.json`, MIT 원문은 각 모델 폴더의 `LICENSE`에 있습니다.
팀 코드의 라이선스 결정은 별도입니다. GitHub 저장소는 공개되어 있으며 [공개 범위](PUBLICATION.md)를 확인하세요.

## 환경과 설정

각 환경에 동일한 Studio/Runtime 버전을 설치합니다. SAM은 [SAM-WORKER.md](SAM-WORKER.md)의
원본 SAM + NumPy 1.x 환경을 그대로 사용합니다. 앱/제어기의 NumPy 2.x와 섞지 않습니다.

```toml
[paths]
home = "~/.local/share/showonce-studio"
calibration = "~/.local/share/showonce-studio/calib/current.json"
sam_checkpoint = "~/models/sam3/sam3.pt"

[python]
capture = "/path/to/capture/bin/python"
geometry = "/path/to/sam/bin/python"
calibration = "/path/to/native-calibration/bin/python"
```

- `capture`: NumPy/OpenCV/pyrealsense2와 Studio/Runtime. SDK는 필요 없습니다.
- `geometry`: 기존 SAM 3 이미지 추론 환경과 Studio/Runtime.
- `calibration`: CUDA PyTorch, nvdiffrast, PyTorch3D, MuJoCo, trimesh, SciPy, OpenCV,
  Studio/Runtime. 기본값은 `python.foundationpose`이므로 기존 native 환경을 재사용할 수 있습니다.
  별도 설정은 `SHOWONCE_CALIBRATION_PYTHON`으로도 가능합니다.
- Thor에서 검증한 native 버전은 `requirements/requirements-showonce-calibration.txt`에 있습니다.
  native 확장은 upstream 빌드가 필요하며 이 목록만으로 새 장비 설치가 끝나지는 않습니다.

받침대는 기존 실측 **15mm**, `table_z_in_base_m=-0.015`를 유지합니다.
받침대가 달라진 경우 로컬 JSON을 만들고 `--mount /path/to/robot_mount.json`으로 지정합니다.

## 실행

팔과 카메라를 정지시키고 팔 전체가 잘 보이게 합니다. 카메라를 점유한 capture/SAM 앱은
먼저 종료합니다. 보정과 capture는 같은 카메라 소유권 잠금을 사용합니다.

```bash
calibrate-camera --serial <D435i_SERIAL> --can can_piper
# 같은 기능, Python 모듈 진입점
python -m showonce_studio.calibration.cli --serial <D435i_SERIAL> --can can_piper
```

기존 `paths.calibration` 파일에 serial이 있으면 `--serial`은 생략합니다.
여러 로봇이 보이면 `--robot-box X0 Y0 X1 Y1`로 Piper 후보 영역을 지정합니다(848×480).
명령은 후보만 생성하며 `--out`으로 지정한 디렉터리가 이미 있으면 거절합니다.

- `<out>/capture/`: RGB-D·메타데이터·촬영 시점 관절 피드백·CAN 송신 카운터 기록.
- `<out>/registration/T_base_cam_robot.json`: 기존과 동일한 보정 형식.
- `<out>/registration/check.png`: 초록 모델·빨강 SAM 윤곽 비교.
- `<out>/registration.log`, `<out>/summary.json`: 계산 결과와 자동 검사.

통과 조건은 기존 그대로입니다. 등록기의 `trusted`, serial 일치, 테이블 높이 차이 ≤10mm,
기울기 ≤5°, RMS ≤5mm, 평면 점 ≥1000개를 검사합니다. 현재 보정은 덮어쓰지 않습니다.
검토한 후보를 Studio 실행 시 `--calib <후보 JSON>`으로 선택하거나 로컬 설정 경로를 변경합니다.
카메라/로봇 베이스 배치가 바뀌지 않았다면 매번 새 보정을 할 필요는 없습니다.

저장 입력으로 전체 알고리즘을 검사할 때는 다음과 같이 실행합니다. 이 모드는 카메라/CAN을 열지 않으며
과거 촬영본의 결과를 현재 배치에 대한 새 보정으로 간주하지 않습니다.

```bash
calibrate-camera --src /path/to/capture/camera \
  --feedback /path/to/capture/robot_feedback.json --out /path/to/new-candidate
```

운영 중인 Thor의 기존 `python calibrate_camera.py`도 같은 패키지 진입점을 호출합니다.
기존 웹앱 레이아웃과 인식/정책 실행 API는 변경하지 않았습니다.

## 검증 범위

[calibration-validation.json](calibration-validation.json)에 실제 검증 결과를 기록합니다.
계산 함수·모델 텐서·다중 자세 FK의 동등성, SHOW-33/이전 모델 폴더 접근을 차단한
저장 촬영본 전체 실행을 구분합니다. 기존 Thor의 실제 D435i 촬영→최신 관절 피드백→
SAM 두 단계→재정합→후보 자동 검사까지도 통과했고, CAN 송신 증가량은 0입니다. 이 검증을 새 장비 native 설치 또는 실제 로봇 작업 성공 검증으로 확대하지 않습니다.
