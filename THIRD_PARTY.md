# 외부 코드·모델 고지

자체 코드 라이선스는 아직 미확정입니다. 아래 고지를 자체 코드의 라이선스로 덮어쓰지 않습니다.
모델 가중치·학습 데이터·녹화 원본은 이 저장소에 포함하지 않습니다.

| 구성 | 저장소에 포함된 범위 | 조건/고지 |
|---|---|---|
| Three.js | `src/showonce_studio/web/vendor/three/` | MIT, 내부 LICENSE 및 개별 모듈 고지 보존 |
| Apache ECharts | web/vendor의 JS 및 고지 | Apache-2.0, LICENSE/NOTICE/d3 고지 보존 |
| Kaolin check_tensor | `vendor/kaolin-check-tensor/` 및 wheel 내부 사본 | 해당 함수의 Apache-2.0 원문·출처 보존; 전체 Kaolin의 조건으로 일반화하지 않음 |
| FoundationPose | 로컬 호환 patch 및 LICENSE 사본, 본체·가중치 미포함 | NVIDIA 원본의 비상업 연구·평가 조건. patch에도 upstream 관련 조건을 유지 |
| SAM 3 | 원본 source를 별도 설치, 본체·가중치 미포함 | SAM License. 의존성/모델 코드/버전/가중치를 변경하지 않음 |
| SAM 3D/PyTorch3D | 호환 patch, 본체·가중치 미포함 | 각각 upstream 조건 적용. SAM 및 PyTorch3D 고지는 `vendor/licenses/`에 보존 |
| Piper 모델·제어 커널 | 별도 Runtime 패키지 의존성 | Runtime의 모델 MIT 고지와 자체 코드 라이선스 범위 구별 |
| Ruckig | 별도 의존성, 본체 미포함 | MIT. 자세 A의 시간 최적 참조에 0.19.4 사용 |
| pyAgxArm/python-can | 선택 SDK 의존성, 본체 미포함 | LGPL-3.0 관련 고지와 소스/재배포 조건은 해당 upstream 적용 |
| Qwen | 어댑터만 포함 | 선택한 Qwen 모델의 Apache-2.0; 가중치 별도 다운로드 |
| Kanana/Gemma | 선택 어댑터만 포함 | 각각 자체 약관. `docs/source-inventory/model-licenses/`와 모델 링크 확인 |
| real2sim | import 연결만 포함, 본체 미포함 | 확인 리비전 LICENSE 없음. 공개 설치·재배포 권한 미확정 |
| 팀 실시간 인식 코드 | 필요한 SAM/추적 lifecycle·이미지 추정기·물체 정의를 Studio 내부에 포함 | 출처와 해시는 `provenance/live-perception.json`. 프로젝트 공개 라이선스는 아직 미확정 |
| 팀 카메라 보정 코드 | Studio 내부 render-and-compare·SAM 분할·보정 orchestration | 출처/해시는 `provenance/camera-calibration.json`. 프로젝트 공개 라이선스 미확정 |
| 보정용 Piper 모델 | 제조사 URDF 팔 메시 + Menagerie 그리퍼 모델, 원본 파일 포함 | 모델별 MIT 원문을 `calibration/data/robot/*/LICENSE` 및 wheel 고지에 보존 |
| 팀 자세 A 복귀·스크립트 파지 코드 | 내부 자세 제어기와 독립 튜토리얼 | 출처/해시는 `provenance/posture-and-scripted.json`. 공개 라이선스 미확정 |

정확한 소스 고정값·URL·프로필은 [external-dependencies.json](docs/external-dependencies.json),
설치는 [WORKERS.md](docs/WORKERS.md)에 있습니다.
`docs/source-inventory/`와 기존 requirements는 과거 Thor 환경의 기록이며 이 표와 최신 설치 안내를 우선합니다.

소스 저장소에 있는 `vendor/patches/`는 upstream 코드의 일부 문맥을 포함합니다.
공개 소스·sdist·wheel·별도 Release를 각각 확인하며, 런타임에서 외부 코드를 import한다는 이유만으로
해당 외부 코드의 권리가 자체 코드 라이선스로 변경되지는 않습니다.
