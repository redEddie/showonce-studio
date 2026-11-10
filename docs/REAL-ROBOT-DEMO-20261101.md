# SHOW-108 단일 실기 영상 기록 — 2026-11-01

관측 DR·넓은 스폰 범위 MLP+LSTM 정책 `ee9da8b8f493eaa56c3c`를 Piper에서 10초 실행하고
D435i RGB 영상을 촬영했습니다. 영상에서 큐브 파지 → 들어올림 → 그릇 위 운반 → 개방 →
그릇 안 배치를 확인했습니다. 마지막 프레임에서도 큐브가 그릇 안에 있습니다.

- 실행 완료, 정책 추론 98회, 런타임 오류 없음.
- 영상 848×480, 523프레임, 약 17.45초. 실행 전후 구간 포함.
- 파지 중 실측 그리퍼 폭 약 52.2mm, 파지 후 큐브 위치는 TCP 상태 추정으로 갱신.
- 관절 속도 1rad/s, 가속도 5rad/s², jerk 100rad/s³, SDK force 0.5, TCP 138.5mm.
- 단일 시도의 시각 확인이며 성공률 평가는 아닙니다. 자세 A까지 완전 복귀는 검증하지 않았습니다.

정책 가중치·정규화 통계의 원본은
[piper-sim-lab Release](https://github.com/redEddie/piper-sim-lab/releases/tag/show108-mlp-lstm-obsdr-wide-14plus2u-20261101)에 있습니다.
배포 상태 추정은 `showonce_scene_state_v2`이며 정책 가중치와 QP를 변경하지 않았습니다.
실행 조건·파일 해시는 [최소 실행 기록](show108-real-demo-summary.json)에 있습니다.

## 보관 파일

[Studio 실기 기록 Release](https://github.com/redEddie/showonce-studio/releases/tag/show108-obsdr-wide-real-demo-20261101):

- `show108-obsdr-wide-real-pick-place-20261101.mp4`: 실기 영상.
- `show108-real-demo-summary.json`: 실행 조건·해시·검증 범위.
- `showonce-thor-operating-sources-20261101.tar.gz`: 당시 Thor 평면 배치 코드와 기존 Home/인식 보조 소스.
- `operating-source-inventory.json`: 운영 소스 경로·개별 SHA256 목록.
- `SHA256SUMS`: 위 네 파일의 SHA256.

운영 소스 사본은 머신 경로가 남아 있는 참고 기록이며 독립 설치 패키지가 아닙니다.
표준 패키지 소스는 Studio와 Runtime 저장소에서 관리합니다.
원본 RGB-D 데이터, 모델 캐시, 로컬 보정, 인증정보와 외부 프로젝트는 업로드하지 않았습니다.

정리 후 ShowOnce 웹·카메라·실시간 인식/SAM·씬·형상 노드를 종료했습니다.
종료 과정에서 로봇 SDK 명령을 보내거나 모터 상태를 변경하지 않았습니다.
