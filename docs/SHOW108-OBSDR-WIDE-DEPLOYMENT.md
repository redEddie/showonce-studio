> 현재 최종 기준은24배입니다. [최종 배포 안내](FINAL24-INTEGRATION.md)를 먼저 보세요. 아래 내용은 이전 모델의 배포·실기 이력입니다.

# SHOW-108 관측 DR·넓은 스폰 실기 준비

2026-11-01, 정책 ID **ee9da8b8f493eaa56c3c**.
Studio04에서 **SHOW-108 · 관측 DR · 넓은 범위 · 14+2배**를 선택한다.
원본16배 정책 `e62899a3bc91c4c241d6`과 SHOW-100 연결 검증용 정책은 그대로 보존한다.

## 원본과 내보내기

- [Release](https://github.com/redEddie/piper-sim-lab/releases/tag/show108-mlp-lstm-obsdr-wide-14plus2u-20261101)
- 게시 source `9b006cf9e2b8bb63b0b0ada98c5368f0013e1dbf`, 학습 source `0f5aa042416f49c7a8839396179e230a0e853242`.
- learner SHA256 `19438a198e31345d04e4234073028c1e0d9f20b48b90f9cb917af48b55f09c2c`.
- config SHA256 `0a083727cb5f14a2a4bf923ce53e318942f1c5ffe5443da9b96f2fd5dadfab2a`.
- 추가16,777,216전이, 부모117,440,512전이, 누적134,217,728전이. 체크포인트 폴더 step은 추가량이다.

Release 파일12개(SHA256SUMS 자체 포함)를 다운로드하고, 명세에 있는11개 파일 및 압축 내부11개 파일 해시를 검증했다.
새 learner의 Flax loader에서 정책 가중치와 **갱신된 normalizer mean/std**를 같이 내보냈다.
기존 normalizer를 재사용하지 않는다. 런타임에서 실측 입력에 인위적인 DR 잡음을 추가하지 않는다.
내보내기 재현은 `scripts/export_show108.py --release obsdr-wide --source-repo ... --run ... --output ...`를 사용한다.
기존16배 내보내기는 `--release 16u`로 보존하며 두 Release의 source/learner/config 해시를 각각 고정 검사한다.

Thor 원본은 `releases/show108-mlp-lstm-obsdr-wide-14plus2u-20261101/`,
격리 source는 `vendor/show108-obsdr-wide-source/`, 파생 패키지는 `artifacts/show108-obsdr-wide-14plus2u-v1/`에 있다.
체크포인트/영상/모델 파일은 Git에 추가하지 않는다.

## 실기 입력·제어

| 항목 | 값 |
|---|---|
| 구조 | actor49 → MLP256/64 → LayerNorm → LSTM128 → 선형14, 행동7 |
| 큐브·그릇 초기 XY | X100–500mm, Y−300–300mm |
| 반경·최소 분리 | 600mm / 140mm |
| TCP | 138.5mm |
| 관절 속도/가속도/jerk | 1rad/s / 5rad/s² / 100rad/s³ |
| SDK 그리퍼 force | 0.5 |
| 주기 | 정책10Hz / 제어200Hz |
| TCP 하한 | 실기 테이블+10mm |
| 에피소드 | 최대20초, 초과 요청은 SDK 연결 전에 거부 |

초기 배치 검사는 **선택한 정책의 environment**에서 읽는다. 이전 모델의 범위를 완화한 것이 아니다.
사각형의 모든 점과 모든 자세에서 도달 가능함을 보장하지 않는다. 기존 QP·관절·테이블 검사는 유지한다.
파지 후 위치 갱신은 기존 `showonce_scene_state_v2`를 유지한다. 상태 예측 horizon은 학습 에피소드20초와 맞췄다.
팔 고정 규칙이나 정책 행동 덮어쓰기를 새로 추가하지 않았다. 20초 뒤 자동 A 복귀를 지시하지 않는다.

## 실행

Studio04에서 새 모델 선택 → 큐브·그릇 최신 관측 확인 → Enable/Home → 실제 실행 준비 → run.
새 정책 직접 링크: `http://localhost:8765/policies?policy=ee9da8b8f493eaa56c3c`.
과거 정책 선택은 유지되므로 이전 탭에서는 새 모델을 명시적으로 선택한다.

```bash
cd ~/piper_ws
source env.sh
python showonce_policy.py run --policy-id ee9da8b8f493eaa56c3c --seconds 10 \
  --calib /home/pi_lab/songjisu/show33/runs/snap-capture-20261028-195434/T_base_cam_robot.json \
  --perception-dir /home/pi_lab/piper_ws/runs/show100-vision-studio-20261101-152530-63be78
```

위 Thor 경로는 현재 장비 구성 기준이며 카메라 배치가 바뀌면 그 배치에 맞는 보정·인식 경로를 사용한다.
SAM·촬영 노드는 재로딩하지 않는다. Studio는 새 모델명·관측 DR 여부·스폰 범위·누적 학습량을 표시한다.

## 검증 범위

원본 actor1,024입력에서 최대 행동 차이9.54e−7, 원본49차원 관측64사례에서 최대 차이4.77e−7, QP32사례의 채택 여부가 일치했다.
CPU mock과 웹 경유 추론/mock이 완료됐으며 모두 실기 명령은 없었다. 관측·내보내기·프로세스 관리 회귀 검사를 수행했다.
브라우저에서는 신구 모델 선택, 복사 명령의 새 ID,20초 상한, 모바일 표시를 확인했다.
이후 2026-11-01 실제 10초 실행을 촬영해 큐브 파지·운반·그릇 안 배치를 시각 확인했다.
단일 시도이므로 성공률 검증과 구분한다. [영상·실행 기록](REAL-ROBOT-DEMO-20261101.md)을 참고한다.

Release의 시뮬레이션 결과는 초기A→배치76.6%, A까지 전체 완료5.5%이며 A 복귀는 남은 과제다.
기존 DR 없는16배 모델과 평가 조건이 다르므로 단순 성공률 비교를 하지 않는다.
