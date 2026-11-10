# 기존 작업 폴더 의존성 최종 점검 — 2026-11-05

**새 설치에서 Studio/Runtime 자체 코드를 실행하는 데 기존 `piper_ws`, SHOW-33,
이전 Codex 작업 폴더나 학습 checkout은 필요하지 않습니다.**
두 저장소 설치 후 사용할 기능의 외부 패키지/원본 코드, 가중치와 사용자 데이터를 준비하고 로컬 설정으로 연결합니다.
모델·외부 원본 코드까지 두 저장소에 포함했다는 뜻은 아닙니다.

| 구분 | 현재 상태 |
|---|---|
| 수집·VLM/인식 orchestration·관계 추출·GUI | Studio 패키지 내부 |
| 실시간 큐브/그릇 인식·카메라 보정·자세 A 복귀 | Studio 내부, 이전 실행기/SHOW-33 코드 불필요 |
| 스크립트 파지 | 독립 터미널 튜토리얼, 이전 스크립트와 웹앱 불필요 |
| 정책 추론·QP·기구학·관절/간격 모델 | Runtime 패키지 내부 |
| SAM/VLM/SDK/Ruckig 등 | 각 Python 환경에 원본 패키지 설치 |
| 참조 에셋 포즈 추적 | FoundationPose 원본 코드·native 확장·scorer/refiner 필요, `paths.third_party` 아래에 준비 |
| 신규 메시 복원 | SAM 3D + real2sim 원본 코드/가중치 필요, `paths.third_party`와 모델 경로로 연결 |
| 보정·에셋 보관함·배포 정책·촬영 데이터 | Git 제외 사용자 데이터. 새로 만들거나 별도 복사/다운로드 |

원본 외부 코드의 리비전과 설치 범위는 [WORKERS.md](WORKERS.md),
[external-dependencies.json](external-dependencies.json)에 있습니다.
FoundationPose 또는 신규 메시 복원 기능을 사용할 때 그 원본 코드는 필수입니다.

## 실제 Thor 배치에서 남아 있는 경로

현재 운영 환경은 유지했습니다. 로컬 설정은 Git에 포함되지 않습니다.

- `paths.home`, worker Python, 데이터 경로는 기존 `piper_ws` 배치를 사용합니다.
- SAM checkpoint는 기존 팀 모델 디렉터리의 파일을 사용합니다.
- FoundationPose의 `weights`는 기존 팀 가중치 디렉터리를 가리키는 심볼릭 링크입니다.
- 기존 FoundationPose venv의 `.pth`는 다른 기존 geometry venv의 site-packages를 공유합니다.

이들은 현재 머신의 환경/데이터 연결입니다. 기존 폴더 없이 **새로 설치**할 때에는
새 worker 환경을 설치하고 원본 가중치를 새 위치에 준비해 `configs/example.toml`을 설정합니다.
기존 venv나 그 심볼릭 링크를 그대로 복사하는 방법은 독립 설치로 보지 않습니다.
기존 에셋·정책·보정·촬영 기록도 새 저장소 clone만으로 자동 복원되지 않습니다.

`compat/thor/`와 과거 실기 기록에는 기존 배치 경로가 남아 있습니다.
이는 운영 호환/출처 기록이고, 일반 설치는 `showonce-studio`, `showonce-robot`,
`showonce-policy`, `showonce-scripted`, `calibrate-camera` 패키지 진입점을 사용합니다.

## 검사와 수정

- 두 패키지 소스의 이전 절대 경로·외부 실행기 import·소스 심볼릭 링크를 검사했습니다.
- 설치 wheel을 사용하고, 별도로 추출한 저장소 스냅샷·빈 모델/third_party 디렉터리·새 데이터 위치에서
  합성 예제를 만들고 네 페이지를 검사했습니다.
- 각 검사 자식 프로세스에서 기존 작업 디렉터리 접근과 SDK/카메라 모듈 import를 차단합니다.
- 이전에 제거된 setup shell script/flat Python 파일을 가리키던 설치 안내를 현재 모듈 명령/문서로 수정했습니다.
- SAM 3D의 기본 가중치 경로도 다른 모델처럼 `paths.models`를 따르게 했습니다.

결과는 [path-audit-validation.json](path-audit-validation.json)에 기록합니다.
위 새 위치 검사는 CPU/offline 검사입니다. GPU 인식·실제 카메라·실기 동작의 새 장비 전체 설치 검증으로 확대하지 않습니다.
기존 인식·보정·자세 A의 코드 분리 검증은 각 기능의 검증 문서에 따로 기록되어 있습니다.

다른 설치에서도 기존 폴더 접근을 막아 검사할 수 있습니다.

```bash
python scripts/smoke_cpu.py --deny-path /old/workspace --report /tmp/path-check.json
```
