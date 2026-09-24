# Wave Growth Statistics

Wave의 일별 GitHub 스타 이력으로 **과거 예측 검증 → 모델별 미래 분포 → 대규모 몬테카를로 → 가정 민감도 비교**를 수행하는 로컬 도구.

목표는 10만 스타 같은 장기 목표를 여러 수학적 가정 아래 탐색하는 것이다. 예측 가중치나 시뮬레이션 성공 비율을 실제 언어 채택의 확률로 부르지 않는다.

## 바로 실행

Python 3.11 이상. 기존 CSV는 저장소 밖에 두어도 된다. 입력 파일이나 실행 결과를 GitHub로 보내는 기능은 없다.

```bash
git clone https://github.com/LunaStev/statistics.git
cd statistics
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
```

먼저 모델 검증만 실행한다. 미래 몬테카를로는 돌리지 않는다.

```bash
python wave_growth_forecast.py ~/star-history-2026924.csv \
  --validate-only --output results/validation
```

10만 회 시험 및 100만 회 실행:

```bash
python wave_growth_forecast.py ~/star-history-2026924.csv \
  --simulations 100000 --target 100000 --years 30 \
  --chunk-size 25000 --jobs 4 --output results/100k-test

python wave_growth_forecast.py ~/star-history-2026924.csv \
  --simulations 1000000 --target 100000 --years 30 \
  --chunk-size 25000 --jobs 6 --output results/1m-baseline
```

`python -m growth ...`와 설치 후 `wave-statistics ...`도 같은 진입점이다. 계산 횟수 `--simulations`와 목표 스타 수 `--target`은 별개다. 기본 100만 회는 **전체 혼합분포에서 100만 경로**이며 각 모델당 100만 회가 아니다.

중단한 실행은 같은 명령에 `--resume`을 붙인다. 완료한 청크는 다시 계산하지 않는다. 입력, 설정, 모델 매개변수, 소스 코드, 주요 라이브러리 버전이 바뀌면 별도의 출력 디렉터리를 사용한다. `--jobs`만 변경하는 것은 허용한다. 같은 설정·청크 크기·시드에서는 작업자 수와 실행 순서가 결과를 바꾸지 않는다.

```bash
python wave_growth_forecast.py ~/star-history-2026924.csv \
  --simulations 1000000 --target 100000 --years 30 \
  --chunk-size 25000 --jobs 6 --output results/1m-baseline --resume

xdg-open results/1m-baseline/report.html
```

원본 CSV는 복사·게시하지 않으며 Git에서 기본적으로 제외한다. 결과도 `results/` 아래에서 제외한다. 기존 v1 결과 디렉터리는 덮어쓰지 않는다.

## 비교할 실험

같은 입력과 시드로 출력 디렉터리를 분리한다.

```bash
# 데이터의 검증 점수 대신 동일한 모델 비중을 적용하는 민감도 비교
python wave_growth_forecast.py ~/star-history-2026924.csv \
  -n 1000000 --weights equal -j 6 -o results/1m-equal

# 복리 모델만 가정한 조건부 실험
python wave_growth_forecast.py ~/star-history-2026924.csv \
  -n 1000000 --model birth -j 6 -o results/1m-birth

# 복리 피드백이 영구적이지 않을 수 있다는 명시적 가정
python wave_growth_forecast.py ~/star-history-2026924.csv \
  -n 1000000 --feedback-half-life-years 3 -j 6 -o results/1m-feedback-3y
```

마지막 옵션은 **피드백 지속 시간의 반감기가 3년**이라는 사용자가 정한 가정이다. Wave 자료에서 추정한 국면 전환 확률이 아니다. 전환 이후에는 장기 포아송 유입률로 돌아가며 개발 중단을 가정하지 않는다. 그 외 모델에는 이 전환을 적용하지 않는다. 피드백 소멸 시나리오의 단기 검증 가중치는 기본 모델에서 가져온 것이므로 시나리오 자체가 검증되었다고 해석하지 않는다.

## 모델

| 이름 | 생성 과정 | 포함한 불확실성 |
|---|---|---|
| `long_poisson` | 최근 365일의 일정한 일별 유입 | Gamma 사후 유입률 + Poisson 사건 |
| `recent_poisson` | 최근 90일의 일정한 일별 유입 | Gamma 사후 유입률 + Poisson 사건 |
| `changepoint_poisson` | 최근 180일의 변화 없음 / 한 번 변화 혼합 | 변화 유무·날짜·현재 유입률 |
| `negative_binomial` | 과산포가 있는 정수 유입 | 유입률·분산 매개변수의 격자 사후분포 |
| `birth` | 현재 스타 수에 비례하는 선형 출생 과정 | 피드백 계수 사후분포 + 정수 출생 사건 |

모든 모델은 **같은 미래 기간의 정수 증가량**에 대한 확률질량으로 비교한다. 연속 확률밀도와 정수 확률을 직접 섞지 않는다. `birth`는 `S+1`을 사용하는 출생·이민 과정으로 0스타에서도 시작할 수 있다. 스타 사이의 실제 인과관계를 입증하는 모델은 아니다.

30일짜리 서로 겹치지 않는 검증 구간을 순차적으로 이동한다. 변화점과 매개변수는 매번 과거 데이터만 사용해 다시 적합한다. 최종 혼합에는 predictive stacking을 사용한다. 앙상블의 검증 성능은 해당 검증 구간보다 **이전 구간에서만 학습한 가중치**로 계산한다. 가중치에 강제 5% 하한을 두지 않는다. 이동 블록 재표본화는 가중치 민감도를 보여주며, 실제 미래에 대한 신뢰구간은 아니다.

## 출력

```text
results/1m-baseline/
├── report.html                    # 로컬 브라우저 보고서
├── run.json                       # 입력 해시·설정·버전·체크포인트 해시
├── model_diagnostics.json         # 변화점 확률, 격자 경계 질량, 사전분포
├── model_weights.csv              # 검증 가중치·동일 가중치·재표본화 민감도
├── validation_folds.csv           # 각 과거 검증 구간의 실제값과 예측
├── validation_summary.csv         # 로그 점수, 중앙값 MAE, 80% 구간 포괄률
├── summary.csv                    # 조건부/전체 도달 중앙값, 기간 내 미도달
├── target_probabilities.csv       # 연도별 목표 확률 + 해석식 교차 확인
├── milestone_probabilities.csv    # 100·250·500·1k·…·100k 도달 확률
├── forecast_quantiles.csv         # 목표에서 잘린 스타 수의 분위수
├── target_cdf.csv                 # 미도달을 포함한 누적 도달 확률
├── charts/                        # 여섯 종류의 PNG 그래프
└── chunks/chunk-000000.npz         # 압축된 경로별 결과와 재개 체크포인트
```

`hit_days_upper=-1`은 관측한 미래 기간 내 미도달, `0`은 시작 전에 이미 도달한 경우다. `given_reached_*`는 기간 내 도달한 경로만의 통계다. `all_paths_*`는 미도달을 포함한다. 절반이 기간 내 도달하지 않았다면 전체 중앙 도달일은 비어 있다.

`mc95_low/high`는 **몬테카를로 표본 오차만** 나타낸다. 0회 관측되어도 상한이 0인 것처럼 표시하지 않는다. 모델 오류·장기 가정의 불확실성은 이 구간에 포함되지 않는다. 기본 모델의 목표 도달 확률은 해석식으로도 계산하여 시뮬레이션과 대조한다. 영구 복리에서 벗어나는 가정은 별도 계산이 필요하므로 해당 해석식 값을 빈칸으로 둔다.

시간 간격은 기본 최대 30일이며 도달 시간은 구간의 **오른쪽 끝**으로 기록한다. `--step-days 7` 또는 `1`로 시간 해상도를 높일 수 있다. 연도별 체크포인트는 실제 달력 날짜를 사용한다. 12×30일을 1년으로 취급하지 않는다. 그래프의 미래 스타 수는 목표에 도달하면 멈추는 `min(S, target)`이며, 그 이후 성장이 멈춘다는 뜻이 아니다.

## 설정과 수치 검증

```bash
python wave_growth_forecast.py ~/star-history-2026924.csv \
  --config config.example.toml --validate-only -o results/config-validation

# 격자 적분 해상도 민감도 확인
python wave_growth_forecast.py ~/star-history-2026924.csv \
  --grid-size 241 --validate-only -o results/finer-grid
```

CLI 인수가 TOML보다 우선한다. `model_diagnostics.json`의 `grid_edge_mass_bound`가 크면 경계·사전분포·격자 범위를 점검한다. 실행 횟수를 늘리는 것은 몬테카를로 오차를 줄일 뿐이며, 이 수치 적분 오차나 모델 가정까지 제거하지 않는다.

입력은 `YYYY-MM-DD`와 비음수 정수 누적 스타 수다. 여러 저장소를 담은 CSV는 `--repository wavefnd/Wave`로 선택한다. 누락 날짜, 서로 충돌하는 중복 날짜, 누적값 감소는 명시적 오류로 처리한다. 첫 행의 스타 수는 초기값이며 일별 신규 유입으로 세지 않는다. 내보내기 도구가 현재 stargazer 목록에서 과거를 재구성한 경우, 취소된 과거 스타는 자료에 없을 수 있다.

## 개발

```bash
python -m pytest -q
ruff check .
```

테스트는 합성 데이터의 수학적 일관성, 미래 정보 누출, 목표 미도달 처리, 해석식과 소규모 시뮬레이션의 일치, 병렬 재현성과 재개, CLI 전체 흐름을 확인한다. 사용자 자료로 100만 회 실행하는 작업은 CI에 넣지 않는다.

자세한 식과 출처는 [통계 방법](docs/methodology.md), 이전 구현과의 차이는 [변경 기록](CHANGELOG.md)을 참고한다. Hawkes/HMM, 실제 노출·개발량·외부 프로젝트 표본 기반 채택 모델은 아직 구현하지 않았다.
