"""사람이 읽는 결과는 한국어, 기존 기계용 CSV/JSON 스키마는 유지한다."""
from pathlib import Path

import pandas as pd

MODELS = {
    "ensemble": "전체 조합", "prequential_ensemble": "시점별 예측 조합",
    "long_poisson": "장기 일정 유입", "recent_poisson": "최근 일정 유입",
    "changepoint_poisson": "유입률 변화 모형", "negative_binomial": "불규칙 유입 모형",
    "birth": "스타 비례 성장", "equal_weight_sensitivity": "동일 비중 비교",
    "baseline": "기존 유입만", "ecosystem": "생태계 조건 추가",
}
LABELS = {
    "model": "모형", "group": "구분", "simulations": "가상 미래 수", "fold": "검증 구간",
    "train_end": "학습 종료일", "test_end": "검증 종료일", "observed_gain": "실제 증가량",
    "log_score": "예측 로그 점수", "mean_log_score": "평균 예측 점수", "windows": "검증 횟수",
    "median_prediction_MAE": "평균 절대 오차(스타)", "coverage80": "중간 80% 범위 적중률",
    "covered80": "예측 범위 안에 포함", "absolute_error": "절대 오차", "p10": "하위 10% 값",
    "median": "중앙값", "p90": "상위 10% 경계", "stacking_weight": "과거 검증 비중",
    "equal_weight": "동일 비중", "selected_weight": "선택 비중", "simulation_weight": "실행 비중",
    "block_bootstrap_p10": "재표본화 하위 비중", "block_bootstrap_p90": "재표본화 상위 비중",
    "reached_within_horizon": "기간 내 도달 수", "unreached_within_horizon": "기간 내 미도달 수",
    "mc_hit_probability": "기간 내 도달 비율", "mc_probability": "도달 비율",
    "mc95_low": "계산 표본오차 95% 하한", "mc95_high": "계산 표본오차 95% 상한",
    "all_paths_median_days_upper": "전체 중앙 도달 소요일(상한)",
    "all_paths_median_date_upper": "전체 중앙 도달일(상한)", "all_paths_median_status": "전체 중앙값 상태",
    "horizon_days": "예측 기간(일)", "date": "날짜", "target": "목표 스타", "successes": "도달 수",
    "analytic_probability": "해석식 도달 비율", "capped_p10": "하위 10% 스타(목표까지)",
    "capped_median": "중앙 스타(목표까지)", "capped_p90": "상위 10% 경계 스타(목표까지)",
    "mc_cumulative_probability": "누적 도달 비율", "parameter": "설정 항목", "value": "설정값",
    "origin": "수치의 출처", "baseline_probability": "기존 유입 도달 비율",
    "ecosystem_probability": "생태계 조건 도달 비율", "difference_pp": "도달 비율 차이(%포인트)",
    "baseline_median": "기존 유입 중앙 스타", "ecosystem_median": "생태계 조건 중앙 스타",
    "paired_added_median": "같은 경로의 추가 스타 중앙값(목표까지)",
    "extra_stars": "생태계에서 추가된 스타", "discovered": "누적 신규 발견자",
    "active_developers": "활동 중인 응용 개발자", "active_projects": "활동 중인 파생 프로젝트",
    "completed_projects": "누적 완성 프로젝트", "successful_projects": "누적 확산력 큰 파생 프로젝트",
    "active_contributors": "언어에 직접 기여 중인 개발자", "metric": "지표",
    "rate_threshold": "추가 평균 유입 기준(스타/일)", "sustained_days": "유지 기간(일)",
    "given_reached_median_date": "기준 충족 경로만의 중앙일",
    "scenario": "시나리오", "name": "이름", "additional_audience": "기존 유입과 겹치지 않는 잠재 독자 수",
    "initial_developers": "초기 응용 개발자 수", "star_probability": "발견자의 스타 전환 확률",
    "adoption_probability": "발견자의 사용 전환 확률", "project_start_rate_per_day": "개발자당 하루 프로젝트 시작률",
    "development_mean_days": "평균 개발 기간(일)", "developer_half_life_days": "개발 참여 반감기(일)",
    "project_half_life_days": "공개 프로젝트 활동 반감기(일)", "regular_leads_per_day": "일반 프로젝트 하루 도달 시도",
    "successful_leads_per_day": "확산력 큰 프로젝트 하루 도달 시도", "project_success_probability": "파생 프로젝트의 큰 확산 조건 확률",
    "contribution_probability": "프로젝트 완성자의 언어 기여 확률", "contributor_half_life_days": "언어 기여 활동 반감기(일)",
    "contribution_max_boost": "언어 기여로 인한 프로젝트 시작률 최대 추가 배수", "contribution_scale": "기여 효과가 절반에 이르는 인원",
}
for prefix, text in [("p10", "하위 10%"), ("median", "중앙"), ("p90", "상위 10% 경계")]:
    LABELS[f"given_reached_{prefix}_days_upper"] = f"도달 경로만의 {text} 소요일(상한)"
    LABELS[f"given_reached_{prefix}_date_upper"] = f"도달 경로만의 {text} 날짜(상한)"
PERCENT_COLUMNS = {"coverage80", "stacking_weight", "equal_weight", "selected_weight", "simulation_weight",
                   "block_bootstrap_p10", "block_bootstrap_p90", "mc_hit_probability", "mc_probability",
                   "mc95_low", "mc95_high", "analytic_probability", "mc_cumulative_probability",
                   "baseline_probability", "ecosystem_probability"}
FILES = {
    "summary": "도달_요약", "target_probabilities": "목표_도달_확률", "milestone_probabilities": "중간목표_도달_확률",
    "forecast_quantiles": "연도별_스타_예상", "target_cdf": "누적_도달_확률", "validation_folds": "과거_검증_상세",
    "validation_summary": "과거_검증_요약", "model_weights": "모형별_비중", "ecosystem_assumptions": "생태계_입력_가정",
    "ecosystem_comparison": "기준선과_생태계_비교", "ecosystem_metrics": "생태계_추이",
    "sustained_arrivals": "추가_유입_유지_시점", "milestone_comparison": "중간목표_비교",
}


def table(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for col in result:
        if col in PERCENT_COLUMNS:
            result[col] = result[col].map(lambda x: "해당 없음" if pd.isna(x) else f"{x:.4%}")
        elif col in {"model", "group", "metric", "parameter"}:
            result[col] = result[col].map(lambda x: MODELS.get(x, LABELS.get(x, x)))
        elif col == "all_paths_median_status":
            result[col] = result[col].map({"within_horizon": "기간 내 도달", "beyond_horizon_or_no_samples": "기간 밖 또는 표본 없음"})
        elif result[col].dtype == bool:
            result[col] = result[col].map({True: "예", False: "아니오"})
    return result.rename(columns=LABELS).fillna("해당 없음").replace("", "기간 내 확인되지 않음")


def save_csv(output: Path, name: str, frame: pd.DataFrame) -> None:
    # 기존 분석 코드와 호환되는 원본. 사용자는 한글 이름의 파일과 HTML을 읽는다.
    frame.to_csv(output / f"{name}.csv", index=False)
    table(frame).to_csv(output / f"{FILES.get(name, name)}.csv", index=False, encoding="utf-8-sig")


def html_table(frame: pd.DataFrame) -> str:
    return table(frame).to_html(index=False, escape=True, float_format=lambda x: f"{x:,.4g}")


def configure_font() -> str:
    import matplotlib
    from matplotlib import font_manager, ft2font
    candidates = ["Noto Sans CJK KR", "Noto Sans KR", "NanumGothic", "Malgun Gothic", "AppleGothic",
                  "Noto Sans CJK JP", "UnDotum"]
    for family in candidates:
        try:
            path = font_manager.findfont(family, fallback_to_default=False)
            if ord("한") not in ft2font.FT2Font(path).get_charmap():
                continue
            matplotlib.rcParams["font.family"] = family
            matplotlib.rcParams["axes.unicode_minus"] = False
            return family
        except (ValueError, RuntimeError, OSError):
            continue
    raise ValueError("한글 그래프용 글꼴이 없어. Noto Sans CJK KR 또는 나눔고딕을 설치한 뒤 다시 실행해. "
                     "그래프 없이 계산하려면 --no-plots를 사용해.")


def summary_lines(history, cfg, summary, quantiles, comparison=None) -> list[str]:
    row = summary.iloc[0]
    lines = [f"{history.repository} 성장 시뮬레이션",
             f"자료 기준일: {history.dates[-1].date()} / 현재 {int(history.stars[-1]):,} 스타",
             f"가상 미래 {cfg.simulations:,}개 / 최대 {cfg.years}년 / 목표 {cfg.target:,} 스타", ""]
    if comparison is not None:
        final = comparison.iloc[-1]
        lines += ["조건을 넣었을 때와 넣지 않았을 때의 비교야. 사건 발생·전환율은 입력한 가정이야.",
                  f"{cfg.years}년 내 목표 도달: 기존 {final.baseline_probability:.2%} → 생태계 조건 {final.ecosystem_probability:.2%}",
                  f"같은 기준선 대비 차이: {final.difference_pp:+.2f}%포인트",
                  "추가 유입만 더하는 모형이므로 기준선보다 나빠지지 않도록 구성돼 있어.", ""]
    else:
        lines.append(f"{cfg.years}년 내 목표 도달 비율: {row.mc_hit_probability:.2%} (선택한 모형 아래의 값)")
    lines += [f"전체 경로의 중앙 도달일: {row.all_paths_median_date_upper or '기간 내 확인되지 않음'}",
              f"기간 내 도달한 경로만의 중앙 도달일: {row.given_reached_median_date_upper or '도달 표본 없음'}", ""]
    for item in quantiles.itertuples():
        lines.append(f"{item.date}: 중심 {int(item.capped_median):,} 스타 / 중간 80% 범위 {int(item.capped_p10):,}~{int(item.capped_p90):,}")
    lines += ["", "스타 수는 목표에서 표시를 멈춰. 그 이후 실제 성장이 멈춘다는 뜻은 아니야.",
              "미도달은 설정한 기간 안에 도달하지 않았다는 뜻이지, 영원히 불가능하다는 뜻이 아니야.",
              "계산 표본오차는 현실의 미래를 보장하는 구간이 아니야."]
    return lines
