"""도달 조건·미도달·모의 표본오차를 분리한 한국어 보고서."""
from dataclasses import asdict
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import Validation
from .ecosystem import EcosystemConfig, METRICS, RATE_THRESHOLDS, assumption_rows
from .history import History
from .korean import MODELS, configure_font, html_table, save_csv, summary_lines
from .models import CountModel, ModelConfig
from .simulation import SimulationConfig, atomic_json


def wilson(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    p = successes / total
    den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    radius = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total**2)) / den
    return max(0.0, center - radius), min(1.0, center + radius)


def hitting_quantile(hit: np.ndarray, q: float) -> int | None:
    index = max(0, int(np.ceil(q * len(hit))) - 1)
    reached = np.sort(hit[hit >= 0])
    return int(reached[index]) if len(reached) > index else None


def save_validation(output: Path, validation: Validation, models: list[CountModel], cfg: ModelConfig) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in [("validation_folds", validation.folds), ("validation_summary", validation.summary),
                         ("model_weights", validation.weights)]:
        save_csv(output, name, frame)
    atomic_json(output / "model_diagnostics.json", {
        "model_config": asdict(cfg), "models": {m.name: m.info() for m in models},
        "weights_meaning": "과거 예측 성능에 따른 조합 비중이며, 모형이 참일 확률이 아니다.",
        "bootstrap_meaning": "과거 구간 재표본화에 따른 비중 민감도이며, 장기 예측의 신뢰구간이 아니다."
    })
    cp = models[2].diagnostics
    lines = ["과거 기록에 대한 진단", "", "미래 킬러 프로젝트의 등장 확률을 계산한 결과가 아니야.",
             f"최근 구간에서 한 번의 유입률 변화를 가정한 조건부 비중: {cp.get('change_probability', 0):.2%}",
             f"변화가 있다고 가정할 때의 후보 날짜: {cp.get('candidate_date_given_change', '자료 부족')}",
             f"변화 후보 이전 유입: {cp.get('candidate_before_rate_mle', 0):.4g} 스타/일",
             f"변화 후보 이후 유입: {cp.get('candidate_after_rate_mle', 0):.4g} 스타/일",
             "이 날짜는 과거 변화 후보이지, 미래 폭발 시점이나 보장된 임계점이 아니야."]
    for m in models:
        lines.append(f"{MODELS.get(m.name, m.name)}: 관측 {m.diagnostics.get('exposure_days', 0)}일; "
                     f"수치 격자 경계 질량 상한 {m.diagnostics.get('grid_edge_mass_bound', 0):.4g}")
    (output / "진단_설명.txt").write_text("\n".join(lines), encoding="utf-8")


def load_chunks(paths: list[Path], cfg: SimulationConfig, snapshots: np.ndarray,
                thresholds: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ids = np.empty(cfg.simulations, dtype=np.uint8)
    hit = np.empty((cfg.simulations, len(thresholds)), dtype=np.int32)
    totals = np.empty((cfg.simulations, len(snapshots)), dtype=np.uint32)
    sample_paths = np.empty((0, 0), dtype=np.uint32)
    offset = 0
    for index, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as part:
            n = len(part["model_id"])
            ids[offset:offset+n] = part["model_id"]
            hit[offset:offset+n] = part["hit_days_upper"]
            totals[offset:offset+n] = part["capped_snapshot_stars"]
            if index == 0:
                sample_paths = part["sample_paths"].copy()
        offset += n
    if offset != cfg.simulations:
        raise ValueError("체크포인트의 시뮬레이션 개수가 설정과 달라.")
    return ids, hit, totals, sample_paths


def _date(last, days):
    return str((last + pd.Timedelta(days=int(days))).date()) if days is not None else ""


def _prob(hit, days):
    success = int(((hit >= 0) & (hit <= days)).sum())
    low, high = wilson(success, len(hit))
    return {"simulations": len(hit), "successes": success,
            "mc_probability": success / len(hit) if len(hit) else np.nan,
            "mc95_low": low, "mc95_high": high}


def _summary(hit, name, last):
    reached = hit[hit >= 0]
    row = {"group": name, "simulations": len(hit), "reached_within_horizon": len(reached),
           "unreached_within_horizon": len(hit)-len(reached),
           "mc_hit_probability": len(reached)/len(hit) if len(hit) else np.nan}
    row["mc95_low"], row["mc95_high"] = wilson(len(reached), len(hit))
    median = hitting_quantile(hit, .5) if len(hit) else None
    row.update(all_paths_median_days_upper=median,
               all_paths_median_status="within_horizon" if median is not None else "beyond_horizon_or_no_samples",
               all_paths_median_date_upper=_date(last, median))
    for q, label in [(0.1, "p10"), (0.5, "median"), (0.9, "p90")]:
        day = hitting_quantile(reached, q) if len(reached) else None
        row[f"given_reached_{label}_days_upper"] = day
        row[f"given_reached_{label}_date_upper"] = _date(last, day)
    return row


def build_reports(output: Path, history: History, cfg: SimulationConfig, models: list[CountModel],
                  weights: np.ndarray, validation: Validation, grid: np.ndarray, snapshots: np.ndarray,
                  thresholds: np.ndarray, chunks: list[Path], *, plots: bool = True,
                  ecosystem: EcosystemConfig | None = None) -> pd.DataFrame:
    ids, hits, totals, sample_paths = load_chunks(chunks, cfg, snapshots, thresholds)
    final_hit = hits[:, -1]
    last, start = history.dates[-1], int(history.stars[-1])
    groups = [("ensemble", np.ones(len(ids), dtype=bool))] + [(m.name, ids == i) for i, m in enumerate(models)]
    summary = pd.DataFrame([_summary(final_hit[mask], name, last) for name, mask in groups])
    probability_rows, milestone_rows, cdf_rows = [], [], []
    for days in snapshots:
        analytic = np.array([m.target_probability(cfg.target, int(days), start) for m in models])
        if ecosystem is not None:
            analytic[:] = np.nan  # 생태계 조건에는 기준선 해석값을 표시하지 않는다.
        elif cfg.feedback_half_life_years is not None:
            analytic[[m.kind == "birth" for m in models]] = np.nan
        for i, (name, mask) in enumerate(groups):
            expected = (float(np.dot(weights, np.nan_to_num(analytic))) if i == 0
                        and not np.any((weights > 0) & np.isnan(analytic)) else analytic[i-1] if i else np.nan)
            probability_rows.append({"group": name, "horizon_days": int(days), "date": _date(last, days),
                                     **_prob(final_hit[mask], days), "analytic_probability": expected})
        probability_rows.append({"group": "equal_weight_sensitivity", "horizon_days": int(days),
                                 "date": _date(last, days), "simulations": 0, "successes": 0,
                                 "mc_probability": np.nan, "mc95_low": np.nan, "mc95_high": np.nan,
                                 "analytic_probability": float(analytic.mean())})
        for col, target in enumerate(thresholds):
            milestone_rows.append({"target": int(target), "horizon_days": int(days), "date": _date(last, days),
                                   **_prob(hits[:, col], days)})
    probability = pd.DataFrame(probability_rows)
    quantiles = np.quantile(totals, [.1, .5, .9], axis=0, method="inverted_cdf")
    quantile_frame = pd.DataFrame({"horizon_days": snapshots, "date": [_date(last, d) for d in snapshots],
                                  "capped_p10": quantiles[0], "capped_median": quantiles[1], "capped_p90": quantiles[2]})
    for name, mask in groups:
        sorted_hit = np.sort(final_hit[mask & (final_hit >= 0)])
        n = int(mask.sum())
        fractions = np.searchsorted(sorted_hit, grid, side="right")/n if n else np.full(len(grid), np.nan)
        cdf_rows.extend({"group": name, "horizon_days": int(d), "mc_cumulative_probability": float(p)} for d, p in zip(grid, fractions))
    cdf_frame = pd.DataFrame(cdf_rows)
    for name, frame in [("summary", summary), ("target_probabilities", probability),
                        ("milestone_probabilities", pd.DataFrame(milestone_rows)),
                        ("forecast_quantiles", quantile_frame), ("target_cdf", cdf_frame)]:
        save_csv(output, name, frame)
    comparison = None
    ecosystem_body = ""
    if ecosystem is not None:
        comparison, ecosystem_body = ecosystem_reports(output, ecosystem, cfg, last, snapshots,
                                                        thresholds, chunks, hits, totals, plots)
    lines = summary_lines(history, cfg, summary, quantile_frame, comparison)
    (output / "요약.txt").write_text("\n".join(lines), encoding="utf-8")
    if plots:
        make_plots(output, history, cfg, validation, grid, snapshots, quantiles, cdf_frame, sample_paths, final_hit)
    body = f"<h1>{escape(history.repository)} 성장 시뮬레이션</h1><h2>결론부터 보기</h2><pre>{escape(chr(10).join(lines))}</pre>"
    body += ecosystem_body
    body += "<h2>연도별 스타 예상</h2>" + html_table(quantile_frame)
    body += "<h2>목표별 도달 비율</h2>" + html_table(pd.DataFrame(milestone_rows))
    if plots:
        for name, label in [("forecast_fan", "앞으로의 스타 분포"), ("target_cdf", "목표까지 걸리는 시간"),
                            ("sample_paths", "일부 가상 미래"), ("hitting_mass", "미도달을 포함한 도달 시점 분포"),
                            ("validation_scores", "과거에 예측해 봤을 때의 성능"), ("model_weights", "모형별 예측 비중")]:
            body += f'<h2>{label}</h2><img src="charts/{name}.png" alt="{label}">'
    body += "<details><summary>모형별 결과와 과거 검증 자세히 보기</summary>"
    body += html_table(summary) + html_table(validation.summary) + html_table(validation.weights) + "</details>"
    notes = ["모든 비율은 선택한 모형·사전분포·장기 가정 아래에서 계산한 값이야.",
             "단기 검증 비중은 그 모형이 진실일 확률이나 30년 뒤 성공 확률 자체가 아니야.",
             f"도달일은 최대 {cfg.step_days}일 간격의 오른쪽 끝이야. 실제 모의 도달는 그 구간 안에 있어.",
             "계산 표본오차 95% 구간은 무작위 계산 오차만 보여줘. 모델 오류나 현실의 변화를 포함하지 않아.",
             "계속 개발한다는 조건을 두지만, 실제 개발량이나 홍보량을 관측해 추정한 것은 아니야.",
             "과거 스타 수만으로 미래 킬러 프로젝트의 발생·성공 확률이나 인과관계를 추정하지 않았어.",
             "확인하기 쉬운 CSV는 한글 이름으로 저장했어. 영문 이름 CSV와 JSON은 기존 도구와 연결하기 위한 원본이야."]
    if ecosystem is not None:
        notes.append("이번 생태계층은 기존 유입과 겹치지 않는 추가 독자만 더해. 따라서 기준선보다 낮아지지 않는 것은 모형 구조 때문이야. 기존 사용자 이탈·유입 대체 효과는 별도 가정이 필요해.")
    if cfg.feedback_half_life_years is not None:
        notes.append(f"별도 조건: 스타 비례 성장의 지속 반감기 {cfg.feedback_half_life_years:g}년 이후 장기 유입으로 복귀해. 개발 중단을 뜻하지 않아.")
    body += "<h2>해석할 때 기억할 점</h2><ul>" + "".join(f"<li>{escape(n)}</li>" for n in notes) + "</ul>"
    html = ('<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>Wave 성장 실험 결과</title>'
            '<style>body{font:16px system-ui;max-width:1200px;margin:2rem auto;padding:1rem;line-height:1.7}'
            'pre{white-space:pre-wrap;font:inherit}table{display:block;overflow:auto;border-collapse:collapse}'
            'th,td{padding:.5rem;text-align:right;border-bottom:1px solid #ddd}img{max-width:100%}'
            'summary{cursor:pointer}h2{margin-top:2rem}</style></head><body>' + body + '</body></html>')
    (output / "report.html").write_text(html, encoding="utf-8")
    return summary


def ecosystem_reports(output, eco, cfg, last, snapshots, thresholds, chunks, hits, totals, plots):
    base_hits = np.empty_like(hits)
    base_totals = np.empty_like(totals)
    metric_values = np.empty((cfg.simulations, len(snapshots), len(METRICS)), dtype=np.uint32)
    rate_hits = np.empty((cfg.simulations, len(RATE_THRESHOLDS)), dtype=np.int32)
    offset = 0
    for path in chunks:
        with np.load(path, allow_pickle=False) as part:
            n = len(part["model_id"])
            base_hits[offset:offset+n] = part["baseline_hit_days_upper"]
            base_totals[offset:offset+n] = part["baseline_snapshot_stars"]
            metric_values[offset:offset+n] = part["ecosystem_snapshots"]
            rate_hits[offset:offset+n] = part["sustained_rate_hit_days_upper"]
        offset += n
    rows, milestone_rows, metric_rows, rate_rows = [], [], [], []
    for i, days in enumerate(snapshots):
        for col, target in enumerate(thresholds):
            before, after = _prob(base_hits[:, col], days), _prob(hits[:, col], days)
            r = {"date": _date(last, days), "horizon_days": int(days), "target": int(target),
                 "baseline_probability": before["mc_probability"], "ecosystem_probability": after["mc_probability"],
                 "difference_pp": 100 * (after["mc_probability"] - before["mc_probability"])}
            milestone_rows.append(r)
            if col == len(thresholds)-1:
                rows.append({**r, "baseline_median": float(np.quantile(base_totals[:, i], .5, method="inverted_cdf")),
                             "ecosystem_median": float(np.quantile(totals[:, i], .5, method="inverted_cdf")),
                             "paired_added_median": float(np.quantile(totals[:, i].astype(np.int64)-base_totals[:, i], .5, method="inverted_cdf"))})
        for j, metric in enumerate(METRICS):
            q = np.quantile(metric_values[:, i, j], [.1, .5, .9], method="inverted_cdf")
            metric_rows.append({"date": _date(last, days), "metric": metric, "p10": q[0], "median": q[1], "p90": q[2]})
    for j, threshold in enumerate(RATE_THRESHOLDS):
        h = rate_hits[:, j]
        for days in snapshots:
            reached = h[(h >= 0) & (h <= days)]
            rate_rows.append({"date": _date(last, days), "rate_threshold": threshold, "sustained_days": eco.sustained_days,
                              **_prob(h, days), "given_reached_median_date": _date(last, hitting_quantile(reached, .5)) if len(reached) else ""})
    comparison = pd.DataFrame(rows)
    assumption_frame = pd.DataFrame(assumption_rows(eco, last))
    metric_frame = pd.DataFrame(metric_rows)
    rate_frame = pd.DataFrame(rate_rows)
    for name, frame in [("ecosystem_comparison", comparison), ("milestone_comparison", pd.DataFrame(milestone_rows)),
                        ("ecosystem_assumptions", assumption_frame), ("ecosystem_metrics", metric_frame),
                        ("sustained_arrivals", rate_frame)]:
        save_csv(output, name, frame)
    body = f"<h2>생태계 조건: {escape(eco.name)}</h2><p>아래는 사건이 생길 가능성을 알아낸 예언이 아니라, 입력한 조건의 효과를 비교한 결과야.</p>"
    body += "<h3>같은 직접 유입 경로와 비교</h3>" + html_table(comparison)
    body += "<h3>추가 유입이 유지되는 시점</h3><p>생태계의 평균 추가 유입이 기준 이상으로 일정 기간 유지되는 운영상 지표야. 언어의 보편적 임계점이나 실제 등장 예정일은 아니야. 평균 유입률에 대한 격자 판정이며 매일 정확히 그 수만큼 별이 붙는다는 뜻도 아니야.</p>"
    body += html_table(rate_frame[rate_frame.date == rate_frame.date.iloc[-1]])
    body += "<details open><summary>사용한 가정 확인</summary>" + html_table(assumption_frame) + "</details>"
    body += "<details><summary>프로젝트·개발자·기여자 변화</summary>" + html_table(metric_frame) + "</details>"
    if plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        configure_font()
        folder = output / "charts"
        folder.mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(11, 6))
        future = last + pd.to_timedelta(np.r_[0, snapshots], unit="D")
        # 목표까지 표시하는 두 조건의 분포를 동일한 시간축에서 비교한다.
        for values, label in [(base_totals, "기존 유입만"), (totals, "생태계 조건 추가")]:
            qs = np.quantile(values, [.1, .5, .9], axis=0, method="inverted_cdf")
            ax.plot(future[1:], qs[1], label=f"{label} 중앙값")
            ax.fill_between(future[1:], qs[0], qs[2], alpha=.18, label=f"{label} 중간 80%")
        ax.set_yscale("symlog", linthresh=1)
        ax.set(xlabel="날짜", ylabel="스타 수(목표까지)", title="킬러 프로젝트·생태계 조건의 차이")
        ax.legend()
        ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(folder / "ecosystem_comparison.png", dpi=160)
        plt.close(fig)
        body += '<h3>기준선과 생태계 조건의 성장 분포</h3><img src="charts/ecosystem_comparison.png" alt="조건별 스타 분포 비교">'
    return comparison, body


def make_plots(output: Path, history: History, cfg: SimulationConfig, validation: Validation,
               grid: np.ndarray, snapshots: np.ndarray, quantiles: np.ndarray,
               cdf: pd.DataFrame, paths: np.ndarray, hit: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    configure_font()
    folder = output / "charts"
    folder.mkdir(exist_ok=True)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(folder / f"{name}.png", dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    for name, group in cdf.groupby("group", sort=False):
        if group.mc_cumulative_probability.notna().any():
            ax.plot(group.horizon_days/365.2425, group.mc_cumulative_probability, label=MODELS.get(name, name))
    ax.set(xlabel="자료 기준일로부터 지난 해", ylabel="전체 경로 중 도달 비율", title=f"{cfg.target:,} 스타까지의 누적 도달 비율", ylim=(0, 1))
    ax.legend()
    ax.grid(alpha=.2)
    save(fig, "target_cdf")

    future = history.dates[-1] + pd.to_timedelta(np.r_[0, snapshots], unit="D")
    qs = np.column_stack([np.full(3, min(int(history.stars[-1]), cfg.target)), quantiles])
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(history.dates, np.minimum(history.stars, cfg.target), label="실제 관측")
    ax.plot(future, qs[1], label="가상 미래의 중앙값")
    ax.fill_between(future, qs[0], qs[2], alpha=.25, label="가상 미래의 중간 80%")
    ax.set_yscale("symlog", linthresh=1)
    ax.set(xlabel="날짜", ylabel="스타 수(목표까지)", title="모형과 가정에 따른 미래 분포")
    ax.legend()
    ax.grid(alpha=.2)
    save(fig, "forecast_fan")

    fig, ax = plt.subplots(figsize=(11, 6))
    for path in paths:
        ax.plot(grid/365.2425, path, alpha=.18, linewidth=.7)
    ax.set_yscale("symlog", linthresh=1)
    ax.set(xlabel="자료 기준일로부터 지난 해", ylabel="스타 수(목표까지)", title=f"가상 미래 {len(paths)}개 예시 — 전체 결과의 일부")
    ax.grid(alpha=.2)
    save(fig, "sample_paths")

    fig, ax = plt.subplots(figsize=(11, 6))
    for name, group in validation.folds.groupby("model", sort=False):
        ax.plot(pd.to_datetime(group.test_end), group.log_score, marker=".", label=MODELS.get(name, name))
    ax.set(xlabel="과거 검증 구간 종료일", ylabel="실제 증가량의 예측 로그 점수", title="과거 예측 시험 — 위쪽일수록 좋은 점수")
    ax.legend()
    ax.grid(alpha=.2)
    save(fig, "validation_scores")

    fig, ax = plt.subplots(figsize=(11, 6))
    w = validation.weights
    ax.bar([MODELS.get(x, x) for x in w.model], w.get("simulation_weight", w.selected_weight))
    ax.tick_params(axis="x", rotation=15)
    ax.set(ylabel="예측에 사용하는 비중", title="모형별 비중 — 참일 확률을 뜻하지 않음", ylim=(0, 1))
    save(fig, "model_weights")

    fig, ax = plt.subplots(figsize=(11, 6))
    reached = hit[hit >= 0]/365.2425
    if len(reached):
        ax.hist(reached, bins=np.linspace(0, grid[-1]/365.2425, 61), weights=np.ones(len(reached))/len(hit))
    ax.set(xlabel="자료 기준일로부터 지난 해", ylabel="전체 경로 중 해당 구간의 비율", title=f"목표 도달 시점 분포 — 기간 내 미도달 {(hit < 0).mean():.1%}")
    ax.grid(alpha=.2)
    save(fig, "hitting_mass")
