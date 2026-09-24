"""Reports keep conditional times, censoring and Monte Carlo error distinct."""
from dataclasses import asdict
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import Validation
from .history import History
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
    # Unreached paths remain in the denominator, not silently discarded.
    index = max(0, int(np.ceil(q * len(hit))) - 1)
    reached = np.sort(hit[hit >= 0])
    return int(reached[index]) if len(reached) > index else None


def save_validation(output: Path, validation: Validation, models: list[CountModel],
                    cfg: ModelConfig) -> None:
    output.mkdir(parents=True, exist_ok=True)
    validation.folds.to_csv(output / "validation_folds.csv", index=False)
    validation.summary.to_csv(output / "validation_summary.csv", index=False)
    validation.weights.to_csv(output / "model_weights.csv", index=False)
    atomic_json(output / "model_diagnostics.json", {
        "model_config": asdict(cfg), "models": {m.name: m.info() for m in models},
        "weights_meaning": "Predictive stacking weights, not posterior probabilities of models being true.",
        "bootstrap_meaning": "Moving-block sensitivity of fitted weights; not a long-horizon confidence interval."
    })


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
            ids[offset:offset + n] = part["model_id"]
            hit[offset:offset + n] = part["hit_days_upper"]
            totals[offset:offset + n] = part["capped_snapshot_stars"]
            if index == 0:
                sample_paths = part["sample_paths"].copy()
        offset += n
    if offset != cfg.simulations:
        raise ValueError("Checkpoint simulation count mismatch.")
    return ids, hit, totals, sample_paths


def build_reports(output: Path, history: History, cfg: SimulationConfig,
                  models: list[CountModel], weights: np.ndarray,
                  validation: Validation, grid: np.ndarray, snapshots: np.ndarray,
                  thresholds: np.ndarray, chunks: list[Path], *, plots: bool = True) -> pd.DataFrame:
    ids, hits, totals, sample_paths = load_chunks(chunks, cfg, snapshots, thresholds)
    final_hit = hits[:, -1]
    last, start = history.dates[-1], int(history.stars[-1])
    groups = [("ensemble", np.ones(len(ids), dtype=bool))] + [
        (m.name, ids == i) for i, m in enumerate(models)]
    summary_rows, probability_rows, milestone_rows = [], [], []
    for name, mask in groups:
        hit = final_hit[mask]
        reached = hit[hit >= 0]
        row = {"group": name, "simulations": len(hit),
               "reached_within_horizon": len(reached),
               "unreached_within_horizon": len(hit) - len(reached),
               "mc_hit_probability": len(reached) / len(hit) if len(hit) else np.nan}
        row["mc95_low"], row["mc95_high"] = wilson(len(reached), len(hit))
        median = hitting_quantile(hit, 0.5) if len(hit) else None
        row["all_paths_median_days_upper"] = median
        row["all_paths_median_status"] = ("within_horizon" if median is not None
                                             else "beyond_horizon_or_no_samples")
        row["all_paths_median_date_upper"] = (str((last + pd.Timedelta(days=median)).date())
                                                 if median is not None else "")
        for q, label in [(0.1, "p10"), (0.5, "median"), (0.9, "p90")]:
            days = hitting_quantile(reached, q) if len(reached) else None
            row[f"given_reached_{label}_days_upper"] = days
            row[f"given_reached_{label}_date_upper"] = (
                str((last + pd.Timedelta(days=days)).date()) if days is not None else "")
        summary_rows.append(row)
    for days in snapshots:
        analytic = np.array([m.target_probability(cfg.target, int(days), start) for m in models])
        if cfg.feedback_half_life_years is not None:
            analytic[[m.kind == "birth" for m in models]] = np.nan
        for name, mask in groups:
            hit = final_hit[mask]
            success = int(((hit >= 0) & (hit <= days)).sum())
            n = len(hit)
            lower, upper = wilson(success, n)
            expected = (float(np.dot(weights, np.nan_to_num(analytic))) if name == "ensemble"
                        and not np.any((weights > 0) & np.isnan(analytic)) else
                        analytic[[m.name for m in models].index(name)] if name != "ensemble" else np.nan)
            probability_rows.append({"group": name, "horizon_days": int(days),
                                     "date": str((last + pd.Timedelta(days=int(days))).date()),
                                     "simulations": n, "successes": success,
                                     "mc_probability": success / n if n else np.nan,
                                     "mc95_low": lower, "mc95_high": upper,
                                     "analytic_probability": expected})
        probability_rows.append({"group": "equal_weight_sensitivity", "horizon_days": int(days),
                                 "date": str((last + pd.Timedelta(days=int(days))).date()),
                                 "simulations": 0, "successes": 0, "mc_probability": np.nan,
                                 "mc95_low": np.nan, "mc95_high": np.nan,
                                 "analytic_probability": float(analytic.mean())})
        for col, target in enumerate(thresholds):
            hit = hits[:, col]
            success = int(((hit >= 0) & (hit <= days)).sum())
            lower, upper = wilson(success, len(hit))
            milestone_rows.append({"target": int(target), "horizon_days": int(days),
                                   "date": str((last + pd.Timedelta(days=int(days))).date()),
                                   "mc_probability": success / len(hit),
                                   "mc95_low": lower, "mc95_high": upper})
    summary = pd.DataFrame(summary_rows)
    probability = pd.DataFrame(probability_rows)
    summary.to_csv(output / "summary.csv", index=False)
    probability.to_csv(output / "target_probabilities.csv", index=False)
    pd.DataFrame(milestone_rows).to_csv(output / "milestone_probabilities.csv", index=False)
    quantiles = np.quantile(totals, [0.1, 0.5, 0.9], axis=0, method="inverted_cdf")
    quantile_frame = pd.DataFrame({"horizon_days": snapshots,
                                  "date": [str((last + pd.Timedelta(days=int(d))).date()) for d in snapshots],
                                  "capped_p10": quantiles[0], "capped_median": quantiles[1],
                                  "capped_p90": quantiles[2]})
    quantile_frame.to_csv(output / "forecast_quantiles.csv", index=False)
    cdf_rows = []
    for name, mask in groups:
        hit = np.sort(final_hit[mask & (final_hit >= 0)])
        n = int(mask.sum())
        probabilities = np.searchsorted(hit, grid, side="right") / n if n else np.full(len(grid), np.nan)
        cdf_rows.extend({"group": name, "horizon_days": int(d), "mc_cumulative_probability": float(p)}
                        for d, p in zip(grid, probabilities))
    cdf_frame = pd.DataFrame(cdf_rows)
    cdf_frame.to_csv(output / "target_cdf.csv", index=False)
    if plots:
        make_plots(output, history, cfg, validation, grid, snapshots, quantiles,
                   cdf_frame, sample_paths, final_hit)
    notes = [
        "All probabilities are conditional on these count models, priors and their extrapolation assumptions.",
        "30-day predictive stacking is not a calibrated probability of 30-year adoption or of a model being true.",
        "Unreached within the horizon does not mean never. Conditional hitting quantiles exclude unreached paths; all-path quantiles do not.",
        f"Hitting times are upper endpoints of intervals no wider than {cfg.step_days} days. Forecast totals and sample paths are capped at the target.",
        "mc95 is a Wilson interval for Monte Carlo sampling error ONLY, not uncertainty about the real future.",
        "The first total is an initial condition. No missing days or removals are imputed. Exporter reconstructions may omit historical unstars.",
        "The ongoing-development assumption is qualitative: no development-effort or exposure covariates are observed.",
        "Finite-grid posterior integration should be checked by increasing --grid-size and inspecting grid-edge mass.",
    ]
    if cfg.feedback_half_life_years is not None:
        notes.append(f"User scenario: feedback survival has half-life {cfg.feedback_half_life_years:g} years, then returns to the long-arrival rate. This hazard was NOT inferred from Wave data.")
    body = f"<h1>{escape(history.repository)}: growth scenarios</h1><p>As of {last.date()}; {start:,} stars; target {cfg.target:,}; {cfg.simulations:,} paths.</p>"
    body += "<h2>Interpretation</h2><ul>" + "".join(f"<li>{escape(n)}</li>" for n in notes) + "</ul>"
    body += "<h2>Historical out-of-sample validation</h2>" + validation.summary.to_html(index=False)
    body += "<h2>Predictive weights and resampling sensitivity</h2>" + validation.weights.to_html(index=False)
    body += "<h2>Target probabilities</h2>" + probability.to_html(index=False, float_format=lambda x: f"{x:.6g}")
    body += "<h2>Hitting times: check the conditional / all-path columns</h2>" + summary.to_html(index=False)
    if plots:
        for name in ["target_cdf", "forecast_fan", "sample_paths", "validation_scores", "model_weights", "hitting_mass"]:
            body += f'<h2>{escape(name.replace("_", " "))}</h2><img src="charts/{name}.png" alt="{name}">'
    html = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Growth scenarios</title>'
            '<style>body{font:16px system-ui;max-width:1200px;margin:2rem auto;padding:1rem;line-height:1.6}'
            'table{display:block;overflow:auto;border-collapse:collapse}th,td{padding:.4rem;text-align:right}'
            'img{max-width:100%}li{margin:.6rem 0}</style><body>' + body + '</body></html>')
    (output / "report.html").write_text(html, encoding="utf-8")
    return summary


def make_plots(output: Path, history: History, cfg: SimulationConfig, validation: Validation,
               grid: np.ndarray, snapshots: np.ndarray, quantiles: np.ndarray,
               cdf: pd.DataFrame, paths: np.ndarray, hit: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = output / "charts"
    folder.mkdir(exist_ok=True)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(folder / f"{name}.png", dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    for name, group in cdf.groupby("group", sort=False):
        if group["mc_cumulative_probability"].notna().any():
            ax.plot(group["horizon_days"] / 365.2425, group["mc_cumulative_probability"], label=name)
    ax.set(xlabel="Years from last observation", ylabel="Fraction of ALL paths reaching target",
           title=f"Time to {cfg.target:,}: conditional model scenarios", ylim=(0, 1))
    ax.legend()
    ax.grid(alpha=0.2)
    save(fig, "target_cdf")

    future = history.dates[-1] + pd.to_timedelta(np.r_[0, snapshots], unit="D")
    qs = np.column_stack([np.full(3, min(int(history.stars[-1]), cfg.target)), quantiles])
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(history.dates, np.minimum(history.stars, cfg.target), label="Observed")
    ax.plot(future, qs[1], label="Simulated median")
    ax.fill_between(future, qs[0], qs[2], alpha=0.25, label="Middle 80% of simulated capped counts")
    ax.set_yscale("symlog", linthresh=1)
    ax.set(xlabel="Date", ylabel="Stars (capped at target)", title="Ensemble fan: model-conditional, not adoption certainty")
    ax.legend()
    ax.grid(alpha=0.2)
    save(fig, "forecast_fan")

    fig, ax = plt.subplots(figsize=(11, 6))
    for path in paths:
        ax.plot(grid / 365.2425, path, alpha=0.18, linewidth=0.7)
    ax.set_yscale("symlog", linthresh=1)
    ax.set(xlabel="Years from last observation", ylabel="Stars (absorbing target)",
           title=f"{len(paths)} sampled futures; display sample only")
    ax.grid(alpha=0.2)
    save(fig, "sample_paths")

    fig, ax = plt.subplots(figsize=(11, 6))
    for name, group in validation.folds.groupby("model", sort=False):
        ax.plot(pd.to_datetime(group["test_end"]), group["log_score"], marker=".", label=name)
    ax.set(xlabel="End of historical test window", ylabel="Log probability of observed integer gain",
           title="Rolling-origin validation (higher is better)")
    ax.legend()
    ax.grid(alpha=0.2)
    save(fig, "validation_scores")

    fig, ax = plt.subplots(figsize=(11, 6))
    w = validation.weights
    ax.bar(w["model"], w["selected_weight"])
    ax.tick_params(axis="x", rotation=20)
    ax.set(ylabel="Forecast combination weight", title="Predictive weights: NOT probabilities that models are true", ylim=(0, 1))
    save(fig, "model_weights")

    fig, ax = plt.subplots(figsize=(11, 6))
    reached = hit[hit >= 0] / 365.2425
    edges = np.linspace(0, grid[-1] / 365.2425, 61)
    if len(reached):
        ax.hist(reached, bins=edges, weights=np.ones(len(reached)) / len(hit))
    ax.set(xlabel="Years from last observation", ylabel="Fraction of ALL simulated paths per bin",
           title=f"Target-time probability mass; {(hit < 0).mean():.1%} unreached within horizon")
    ax.grid(alpha=0.2)
    save(fig, "hitting_mass")
