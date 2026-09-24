"""Rolling-origin count forecasts and predictive stacking, NOT model probabilities."""
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

from .history import History
from .models import MODEL_NAMES, ModelConfig, fit_models


def stacking(log_scores: np.ndarray) -> np.ndarray:
    if log_scores.ndim != 2 or not np.isfinite(log_scores).all():
        raise ValueError("Stacking needs a finite folds-by-models log-PMF matrix.")
    m = log_scores.shape[1]
    if len(log_scores) < 2:
        return np.full(m, 1.0 / m)
    # Row shifts preserve the optimizer, avoiding underflow in probabilities.
    p = np.exp(log_scores - log_scores.max(axis=1, keepdims=True))

    def objective(w: np.ndarray) -> tuple[float, np.ndarray]:
        mixture = np.maximum(p @ w, np.finfo(float).tiny)
        return -float(np.log(mixture).mean()), -(p / mixture[:, None]).mean(axis=0)

    fit = minimize(objective, np.full(m, 1.0 / m), jac=True, method="SLSQP",
                   bounds=[(0.0, 1.0)] * m,
                   constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0,
                                "jac": lambda w: np.ones_like(w)},
                   options={"ftol": 1e-10, "maxiter": 1000})
    if not fit.success or not np.isfinite(fit.x).all():
        raise RuntimeError(f"Stacking optimization failed: {fit.message}")
    w = np.maximum(fit.x, 0)
    return w / w.sum()


def count_quantile(cdf: Callable[[int], float], probability: float) -> int:
    if cdf(0) >= probability:
        return 0
    low, high = 0, 1
    while cdf(high) < probability:
        high *= 2
        if high > 2**52:
            raise ArithmeticError("Predictive quantile exceeds numerical count range.")
    while high - low > 1:
        mid = (low + high) // 2
        if cdf(mid) >= probability:
            high = mid
        else:
            low = mid
    return high


@dataclass
class Validation:
    folds: pd.DataFrame
    summary: pd.DataFrame
    weights: pd.DataFrame
    selected_weights: np.ndarray


def validate(history: History, cfg: ModelConfig, *, horizon: int = 30,
             min_train: int = 180, bootstrap: int = 100, seed: int = 42,
             weighting: str = "stacking", progress: bool = True) -> Validation:
    n = len(history.gains)
    origins = list(range(n - horizon, min_train - 1, -horizon))[::-1]
    if len(origins) < 3:
        raise ValueError("Need at least three non-overlapping validation windows. "
                         "Supply more history or reduce --min-train-days/--validation-days.")
    score_rows: list[np.ndarray] = []
    rows: list[dict] = []
    for fold, origin in enumerate(origins):
        # No future observation enters fitting, change-point selection or weights.
        models = fit_models(history.prefix(origin), cfg)
        truth = int(history.stars[origin + horizon] - history.stars[origin])
        start = int(history.stars[origin])
        w = (stacking(np.vstack(score_rows)) if len(score_rows) >= 2 and weighting == "stacking"
             else np.full(len(models), 1 / len(models)))
        scores = np.array([model.logpmf(truth, horizon, start) for model in models])
        if not np.isfinite(scores).all():
            raise ArithmeticError("Non-finite count predictive score.")
        for model, score in zip(models, scores):
            def cdf(k, model=model):
                return model.cdf(k, horizon, start)
            lo, median, hi = [count_quantile(cdf, q) for q in (0.1, 0.5, 0.9)]
            rows.append(dict(fold=fold, train_end=str(history.dates[origin].date()),
                             test_end=str(history.dates[origin + horizon].date()),
                             model=model.name, observed_gain=truth, log_score=score,
                             p10=lo, median=median, p90=hi,
                             covered80=lo <= truth <= hi, absolute_error=abs(median - truth)))
        def cdf(k):
            return sum(wi * mi.cdf(k, horizon, start) for wi, mi in zip(w, models))
        lo, median, hi = [count_quantile(cdf, q) for q in (0.1, 0.5, 0.9)]
        with np.errstate(divide="ignore"):
            ensemble_score = float(logsumexp(np.log(w) + scores))
        rows.append(dict(fold=fold, train_end=str(history.dates[origin].date()),
                         test_end=str(history.dates[origin + horizon].date()),
                         model="prequential_ensemble", observed_gain=truth,
                         log_score=ensemble_score, p10=lo, median=median, p90=hi,
                         covered80=lo <= truth <= hi, absolute_error=abs(median - truth)))
        score_rows.append(scores)
        if progress:
            print(f"\rValidated {fold + 1}/{len(origins)} historical windows", end="", flush=True)
    if progress:
        print()
    scores = np.vstack(score_rows)
    fitted = stacking(scores)
    equal = np.full(len(MODEL_NAMES), 1 / len(MODEL_NAMES))
    selected = fitted if weighting == "stacking" else equal
    weight_rows = pd.DataFrame({"model": MODEL_NAMES, "stacking_weight": fitted,
                                "equal_weight": equal, "selected_weight": selected})
    if bootstrap:
        # Moving-block sensitivity resampling; not a posterior confidence interval.
        rng = np.random.default_rng(seed)
        block = min(3, len(origins))
        draws = []
        for _ in range(bootstrap):
            starts = rng.integers(0, len(origins) - block + 1,
                                  size=int(np.ceil(len(origins) / block)))
            indices = np.concatenate([np.arange(s, s + block) for s in starts])[:len(origins)]
            draws.append(stacking(scores[indices]))
        lo, hi = np.quantile(draws, [0.1, 0.9], axis=0)
        weight_rows["block_bootstrap_p10"] = lo
        weight_rows["block_bootstrap_p90"] = hi
    folds = pd.DataFrame(rows)
    summary = folds.groupby("model", sort=False).agg(
        windows=("fold", "count"), mean_log_score=("log_score", "mean"),
        median_prediction_MAE=("absolute_error", "mean"), coverage80=("covered80", "mean")
    ).reset_index()
    return Validation(folds, summary, weight_rows, selected)
