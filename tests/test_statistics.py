import numpy as np
import pandas as pd
import pytest
from scipy.stats import nbinom

from growth.backtest import count_quantile, stacking, validate
from growth.history import History
from growth.models import CountModel, ModelConfig, changepoint_model, fit_models
from growth.report import hitting_quantile, wilson


def history(gains, initial=1):
    stars = np.r_[initial, initial + np.cumsum(gains)].astype(np.int64)
    return History(pd.date_range("2025-01-01", periods=len(stars)), stars, "synthetic/test", "synthetic")


def test_changepoint_preserves_daily_information():
    cfg = ModelConfig(change_days=120, min_segment=20)
    clustered = changepoint_model(history(np.r_[np.zeros(60), np.ones(60)]), cfg)
    uniform = changepoint_model(history(np.tile([0, 1], 60)), cfg)
    assert clustered.diagnostics["change_probability"] > 0.999
    assert uniform.diagnostics["change_probability"] < 0.5
    assert clustered.diagnostics["candidate_date_given_change"] == "2025-03-03"
    assert clustered.diagnostics["candidate_before_rate_mle"] == 0
    assert np.isclose(clustered.weights.sum(), 1)


def test_constant_sequence_favors_no_change():
    m = changepoint_model(history(np.ones(120)), ModelConfig())
    assert m.diagnostics["change_probability"] < 0.5


def test_short_change_window_falls_back():
    m = changepoint_model(history([0, 1, 0]), ModelConfig())
    assert m.diagnostics["change_probability"] == 0
    assert len(m.weights) == 1


def test_gamma_poisson_predictive_matches_negative_binomial():
    m = CountModel("test", "poisson", np.ones(1), np.array([3.5]), np.array([9.0]))
    assert m.logpmf(5, 30, 56) == pytest.approx(nbinom.logpmf(5, 3.5, 9 / 39))
    assert m.target_probability(61, 30, 56) == pytest.approx(nbinom.sf(4, 3.5, 9 / 39))


def test_all_models_score_discrete_probability_mass():
    h = history(np.tile([0, 0, 0, 1], 30))
    for m in fit_models(h, ModelConfig(grid_size=41)):
        assert np.isclose(m.weights.sum(), 1)
        assert m.logpmf(0, 30, 31) <= 0
        mass = sum(np.exp(m.logpmf(k, 30, 31)) for k in range(100))
        assert mass == pytest.approx(m.cdf(99, 30, 31), abs=1e-9)
        assert m.target_probability(31, 0, 31) == 1
        assert m.target_probability(32, 0, 31) == 0


def test_stacking_identical_predictions_and_row_shift():
    identical = np.tile(np.arange(-5, -1)[:, None], (1, 5))
    assert np.allclose(stacking(identical), 0.2)
    scores = np.array([[-1, -3], [-2, -0.1], [-0.2, -0.8]])
    assert np.allclose(stacking(scores), stacking(scores + np.arange(3)[:, None] * 100))


def test_no_holdout_leakage():
    h = history(np.tile([0, 0, 0, 1], 45))
    cfg = ModelConfig(grid_size=41)
    a = validate(h, cfg, min_train=90, horizon=30, bootstrap=0, progress=False)
    changed = History(h.dates, h.stars.copy(), h.repository, h.source_sha256)
    changed.stars[-1] += 100
    b = validate(changed, cfg, min_train=90, horizon=30, bootstrap=0, progress=False)
    # Even the last fold's predictive distribution cannot see its final outcome.
    assert np.array_equal(a.folds[["p10", "median", "p90"]], b.folds[["p10", "median", "p90"]])
    assert np.array_equal(a.folds.iloc[:-6]["log_score"], b.folds.iloc[:-6]["log_score"])


def test_censored_and_conditional_medians_are_different():
    hit = np.array([-1, -1, -1, 12, 15])
    assert hitting_quantile(hit, 0.5) is None
    assert hitting_quantile(hit[hit >= 0], 0.5) == 12
    assert hitting_quantile(np.array([0, 0, 0]), 0.5) == 0


def test_zero_success_does_not_imply_zero_upper_error_bound():
    lo, hi = wilson(0, 100)
    assert lo == pytest.approx(0)
    assert 0 < hi < 0.05


def test_discrete_quantile_inversion():
    for q in [0.1, 0.5, 0.9]:
        result = count_quantile(lambda k: nbinom.cdf(k, 4, 0.2), q)
        assert result == nbinom.ppf(q, 4, 0.2)
