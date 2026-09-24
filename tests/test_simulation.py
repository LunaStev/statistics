from dataclasses import replace
import numpy as np
import pandas as pd
import pytest

from growth.models import CountModel
from growth.simulation import SimulationConfig, run_simulations, schedule, simulate_chunk


def model(kind):
    values = {"poisson": (20.0, 40.0), "nb": (0.5, 0.25), "birth": (0.025, 0.0)}
    a, b = values[kind]
    return CountModel(kind, kind, np.ones(1), np.array([a]), np.array([b]))


@pytest.mark.parametrize("kind", ["poisson", "nb", "birth"])
def test_simulated_target_probability_matches_analytic(kind):
    m = model(kind)
    cfg = SimulationConfig(simulations=3000, target=35, years=1, plot_paths=20)
    grid = np.array([0, 10, 20, 30])
    result = simulate_chunk(0, 3000, 20, cfg, [m], np.ones(1), grid,
                            np.array([30]), np.array([25, 35]))
    estimate = (result["hit_days_upper"][:, -1] >= 0).mean()
    assert estimate == pytest.approx(m.target_probability(35, 30, 20), abs=0.035)
    assert (np.diff(result["sample_paths"].astype(np.int64), axis=1) >= 0).all()
    assert result["capped_snapshot_stars"].max() <= 35
    hits = result["hit_days_upper"]
    assert np.all((hits[:, 1] < 0) | ((hits[:, 0] >= 0) & (hits[:, 0] <= hits[:, 1])))


def test_already_reached():
    cfg = SimulationConfig(simulations=10, target=10, plot_paths=5)
    out = simulate_chunk(0, 10, 20, cfg, [model("poisson")], np.ones(1),
                         np.array([0, 30]), np.array([30]), np.array([10]))
    assert (out["hit_days_upper"] == 0).all()
    assert (out["capped_snapshot_stars"] == 10).all()


def test_calendar_schedule_does_not_treat_year_as_360_days():
    grid, snapshots, _ = schedule(pd.Timestamp("2026-09-24"), SimulationConfig(years=2))
    assert snapshots.tolist() == [365, 731]
    assert grid[-1] == 731
    assert np.diff(grid).max() <= 30


def test_reproducible_and_resumable_across_worker_count(tmp_path):
    cfg = SimulationConfig(simulations=41, chunk_size=13, target=40, years=1, plot_paths=5)
    grid, snapshots, thresholds = schedule(pd.Timestamp("2026-09-24"), cfg)
    args = (20, cfg, [model("poisson")], np.ones(1), grid, snapshots, thresholds)
    one = run_simulations(tmp_path / "one", *args, jobs=1)
    two = run_simulations(tmp_path / "two", *args, jobs=2)
    for p, q in zip(one, two):
        with np.load(p) as a, np.load(q) as b:
            for key in a.files:
                assert np.array_equal(a[key], b[key])
    old = [p.read_bytes() for p in one]
    run_simulations(tmp_path / "one", *args, jobs=2, resume=True)
    assert old == [p.read_bytes() for p in one]
    with pytest.raises(ValueError, match="mismatch"):
        run_simulations(tmp_path / "one", 20, replace(cfg, seed=7), *args[2:], resume=True)
    one[0].write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupted"):
        run_simulations(tmp_path / "one", *args, resume=True)


def test_feedback_scenario_can_revert_without_stopping_development():
    cfg = SimulationConfig(simulations=100, target=1000, plot_paths=10,
                           feedback_half_life_years=1e-9)
    models = [model("poisson"), model("birth")]
    out = simulate_chunk(0, 100, 20, cfg, models, np.array([0.0, 1.0]),
                         np.array([0, 30]), np.array([30]), np.array([1000]))
    # Fallback still receives additive arrivals, despite vanishing feedback duration.
    assert 25 < out["capped_snapshot_stars"].mean() < 45
