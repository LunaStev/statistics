from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from growth.cli import configs, main, parse_args
from growth.ecosystem import EcosystemConfig, EcosystemEvent, EcosystemState, load_scenario
from growth.models import CountModel
from growth.simulation import SimulationConfig, run_simulations, schedule, simulate_chunk


def event(**kw):
    return replace(EcosystemEvent("대표작", "killer", 7, 7, 1.0, 100.0, 0.0, 100.0), **kw)


def baseline():
    return [CountModel("long_poisson", "poisson", np.ones(1), np.array([20.0]), np.array([40.0]))]


def run(eco=None, target=1000):
    cfg = SimulationConfig(simulations=80, years=1, step_days=7, target=target, plot_paths=10)
    grid, snapshots, thresholds = schedule(pd.Timestamp("2026-09-24"), cfg)
    return simulate_chunk(0, cfg.simulations, 56, cfg, baseline(), np.ones(1), grid, snapshots, thresholds, eco)


@pytest.mark.parametrize("kw", [{"star_probability": -1}, {"adoption_probability": 1.1},
                               {"additional_audience": True}, {"initial_developers": -1},
                               {"project_half_life_days": 0}, {"regular_leads_per_day": np.inf},
                               {"contribution_max_boost": np.nan}, {"events": (event(probability=2),)},
                               {"events": (event(first_day=0),)}, {"events": (event(kind="typo"),)},
                               {"events": (event(log_sigma=3),)}, {"name": ""}])
def test_invalid_assumptions_rejected(kw):
    with pytest.raises(ValueError):
        replace(EcosystemConfig(), **kw).validate()


def test_no_sources_gives_exact_baseline_even_after_target():
    direct = run(target=90)
    empty = run(EcosystemConfig(), target=90)
    for key in direct:
        assert np.array_equal(direct[key], empty[key])
    assert not empty["ecosystem_snapshots"].any()
    assert (empty["sustained_rate_hit_days_upper"] == -1).all()


def test_disabled_event_is_not_forced_to_happen():
    out = run(EcosystemConfig(events=(event(probability=0),)))
    assert (out["first_event_days"] == -1).all()
    assert (out["ecosystem_snapshots"][:, :, 0] == 0).all()
    assert np.array_equal(out["capped_snapshot_stars"], out["baseline_snapshot_stars"])


def test_baseline_random_stream_is_unchanged_and_pair_is_monotone():
    direct = run()
    eco = run(EcosystemConfig(events=(event(),)))
    assert np.array_equal(direct["capped_snapshot_stars"], eco["baseline_snapshot_stars"])
    assert np.array_equal(direct["hit_days_upper"], eco["baseline_hit_days_upper"])
    assert np.array_equal(eco["capped_snapshot_stars"],
                          np.minimum(eco["baseline_snapshot_stars"].astype(np.int64) + eco["ecosystem_snapshots"][:, :, 0], 1000))
    assert (eco["capped_snapshot_stars"] >= direct["capped_snapshot_stars"]).all()
    assert (eco["ecosystem_snapshots"][:, :, 0] > 0).all()


def test_event_exposure_integrates_only_after_appearance():
    cfg = EcosystemConfig(star_probability=1, adoption_probability=0, events=(event(first_day=6, last_day=6),))
    state = EcosystemState(10, cfg, np.random.default_rng(5))
    state.advance(0, 5)
    assert not state.discovered.any()
    assert (state.first_event == -1).all()
    state.advance(5, 7)
    h = np.log(2)/100
    area = 100 * (-np.expm1(-h))/h  # 한 날짜의 노출만 적분한다.
    expected = cfg.additional_audience * (-np.expm1(-area/cfg.additional_audience))/2
    assert np.allclose(state.previous_rate, expected)
    assert (state.first_event == 6).all()
    assert (state.rate_hits == -1).all()


def test_audience_deduplicates_and_caps_without_capping_user_activity_at_star_target():
    cfg = EcosystemConfig(additional_audience=100, star_probability=1, adoption_probability=1,
                          events=(event(leads_per_day=1_000_000),))
    state = EcosystemState(30, cfg, np.random.default_rng(0))
    previous = np.zeros(30)
    for day in range(7, 301, 7):
        state.advance(day-7, day)
        assert (state.extra_stars >= previous).all()
        assert (state.discovered <= 100).all()
        assert (state.extra_stars <= state.discovered).all()
        previous = state.extra_stars.copy()
    assert (state.extra_stars == 100).all()


def test_development_requires_time_and_completed_projects_can_contribute():
    cfg = EcosystemConfig(initial_developers=1000, project_start_rate_per_day=1,
                          development_mean_days=1, contribution_probability=1,
                          adoption_probability=0, regular_leads_per_day=0,
                          successful_leads_per_day=0)
    state = EcosystemState(5, cfg, np.random.default_rng(0))
    state.advance(0, 1)
    assert not state.completed.any()  # 첫 구간에는 개발을 시작만 한다.
    state.advance(1, 2)
    assert (state.completed > 0).all()
    assert (state.contributors > 0).all()
    assert (state.contributors <= state.completed).all()


def test_rate_proxy_is_not_triggered_by_a_single_instant_spike():
    cfg = EcosystemConfig(additional_audience=1_000_000, star_probability=1,
                          adoption_probability=0, sustained_days=30,
                          events=(event(first_day=1, last_day=1, leads_per_day=100, half_life_days=.1),))
    state = EcosystemState(10, cfg, np.random.default_rng(2))
    for day in range(1, 41):
        state.advance(day-1, day)
    assert (state.rate_hits == -1).all()


def test_rate_proxy_requires_whole_sustained_duration():
    cfg = EcosystemConfig(additional_audience=10_000_000, star_probability=1,
                          adoption_probability=0, sustained_days=30,
                          events=(event(first_day=1, last_day=1, leads_per_day=100, half_life_days=100000),))
    state = EcosystemState(10, cfg, np.random.default_rng(3))
    for day in range(1, 40):
        state.advance(day-1, day)
    assert (state.rate_hits >= 31).all()
    assert (state.rate_hits <= 33).all()


def test_scenario_parallel_and_resume_fingerprint(tmp_path):
    cfg = SimulationConfig(simulations=31, chunk_size=13, target=100, years=1, step_days=7, plot_paths=5)
    eco = EcosystemConfig(events=(event(),))
    grid, snapshots, thresholds = schedule(pd.Timestamp("2026-09-24"), cfg)
    args = (56, cfg, baseline(), np.ones(1), grid, snapshots, thresholds)
    one = run_simulations(tmp_path/'one', *args, jobs=1, ecosystem=eco)
    two = run_simulations(tmp_path/'two', *args, jobs=2, ecosystem=eco)
    for p, q in zip(one, two):
        with np.load(p) as a, np.load(q) as b:
            for key in a.files:
                assert np.array_equal(a[key], b[key])
    old = [p.read_bytes() for p in one]
    run_simulations(tmp_path/'one', *args, resume=True, ecosystem=eco)
    assert old == [p.read_bytes() for p in one]
    with pytest.raises(ValueError, match="mismatch"):
        run_simulations(tmp_path/'one', *args, resume=True, ecosystem=replace(eco, star_probability=.2))
    manifest = json.loads((tmp_path/'one'/'run.json').read_text())
    assert manifest['specification']['ecosystem']['events'][0]['name'] == '대표작'


def test_scenario_file_unknown_keys_and_dates(tmp_path):
    path = tmp_path/'case.toml'
    path.write_text('[scenario]\nunknown = 1\n', encoding='utf-8')
    with pytest.raises(ValueError):
        load_scenario(path, pd.Timestamp('2026-09-24'))
    path.write_text('''[scenario]
name = "시험"
[[events]]
name = "대표작"
kind = "killer"
earliest = "2026-09-24"
latest = "2027-01-01"
probability = 1.0
leads_per_day = 20.0
log_sigma = 0.0
half_life_days = 200.0
''', encoding='utf-8')
    with pytest.raises(ValueError):
        load_scenario(path, pd.Timestamp('2026-09-24'))
    path.write_text(path.read_text().replace('earliest = "2026-09-24"', 'earliest = "2026-09-25"'))
    assert load_scenario(path, pd.Timestamp('2026-09-24')).events[0].first_day == 1


def test_ecosystem_step_defaults_and_validation():
    assert configs(parse_args(['file.csv', '--scenario', 'case.toml']))[1].step_days == 7
    with pytest.raises(ValueError):
        configs(parse_args(['file.csv', '--scenario', 'case.toml', '--step-days', '30']))


def test_korean_scenario_report_end_to_end(tmp_path):
    source = tmp_path/'synthetic.csv'
    y = np.tile([0, 0, 0, 1], 30)
    pd.DataFrame({'Date': pd.date_range('2025-01-01', periods=121),
                  'Stars': np.r_[1, 1+np.cumsum(y)]}).to_csv(source, index=False)
    scenario = tmp_path/'scenario.toml'
    scenario.write_text('''[scenario]
name = "합성 자료 시험"
initial_developers = 30
project_start_rate_per_day = 0.1
''', encoding='utf-8')
    output = tmp_path/'output'
    args = [str(source), '--scenario', str(scenario), '--simulations', '30', '--chunk-size', '11',
            '--years', '1', '--min-train-days', '30', '--grid-size', '41', '--weight-bootstrap', '0',
            '--target', '100', '--plot-paths', '3', '--output', str(output)]
    assert main(args) == 0
    for name in ['요약.txt', '생태계_입력_가정.csv', '기준선과_생태계_비교.csv',
                 '생태계_추이.csv', '추가_유입_유지_시점.csv', 'charts/ecosystem_comparison.png']:
        assert (output/name).is_file()
    body = (output/'report.html').read_text()
    assert 'lang="ko"' in body
    assert '결론부터 보기' in body and '입력한 조건' in body
    assert 'Historical out-of-sample validation' not in body
    assert 'Observed' not in body
    probs = pd.read_csv(output/'target_probabilities.csv')
    assert probs['analytic_probability'].isna().all()
    cmp = pd.read_csv(output/'ecosystem_comparison.csv')
    assert (cmp['difference_pp'] >= 0).all()
    assert main(args+['--resume', '--no-plots']) == 0
