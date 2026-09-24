"""Local CLI: fit, validate, simulate, checkpoint, and report."""
import argparse
from dataclasses import asdict, fields
import json
from pathlib import Path
import sys
import tomllib

import numpy as np

from .backtest import validate
from .history import load_history
from .models import MODEL_NAMES, ModelConfig, fit_models
from .report import build_reports, save_validation
from .simulation import SimulationConfig, run_simulations, schedule


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Count-based GitHub growth scenarios with rolling validation.")
    p.add_argument("csv", help="Daily cumulative CSV: Repository (optional), Date, Stars")
    p.add_argument("--repository")
    p.add_argument("--config", type=Path, help="Optional TOML: [model], [simulation], [validation]")
    p.add_argument("--simulations", "-n", type=int)
    p.add_argument("--chunk-size", type=int)
    p.add_argument("--years", type=int)
    p.add_argument("--target", type=int)
    p.add_argument("--step-days", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--plot-paths", type=int)
    p.add_argument("--feedback-half-life-years", type=float,
                   help="Explicit sensitivity assumption: birth feedback reverts to long arrivals with this survival half-life.")
    p.add_argument("--jobs", "-j", type=int, default=1)
    p.add_argument("--grid-size", type=int)
    p.add_argument("--min-train-days", type=int)
    p.add_argument("--validation-days", type=int)
    p.add_argument("--weight-bootstrap", type=int)
    p.add_argument("--weights", choices=["stacking", "equal"], default="stacking")
    p.add_argument("--model", choices=["ensemble", *MODEL_NAMES], default="ensemble")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--output", "-o", type=Path, default=Path("results/run"))
    return p.parse_args(argv)


def configs(args):
    config = tomllib.loads(args.config.read_text()) if args.config else {}
    unknown = set(config) - {"model", "simulation", "validation"}
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    if any(not isinstance(value, dict) for value in config.values()):
        raise ValueError("Configuration sections must be TOML tables.")
    model = config.get("model", {}).copy()
    sim = config.get("simulation", {}).copy()
    val = {"min_train": 180, "horizon": 30, "bootstrap": 100, **config.get("validation", {})}
    for key in fields(SimulationConfig):
        override = getattr(args, key.name, None)
        if override is not None:
            sim[key.name] = override
    if args.grid_size is not None:
        model["grid_size"] = args.grid_size
    for option, key in [("min_train_days", "min_train"), ("validation_days", "horizon"),
                        ("weight_bootstrap", "bootstrap")]:
        if getattr(args, option) is not None:
            val[key] = getattr(args, option)
    for data, names, label in [(model, {f.name for f in fields(ModelConfig)}, "model"),
                               (sim, {f.name for f in fields(SimulationConfig)}, "simulation"),
                               (val, {"min_train", "horizon", "bootstrap"}, "validation")]:
        if set(data) - names:
            raise ValueError(f"Unknown {label} configuration keys: {sorted(set(data) - names)}")
    m, s = ModelConfig(**model), SimulationConfig(**sim)
    integers = {"jobs": args.jobs, **{k: getattr(s, k) for k in
                ["simulations", "chunk_size", "years", "target", "step_days", "seed", "plot_paths"]},
                **{k: getattr(m, k) for k in ["long_days", "recent_days", "change_days", "min_segment", "grid_size"]},
                **val}
    for name, value in integers.items():
        minimum = 0 if name in {"seed", "plot_paths", "bootstrap"} else 1
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    for name in ["rate_shape", "rate_rate", "birth_shape", "birth_rate"]:
        value = getattr(m, name)
        if not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    if not 0 < m.change_prior < 1:
        raise ValueError("change_prior must be between 0 and 1 (exclusive).")
    if not 41 <= m.grid_size <= 501:
        raise ValueError("grid_size must be between 41 and 501.")
    if s.step_days > 30 or s.years > 100 or s.target > 2_000_000_000:
        raise ValueError("Use step_days <= 30, years <= 100 and target <= 2,000,000,000.")
    if val["horizon"] > 365 or val["min_train"] < 30:
        raise ValueError("validation horizon must be <=365 and min_train >=30.")
    if s.feedback_half_life_years is not None and (not np.isfinite(s.feedback_half_life_years)
                                                 or s.feedback_half_life_years <= 0):
        raise ValueError("feedback_half_life_years must be finite and positive.")
    return m, s, val


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        model_cfg, sim_cfg, val_cfg = configs(args)
        history = load_history(args.csv, args.repository)
        print(f"{history.repository}: {history.dates[0].date()} to {history.dates[-1].date()}, "
              f"{int(history.stars[-1]):,} stars, {len(history.gains)} observed daily increments.")
        validation = validate(history, model_cfg, **val_cfg, seed=sim_cfg.seed, weighting=args.weights)
        models = fit_models(history, model_cfg)
        weights = validation.selected_weights.copy()
        if args.model != "ensemble":
            weights[:] = 0
            weights[list(MODEL_NAMES).index(args.model)] = 1
        validation.weights["simulation_weight"] = weights
        print("\n=== OUT-OF-SAMPLE VALIDATION ===")
        print(validation.summary.to_string(index=False))
        print("\n=== PREDICTIVE WEIGHTS (not model truth probabilities) ===")
        print(validation.weights.to_string(index=False))
        print("\n=== CHANGE / NO-CHANGE POSTERIOR ===")
        print(json.dumps(models[2].diagnostics, indent=2))
        for model in models:
            if model.diagnostics.get("grid_edge_mass_bound", 0) > 0.01:
                print(f"WARNING: {model.name} posterior has boundary mass; inspect grid/prior sensitivity.")
        if args.validate_only:
            if args.output.exists() and any(args.output.iterdir()):
                raise ValueError("Choose an empty --output directory for --validate-only.")
            save_validation(args.output, validation, models, model_cfg)
            print(f"Validation only: {args.output.resolve()}")
            return 0
        grid, snapshots, thresholds = schedule(history.dates[-1], sim_cfg)
        metadata = {"input_sha256": history.source_sha256, "repository": history.repository,
                    "last_date": str(history.dates[-1].date()), "model_config": asdict(model_cfg),
                    "validation_config": val_cfg, "weighting": args.weights, "model_choice": args.model}
        estimated = sim_cfg.simulations * (1 + 4 * len(thresholds) + 4 * len(snapshots)) / 2**20
        print(f"\n{sim_cfg.simulations:,} paths; {len(grid)-1} time intervals; "
              f"report arrays about {estimated:.1f} MiB, plus working memory.")
        chunks = run_simulations(args.output, int(history.stars[-1]), sim_cfg, models, weights,
                                 grid, snapshots, thresholds, jobs=args.jobs, resume=args.resume,
                                 metadata=metadata)
        save_validation(args.output, validation, models, model_cfg)
        summary = build_reports(args.output, history, sim_cfg, models, weights, validation,
                                grid, snapshots, thresholds, chunks, plots=not args.no_plots)
        print("\n=== MODEL-CONDITIONAL RESULTS ===")
        print(summary[["group", "simulations", "mc_hit_probability", "all_paths_median_date_upper",
                       "given_reached_median_date_upper", "unreached_within_horizon"]].to_string(index=False))
        print("\nUnreached means censored at the horizon, not never. mc95 measures simulation error only.")
        print(f"Report: {(args.output / 'report.html').resolve()}")
        return 0
    except (ValueError, TypeError, OSError, RuntimeError, ArithmeticError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted. Completed checkpoints are safe; repeat with --resume.", file=sys.stderr)
        return 130
