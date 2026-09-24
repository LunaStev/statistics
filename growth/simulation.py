"""Chunked, reproducible, resumable count paths with right-censored hitting times."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .models import CountModel


@dataclass(frozen=True)
class SimulationConfig:
    simulations: int = 1_000_000
    chunk_size: int = 25_000
    years: int = 30
    target: int = 100_000
    step_days: int = 30
    seed: int = 20260924
    plot_paths: int = 120
    feedback_half_life_years: float | None = None


def schedule(last_date: pd.Timestamp, cfg: SimulationConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    years = sorted(set(y for y in [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, cfg.years] if y <= cfg.years))
    snapshots = np.array([(last_date + pd.DateOffset(years=y) - last_date).days for y in years])
    horizon = int(snapshots[-1])
    grid = np.unique(np.r_[0, np.arange(cfg.step_days, horizon, cfg.step_days), snapshots]).astype(int)
    thresholds = np.array(sorted(set(x for x in [100, 250, 500, 1000, 2500, 5000,
                                                   10000, 25000, 50000, cfg.target] if x <= cfg.target)))
    return grid, snapshots, thresholds


def simulate_chunk(index: int, size: int, start: int, cfg: SimulationConfig,
                   models: list[CountModel], weights: np.ndarray, grid: np.ndarray,
                   snapshots: np.ndarray, thresholds: np.ndarray) -> dict[str, np.ndarray]:
    # One stable stream per chunk. Worker scheduling/resume does not alter results.
    rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, index]))
    ids = rng.choice(len(models), size=size, p=weights).astype(np.uint8)
    theta, aux = np.zeros(size), np.zeros(size)
    for j, model in enumerate(models):
        mask = ids == j
        theta[mask], aux[mask] = model.draw(int(mask.sum()), rng)
    stars = np.full(size, min(start, cfg.target), dtype=np.int64)
    hits = np.full((size, len(thresholds)), -1, dtype=np.int32)
    hits[:, thresholds <= start] = 0
    totals = np.full((size, len(snapshots)), cfg.target, dtype=np.uint32)
    sample_n = min(size, cfg.plot_paths) if index == 0 else 0
    paths = np.full((sample_n, len(grid)), cfg.target, dtype=np.uint32)
    paths[:, 0] = stars[:sample_n]
    snap_cols = {int(day): i for i, day in enumerate(snapshots)}
    switch, fallback = np.full(size, np.inf), np.zeros(size)
    if cfg.feedback_half_life_years is not None:
        switch = rng.exponential(cfg.feedback_half_life_years * 365.2425 / np.log(2), size=size)
        fallback, _ = models[0].draw(size, rng)  # long-arrival fallback, not zero development
    for step, day in enumerate(grid[1:], start=1):
        previous = int(grid[step - 1])
        dt = int(day - previous)
        for j, model in enumerate(models):
            mask = (ids == j) & (stars < cfg.target)
            count = int(mask.sum())
            if not count:
                continue
            a, b = theta[mask], aux[mask]
            if model.kind == "poisson":
                added = rng.poisson(a * dt)
            elif model.kind == "nb":
                added = rng.negative_binomial(b * dt, b / (a + b))
            else:
                growth_dt = np.clip(switch[mask] - previous, 0.0, dt)
                added = rng.negative_binomial(stars[mask] + 1.0, np.exp(-a * growth_dt))
                if cfg.feedback_half_life_years is not None:
                    added += rng.poisson(fallback[mask] * (dt - growth_dt))
            stars[mask] = np.minimum(stars[mask] + added, cfg.target)
        for col, target in enumerate(thresholds):
            new = (hits[:, col] < 0) & (stars >= target)
            hits[new, col] = day
        if int(day) in snap_cols:
            totals[:, snap_cols[int(day)]] = stars
        paths[:, step] = stars[:sample_n]
        if np.all(stars >= cfg.target):
            break  # remaining totals/paths were prefilled with the absorbing target
    return {"model_id": ids, "hit_days_upper": hits,
            "capped_snapshot_stars": totals, "sample_paths": paths}


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def run_simulations(output: Path, start: int, cfg: SimulationConfig,
                    models: list[CountModel], weights: np.ndarray, grid: np.ndarray,
                    snapshots: np.ndarray, thresholds: np.ndarray, *, jobs: int = 1,
                    resume: bool = False, metadata: dict | None = None) -> list[Path]:
    import scipy
    signature = {"version": __version__, "numpy": np.__version__, "scipy": scipy.__version__,
                 "pandas": pd.__version__, "source_code_sha256": sha256(b"".join(
                     p.name.encode() + p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py"))
                 )).hexdigest(), "config": asdict(cfg), "start": start,
                 "weights": weights.tolist(), "grid": grid.tolist(),
                 "snapshot_days": snapshots.tolist(), "thresholds": thresholds.tolist(),
                 "metadata": metadata or {}, "model_hashes": [
                     sha256(b"".join(np.asarray(a, dtype=np.float64).tobytes()
                                    for a in [m.weights, m.a, m.b])).hexdigest() for m in models]}
    canonical = json.dumps(signature, sort_keys=True, allow_nan=False).encode()
    fingerprint = sha256(canonical).hexdigest()
    manifest_path = output / "run.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if not resume:
            raise ValueError("Output already has a run. Use --resume or a new --output directory.")
        if manifest.get("fingerprint") != fingerprint:
            raise ValueError("Resume configuration/input/dependency mismatch. Use a new output directory.")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Output directory is not empty and has no matching run.json.")
        output.mkdir(parents=True, exist_ok=True)
        manifest = {"fingerprint": fingerprint, "specification": signature, "chunks": {}}
        atomic_json(manifest_path, manifest)
    chunks = output / "chunks"
    chunks.mkdir(exist_ok=True)
    tasks = []
    paths = []
    for index, offset in enumerate(range(0, cfg.simulations, cfg.chunk_size)):
        size = min(cfg.chunk_size, cfg.simulations - offset)
        path = chunks / f"chunk-{index:06d}.npz"
        paths.append(path)
        old_hash = manifest["chunks"].get(str(index))
        if old_hash is not None:
            if not path.exists() or sha256(path.read_bytes()).hexdigest() != old_hash:
                raise ValueError(f"Missing/corrupted checkpoint: {path}. Restore it or start a new run.")
        else:
            tasks.append((index, size, start, cfg, models, weights, grid, snapshots, thresholds))
    done = len(paths) - len(tasks)
    print(f"Simulation chunks: {done}/{len(paths)} already verified; {jobs} worker(s).")

    def persist(index: int, result: dict[str, np.ndarray]) -> None:
        nonlocal done
        path = paths[index]
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **result)
        os.replace(temporary, path)
        manifest["chunks"][str(index)] = sha256(path.read_bytes()).hexdigest()
        atomic_json(manifest_path, manifest)
        done += 1
        print(f"\rSimulation chunks completed: {done}/{len(paths)}", end="", flush=True)

    if jobs == 1:
        for args in tasks:
            persist(args[0], simulate_chunk(*args))
    elif tasks:
        # Bounded in-flight results: no unbounded futures or N x days allocation.
        with ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
            iterator = iter(tasks)
            pending = {}
            for _ in range(min(2 * jobs, len(tasks))):
                args = next(iterator)
                pending[pool.submit(simulate_chunk, *args)] = args[0]
            while pending:
                future = next(as_completed(pending))
                index = pending.pop(future)
                persist(index, future.result())
                try:
                    args = next(iterator)
                except StopIteration:
                    continue
                pending[pool.submit(simulate_chunk, *args)] = args[0]
    print()
    return paths
