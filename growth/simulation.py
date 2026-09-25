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
from .ecosystem import EcosystemConfig, EcosystemState, METRICS


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
                   snapshots: np.ndarray, thresholds: np.ndarray,
                   ecosystem: EcosystemConfig | None = None) -> dict[str, np.ndarray]:
    # 기준선 난수는 생태계 유무와 무관하다. 쌍을 이룬 비교에서 같은 직접 유입을 사용한다.
    rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, index]))
    ids = rng.choice(len(models), size=size, p=weights).astype(np.uint8)
    theta, aux = np.zeros(size), np.zeros(size)
    for j, model in enumerate(models):
        mask = ids == j
        theta[mask], aux[mask] = model.draw(int(mask.sum()), rng)
    base_stars = np.full(size, min(start, cfg.target), dtype=np.int64)
    hits = np.full((size, len(thresholds)), -1, dtype=np.int32)
    hits[:, thresholds <= start] = 0
    totals = np.full((size, len(snapshots)), cfg.target, dtype=np.uint32)
    sample_n = min(size, cfg.plot_paths) if index == 0 else 0
    paths = np.full((sample_n, len(grid)), cfg.target, dtype=np.uint32)
    paths[:, 0] = base_stars[:sample_n]
    snap_cols = {int(day): i for i, day in enumerate(snapshots)}
    switch, fallback = np.full(size, np.inf), np.zeros(size)
    if cfg.feedback_half_life_years is not None:
        switch = rng.exponential(cfg.feedback_half_life_years * 365.2425 / np.log(2), size=size)
        fallback, _ = models[0].draw(size, rng)
    eco = None
    if ecosystem is not None:
        ecosystem.validate()
        if np.diff(grid).max() > 7:
            raise ValueError("생태계 모형은 --step-days 7 이하가 필요해.")
        eco = EcosystemState(size, ecosystem, np.random.default_rng(
            np.random.SeedSequence([cfg.seed, index, 0xEC05])))
        base_hits, base_totals, base_paths = hits.copy(), totals.copy(), paths.copy()
        eco_totals = np.zeros((size, len(snapshots), len(METRICS)), dtype=np.uint32)
    for step, day in enumerate(grid[1:], start=1):
        previous = int(grid[step - 1])
        dt = int(day - previous)
        for j, model in enumerate(models):
            mask = (ids == j) & (base_stars < cfg.target)
            if not mask.any():
                continue
            a, b = theta[mask], aux[mask]
            if model.kind == "poisson":
                added = rng.poisson(a * dt)
            elif model.kind == "nb":
                added = rng.negative_binomial(b * dt, b / (a + b))
            else:
                growth_dt = np.clip(switch[mask] - previous, 0.0, dt)
                added = rng.negative_binomial(base_stars[mask] + 1.0, np.exp(-a * growth_dt))
                if cfg.feedback_half_life_years is not None:
                    added += rng.poisson(fallback[mask] * (dt - growth_dt))
            base_stars[mask] = np.minimum(base_stars[mask] + added, cfg.target)
        if eco is not None:
            eco.advance(previous, int(day))
            # 추가 스타를 기존 birth 모형에 다시 넣지 않는다.
            stars = np.minimum(base_stars + eco.extra_stars, cfg.target)
            for col, target in enumerate(thresholds):
                base_hits[(base_hits[:, col] < 0) & (base_stars >= target), col] = day
            if int(day) in snap_cols:
                base_totals[:, snap_cols[int(day)]] = base_stars
                eco_totals[:, snap_cols[int(day)], :] = eco.snapshot()
            base_paths[:, step] = base_stars[:sample_n]
        else:
            stars = base_stars
        for col, target in enumerate(thresholds):
            hits[(hits[:, col] < 0) & (stars >= target), col] = day
        if int(day) in snap_cols:
            totals[:, snap_cols[int(day)]] = stars
        paths[:, step] = stars[:sample_n]
        if eco is None and np.all(stars >= cfg.target):
            break
    result = {"model_id": ids, "hit_days_upper": hits,
              "capped_snapshot_stars": totals, "sample_paths": paths}
    if eco is not None:
        result.update(baseline_hit_days_upper=base_hits,
                      baseline_snapshot_stars=base_totals, baseline_sample_paths=base_paths,
                      ecosystem_snapshots=eco_totals, sustained_rate_hit_days_upper=eco.rate_hits,
                      first_event_days=eco.first_event)
    return result


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def run_simulations(output: Path, start: int, cfg: SimulationConfig,
                    models: list[CountModel], weights: np.ndarray, grid: np.ndarray,
                    snapshots: np.ndarray, thresholds: np.ndarray, *, jobs: int = 1,
                    resume: bool = False, metadata: dict | None = None,
                    ecosystem: EcosystemConfig | None = None) -> list[Path]:
    import scipy
    signature = {"version": __version__, "numpy": np.__version__, "scipy": scipy.__version__,
                 "pandas": pd.__version__, "source_code_sha256": sha256(b"".join(
                     p.name.encode() + p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py"))
                 )).hexdigest(), "config": asdict(cfg), "start": start,
                 "weights": weights.tolist(), "grid": grid.tolist(),
                 "snapshot_days": snapshots.tolist(), "thresholds": thresholds.tolist(),
                 "ecosystem": asdict(ecosystem) if ecosystem is not None else None,
                 "metadata": metadata or {}, "model_hashes": [
                     sha256(b"".join(np.asarray(a, dtype=np.float64).tobytes()
                                    for a in [m.weights, m.a, m.b])).hexdigest() for m in models]}
    canonical = json.dumps(signature, sort_keys=True, allow_nan=False).encode()
    fingerprint = sha256(canonical).hexdigest()
    manifest_path = output / "run.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if not resume:
            raise ValueError("실행 결과가 이미 있어. --resume으로 재개하거나 새 --output 폴더를 지정해.")
        if manifest.get("fingerprint") != fingerprint:
            raise ValueError("재개 설정·입력·버전 불일치 [mismatch]. 새 출력 폴더를 사용해.")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("출력 폴더가 비어 있지 않고 일치하는 run.json도 없어.")
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
                raise ValueError(f"체크포인트 누락·손상 [corrupted]: {path}. 복원하거나 새로 실행해.")
        else:
            tasks.append((index, size, start, cfg, models, weights, grid, snapshots, thresholds, ecosystem))
    done = len(paths) - len(tasks)
    print(f"계산 묶음: {done}/{len(paths)}개 확인 완료; 작업자 {jobs}개.")

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
        print(f"\r계산 완료: {done}/{len(paths)}개 묶음", end="", flush=True)

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
