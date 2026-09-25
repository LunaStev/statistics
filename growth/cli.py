"""한국어 실행 화면: 과거 검증, 조건부 생태계 비교, 병렬 계산, 재개, 보고서."""
import argparse
from dataclasses import asdict, fields, replace
from pathlib import Path
import sys
import tomllib

import numpy as np

from .backtest import validate
from .ecosystem import assumption_rows, load_scenario
from .history import load_history
from .korean import MODELS, configure_font, save_csv, table
from .models import MODEL_NAMES, ModelConfig, fit_models
from .report import build_reports, save_validation
from .simulation import SimulationConfig, run_simulations, schedule


class KoreanParser(argparse.ArgumentParser):
    def format_help(self):
        return super().format_help().replace("usage:", "사용법:").replace("show this help message and exit", "도움말을 표시하고 종료")

    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(2, f"인수를 확인해: {message}\n")


def parse_args(argv=None):
    p = KoreanParser(description="Wave 성장 시뮬레이션: 실제 스타 이력과 별도로 입력한 생태계 조건을 비교해.")
    p._positionals.title, p._optionals.title = "필수 인수", "설정 옵션"
    p.add_argument("csv", help="일별 누적 스타 CSV: Date, Stars, 선택적으로 Repository 열")
    p.add_argument("--repository", help="여러 저장소가 들어 있으면 분석할 저장소 이름")
    p.add_argument("--config", type=Path, help="기존 모형·계산·검증 설정 TOML")
    p.add_argument("--scenario", type=Path, help="생태계 가정 TOML. 생략하면 기존 기준선만 계산해")
    for flag, alias, help_text in [
        ("simulations", "-n", "가상 미래 개수. 기본 100만"),
        ("jobs", "-j", "병렬 작업자 수. 기본 1"),
        ("chunk-size", None, "한 번에 처리할 경로 수. 기본 25000"),
        ("years", None, "예측 기간(년). 기본 30"),
        ("target", None, "목표 스타 수. 기본 100000"),
        ("step-days", None, "계산 간격(일). 기본 30, 생태계 사용 시 기본 7"),
        ("seed", None, "재현을 위한 난수 시드"),
        ("plot-paths", None, "그래프에 표시할 일부 경로 개수"),
        ("grid-size", None, "사후분포 수치 적분 격자 크기"),
        ("min-train-days", None, "첫 검증에 필요한 최소 학습일. 기본 180"),
        ("validation-days", None, "각 과거 검증 구간 길이(일). 기본 30"),
        ("weight-bootstrap", None, "검증 비중 재표본화 횟수. 기본 100"),
    ]:
        p.add_argument(*([f"--{flag}", alias] if alias else [f"--{flag}"]), type=int,
                       default=1 if flag == "jobs" else None, help=help_text)
    p.add_argument("--feedback-half-life-years", type=float, help="별도 가정: 스타 비례 성장의 지속 반감기(년)")
    p.add_argument("--weights", choices=["stacking", "equal"], default="stacking", help="stacking=과거 검증 비중, equal=동일 비중")
    p.add_argument("--model", choices=["ensemble", *MODEL_NAMES], default="ensemble", help="기준선 모형 선택. ensemble=전체 조합")
    p.add_argument("--resume", action="store_true", help="같은 버전·입력·설정의 완료한 묶음을 재사용")
    p.add_argument("--validate-only", action="store_true", help="대규모 계산 없이 과거 예측 시험과 설정 확인만 수행")
    p.add_argument("--no-plots", action="store_true", help="이미지 없이 한국어 표·텍스트·HTML만 저장")
    p.add_argument("--output", "-o", type=Path, default=Path("results/run"), help="새 결과 폴더. 기본 results/run")
    return p.parse_args(argv)


def configs(args):
    config = tomllib.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    if set(config) - {"model", "simulation", "validation"}:
        raise ValueError("기본 설정은 [model], [simulation], [validation]만 지원해.")
    if any(not isinstance(value, dict) for value in config.values()):
        raise ValueError("설정 구역은 TOML 표로 작성해.")
    model, sim = config.get("model", {}).copy(), config.get("simulation", {}).copy()
    val = {"min_train": 180, "horizon": 30, "bootstrap": 100, **config.get("validation", {})}
    for key in fields(SimulationConfig):
        override = getattr(args, key.name, None)
        if override is not None:
            sim[key.name] = override
    if args.grid_size is not None:
        model["grid_size"] = args.grid_size
    for option, key in [("min_train_days", "min_train"), ("validation_days", "horizon"), ("weight_bootstrap", "bootstrap")]:
        if getattr(args, option) is not None:
            val[key] = getattr(args, option)
    for data, names, label in [(model, {f.name for f in fields(ModelConfig)}, "모형"),
                               (sim, {f.name for f in fields(SimulationConfig)}, "계산"),
                               (val, {"min_train", "horizon", "bootstrap"}, "검증")]:
        if set(data) - names:
            raise ValueError(f"알 수 없는 {label} 설정: {sorted(set(data)-names)}")
    m, s = ModelConfig(**model), SimulationConfig(**sim)
    integers = {"jobs": args.jobs, **{k: getattr(s, k) for k in
                ["simulations", "chunk_size", "years", "target", "step_days", "seed", "plot_paths"]},
                **{k: getattr(m, k) for k in ["long_days", "recent_days", "change_days", "min_segment", "grid_size"]}, **val}
    for name, value in integers.items():
        minimum = 0 if name in {"seed", "plot_paths", "bootstrap"} else 1
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name}: {minimum} 이상 정수가 필요해.")
    for name in ["rate_shape", "rate_rate", "birth_shape", "birth_rate"]:
        value = getattr(m, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name}: 유한한 양수가 필요해.")
    if isinstance(m.change_prior, bool) or not isinstance(m.change_prior, (float, int)) or not 0 < m.change_prior < 1:
        raise ValueError("change_prior는 0과 1 사이여야 해.")
    if not 41 <= m.grid_size <= 501:
        raise ValueError("grid_size는 41~501 범위로 설정해.")
    if s.step_days > 30 or s.years > 100 or s.target > 2_000_000_000:
        raise ValueError("계산 간격은 최대 30일, 예측은 최대 100년, 목표는 최대 20억 스타야.")
    if val["horizon"] > 365 or val["min_train"] < 30:
        raise ValueError("검증 기간은 최대 365일, 최초 학습 기간은 최소 30일이야.")
    h = s.feedback_half_life_years
    if h is not None and (isinstance(h, bool) or not isinstance(h, (float, int)) or not np.isfinite(h) or h <= 0):
        raise ValueError("피드백 지속 반감기는 유한한 양수여야 해.")
    if args.scenario:
        if "step_days" not in sim:
            s = replace(s, step_days=7)
        elif s.step_days > 7:
            raise ValueError("생태계 계산에는 --step-days 7 이하를 사용해.")
    return m, s, val


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        model_cfg, sim_cfg, val_cfg = configs(args)
        history = load_history(args.csv, args.repository)
        ecosystem = load_scenario(args.scenario, history.dates[-1]) if args.scenario else None
        if not args.no_plots and not args.validate_only:
            configure_font()  # 긴 계산 전에 한글 표시 환경을 확인한다.
        print(f"분석 대상: {history.repository}\n자료 기간: {history.dates[0].date()} ~ {history.dates[-1].date()}\n"
              f"현재 스타: {int(history.stars[-1]):,}개 / 관측 증가량 {len(history.gains):,}일")
        if ecosystem:
            print(f"생태계 조건: {ecosystem.name}\n사건·전환율은 입력한 가정이며 실제 발생 확률을 추정한 값이 아니야.")
        validation = validate(history, model_cfg, **val_cfg, seed=sim_cfg.seed, weighting=args.weights)
        models = fit_models(history, model_cfg)
        weights = validation.selected_weights.copy()
        if args.model != "ensemble":
            weights[:] = 0
            weights[list(MODEL_NAMES).index(args.model)] = 1
        validation.weights["simulation_weight"] = weights
        print("\n모형별 실행 비중 — 모형이 참일 확률은 아니야")
        print(table(validation.weights[["model", "simulation_weight"]]).to_string(index=False))
        for model in models:
            if model.diagnostics.get("grid_edge_mass_bound", 0) > .01:
                print(f"확인 필요: {MODELS[model.name]}의 격자 경계 질량이 커. 격자·사전분포 민감도를 확인해.")
        if args.validate_only:
            if args.output.exists() and any(args.output.iterdir()):
                raise ValueError("검증 전용 실행에는 빈 출력 폴더를 지정해.")
            save_validation(args.output, validation, models, model_cfg)
            if ecosystem:
                import pandas as pd
                save_csv(args.output, "ecosystem_assumptions", pd.DataFrame(assumption_rows(ecosystem, history.dates[-1])))
            print("\n과거 예측 시험 결과")
            print(table(validation.summary).to_string(index=False))
            print(f"검증 결과 저장: {args.output.resolve()}")
            return 0
        grid, snapshots, thresholds = schedule(history.dates[-1], sim_cfg)
        if ecosystem:
            for event in ecosystem.events:
                if event.first_day > grid[-1]:
                    print(f"참고: '{event.name}'은 예측 기간 밖의 사건이라 이번 기간에는 발생하지 않아.")
        metadata = {"input_sha256": history.source_sha256, "repository": history.repository,
                    "last_date": str(history.dates[-1].date()), "model_config": asdict(model_cfg),
                    "validation_config": val_cfg, "weighting": args.weights, "model_choice": args.model}
        print(f"\n가상 미래 {sim_cfg.simulations:,}개 / {len(grid)-1:,}개 시간 구간 / 작업자 {args.jobs}개")
        chunks = run_simulations(args.output, int(history.stars[-1]), sim_cfg, models, weights,
                                 grid, snapshots, thresholds, jobs=args.jobs, resume=args.resume,
                                 metadata=metadata, ecosystem=ecosystem)
        save_validation(args.output, validation, models, model_cfg)
        build_reports(args.output, history, sim_cfg, models, weights, validation, grid, snapshots,
                      thresholds, chunks, plots=not args.no_plots, ecosystem=ecosystem)
        print("\n" + (args.output / "요약.txt").read_text(encoding="utf-8"))
        print(f"\n한국어 보고서: {(args.output / 'report.html').resolve()}")
        return 0
    except (ValueError, TypeError, OSError, RuntimeError, ArithmeticError) as error:
        print(f"실행을 마치지 못했어: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n중단했어. 완료된 묶음은 보관했으니 같은 명령에 --resume을 붙여 재개해.", file=sys.stderr)
        return 130
