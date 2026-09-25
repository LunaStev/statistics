"""사용자가 정한 생태계 가정. 스타 이력에서 학습한 확률로 취급하지 않는다.

기존 유입과 겹치지 않는 추가 잠재 독자 집단을 가정한다. 이 집단에서
발견 -> 사용 -> 프로젝트 개발 -> 공개 -> 새 발견 / 언어 기여를 모의한다.
기존 모델의 스타와 추가 스타를 분리하여 두 피드백을 중복 적용하지 않는다.
시간 이산화된 집단 모형이며 1일/7일 간격 민감도 검사가 필요하다.
"""
from dataclasses import asdict, dataclass, fields
from datetime import date
from pathlib import Path
import tomllib

import numpy as np
import pandas as pd


EVENT_KINDS = {"killer": "대표 프로젝트", "rewrite": "기존 프로젝트 재작성", "module": "모듈 도입"}
METRICS = ("extra_stars", "discovered", "active_developers", "active_projects",
           "completed_projects", "successful_projects", "active_contributors")
RATE_THRESHOLDS = np.array([1.0, 5.0, 10.0, 30.0])


@dataclass(frozen=True)
class EcosystemEvent:
    name: str
    kind: str
    first_day: int
    last_day: int
    probability: float
    leads_per_day: float
    log_sigma: float
    half_life_days: float


@dataclass(frozen=True)
class EcosystemConfig:
    name: str = "생태계 가정 실험"
    additional_audience: int = 1_000_000
    initial_developers: int = 0
    star_probability: float = 0.10
    adoption_probability: float = 0.02
    project_start_rate_per_day: float = 0.001
    development_mean_days: float = 90.0
    developer_half_life_days: float = 365.0
    project_half_life_days: float = 730.0
    regular_leads_per_day: float = 0.10
    successful_leads_per_day: float = 20.0
    project_success_probability: float = 0.01
    contribution_probability: float = 0.10
    contributor_half_life_days: float = 365.0
    contribution_max_boost: float = 1.0
    contribution_scale: float = 20.0
    sustained_days: int = 30
    events: tuple[EcosystemEvent, ...] = ()

    def validate(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("시나리오 이름은 비어 있지 않은 문자열이어야 해.")
        for key, minimum in [("additional_audience", 1), ("initial_developers", 0), ("sustained_days", 1)]:
            value = getattr(self, key)
            if type(value) is not int or not minimum <= value <= 2_000_000_000:
                raise ValueError(f"{key}: {minimum} 이상 20억 이하 정수가 필요해.")
        if self.initial_developers > self.additional_audience:
            raise ValueError("초기 개발자 수는 추가 잠재 독자 수를 넘을 수 없어.")
        probabilities = ["star_probability", "adoption_probability", "project_success_probability",
                         "contribution_probability"]
        positive = ["development_mean_days", "developer_half_life_days", "project_half_life_days",
                    "contributor_half_life_days", "contribution_scale"]
        nonnegative = ["project_start_rate_per_day", "regular_leads_per_day",
                       "successful_leads_per_day", "contribution_max_boost"]
        for key in probabilities + positive + nonnegative:
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f"{key}: 유한한 숫자가 필요해.")
            if key in probabilities and not 0 <= value <= 1:
                raise ValueError(f"{key}: 확률은 0~1이어야 해.")
            if key in positive and value <= 0:
                raise ValueError(f"{key}: 0보다 커야 해.")
            if key in nonnegative and not 0 <= value <= 1_000_000:
                raise ValueError(f"{key}: 0~100만 범위가 필요해.")
        if self.project_start_rate_per_day > 1:
            raise ValueError("개발자당 프로젝트 시작률은 하루 1 이하로 설정해.")
        if len(self.events) > 100:
            raise ValueError("사건은 한 시나리오에 최대 100개까지 설정해.")
        for event in self.events:
            if event.kind not in EVENT_KINDS or not isinstance(event.name, str) or not event.name.strip():
                raise ValueError("사건 이름과 종류(killer/rewrite/module)를 확인해.")
            if type(event.first_day) is not int or type(event.last_day) is not int or not 0 < event.first_day <= event.last_day:
                raise ValueError("사건은 관측 종료일 이후여야 하며 시작일이 종료일보다 늦을 수 없어.")
            for key in ["probability", "leads_per_day", "log_sigma", "half_life_days"]:
                value = getattr(event, key)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                    raise ValueError(f"사건 {event.name}: {key}에 유한한 숫자가 필요해.")
            if not 0 <= event.probability <= 1 or not 0 <= event.leads_per_day <= 1_000_000:
                raise ValueError("사건 발생 확률은 0~1, 하루 도달 시도는 0~100만이어야 해.")
            if not 0 <= event.log_sigma <= 2 or event.half_life_days <= 0:
                raise ValueError("사건 변동성은 0~2, 영향 반감기는 양수여야 해.")


def load_scenario(path: Path, last_date: pd.Timestamp) -> EcosystemConfig:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    if set(data) - {"scenario", "events"}:
        raise ValueError("시나리오 파일은 [scenario], [[events]]만 지원해.")
    params = data.get("scenario", {})
    if not isinstance(params, dict) or set(params) - {f.name for f in fields(EcosystemConfig) if f.name != "events"}:
        raise ValueError("[scenario] 설정 이름을 확인해.")
    events = data.get("events", [])
    if not isinstance(events, list):
        raise ValueError("사건은 [[events]] 배열로 작성해.")
    parsed = []
    for item in events:
        required = {"name", "kind", "earliest", "latest", "probability", "leads_per_day", "log_sigma", "half_life_days"}
        if not isinstance(item, dict) or set(item) != required:
            raise ValueError(f"사건에는 다음 항목이 모두 필요해: {', '.join(sorted(required))}")
        def offset(key):
            value = str(item[key])
            parsed_date = date.fromisoformat(value)
            if parsed_date.isoformat() != value:
                raise ValueError("사건 날짜는 YYYY-MM-DD로 작성해.")
            return (pd.Timestamp(parsed_date) - last_date).days
        parsed.append(EcosystemEvent(item["name"], item["kind"], int(offset("earliest")),
                                     int(offset("latest")), item["probability"], item["leads_per_day"],
                                     item["log_sigma"], item["half_life_days"]))
    result = EcosystemConfig(**params, events=tuple(parsed))
    result.validate()
    return result


def assumption_rows(cfg: EcosystemConfig, last_date: pd.Timestamp) -> list[dict]:
    rows = [{"parameter": k, "value": v, "origin": "사용자 입력 가정(스타 이력에서 추정하지 않음)"}
            for k, v in asdict(cfg).items() if k != "events"]
    for event in cfg.events:
        first = (last_date + pd.Timedelta(days=event.first_day)).date()
        last = (last_date + pd.Timedelta(days=event.last_day)).date()
        rows.append({"parameter": f"{EVENT_KINDS[event.kind]}: {event.name}",
                     "value": f"{first}~{last}; 발생 조건 {event.probability:.1%}; 초기 도달 시도 {event.leads_per_day:g}명/일; 영향 반감기 {event.half_life_days:g}일",
                     "origin": "조건부 사건: 실제 등장 확률을 추정한 값이 아님"})
    return rows


class EcosystemState:
    """한 청크의 집단을 벡터로 저장한다. 프로젝트별 객체/전체 일별 경로는 만들지 않는다."""

    def __init__(self, size: int, cfg: EcosystemConfig, rng: np.random.Generator):
        self.cfg, self.rng = cfg, rng
        self.idle = np.full(size, cfg.initial_developers, dtype=np.int64)
        self.building = np.zeros(size, dtype=np.int64)
        self.regular = np.zeros(size, dtype=np.int64)
        self.successful = np.zeros(size, dtype=np.int64)
        self.contributors = np.zeros(size, dtype=np.int64)
        # 초기 개발자는 이미 발견된 사람이며 다시 별/채택으로 세지 않는다.
        self.discovered = self.idle.copy()
        self.extra_stars = np.zeros(size, dtype=np.int64)
        self.completed = np.zeros(size, dtype=np.int64)
        self.success_total = np.zeros(size, dtype=np.int64)
        self.first_event = np.full(size, -1, dtype=np.int32)
        self.streak = np.zeros((size, len(RATE_THRESHOLDS)), dtype=np.int32)
        self.rate_hits = np.full_like(self.streak, -1)
        self.previous_rate = np.zeros(size)
        self.events = []
        for event in cfg.events:
            times = rng.integers(event.first_day, event.last_day + 1, size=size)
            occurs = rng.random(size) < event.probability
            times = np.where(occurs, times, np.iinfo(np.int32).max)
            if event.leads_per_day == 0:
                amplitude = np.zeros(size)
            else:
                # 입력 도달 시도는 로그정규 분포의 중앙값이 아니라 평균이다.
                amplitude = rng.lognormal(np.log(event.leads_per_day) - event.log_sigma**2 / 2,
                                          event.log_sigma, size=size)
            self.events.append((event, times, amplitude))

    def advance(self, previous: int, day: int) -> None:
        cfg, rng = self.cfg, self.rng
        dt = day - previous
        if not 0 < dt <= 7:
            raise ValueError("생태계 모형의 계산 간격은 1~7일이어야 해.")
        # 기존 프로젝트의 기대 노출을 수명 분포로 적분한다. 집단 갱신은 구간 끝에서 한다.
        hazard = np.log(2) / cfg.project_half_life_days
        project_area = -np.expm1(-hazard * dt) / hazard
        leads = (self.regular * cfg.regular_leads_per_day + self.successful * cfg.successful_leads_per_day) * project_area
        for event, times, amplitude in self.events:
            duration = np.maximum(day - np.maximum(previous, times), 0)
            age = np.maximum(previous - times, 0)
            h = np.log(2) / event.half_life_days
            leads += amplitude * np.exp(-h * age) * (-np.expm1(-h * duration)) / h
            appeared = (times <= day) & ((self.first_event < 0) | (times < self.first_event))
            self.first_event[appeared] = times[appeared]
        # 포아송 도달 시도를 잠재 집단에 균등 배분하면 사람별 최초 발견 확률은 아래와 같다.
        remaining = cfg.additional_audience - self.discovered
        discovery_p = -np.expm1(-leads / cfg.additional_audience)
        expected_rate = remaining * discovery_p * cfg.star_probability / dt
        new_people = rng.binomial(remaining, discovery_p)
        self.discovered += new_people
        self.extra_stars += rng.binomial(new_people, cfg.star_probability)
        new_users = rng.binomial(new_people, cfg.adoption_probability)
        # 역할 이탈/프로젝트 중단. 기여자는 앱 유지와 별개로 언어에 계속 기여할 수 있다.
        self.idle = rng.binomial(self.idle, np.exp(-np.log(2) * dt / cfg.developer_half_life_days))
        self.regular = rng.binomial(self.regular, np.exp(-hazard * dt))
        self.successful = rng.binomial(self.successful, np.exp(-hazard * dt))
        self.contributors = rng.binomial(self.contributors, np.exp(-np.log(2) * dt / cfg.contributor_half_life_days))
        # 진행 중인 개발은 완료와 포기의 경쟁 위험을 가진다.
        mature_h = 1.0 / cfg.development_mean_days
        abort_h = np.log(2) / cfg.developer_half_life_days
        exits = rng.binomial(self.building, -np.expm1(-(mature_h + abort_h) * dt))
        completed = rng.binomial(exits, mature_h / (mature_h + abort_h))
        self.building -= exits
        strong = rng.binomial(completed, cfg.project_success_probability)
        self.regular += completed - strong
        self.successful += strong
        self.completed += completed
        self.success_total += strong
        # 기여가 늘면 후속 프로젝트를 시작하기 쉬워진다는 유한한 배수 가정이다.
        # 이번 구간의 새 기여자는 다음 구간부터 영향을 준다.
        boost = 1.0 + cfg.contribution_max_boost * self.contributors / (self.contributors + cfg.contribution_scale)
        starting = rng.binomial(self.idle, -np.expm1(-cfg.project_start_rate_per_day * boost * dt))
        self.idle -= starting
        self.building += starting
        self.idle += new_users
        self.contributors += rng.binomial(completed, cfg.contribution_probability)
        # 경계 구간을 소급하여 '유지'로 세지 않는 보수적 격자 판정이다.
        above = (expected_rate[:, None] >= RATE_THRESHOLDS) & (self.previous_rate[:, None] >= RATE_THRESHOLDS)
        self.streak = np.where(above, self.streak + dt, 0)
        newly = (self.rate_hits < 0) & (self.streak >= cfg.sustained_days)
        self.rate_hits[newly] = day
        self.previous_rate = expected_rate

    def snapshot(self) -> np.ndarray:
        return np.column_stack((self.extra_stars, self.discovered,
                                self.idle + self.building + self.regular + self.successful,
                                self.regular + self.successful, self.completed,
                                self.success_total, self.contributors)).astype(np.uint32)
