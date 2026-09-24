"""All candidates predict the SAME integer count over the SAME future interval.

Gamma parameterization is shape/rate. Grid posteriors use log-coordinate
quadrature weights (including the Jacobian), not continuous density scores.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.stats import nbinom

from .history import History

MODEL_NAMES = ("long_poisson", "recent_poisson", "changepoint_poisson",
               "negative_binomial", "birth")


@dataclass(frozen=True)
class ModelConfig:
    long_days: int = 365
    recent_days: int = 90
    change_days: int = 180
    min_segment: int = 21
    rate_shape: float = 0.5
    rate_rate: float = 5.0
    change_prior: float = 0.5
    birth_shape: float = 1.0
    birth_rate: float = 333.3333333333333
    grid_size: int = 121


def normalized(log_weights: np.ndarray) -> np.ndarray:
    w = np.exp(log_weights - logsumexp(log_weights))
    if not np.isfinite(w).all():
        raise ValueError("Non-finite posterior; check input and prior settings.")
    return w / w.sum()


def quadrature_prior(grid: np.ndarray, shape: float, rate: float) -> np.ndarray:
    # Uniform log-grid: Gamma density * dx = Gamma density * x d(log x).
    result = shape * np.log(grid) - rate * grid
    result[[0, -1]] -= np.log(2.0)  # trapezoidal endpoint weights
    return result


@dataclass
class CountModel:
    name: str
    kind: str
    weights: np.ndarray
    a: np.ndarray
    b: np.ndarray
    diagnostics: dict = field(default_factory=dict)

    def distribution(self, days: float, start: int) -> tuple[np.ndarray, np.ndarray]:
        if days <= 0:
            raise ValueError("Predictive interval must be positive.")
        if self.kind == "poisson":
            return self.a, self.b / (self.b + days)
        if self.kind == "nb":
            return self.b * days, self.b / (self.b + self.a)
        if self.kind == "birth":
            # Linear birth with immigration offset 1: n=S+1, p=exp(-r*t).
            return np.full_like(self.a, start + 1.0), np.exp(-np.minimum(self.a * days, 700.0))
        raise ValueError(f"Unknown model kind: {self.kind}")

    def logpmf(self, count: int, days: float, start: int) -> float:
        n, p = self.distribution(days, start)
        with np.errstate(divide="ignore"):
            return float(logsumexp(np.log(self.weights) + nbinom.logpmf(count, n, p)))

    def cdf(self, count: int, days: float, start: int) -> float:
        n, p = self.distribution(days, start)
        return float(np.dot(self.weights, nbinom.cdf(count, n, p)))

    def target_probability(self, target: int, days: float, start: int) -> float:
        if target <= start:
            return 1.0
        if days <= 0:
            return 0.0
        n, p = self.distribution(days, start)
        return float(np.dot(self.weights, nbinom.sf(target - start - 1, n, p)))

    def mean_gain(self, days: float, start: int) -> float:
        if self.kind == "poisson":
            return float(np.dot(self.weights, self.a / self.b) * days)
        if self.kind == "nb":
            return float(np.dot(self.weights, self.a) * days)
        return float(np.dot(self.weights, (start + 1) * np.expm1(self.a * days)))

    def draw(self, size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        index = rng.choice(len(self.weights), size=size, p=self.weights)
        if self.kind == "poisson":
            return rng.gamma(self.a[index], 1.0 / self.b[index]), np.zeros(size)
        return self.a[index].copy(), self.b[index].copy()

    def info(self) -> dict:
        return {"kind": self.kind, "posterior_components": len(self.weights),
                **self.diagnostics}


def gamma_model(name: str, gains: np.ndarray, cfg: ModelConfig) -> CountModel:
    events, days = int(gains.sum()), len(gains)
    return CountModel(name, "poisson", np.ones(1),
                      np.array([cfg.rate_shape + events]),
                      np.array([cfg.rate_rate + days]),
                      {"events": events, "exposure_days": days})


def log_evidence(events: np.ndarray, days: np.ndarray, cfg: ModelConfig) -> np.ndarray:
    # The common sum(log(y_i!)) cancels between segmentations of the same days.
    a, b = cfg.rate_shape, cfg.rate_rate
    return gammaln(a + events) - gammaln(a) + a * np.log(b) - (a + events) * np.log(b + days)


def changepoint_model(history: History, cfg: ModelConfig) -> CountModel:
    y = history.gains[-cfg.change_days:]
    n, total = len(y), int(y.sum())
    null = gamma_model("changepoint_poisson", y, cfg)
    positions = np.arange(cfg.min_segment, n - cfg.min_segment + 1)
    if len(positions) == 0:
        null.diagnostics.update(change_probability=0.0, note="Insufficient segment exposure.")
        return null
    sums = np.cumsum(y)
    left = sums[positions - 1]
    right = total - left
    l0 = float(log_evidence(total, n, cfg))
    l1 = log_evidence(left, positions, cfg) + log_evidence(right, n - positions, cfg)
    log_weights = np.r_[np.log1p(-cfg.change_prior) + l0,
                        np.log(cfg.change_prior) - np.log(len(positions)) + l1]
    weights = normalized(log_weights)
    conditional = normalized(l1)
    best = int(np.argmax(conditional))
    event_dates = history.dates[-n:]
    cumulative = np.cumsum(conditional)
    date_quantiles = [str(event_dates[positions[min(np.searchsorted(cumulative, q),
                                                          len(positions) - 1)]].date())
                      for q in (0.1, 0.5, 0.9)]
    return CountModel("changepoint_poisson", "poisson", weights,
                      cfg.rate_shape + np.r_[total, right],
                      cfg.rate_rate + np.r_[n, n - positions], {
        "events": total, "exposure_days": n,
        "change_probability": float(weights[1:].sum()),
        "log_bayes_factor_change_vs_none": float(logsumexp(l1) - np.log(len(l1)) - l0),
        "candidate_date_given_change": str(event_dates[positions[best]].date()),
        "date_p10_p50_p90_given_change": date_quantiles,
        "candidate_before_rate_mle": float(left[best] / positions[best]),
        "candidate_after_rate_mle": float(right[best] / (n - positions[best])),
        "note": "One-change vs no-change posterior, conditional on priors and window; not a viral-state probability."
    })


def nb_model(gains: np.ndarray, cfg: ModelConfig) -> CountModel:
    upper = max(10.0, float(gains.mean()) * 50.0, float(gains.max(initial=0)) * 5.0)
    rates = np.geomspace(1e-8, upper, cfg.grid_size)
    dispersion = np.geomspace(0.01, 10000.0, cfg.grid_size)
    lam, kappa = np.meshgrid(rates, dispersion, indexing="ij")
    prior_k = -0.5 * (np.log(dispersion) / 1.5) ** 2
    prior_k[[0, -1]] -= np.log(2.0)
    logw = quadrature_prior(rates, cfg.rate_shape, cfg.rate_rate)[:, None] + prior_k[None, :]
    for count, frequency in zip(*np.unique(gains, return_counts=True)):
        logw += frequency * nbinom.logpmf(count, kappa, kappa / (kappa + lam))
    weights = normalized(logw.ravel())
    w2 = weights.reshape(logw.shape)
    edge = float(w2[:2].sum() + w2[-2:].sum() + w2[:, :2].sum() + w2[:, -2:].sum())
    return CountModel("negative_binomial", "nb", weights, lam.ravel(), kappa.ravel(), {
        "events": int(gains.sum()), "exposure_days": len(gains),
        "grid_edge_mass_bound": edge,
        "dispersion_prior": "log(kappa) ~ Normal(0, 1.5^2)",
        "note": "Conditional daily variance=lambda+lambda^2/kappa; posterior rate uncertainty is separate."
    })


def birth_model(history: History, cfg: ModelConfig) -> CountModel:
    y = history.gains[-cfg.long_days:]
    n = history.stars[-len(y) - 1:-1] + 1.0
    rates = np.geomspace(1e-7, 0.3, max(241, cfg.grid_size))
    # Exact daily transition: NB(n=S_previous+1, p=exp(-r)).
    # Combinatorial factors do not depend on r and cancel in the posterior.
    logw = quadrature_prior(rates, cfg.birth_shape, cfg.birth_rate)
    logw += -rates * n.sum() + y.sum() * np.log(-np.expm1(-rates))
    weights = normalized(logw)
    return CountModel("birth", "birth", weights, rates, np.zeros_like(rates), {
        "events": int(y.sum()), "exposure_days": len(y),
        "posterior_mean_rate_per_day": float(np.dot(weights, rates)),
        "grid_edge_mass_bound": float(weights[:2].sum() + weights[-2:].sum()),
        "immigration_offset": 1,
        "note": "Unbounded feedback is an assumption, not evidence of causal star-to-star recruitment."
    })


def fit_models(history: History, cfg: ModelConfig) -> list[CountModel]:
    y = history.gains
    if len(y) == 0:
        raise ValueError("No observed increments.")
    return [gamma_model("long_poisson", y[-cfg.long_days:], cfg),
            gamma_model("recent_poisson", y[-cfg.recent_days:], cfg),
            changepoint_model(history, cfg), nb_model(y[-cfg.long_days:], cfg),
            birth_model(history, cfg)]
