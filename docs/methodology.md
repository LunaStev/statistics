# Statistical specification

## Observation and prediction targets

Let `S_t` be the daily cumulative total and `Y_t=S_t-S_(t-1)` the observed increment. The first total is conditioned on; its earlier exposure is not fabricated. Counts must be nonnegative and dates consecutive. These are arrival-only models, not addition/removal models. They do not identify the effects of effort, publicity, recommendations, or causal referral from stars.

A validation target is one integer, `K=S_(t+h)-S_t`, for the same `h` in every model. Predictions are fitted using only observations through `t`. The non-overlapping rolling windows default to `h=30` days. The final horizon can be decades, but validation does not certify extrapolation to those decades.

## 1. Gamma–Poisson arrivals

For a window of `d` daily observations with `k` events:

```
lambda ~ Gamma(a0, rate=b0)
a = a0 + k
b = b0 + d
K_future | data ~ NB(n=a, p=b/(b+h))
```

The code uses SciPy's failures-before-`n`-successes NB parameterization. The default prior is `a0=0.5`, `b0=5` (mean 0.1 arrivals/day). Its influence is explicit in configuration, not described as absent. One lambda is drawn per simulated future, then its future Poisson increments are conditionally independent. Lambda is not resampled every time step.

## 2. Bayesian one-change mixture

Compare no change with one change at each admissible split; each segment has at least 21 days. The default prior probability of some change is 0.5, distributed uniformly over candidate dates. A segment's integrated log likelihood, up to the common daily factorial term, is

```
L(k,d) = lgamma(a0+k)-lgamma(a0)
         +a0*log(b0)-(a0+k)*log(b0+d).
```

Use `L(k,d)` for no change and `L(k_left,d_left)+L(k_right,d_right)` for a split. The omitted `-sum(log(Y_t!))` is the SAME constant for all hypotheses. Average the split evidence over candidate dates, rather than selecting the largest score and treating that search as free. The current rate is a mixture of the no-change posterior and all post-change posteriors.

The change probability is conditional on the Poisson assumptions, window and priors. It is not a probability of being "about to go viral". The reported candidate date and its date quantiles are conditional on a change having occurred. No-change remains available even when a candidate date is printed. Future additional changes are not automatically assumed.

## 3. Overdispersed negative-binomial process

For fixed rate `lambda` and daily shape `kappa`:

```
Y_t ~ NB(n=kappa, p=kappa/(kappa+lambda))
E[Y_t] = lambda
Var[Y_t] = lambda + lambda^2/kappa
K_h ~ NB(n=kappa*h, p=kappa/(kappa+lambda)).
```

The aggregation identity defines a stationary independent-increment count process conditional on the parameters. Parameter uncertainty is integrated using a log-spaced two-dimensional posterior grid. Lambda uses the Gamma prior above. `log(kappa) ~ Normal(0,1.5^2)`. The grid includes the log-coordinate Jacobian and trapezoidal endpoint weights. Parameter draws are held fixed within a simulated future. This overdispersion differs from the negative-binomial marginal uncertainty produced by a Gamma–Poisson mixture.

Grid integration is a numerical approximation, not MCMC. Inspect boundary mass and refit with a denser grid to check numerical sensitivity. The dispersion grid is 0.01–10000. Lambda's upper numerical bound is chosen from training data only. Neither bound is a claimed estimate of a market limit.

## 4. Birth / immigration feedback model

Use a continuous-time integer process with jump rate `r*(S+1)`. The shift `N=S+1` follows a linear pure-birth process. Its exact transition for any duration `h` is

```
S_(t+h)-S_t ~ NB(n=S_t+1, p=exp(-r*h)).
```

The extra 1 is a fixed immigration offset, allowing growth from zero. For daily observations, the r-dependent log likelihood is

```
-r*sum(S_(t-1)+1) + sum(Y_t)*log(1-exp(-r)).
```

A Gamma(shape=1, rate=333.333...) prior on r is integrated on a log grid from 1e-7 to 0.3 per day. The posterior includes the correct coordinate measure. This prior, window and mechanism are configurable/explicit; early `1 -> 2` percentage gains are not automatically promoted into decades of monthly compound growth.

Conditional on positive r persisting forever, this model has unbounded expected growth. That is a model assumption, not evidence that the mechanism will persist. The model's prediction of a high eventual crossing probability must not be interpreted as validation of perpetual feedback.

## 5. Predictive stacking and temporal evaluation

Let `l_im` be model m's log probability of the observed integer gain in held-out window i. All entries are log PMFs on the same support. Fit nonnegative weights summing to one by

```
maximize_w sum_i log(sum_m w_m * exp(l_im)).
```

This is predictive stacking with rolling-origin validation, not posterior model probability and not a softmax over mismatched densities. No arbitrary minimum weight is imposed. Two near-identical predictors can have unstable individual weights; compare predictive distributions and reported sensitivity rather than interpreting each weight causally.

For the ensemble's reported historical performance, each window uses stacking weights learned only from EARLIER held-out windows; use equal weights until there are at least two such windows. Final future forecasts use all available validation windows. Per-model coverage uses discrete central 80% intervals, whose observed coverage may exceed 80% because of discreteness and small sample size.

Moving-block resampling of successive validation windows produces sensitivity ranges for final weights. Default block length is three windows (or all windows if fewer). These are not calibrated posterior credible intervals, especially under nonstationarity. Equal-weight scenarios are also available.

## 6. Monte Carlo, exact checks, censoring

A simulated future chooses one component using the selected weights, draws its parameters from that component's posterior, and evolves integer counts. Parameter and event uncertainty are propagated. Re-selecting the model each month is NOT part of the baseline.

For monotone arrivals, reaching target T by h is equivalent to `S_h >= T`. The marginal crossing probability is therefore a mixture of NB survival functions, available analytically up to grid quadrature. Reports compare this with simulation frequencies. This check detects implementation error; it does not check realism of the model family.

Paths stop at the target for storage/plotting. Smaller milestone crossing times are recorded. Because crossings are checked on the simulation grid, a recorded day d means the crossing lies in `(previous_grid_day,d]`, except a day-0 crossing. The grid contains exact anniversary checkpoints and never advances beyond the selected horizon. Use a finer step for timing precision; endpoint marginal probabilities remain exact under the baseline component processes.

Unreached paths are right-censored at the horizon, not labeled as never reaching the goal. Conditional hitting quantiles use only paths that reach within the horizon. All-path quantiles keep the censored mass; if the cumulative crossing frequency never reaches 50%, no all-path median date is reported. Conditional target-time histograms are not mislabeled as unconditional densities.

Wilson intervals accompany simulation proportions, including zero-event proportions. These measure only Monte Carlo sampling error with fixed model fits and weights, not parameter uncertainty a second time and not long-horizon model misspecification. More simulations reduce numerical Monte Carlo error, not missing information about adoption mechanisms.

## 7. Explicit feedback-duration sensitivity

With `--feedback-half-life-years H`, only birth-component paths draw a feedback end time `T ~ Exponential(rate=log(2)/(H*365.2425))`. Before T, use the exact birth transition. After T, use a fixed rate drawn from the long-window Gamma–Poisson posterior. Split a time step at T if necessary. Development is not modeled as stopping, and the fallback rate is not zero.

H is a user-specified assumption, NOT estimated from this single small history. The scenario does not learn an HMM, does not allow later reacceleration and is not included in the baseline backtest fit. Analytic baseline birth crossing probabilities are deliberately omitted for this modified scenario. Compare several H values as sensitivity experiments, not as competing estimated truths.

## References

- Hyndman & Athanasopoulos, *Forecasting: Principles and Practice*, time-series cross-validation: https://otexts.com/fpp3/tscv.html
- Yao, Vehtari, Simpson & Gelman (2018), *Using stacking to average Bayesian predictive distributions*, discussed in the Stan/loo authors' documentation: https://mc-stan.org/loo/articles/loo2-weights.html
- SciPy authors, negative-binomial parameterization, PMF, CDF and survival function: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nbinom.html

These references support the individual statistical methods, not the realism of applying them to Wave's long-term popularity.
