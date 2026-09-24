# Changelog

## 0.2.0

- Replace the initial monolithic experiment with data, model, validation, simulation and reporting modules.
- Correct change-point inference using the same daily likelihood and integration over split locations plus a no-change alternative.
- Compare all models using the log probability of the same held-out integer count; remove the continuous-density/discrete-probability score mixture and 5% weight floor.
- Refit candidates at each rolling origin; evaluate ensemble weights using only earlier validation outcomes.
- Replace the arbitrary monthly log-growth AR process with an explicit integer birth/immigration model.
- Add a separately overdispersed negative-binomial process with posterior grid integration.
- Add an explicitly assumed finite-feedback-duration scenario, without labeling its hazard as learned.
- Separate conditional hitting-time quantiles from all-path quantiles and right-censored outcomes.
- Add analytic crossing-probability checks, Monte Carlo Wilson intervals, calendar checkpoints and interval-end hitting times.
- Add bounded chunk processing, spawned parallel workers, source/input/version fingerprints and hashed resumable checkpoints.
- Add a local HTML report, six chart types, CSV summaries and synthetic-data regression tests.

The v1 model weights and headline target probabilities are not comparable with v2 as calibrated success probabilities. v2's scenario results are also conditional on its models and long-term assumptions.
