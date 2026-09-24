# Working on this repository

- Keep the existing `python wave_growth_forecast.py CSV ...` interface usable.
- Split implementation by data/model/validation/simulation/report responsibilities.
- Never compare a continuous log-density directly with a discrete log-PMF.
- Refit every data-dependent selection using the training portion of each temporal fold only.
- Predictive stacking weights are not probabilities that a model is true.
- Distinguish right-censoring, conditional-on-reaching quantiles and all-path quantiles.
- Label user-chosen scenarios/priors separately from data-estimated quantities.
- Do not fabricate statistical results or run the user's million-path jobs as part of maintenance.
- Use synthetic data for unit and smoke tests. Do not commit user CSVs or generated results.
- Run `python -m pytest -q` and `ruff check .` before changing inference or simulation code.
- Keep plots on Matplotlib's default color cycle; one figure per chart.
- Preserve deterministic chunk streams, checkpoint compatibility checks and non-destructive outputs.
