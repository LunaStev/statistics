import numpy as np
import pandas as pd
import pytest

from growth.cli import configs, main, parse_args


@pytest.mark.parametrize("option,value", [("--simulations", "0"), ("--chunk-size", "-1"),
                                         ("--years", "0"), ("--jobs", "0"),
                                         ("--grid-size", "10"), ("--step-days", "31")])
def test_cli_validates_ranges(option, value):
    with pytest.raises(ValueError):
        configs(parse_args(["sample.csv", option, value]))


def test_local_end_to_end_smoke(tmp_path):
    source = tmp_path / "synthetic.csv"
    y = np.tile([0, 0, 0, 1], 30)
    pd.DataFrame({"Date": pd.date_range("2025-01-01", periods=121),
                  "Stars": np.r_[1, 1 + np.cumsum(y)]}).to_csv(source, index=False)
    output = tmp_path / "run"
    args = [str(source), "--simulations", "50", "--chunk-size", "17", "--years", "1",
            "--target", "100", "--min-train-days", "30", "--grid-size", "41",
            "--weight-bootstrap", "0", "--output", str(output)]
    assert main(args) == 0
    for path in ["run.json", "summary.csv", "validation_folds.csv", "model_diagnostics.json",
                 "target_probabilities.csv", "milestone_probabilities.csv", "report.html",
                 "charts/forecast_fan.png", "charts/target_cdf.png"]:
        assert (output / path).is_file()
    assert main(args + ["--resume", "--no-plots"]) == 0
