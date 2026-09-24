import numpy as np
import pandas as pd
import pytest

from growth.history import load_history


def csv(tmp_path, values=(1, 2, 2), dates=None, repos=None):
    frame = pd.DataFrame({"Date": dates or ["2026-01-01", "2026-01-02", "2026-01-03"],
                          "Stars": values})
    if repos is not None:
        frame["Repository"] = repos
    path = tmp_path / "history.csv"
    frame.to_csv(path, index=False)
    return path


def test_initial_total_is_not_a_daily_event(tmp_path):
    h = load_history(csv(tmp_path, (20, 21, 21)))
    assert h.gains.tolist() == [1, 0]
    assert len(h.prefix(1).stars) == 2


@pytest.mark.parametrize("values", [(1, 2, 1), (1, 2.1, 3), (1, np.nan, 3), (-1, 2, 3)])
def test_invalid_counts(tmp_path, values):
    with pytest.raises(ValueError):
        load_history(csv(tmp_path, values))


def test_missing_day_rejected(tmp_path):
    with pytest.raises(ValueError, match="Missing calendar"):
        load_history(csv(tmp_path, dates=["2026-01-01", "2026-01-02", "2026-01-04"]))


def test_conflicting_duplicate_rejected(tmp_path):
    with pytest.raises(ValueError, match="Conflicting"):
        load_history(csv(tmp_path, dates=["2026-01-01", "2026-01-01", "2026-01-02"]))


def test_equal_duplicates_collapsed_and_sorted(tmp_path):
    path = tmp_path / "history.csv"
    path.write_text("Date,Stars\n2026-01-03,2\n2026-01-01,1\n2026-01-02,2\n2026-01-02,2\n")
    assert load_history(path).stars.tolist() == [1, 2, 2]


def test_multiple_repositories_require_selection(tmp_path):
    a = pd.DataFrame({"Date": pd.date_range("2026-01-01", periods=3), "Stars": [1, 2, 3], "Repository": "a"})
    b = a.assign(Repository="b", Stars=[10, 20, 30])
    path = tmp_path / "multi.csv"
    pd.concat([a, b]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Multiple"):
        load_history(path)
    assert load_history(path, "b").stars.tolist() == [10, 20, 30]
