"""Strict daily cumulative-count input; never silently invent missing events."""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class History:
    dates: pd.DatetimeIndex
    stars: np.ndarray
    repository: str
    source_sha256: str

    @property
    def gains(self) -> np.ndarray:
        # The first total is an initial condition, not an observed daily arrival.
        return np.diff(self.stars)

    def prefix(self, days: int) -> "History":
        return History(self.dates[:days + 1], self.stars[:days + 1],
                       self.repository, self.source_sha256)


def load_history(path: str | Path, repository: str | None = None) -> History:
    path = Path(path)
    frame = pd.read_csv(path)
    if not {"Date", "Stars"}.issubset(frame.columns):
        raise ValueError("CSV must contain Date and Stars columns.")
    name = repository or "repository"
    if "Repository" in frame:
        if frame["Repository"].isna().any():
            raise ValueError("Repository contains empty values.")
        names = frame["Repository"].astype(str).unique()
        if repository is None and len(names) != 1:
            raise ValueError("Multiple repositories: select one with --repository.")
        name = repository or str(names[0])
        frame = frame[frame["Repository"].astype(str) == name].copy()
    elif repository is not None:
        raise ValueError("--repository requires a Repository column.")
    if frame.empty:
        raise ValueError("No observations for the selected repository.")
    date_text = frame["Date"].astype(str)
    if not date_text.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
        raise ValueError("Date must use YYYY-MM-DD calendar dates.")
    frame["Date"] = pd.to_datetime(date_text, format="%Y-%m-%d", errors="raise")
    values = pd.to_numeric(frame["Stars"], errors="raise").to_numpy(dtype=float)
    if (not np.isfinite(values).all() or (values < 0).any()
            or (values != np.floor(values)).any() or (values > 2_000_000_000).any()):
        raise ValueError("Stars must be finite, nonnegative integers <= 2,000,000,000.")
    frame["Stars"] = values.astype(np.int64)
    if (frame.groupby("Date")["Stars"].nunique() > 1).any():
        raise ValueError("Conflicting star totals on the same date.")
    frame = frame.sort_values("Date").drop_duplicates("Date")
    dates = pd.DatetimeIndex(frame["Date"])
    if len(dates) < 3:
        raise ValueError("At least three daily observations are required.")
    if not (np.diff(dates.values).astype("timedelta64[D]").astype(int) == 1).all():
        raise ValueError("Missing calendar days. Supply daily history; gaps are not forward-filled.")
    stars = frame["Stars"].to_numpy(dtype=np.int64)
    if (np.diff(stars) < 0).any():
        raise ValueError("Decreasing totals require an addition/removal model; not clipped to zero.")
    return History(dates, stars, name, sha256(path.read_bytes()).hexdigest())
