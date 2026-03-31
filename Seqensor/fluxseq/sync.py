from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Timeline:
    """Uniform timeline."""

    t: np.ndarray  # seconds
    fps: float

    @property
    def dt(self) -> float:
        return 1.0 / float(self.fps)


def build_timeline(
    *streams: pd.DataFrame,
    fps: float = 30.0,
    t_min: float | None = None,
    t_max: float | None = None,
    time_col: str = "time_seconds",
) -> Timeline:
    """Build a uniform timeline over the overlap of provided streams.

    If t_min/t_max are not provided, uses the intersection of [min,max] from streams.
    """
    if fps <= 0:
        raise ValueError("fps must be > 0")

    mins, maxs = [], []
    for df in streams:
        if time_col not in df.columns:
            raise ValueError(f"stream missing '{time_col}'")
        mins.append(float(df[time_col].min()))
        maxs.append(float(df[time_col].max()))

    if t_min is None:
        t_min = max(mins) if mins else 0.0
    if t_max is None:
        t_max = min(maxs) if maxs else 0.0

    if t_max <= t_min:
        raise ValueError(f"invalid overlap: t_max ({t_max}) <= t_min ({t_min})")

    dt = 1.0 / float(fps)
    n = int(np.floor((t_max - t_min) / dt)) + 1
    t = t_min + dt * np.arange(n, dtype=np.float64)
    return Timeline(t=t, fps=float(fps))


def align_to_timeline(
    df: pd.DataFrame,
    timeline: Timeline,
    *,
    time_col: str = "time_seconds",
    method: str = "linear",
    fill_value: float = np.nan,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Resample numeric columns of df onto the timeline.

    - method: 'linear' (interpolate) or 'ffill' (forward-fill)
    """
    if time_col not in df.columns:
        raise ValueError(f"df missing '{time_col}'")

    if columns is None:
        columns = [c for c in df.columns if c != time_col]

    # Keep only numeric columns.
    num_cols = []
    for c in columns:
        if c == time_col:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)

    src = df[[time_col] + num_cols].dropna(subset=[time_col]).sort_values(time_col)

    out = pd.DataFrame({"time_seconds": timeline.t})
    if len(src) == 0 or len(num_cols) == 0:
        for c in num_cols:
            out[c] = fill_value
        return out

    if method == "linear":
        x = src[time_col].to_numpy(dtype=np.float64)
        for c in num_cols:
            y = src[c].to_numpy(dtype=np.float64)
            # np.interp cannot handle nans well. Drop them per column.
            m = ~np.isnan(y) & ~np.isnan(x)
            if m.sum() < 2:
                out[c] = fill_value
                continue
            out[c] = np.interp(timeline.t, x[m], y[m], left=fill_value, right=fill_value)
    elif method == "ffill":
        # Reindex with nearest previous sample.
        src2 = src.set_index(time_col).sort_index()
        out = out.set_index("time_seconds")
        out = out.join(src2, how="left")
        out = out.ffill().reset_index()
    else:
        raise ValueError("method must be 'linear' or 'ffill'")

    return out
