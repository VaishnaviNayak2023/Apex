"""Cleaning rules for race lap data."""

from __future__ import annotations

import pandas as pd


def clean_laps(laps: pd.DataFrame, max_percentile: float = 1.07) -> pd.DataFrame:
    """Remove invalid, pit, and neutralised laps and robustly trim outliers."""
    result = laps.copy()
    result = result[result["LapTime"].notna()]
    result = result[~result["IsPitIn"] & ~result["IsPitOut"]]
    track_status = result["TrackStatus"].astype("string")
    result = result[track_status.isin(["1", "2"]) | track_status.isna()]
    result = result[result["Compound"].isin(["SOFT", "MEDIUM", "HARD"])]
    medians = result.groupby("Driver")["LapTime"].transform("median")
    result = result[result["LapTime"] <= medians * max_percentile]
    return result.reset_index(drop=True)
