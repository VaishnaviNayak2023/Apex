"""Feature engineering shared by training and simulation."""

from __future__ import annotations

import pandas as pd

MODEL_COLUMNS = [
    "tyre_age",
    "lap_number",
    "fuel_proxy",
    "track_temp",
    "air_temp",
    "traffic_pressure",
    "overtakes",
    "team",
    "compound",
]
FEATURE_COLUMNS = MODEL_COLUMNS


def add_features(laps: pd.DataFrame, total_laps: int | None = None) -> pd.DataFrame:
    result = laps.copy()
    race_laps = total_laps or int(result["LapNumber"].max())
    result["tyre_age"] = result["TyreLife"].astype(float)
    result["lap_number"] = result["LapNumber"].astype(float)
    result["fuel_proxy"] = race_laps - result["lap_number"]
    result["track_temp"] = result["TrackTemp"].fillna(result["TrackTemp"].median()).astype(float)
    result["air_temp"] = result["AirTemp"].fillna(result["AirTemp"].median()).astype(float)
    result["traffic_pressure"] = result["TrafficPressure"].fillna(0).astype(float)
    result["overtakes"] = result["Overtakes"].fillna(0).astype(float)
    result["team"] = result["Team"].fillna("Unknown").astype(str)
    result["compound"] = result["Compound"].astype(str)
    return result
