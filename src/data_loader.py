"""FastF1 loading and a deterministic demo dataset for local development."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from fastf1.exceptions import (
    DataNotLoadedError,
    F1AuthError,
    FastF1CriticalError,
    RateLimitExceededError,
)
from src.openf1_loader import _gap_seconds, load_openf1_race

load_dotenv()

COMPOUNDS: Final = ("SOFT", "MEDIUM", "HARD")
REAL_DATA_SOURCES: Final = frozenset({"FastF1", "OpenF1"})


class RealDataUnavailable(RuntimeError):
    """Raised when neither configured real-data provider can load a race."""


def _demo_laps(total_laps: int = 57, drivers: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows: list[dict[str, object]] = []
    for driver_number in range(1, drivers + 1):
        offset = rng.normal(0, 0.35)
        stint_starts = [1, 21, 40]
        compounds = rng.choice(COMPOUNDS, size=3, replace=False)
        for lap in range(1, total_laps + 1):
            stint = 1 + (lap >= 21) + (lap >= 40)
            age = lap - stint_starts[stint - 1] + 1
            compound = str(compounds[stint - 1])
            base = {"SOFT": 91.2, "MEDIUM": 91.8, "HARD": 92.4}[compound]
            traffic_pressure = float(rng.choice([0, 0, 0, 0.5, 1]))
            overtakes = float(rng.choice([0, 0, 0, 1]))
            lap_time = (
                base + offset + 0.075 * age + 0.018 * (total_laps - lap)
                + 0.20 * traffic_pressure - 0.08 * overtakes
            )
            rows.append(
                {
                    "Driver": f"D{driver_number:02d}",
                    "Team": f"Team {(driver_number - 1) % 4 + 1}",
                    "LapNumber": lap,
                    "LapTime": lap_time + rng.normal(0, 0.16),
                    "Compound": compound,
                    "TyreLife": age,
                    "Stint": stint,
                    "TrackTemp": 38.0 + rng.normal(0, 1),
                    "AirTemp": 27.0 + rng.normal(0, 0.5),
                    "TrackStatus": "1",
                    "TrafficPressure": traffic_pressure,
                    "Overtakes": overtakes,
                    "IsPitIn": False,
                    "IsPitOut": False,
                }
            )
    return pd.DataFrame(rows)


def fastf1_cache_dir(cache_dir: str | Path | None = None) -> Path:
    if cache_dir is None:
        cache_dir = Path(os.environ.get("RACE_STORAGE_DIR", "data")) / "raw"
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def load_race(
    year: int, race: str, cache_dir: str | Path | None = None
) -> pd.DataFrame:
    """Load real laps from FastF1/OpenF1; synthetic laps require explicit opt-in."""
    cache_path = fastf1_cache_dir(cache_dir)
    warnings: list[str] = []
    has_credentials = bool(
        os.environ.get("F1_USERNAME") and os.environ.get("F1_PASSWORD")
    )

    if has_credentials:
        try:
            import fastf1

            fastf1.Cache.configure(cache_dir=str(cache_path))
            session = fastf1.get_session(year, race, "R", backend="fastf1")
            session.load(laps=True, telemetry=False, weather=True, messages=False)
            laps = session.laps.copy()
            weather = laps.get_weather_data()
            for column in ("TrackTemp", "AirTemp"):
                if column in weather.columns:
                    if len(weather) != len(laps):
                        raise ValueError(
                            "FastF1 weather rows could not be aligned with lap timing."
                        )
                    laps[column] = weather[column].to_numpy()
            result = normalize_fastf1_laps(laps)
            if not result.empty:
                result.attrs["source"] = "FastF1"
                result.attrs["totalLaps"] = int(
                    session.total_laps or result["LapNumber"].max()
                )
                result.attrs["safetyCarEpisodes"] = _count_safety_car_episodes(laps)
                return result
            warnings.append("FastF1 returned no usable laps.")
        except (
            ImportError,
            OSError,
            RuntimeError,
            ValueError,
            KeyError,
            DataNotLoadedError,
            FastF1CriticalError,
            F1AuthError,
            RateLimitExceededError,
            requests.RequestException,
        ) as error:
            if isinstance(error, F1AuthError):
                warnings.append("FastF1 authentication failed.")
            else:
                warnings.append(f"FastF1 could not load timing: {error}")
    else:
        warnings.append(
            "FastF1 credentials were not configured; tried public OpenF1 historical data."
        )

    try:
        laps, total_laps = load_openf1_race(year, race, cache_path)
        laps.attrs["totalLaps"] = total_laps
        return laps
    except (
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
        requests.RequestException,
    ) as error:
        warnings.append(f"OpenF1 could not load {year} {race} race timing: {error}")

    if os.environ.get("ALLOW_DEMO_DATA", "").strip().casefold() in {"1", "true", "yes"}:
        demo = _demo_laps()
        demo.attrs["source"] = "Demo data"
        demo.attrs["totalLaps"] = int(demo["LapNumber"].max())
        demo.attrs["safetyCarEpisodes"] = 0
        demo.attrs["load_warning"] = " ".join(warnings)
        return demo
    raise RealDataUnavailable(
        f"No real timing data is available for {year} {race}. "
        + " ".join(warnings)
    )


def normalize_fastf1_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Convert FastF1 laps into the columns consumed by the modelling code."""
    required = ["Driver", "Team", "LapNumber", "LapTime", "Compound", "TyreLife", "Stint", "TrackStatus"]
    missing = [column for column in required if column not in laps.columns]
    if missing:
        raise ValueError(f"FastF1 laps missing required columns: {', '.join(missing)}")
    result = laps.copy()
    result["LapTime"] = result["LapTime"].dt.total_seconds()
    result["Team"] = result["Team"].fillna("Unknown").astype(str)
    for column in ("TrackTemp", "AirTemp"):
        if column not in result:
            result[column] = np.nan
    result["IsPitIn"] = result.get("PitInTime", pd.Series(False, index=result.index)).notna()
    result["IsPitOut"] = result.get("PitOutTime", pd.Series(False, index=result.index)).notna()
    gap = result.get("IntervalToPositionAhead", pd.Series(index=result.index, dtype=object))
    result["TrafficGapSeconds"] = gap.map(_gap_seconds)
    result["TrafficPressure"] = result["TrafficGapSeconds"].map(_traffic_pressure)
    if "Position" in result:
        positions = pd.to_numeric(result["Position"], errors="coerce")
        previous = positions.groupby(result["Driver"]).shift()
        result["Position"] = positions
        result["PositionChange"] = previous - positions
        result["Overtakes"] = result["PositionChange"].clip(0, 3).fillna(0)
    else:
        result["Position"] = np.nan
        result["PositionChange"] = np.nan
        result["Overtakes"] = 0.0
    return result[
        required + [
            "TrackTemp",
            "AirTemp",
            "TrafficPressure",
            "TrafficGapSeconds",
            "Position",
            "PositionChange",
            "Overtakes",
            "IsPitIn",
            "IsPitOut",
        ]
    ]


def _traffic_pressure(value: object) -> float:
    if pd.isna(value):
        return 0.0
    if isinstance(value, pd.Timedelta):
        seconds = value.total_seconds()
    elif isinstance(value, (int, float)):
        seconds = float(value)
    else:
        text = str(value).strip().lower()
        if "lap" in text or text in {"leader", "nan", ""}:
            return 0.0
        try:
            seconds = float(text.removesuffix("s").replace("+", "").strip())
        except ValueError:
            try:
                seconds = pd.to_timedelta(text).total_seconds()
            except (TypeError, ValueError):
                return 0.0
    return float(max(0.0, min(1.0, (5.0 - seconds) / 5.0)))


def _count_safety_car_episodes(laps: pd.DataFrame) -> int:
    if "TrackStatus" not in laps or "LapNumber" not in laps:
        return 0
    status_by_lap = (
        laps.assign(_status=laps["TrackStatus"].astype(str))
        .groupby("LapNumber")["_status"]
        .agg(lambda values: any(value.strip() in {"4", "6"} for value in values))
        .sort_index()
    )
    starts = status_by_lap & ~status_by_lap.shift(fill_value=False)
    return int(starts.sum())
