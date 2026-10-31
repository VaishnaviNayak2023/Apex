"""Historical lap timing loader for the public OpenF1 API."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import requests

OPENF1_BASE_URL = "https://api.openf1.org/v1"
COMPOUND_MAP = {
    "SOFT": "SOFT",
    "MEDIUM": "MEDIUM",
    "HARD": "HARD",
    "INTERMEDIATE": "INTERMEDIATE",
    "WET": "WET",
}
_request_lock = threading.Lock()
_request_times: deque[float] = deque()


class OpenF1Unavailable(RuntimeError):
    """Raised when OpenF1 cannot supply a complete historical race session."""


def _name_key(value: str) -> str:
    words = re.sub(r"[^a-z0-9]+", " ", value.casefold()).split()
    ignored = {"formula", "one", "grand", "prix", "gran", "premio"}
    return " ".join(word for word in words if word not in ignored)


def _cache_path(cache_dir: Path, endpoint: str, params: dict) -> Path:
    key = json.dumps(params, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(key).hexdigest()[:20]
    return cache_dir / "openf1" / f"{endpoint}-{digest}.json"


def _respect_rate_limit() -> None:
    with _request_lock:
        now = time.monotonic()
        while _request_times and now - _request_times[0] >= 60:
            _request_times.popleft()
        if len(_request_times) >= 30:
            time.sleep(max(0, 60 - (now - _request_times[0])))
            now = time.monotonic()
            while _request_times and now - _request_times[0] >= 60:
                _request_times.popleft()
        if _request_times:
            time.sleep(max(0, 0.35 - (now - _request_times[-1])))
        _request_times.append(time.monotonic())


def _fetch_json(
    endpoint: str,
    params: dict,
    cache_dir: Path,
    getter: Callable = requests.get,
    *,
    cache_result: bool = True,
) -> list[dict]:
    path = _cache_path(cache_dir, endpoint, params)
    cached_content: list[dict] | None = None
    if path.exists():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(cached, dict) and isinstance(cached.get("data"), list):
                cached_content = cached["data"]
                fetched_at = datetime.fromisoformat(cached["fetchedAt"])
                if datetime.now(timezone.utc) - fetched_at < timedelta(hours=12):
                    return cached_content
            elif isinstance(cached, list):
                cached_content = cached
        except (OSError, json.JSONDecodeError):
            path.unlink(missing_ok=True)

    try:
        if getter is requests.get:
            _respect_rate_limit()
        response = getter(
            f"{OPENF1_BASE_URL}/{endpoint}",
            params=params,
            timeout=(5, 30),
            headers={"User-Agent": "F1-Strategy-Lab/1.0"},
        )
        response.raise_for_status()
        content = response.json()
    except requests.RequestException:
        if cached_content is not None:
            return cached_content
        raise
    if not isinstance(content, list):
        raise OpenF1Unavailable(f"OpenF1 returned an invalid {endpoint} response.")
    if cache_result:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "fetchedAt": datetime.now(timezone.utc).isoformat(),
                    "data": content,
                }
            ),
            encoding="utf-8",
        )
    return content


def _find_race_session(
    year: int,
    race: str,
    cache_dir: Path,
    getter: Callable,
) -> dict:
    meetings = _fetch_json("meetings", {"year": year}, cache_dir, getter)
    sessions = _fetch_json(
        "sessions",
        {"year": year, "session_name": "Race"},
        cache_dir,
        getter,
    )
    if not meetings or not sessions:
        raise OpenF1Unavailable(f"OpenF1 has no race sessions for {year}.")

    wanted = _name_key(race)
    ranked = sorted(
        meetings,
        key=lambda meeting: SequenceMatcher(
            None,
            wanted,
            _name_key(str(meeting.get("meeting_name", ""))),
        ).ratio(),
        reverse=True,
    )
    meeting = ranked[0]
    match = SequenceMatcher(
        None, wanted, _name_key(str(meeting.get("meeting_name", "")))
    ).ratio()
    if not wanted or match < 0.84:
        raise OpenF1Unavailable(
            f"OpenF1 has no unambiguous meeting matching {race!r} in {year}."
        )

    race_sessions = [
        session
        for session in sessions
        if session.get("meeting_key") == meeting.get("meeting_key")
        and str(session.get("session_name", "")).casefold() == "race"
        and not session.get("is_cancelled", False)
    ]
    if not race_sessions:
        raise OpenF1Unavailable(f"OpenF1 has no completed race session for {race} {year}.")
    session = race_sessions[0]
    try:
        end_time = datetime.fromisoformat(
            str(session["date_end"]).replace("Z", "+00:00")
        ).astimezone(timezone.utc)
    except (KeyError, ValueError) as error:
        raise OpenF1Unavailable("OpenF1 race session has no valid end time.") from error
    if end_time > datetime.now(timezone.utc) - timedelta(minutes=30):
        raise OpenF1Unavailable(
            "OpenF1 historical access is available after a session ends; "
            "live-session data requires an OpenF1 subscription."
        )
    return session


def _weather_at_laps(laps: pd.DataFrame, weather: list[dict]) -> pd.DataFrame:
    laps["TrackTemp"] = np.nan
    laps["AirTemp"] = np.nan
    if not weather:
        return laps
    weather_frame = pd.DataFrame(weather)
    if not {"date", "track_temperature", "air_temperature"} <= set(weather_frame.columns):
        return laps
    weather_frame["date"] = pd.to_datetime(weather_frame["date"], utc=True, errors="coerce")
    weather_frame = weather_frame.dropna(subset=["date"]).sort_values("date")
    lap_dates = pd.to_datetime(laps["Date"], utc=True, errors="coerce")
    if weather_frame.empty or lap_dates.isna().all():
        return laps
    left = (
        laps.assign(_date=lap_dates, _weather_row=np.arange(len(laps)))
        .dropna(subset=["_date"])
        .sort_values("_date")
    )
    if left.empty:
        return laps
    merged = pd.merge_asof(
        left,
        weather_frame[["date", "track_temperature", "air_temperature"]],
        left_on="_date",
        right_on="date",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=5),
    )
    merged = merged.sort_values("_weather_row")
    row_indices = merged["_weather_row"].astype(int).to_numpy()
    laps.iloc[row_indices, laps.columns.get_loc("TrackTemp")] = (
        merged["track_temperature"].to_numpy()
    )
    laps.iloc[row_indices, laps.columns.get_loc("AirTemp")] = (
        merged["air_temperature"].to_numpy()
    )
    return laps


def _merge_timed_values(
    laps: pd.DataFrame,
    events: list[dict],
    event_column: str,
    output_column: str,
    *,
    per_driver: bool,
) -> pd.Series:
    values = pd.Series(np.nan, index=laps.index, dtype=object)
    if not events:
        return values

    frame = pd.DataFrame(events)
    required = {"date", event_column}
    if per_driver:
        required.add("driver_number")
    if not required <= set(frame.columns):
        return values

    frame["_date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["_date", event_column]).copy()
    if per_driver:
        frame["driver_number"] = pd.to_numeric(
            frame["driver_number"], errors="coerce"
        )
        frame = frame.dropna(subset=["driver_number"])
        frame["driver_number"] = frame["driver_number"].astype(int)

    for driver_number, indices in laps.groupby("_driver_number").groups.items():
        samples = frame
        if per_driver:
            samples = frame[frame["driver_number"] == driver_number]
        if samples.empty:
            continue
        left = (
            laps.loc[list(indices), ["Date"]]
            .assign(_row=indices)
            .dropna(subset=["Date"])
            .sort_values("Date")
        )
        if left.empty:
            continue
        right_columns = ["_date", event_column]
        merged = pd.merge_asof(
            left,
            samples[right_columns].sort_values("_date"),
            left_on="Date",
            right_on="_date",
            direction="backward",
        )
        values.loc[merged["_row"].astype(int)] = merged[event_column].to_numpy()
    return values.rename(output_column)


def _gap_seconds(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().casefold()
    if not text or "lap" in text or text in {"leader", "nan"}:
        return np.nan
    try:
        seconds = float(text.removesuffix("s").replace("+", "").strip())
    except ValueError:
        try:
            seconds = pd.to_timedelta(text).total_seconds()
        except (TypeError, ValueError):
            return np.nan
    return max(0.0, seconds)


def _track_status_from_race_control(
    laps: pd.DataFrame, events: list[dict]
) -> pd.Series:
    status_by_flag = {
        "GREEN": "1",
        "CLEAR": "1",
        "YELLOW": "2",
        "DOUBLE YELLOW": "2",
        "RED": "5",
        "RED FLAG": "5",
        "CHEQUERED": "8",
        "CHEQUERED FLAG": "8",
    }
    mapped_events = []
    for event in events:
        flag = str(event.get("flag") or "").strip().upper()
        message = str(event.get("message") or "").strip().upper()
        status = status_by_flag.get(flag) or status_by_flag.get(message)
        if status is None and "SAFETY CAR" in message:
            status = "6" if "VIRTUAL" in message else "4"
        if status is not None and event.get("date") is not None:
            mapped_events.append({"date": event["date"], "status": status})
    return _merge_timed_values(
        laps,
        mapped_events,
        "status",
        "TrackStatus",
        per_driver=False,
    )


def _map_stints(laps: pd.DataFrame, stints: list[dict]) -> pd.DataFrame:
    laps["Compound"] = "UNKNOWN"
    laps["TyreLife"] = np.nan
    laps["Stint"] = np.nan
    laps["IsPitIn"] = False
    if not stints:
        return laps
    for stint in stints:
        try:
            first_lap = int(stint["lap_start"])
            last_lap = int(stint["lap_end"])
            stint_number = int(stint["stint_number"])
        except (KeyError, TypeError, ValueError):
            continue
        try:
            driver_number = int(stint["driver_number"])
        except (KeyError, TypeError, ValueError):
            continue
        mask = (
            laps["driver_number"].eq(driver_number)
            & laps["LapNumber"].between(first_lap, last_lap)
        )
        compound = str(stint.get("compound") or "UNKNOWN").upper()
        laps.loc[mask, "Compound"] = COMPOUND_MAP.get(compound, compound)
        laps.loc[mask, "TyreLife"] = (
            pd.to_numeric(stint.get("tyre_age_at_start", 0), errors="coerce")
            + laps.loc[mask, "LapNumber"]
            - first_lap
            + 1
        )
        laps.loc[mask, "Stint"] = stint_number
        if last_lap < laps["LapNumber"].max():
            laps.loc[laps["LapNumber"] == last_lap, "IsPitIn"] = True
    return laps


def load_openf1_race(
    year: int,
    race: str,
    cache_dir: str | Path = "data/raw",
    getter: Callable = requests.get,
) -> tuple[pd.DataFrame, int]:
    """Load real historical race laps and return the lap frame and safety-car count."""
    cache_path = Path(cache_dir)
    session = _find_race_session(year, race, cache_path, getter)
    session_key = session["session_key"]
    query = {"session_key": session_key}
    laps_data = _fetch_json("laps", query, cache_path, getter)
    stints_data = _fetch_json("stints", query, cache_path, getter)
    drivers_data = _fetch_json("drivers", query, cache_path, getter)
    weather_data = _fetch_json("weather", query, cache_path, getter)
    optional_data: dict[str, list[dict]] = {}
    optional_warnings: list[str] = []
    for endpoint in ("race_control", "intervals", "position"):
        try:
            optional_data[endpoint] = _fetch_json(
                endpoint, query, cache_path, getter
            )
        except (OSError, OpenF1Unavailable, requests.RequestException) as error:
            optional_data[endpoint] = []
            optional_warnings.append(f"{endpoint} feed unavailable: {error}")
    if not laps_data:
        raise OpenF1Unavailable(f"OpenF1 has no lap timing for {race} {year}.")

    drivers = {
        int(row["driver_number"]): row
        for row in drivers_data
        if row.get("driver_number") is not None
    }
    laps = pd.DataFrame(laps_data)
    required = {"driver_number", "lap_number", "lap_duration", "date_start"}
    missing = required - set(laps.columns)
    if missing:
        raise OpenF1Unavailable(
            f"OpenF1 lap timing is missing required fields: {', '.join(sorted(missing))}."
        )
    laps = laps.dropna(subset=["driver_number", "lap_number", "lap_duration"]).copy()
    laps["driver_number"] = pd.to_numeric(laps["driver_number"], errors="coerce")
    laps["LapNumber"] = pd.to_numeric(laps["lap_number"], errors="coerce")
    laps["LapTime"] = pd.to_numeric(laps["lap_duration"], errors="coerce")
    laps = laps.dropna(subset=["driver_number", "LapNumber", "LapTime"])
    laps["driver_number"] = laps["driver_number"].astype(int)
    laps["LapNumber"] = laps["LapNumber"].astype(int)
    laps["Date"] = pd.to_datetime(laps["date_start"], utc=True, errors="coerce")
    if "is_pit_out_lap" in laps:
        laps["IsPitOut"] = laps["is_pit_out_lap"].fillna(False).astype(bool)
    else:
        laps["IsPitOut"] = False
    laps["Team"] = laps["driver_number"].map(
        lambda number: drivers.get(number, {}).get("team_name", "Unknown")
    )
    laps = _map_stints(laps, stints_data)
    laps = _weather_at_laps(laps, weather_data)
    laps = laps.rename(columns={"driver_number": "_driver_number"})
    laps["TrackStatus"] = _track_status_from_race_control(
        laps, optional_data["race_control"]
    )
    interval_events = optional_data["intervals"]
    raw_gaps = _merge_timed_values(
        laps,
        interval_events,
        "interval",
        "TrafficGapSeconds",
        per_driver=True,
    )
    laps["TrafficGapSeconds"] = raw_gaps.map(_gap_seconds)
    laps["TrafficPressure"] = (
        (5.0 - laps["TrafficGapSeconds"]) / 5.0
    ).clip(0, 1).fillna(0.0)
    positions = _merge_timed_values(
        laps,
        optional_data["position"],
        "position",
        "Position",
        per_driver=True,
    )
    laps["Position"] = pd.to_numeric(positions, errors="coerce")
    laps = laps.sort_values(["_driver_number", "LapNumber"])
    laps["PositionChange"] = (
        laps.groupby("_driver_number")["Position"].shift(1) - laps["Position"]
    )
    laps["Overtakes"] = laps["PositionChange"].clip(0, 3).fillna(0.0)
    laps["Driver"] = laps["_driver_number"].map(
        lambda number: drivers.get(number, {}).get("name_acronym", str(number))
    )
    laps["IsPitIn"] = laps["IsPitIn"].fillna(False).astype(bool)
    laps = laps.sort_values(["Driver", "LapNumber"]).reset_index(drop=True)
    laps["Stint"] = laps["Stint"].fillna(1).astype(int)
    laps["TyreLife"] = laps["TyreLife"].fillna(0).astype(float)
    result = laps[
        [
            "Driver",
            "Team",
            "LapNumber",
            "LapTime",
            "Compound",
            "TyreLife",
            "Stint",
            "TrackTemp",
            "AirTemp",
            "TrackStatus",
            "TrafficPressure",
            "TrafficGapSeconds",
            "Position",
            "PositionChange",
            "Overtakes",
            "IsPitIn",
            "IsPitOut",
        ]
    ].copy()
    total_laps = int(result["LapNumber"].max())
    result.attrs["source"] = "OpenF1"
    result.attrs["totalLaps"] = total_laps
    status_by_lap = (
        result.assign(_status=result["TrackStatus"].astype(str))
        .groupby("LapNumber")["_status"]
        .agg(lambda values: any(value.strip() in {"4", "6"} for value in values))
        .sort_index()
    )
    safety_car_starts = status_by_lap & ~status_by_lap.shift(fill_value=False)
    result.attrs["safetyCarEpisodes"] = int(safety_car_starts.sum())
    availability = {
        "trackStatus": int(result["TrackStatus"].notna().sum()),
        "trafficGap": int(result["TrafficGapSeconds"].notna().sum()),
        "position": int(result["Position"].notna().sum()),
        "positionChange": int(result["PositionChange"].notna().sum()),
        "lapRows": len(result),
    }
    result.attrs["dataAvailability"] = availability
    result.attrs["load_warning"] = (
        "OpenF1 historical race data (not live-session timing) loaded successfully. "
        "Timing context coverage: "
        f"track status {availability['trackStatus']}/{availability['lapRows']}, "
        f"traffic gap {availability['trafficGap']}/{availability['lapRows']}, "
        f"position {availability['position']}/{availability['lapRows']}, "
        f"position change {availability['positionChange']}/{availability['lapRows']} laps."
    )
    if optional_warnings:
        result.attrs["load_warning"] += " " + " ".join(optional_warnings)
    return result, total_laps
