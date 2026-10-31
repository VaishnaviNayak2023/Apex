"""Race calendar ingestion and local registry persistence."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Callable

import requests
import pandas as pd

JOLPICA_RACES_URL = "https://api.jolpi.ca/ergast/f1/races.json"
PAGE_SIZE = 100
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def database_path() -> Path:
    configured = os.environ.get("RACE_DB_PATH")
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else PROJECT_ROOT / path
    storage_dir = os.environ.get("RACE_STORAGE_DIR")
    if storage_dir:
        return Path(storage_dir) / "processed" / "race_registry.sqlite"
    return PROJECT_ROOT / "data" / "processed" / "race_registry.sqlite"


def connect_registry(path: Path | str | None = None) -> sqlite3.Connection:
    resolved = Path(path) if path is not None else database_path()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(resolved, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    initialize_registry(connection)
    return connection


def initialize_registry(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS races (
            race_id TEXT PRIMARY KEY,
            year INTEGER NOT NULL,
            round INTEGER NOT NULL,
            grand_prix TEXT NOT NULL,
            circuit TEXT NOT NULL,
            date_race TEXT NOT NULL,
            race_start_utc TEXT NOT NULL,
            start_time_known INTEGER NOT NULL,
            qualifying_start_utc TEXT,
            data_tier TEXT NOT NULL CHECK (data_tier IN ('T0', 'T1', 'T2', 'T3')),
            format TEXT NOT NULL,
            status TEXT NOT NULL CHECK (
                status IN ('scheduled', 'predicted', 'live', 'finished', 'evaluated', 'cancelled')
            ),
            prediction_locked_at TEXT,
            notes TEXT,
            source TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            session_schedule_json TEXT NOT NULL DEFAULT '[]',
            UNIQUE (year, round)
        );
        CREATE INDEX IF NOT EXISTS races_year_round_idx ON races(year, round);
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id TEXT PRIMARY KEY,
            race_id TEXT NOT NULL REFERENCES races(race_id),
            prediction_type TEXT NOT NULL CHECK (prediction_type IN ('P0', 'P1', 'P2')),
            version INTEGER NOT NULL DEFAULT 1,
            locked_at TEXT NOT NULL,
            model_version TEXT NOT NULL,
            training_data_cutoff TEXT,
            payload_json TEXT NOT NULL,
            model_hash TEXT,
            supersedes_id TEXT REFERENCES predictions(prediction_id),
            correction_reason TEXT,
            UNIQUE (race_id, prediction_type, version)
        );
        CREATE INDEX IF NOT EXISTS predictions_race_locked_idx
            ON predictions(race_id, locked_at);
        CREATE TABLE IF NOT EXISTS actuals (
            race_id TEXT PRIMARY KEY REFERENCES races(race_id),
            winner_strategy_json TEXT,
            actual_lap_times_json TEXT,
            actual_result_json TEXT NOT NULL,
            data_source TEXT NOT NULL,
            ingested_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS evaluations (
            prediction_id TEXT PRIMARY KEY REFERENCES predictions(prediction_id),
            metrics_json TEXT NOT NULL,
            cause_tag TEXT NOT NULL CHECK (
                cause_tag IN (
                    'safety_car', 'weather', 'traffic', 'reliability',
                    'tyre_surprise', 'model_error', 'insufficient_data'
                )
            ),
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS backtests (
            run_id TEXT NOT NULL,
            race_id TEXT NOT NULL REFERENCES races(race_id),
            prediction_json TEXT NOT NULL,
            evaluation_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (run_id, race_id)
        );
        CREATE TABLE IF NOT EXISTS laps (
            race_id TEXT NOT NULL REFERENCES races(race_id),
            driver TEXT NOT NULL,
            lap INTEGER NOT NULL,
            lap_time REAL,
            compound TEXT,
            tyre_age REAL,
            stint INTEGER,
            track_status TEXT,
            PRIMARY KEY (race_id, driver, lap)
        );
        CREATE TABLE IF NOT EXISTS pit_stops (
            race_id TEXT NOT NULL REFERENCES races(race_id),
            driver TEXT NOT NULL,
            stop_no INTEGER NOT NULL,
            lap INTEGER NOT NULL,
            duration REAL,
            PRIMARY KEY (race_id, driver, stop_no)
        );
        CREATE TABLE IF NOT EXISTS weather (
            race_id TEXT NOT NULL REFERENCES races(race_id),
            time_utc TEXT NOT NULL,
            track_temp REAL,
            air_temp REAL,
            rain INTEGER,
            PRIMARY KEY (race_id, time_utc)
        );
        CREATE TABLE IF NOT EXISTS race_context (
            race_id TEXT PRIMARY KEY REFERENCES races(race_id),
            total_laps INTEGER NOT NULL,
            median_track_temp REAL,
            median_air_temp REAL,
            pit_loss_estimate REAL,
            source TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS registry_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS calendar_changes (
            change_id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id TEXT NOT NULL,
            year INTEGER NOT NULL,
            grand_prix TEXT NOT NULL,
            change_type TEXT NOT NULL CHECK (change_type IN ('moved', 'cancelled')),
            old_value TEXT,
            new_value TEXT,
            detected_at TEXT NOT NULL
        );
        """
    )
    prediction_columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(predictions)")
    }
    if "supersedes_id" not in prediction_columns:
        connection.execute("ALTER TABLE predictions ADD COLUMN supersedes_id TEXT")
    if "correction_reason" not in prediction_columns:
        connection.execute("ALTER TABLE predictions ADD COLUMN correction_reason TEXT")
    race_columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(races)")
    }
    if "qualifying_start_utc" not in race_columns:
        connection.execute("ALTER TABLE races ADD COLUMN qualifying_start_utc TEXT")
    if "session_schedule_json" not in race_columns:
        connection.execute(
            "ALTER TABLE races ADD COLUMN session_schedule_json TEXT NOT NULL DEFAULT '[]'"
        )


def get_race(race_id: str, path: Path | str | None = None) -> dict | None:
    connection = connect_registry(path)
    try:
        row = connection.execute(
            "SELECT * FROM races WHERE race_id = ?", (race_id,)
        ).fetchone()
        return dict(row) if row is not None else None
    finally:
        connection.close()


def races_before(
    race_date: str,
    min_data_tier: str = "T3",
    path: Path | str | None = None,
) -> list[dict]:
    tier_order = {"T0": 0, "T1": 1, "T2": 2, "T3": 3}
    if min_data_tier not in tier_order:
        raise ValueError(f"Unknown data tier: {min_data_tier}.")
    connection = connect_registry(path)
    try:
        rows = connection.execute(
            """
            SELECT * FROM races
            WHERE date_race < ? AND status != 'cancelled'
            ORDER BY date_race, year, round
            """,
            (race_date,),
        ).fetchall()
    finally:
        connection.close()
    return [
        dict(row)
        for row in rows
        if tier_order[row["data_tier"]] >= tier_order[min_data_tier]
    ]


def save_race_context(
    race_id: str,
    total_laps: int,
    median_track_temp: float | None,
    median_air_temp: float | None,
    pit_loss_estimate: float | None,
    source: str,
    path: Path | str | None = None,
    updated_at: datetime | None = None,
) -> None:
    if total_laps < 1:
        raise ValueError("Race context must contain a positive lap count.")
    if source.strip().lower() in {"demo", "demo data", "synthetic"}:
        raise ValueError("Synthetic data cannot be saved as historical race context.")
    timestamp = (updated_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    connection = connect_registry(path)
    try:
        with connection:
            connection.execute(
                """
                INSERT INTO race_context (
                    race_id, total_laps, median_track_temp, median_air_temp,
                    pit_loss_estimate, source, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(race_id) DO UPDATE SET
                    total_laps = excluded.total_laps,
                    median_track_temp = excluded.median_track_temp,
                    median_air_temp = excluded.median_air_temp,
                    pit_loss_estimate = excluded.pit_loss_estimate,
                    source = excluded.source,
                    updated_at = excluded.updated_at
                """,
                (
                    race_id,
                    total_laps,
                    median_track_temp,
                    median_air_temp,
                    pit_loss_estimate,
                    source,
                    timestamp.isoformat(),
                ),
            )
    finally:
        connection.close()


def lock_prediction(
    race_id: str,
    payload: dict,
    model_version: str,
    training_data_cutoff: str | None,
    model_hash: str | None = None,
    prediction_type: str = "P1",
    version: int = 1,
    supersedes_id: str | None = None,
    correction_reason: str | None = None,
    path: Path | str | None = None,
    now: datetime | None = None,
) -> dict:
    if prediction_type not in {"P0", "P1", "P2"}:
        raise ValueError("Prediction type must be P0, P1, or P2.")
    if version < 1:
        raise ValueError("Prediction version must be positive.")
    locked_at = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    payload_json = json.dumps(payload, sort_keys=True, allow_nan=False)
    if path is None and os.environ.get("RACE_LEDGER_DIR"):
        ledger_root = Path(os.environ["RACE_LEDGER_DIR"])
    elif path is None:
        ledger_root = PROJECT_ROOT / "predictions" / "ledger"
    else:
        ledger_root = Path(path).parent / "predictions" / "ledger"
    ledger_file = ledger_root / f"{race_id}-{prediction_type}-v{version}.json"
    created_ledger_file: Path | None = None
    connection = connect_registry(path)
    try:
        with connection:
            race = connection.execute(
                """
                SELECT race_start_utc, start_time_known, date_race,
                       qualifying_start_utc
                FROM races WHERE race_id = ?
                """,
                (race_id,),
            ).fetchone()
            if race is None:
                raise ValueError(f"Race {race_id} is not present in the registry.")
            if prediction_type == "P1":
                race_start = datetime.fromisoformat(race["race_start_utc"])
                if not model_version.strip() or not model_hash:
                    raise ValueError(
                        "A P1 prediction requires a model version and model artifact hash."
                    )
                if not training_data_cutoff:
                    raise ValueError(
                        "A P1 prediction requires the last completed training race date."
                    )
                if (
                    not race["start_time_known"]
                    and locked_at.date().isoformat() >= race["date_race"]
                ):
                    raise ValueError(
                        "P1 prediction is blocked on race day when the start time is unknown."
                    )
                if locked_at >= race_start:
                    raise ValueError("P1 predictions must be locked before race start.")
                if not race["qualifying_start_utc"]:
                    raise ValueError(
                        "P1 prediction is blocked because the calendar has no qualifying start time."
                    )
                qualifying_end = datetime.fromisoformat(
                    race["qualifying_start_utc"]
                ) + timedelta(hours=1)
                if locked_at < qualifying_end + timedelta(minutes=60):
                    raise ValueError(
                        "P1 predictions must be locked at least 60 minutes after qualifying."
                    )
                cutoff = date.fromisoformat(training_data_cutoff[:10])
                if cutoff >= date.fromisoformat(race["date_race"]):
                    raise ValueError(
                        "Training data cutoff must be before the target race date."
                    )
            elif prediction_type == "P0":
                race_start = datetime.fromisoformat(race["race_start_utc"])
                if not model_version.strip() or not model_hash or not training_data_cutoff:
                    raise ValueError(
                        "A P0 forecast requires a model version, model hash, and training cutoff."
                    )
                if locked_at >= race_start:
                    raise ValueError("P0 forecasts must be locked before race start.")
                if date.fromisoformat(training_data_cutoff[:10]) >= date.fromisoformat(
                    race["date_race"]
                ):
                    raise ValueError("P0 training data must precede the target race.")
            latest = connection.execute(
                """
                SELECT prediction_id, version FROM predictions
                WHERE race_id = ? AND prediction_type = ?
                ORDER BY version DESC LIMIT 1
                """,
                (race_id, prediction_type),
            ).fetchone()
            expected_version = 1 if latest is None else latest["version"] + 1
            if version != expected_version:
                raise ValueError(
                    f"Prediction versions are append-only; the next version must be {expected_version}."
                )
            if latest is not None:
                if supersedes_id != latest["prediction_id"] or not correction_reason:
                    raise ValueError(
                        "A correction must identify the prior prediction and explain the change."
                    )
            elif supersedes_id or correction_reason:
                raise ValueError("A first prediction cannot supersede another prediction.")

            prediction_id = str(uuid.uuid4())
            connection.execute(
                """
                INSERT INTO predictions (
                    prediction_id, race_id, prediction_type, version, locked_at,
                    model_version, training_data_cutoff, payload_json, model_hash,
                    supersedes_id, correction_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    prediction_id,
                    race_id,
                    prediction_type,
                    version,
                    locked_at.isoformat(),
                    model_version,
                    training_data_cutoff,
                    payload_json,
                    model_hash,
                    supersedes_id,
                    correction_reason,
                ),
            )
            ledger_root.mkdir(parents=True, exist_ok=True)
            ledger_record = {
                "predictionId": prediction_id,
                "raceId": race_id,
                "predictionType": prediction_type,
                "version": version,
                "lockedAt": locked_at.isoformat(),
                "modelVersion": model_version,
                "trainingDataCutoff": training_data_cutoff,
                "modelHash": model_hash,
                "supersedesId": supersedes_id,
                "correctionReason": correction_reason,
                "payload": payload,
            }
            try:
                with ledger_file.open("x", encoding="utf-8") as ledger:
                    ledger.write(
                        json.dumps(
                            ledger_record,
                            indent=2,
                            sort_keys=True,
                            allow_nan=False,
                        )
                    )
                    ledger.write("\n")
                created_ledger_file = ledger_file
            except OSError:
                if ledger_file.exists():
                    ledger_file.unlink()
                raise
    except sqlite3.Error:
        if created_ledger_file is not None and created_ledger_file.exists():
            created_ledger_file.unlink()
        raise
    finally:
        connection.close()

    return {
        "predictionId": prediction_id,
        "raceId": race_id,
        "predictionType": prediction_type,
        "version": version,
        "lockedAt": locked_at.isoformat(),
        "modelVersion": model_version,
        "trainingDataCutoff": training_data_cutoff,
        "modelHash": model_hash,
        "payload": payload,
        "supersedesId": supersedes_id,
    }


def save_actuals(
    race_id: str,
    actual_result: dict,
    data_source: str,
    winner_strategy: dict | None = None,
    actual_lap_times: list[dict] | None = None,
    path: Path | str | None = None,
    ingested_at: datetime | None = None,
) -> None:
    if not data_source or data_source.strip().lower() in {"demo", "demo data", "synthetic"}:
        raise ValueError("Synthetic or demo data cannot be saved as official race actuals.")
    timestamp = (ingested_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    connection = connect_registry(path)
    try:
        with connection:
            race = connection.execute(
                """
                SELECT race_start_utc, start_time_known, date_race
                FROM races WHERE race_id = ?
                """,
                (race_id,),
            ).fetchone()
            if race is None:
                raise ValueError(f"Race {race_id} is not present in the registry.")
            race_start = datetime.fromisoformat(race["race_start_utc"])
            if race["start_time_known"] and timestamp < race_start + timedelta(hours=5):
                raise ValueError("Race actuals cannot be ingested before the race has finished.")
            if not race["start_time_known"] and timestamp.date().isoformat() <= race["date_race"]:
                raise ValueError("Race actuals cannot be ingested before the event date has passed.")
            connection.execute(
                """
                INSERT INTO actuals (
                    race_id, winner_strategy_json, actual_lap_times_json,
                    actual_result_json, data_source, ingested_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(race_id) DO UPDATE SET
                    winner_strategy_json = excluded.winner_strategy_json,
                    actual_lap_times_json = excluded.actual_lap_times_json,
                    actual_result_json = excluded.actual_result_json,
                    data_source = excluded.data_source,
                    ingested_at = excluded.ingested_at
                """,
                (
                    race_id,
                    json.dumps(winner_strategy, sort_keys=True, allow_nan=False)
                    if winner_strategy is not None
                    else None,
                    json.dumps(actual_lap_times, sort_keys=True, allow_nan=False)
                    if actual_lap_times is not None
                    else None,
                    json.dumps(actual_result, sort_keys=True, allow_nan=False),
                    data_source,
                    timestamp.isoformat(),
                ),
            )
    finally:
        connection.close()


def save_evaluation(
    prediction_id: str,
    metrics: dict,
    cause_tag: str,
    path: Path | str | None = None,
    created_at: datetime | None = None,
) -> None:
    allowed_causes = {
        "safety_car", "weather", "traffic", "reliability",
        "tyre_surprise", "model_error", "insufficient_data",
    }
    if cause_tag not in allowed_causes:
        raise ValueError(f"Unsupported evaluation cause tag: {cause_tag}.")
    connection = connect_registry(path)
    try:
        with connection:
            prediction = connection.execute(
                """
                SELECT p.race_id FROM predictions p
                JOIN actuals a ON a.race_id = p.race_id
                WHERE p.prediction_id = ?
                """,
                (prediction_id,),
            ).fetchone()
            if prediction is None:
                raise ValueError(
                    "Evaluation requires a saved prediction and ingested actual race data."
                )
            timestamp = (created_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
            metrics_json = json.dumps(metrics, sort_keys=True, allow_nan=False)
            existing = connection.execute(
                "SELECT metrics_json, cause_tag FROM evaluations WHERE prediction_id = ?",
                (prediction_id,),
            ).fetchone()
            if existing is not None:
                if existing["metrics_json"] != metrics_json or existing["cause_tag"] != cause_tag:
                    raise ValueError(
                        "An evaluation already exists with different results; create an explicit correction."
                    )
                return
            connection.execute(
                """
                INSERT INTO evaluations(prediction_id, metrics_json, cause_tag, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    prediction_id,
                    metrics_json,
                    cause_tag,
                    timestamp.isoformat(),
                ),
            )
            connection.execute(
                "UPDATE races SET status = 'evaluated', updated_at = ? WHERE race_id = ?",
                (timestamp.isoformat(), prediction["race_id"]),
            )
    finally:
        connection.close()


def _data_tier(year: int) -> str:
    if year >= 2018:
        return "T3"
    if year >= 2011:
        return "T2"
    if year >= 1996:
        return "T1"
    return "T0"


def _parse_race_start(event: dict) -> tuple[datetime, bool]:
    race_date = date.fromisoformat(str(event["date"]))
    raw_time = event.get("time")
    if not raw_time:
        return datetime.combine(race_date, time.min, tzinfo=timezone.utc), False

    parsed_time = time.fromisoformat(str(raw_time).replace("Z", "+00:00"))
    race_start = datetime.combine(race_date, parsed_time)
    if race_start.tzinfo is None:
        race_start = race_start.replace(tzinfo=timezone.utc)
    return race_start.astimezone(timezone.utc), True


def _session_start(event: dict, session_name: str) -> str | None:
    session = event.get(session_name)
    if not isinstance(session, dict) or not session.get("date"):
        return None
    session_date = date.fromisoformat(str(session["date"]))
    session_time = session.get("time")
    if not session_time:
        return datetime.combine(session_date, time.min, tzinfo=timezone.utc).isoformat()
    parsed_time = time.fromisoformat(str(session_time).replace("Z", "+00:00"))
    session_start = datetime.combine(session_date, parsed_time)
    if session_start.tzinfo is None:
        session_start = session_start.replace(tzinfo=timezone.utc)
    return session_start.astimezone(timezone.utc).isoformat()


def _session_schedule(event: dict) -> list[dict]:
    candidates = (
        ("FirstPractice", "Practice 1"),
        ("SecondPractice", "Practice 2"),
        ("ThirdPractice", "Practice 3"),
        ("SprintQualifying", "Sprint Qualifying"),
        ("SprintShootout", "Sprint Qualifying"),
        ("Sprint", "Sprint"),
        ("Qualifying", "Qualifying"),
        ("Race", "Race"),
    )
    schedule = []
    seen = set()
    for source_name, label in candidates:
        timestamp = _session_start(event, source_name)
        if timestamp and label not in seen:
            schedule.append({"name": label, "startUtc": timestamp})
            seen.add(label)
    if not any(item["name"] == "Race" for item in schedule):
        race_timestamp = _session_start(event, "Race")
        if race_timestamp:
            schedule.append({"name": "Race", "startUtc": race_timestamp})
    return sorted(schedule, key=lambda item: item["startUtc"])


def _initial_status(race_start: datetime, now: datetime) -> str:
    if race_start > now:
        return "scheduled"
    if now <= race_start + timedelta(hours=5):
        return "live"
    return "finished"


def _normalize_event(event: dict, now: datetime) -> dict:
    year = int(event["season"])
    round_number = int(event["round"])
    race_start, start_time_known = _parse_race_start(event)
    circuit = event.get("Circuit", {})
    circuit_name = str(circuit.get("circuitName") or "Unknown circuit")
    sprint_format = any(
        key in event for key in ("Sprint", "SprintQualifying", "SprintShootout")
    )
    notes = None
    if not start_time_known:
        notes = "Race start time was not supplied by the calendar source; same-day predictions must be blocked."
    return {
        "race_id": f"{year}_{round_number:02d}",
        "year": year,
        "round": round_number,
        "grand_prix": str(event["raceName"]),
        "circuit": circuit_name,
        "date_race": race_start.date().isoformat(),
        "race_start_utc": race_start.isoformat(),
        "start_time_known": int(start_time_known),
        "qualifying_start_utc": _session_start(event, "Qualifying"),
        "data_tier": _data_tier(year),
        "format": "Sprint" if sprint_format else "Standard",
        "session_schedule": _session_schedule(event),
        "status": _initial_status(race_start, now),
        "notes": notes,
        "source": "Jolpica-F1",
    }


def _fastf1_current_schedule(year: int, now: datetime) -> list[dict]:
    import fastf1

    schedule = fastf1.get_event_schedule(year, include_testing=False)
    if schedule.empty:
        raise ValueError(f"FastF1 returned no current-season events for {year}.")
    events = []
    for _, row in schedule.iterrows():
        round_number = int(row["RoundNumber"])
        if round_number < 1:
            continue
        race_date_value = row["EventDate"]
        race_date = race_date_value.date() if hasattr(race_date_value, "date") else race_date_value
        race_start = row.get("Session5DateUtc")
        qualifying_start = row.get("Session4DateUtc")
        if hasattr(race_start, "to_pydatetime"):
            race_start = race_start.to_pydatetime()
        if hasattr(qualifying_start, "to_pydatetime"):
            qualifying_start = qualifying_start.to_pydatetime()
        race_start_known = pd.notna(race_start)
        qualifying_known = pd.notna(qualifying_start)
        if race_start_known and race_start.tzinfo is None:
            race_start = race_start.replace(tzinfo=timezone.utc)
        if not race_start_known:
            race_start = datetime.combine(race_date, time.min, tzinfo=timezone.utc)
        if qualifying_known and qualifying_start.tzinfo is None:
            qualifying_start = qualifying_start.replace(tzinfo=timezone.utc)
        event_format = str(row.get("EventFormat") or "conventional")
        session_schedule = []
        for index, fallback_name in enumerate(
            ("Practice 1", "Practice 2", "Practice 3", "Qualifying", "Race"),
            start=1,
        ):
            session_name = str(row.get(f"Session{index}") or fallback_name)
            session_start = row.get(f"Session{index}DateUtc")
            if hasattr(session_start, "to_pydatetime"):
                session_start = session_start.to_pydatetime()
            if pd.notna(session_start):
                if session_start.tzinfo is None:
                    session_start = session_start.replace(tzinfo=timezone.utc)
                session_schedule.append(
                    {
                        "name": session_name,
                        "startUtc": session_start.astimezone(timezone.utc).isoformat(),
                    }
                )
        events.append(
            {
                "race_id": f"{year}_{round_number:02d}",
                "year": year,
                "round": round_number,
                "grand_prix": str(row["EventName"]),
                "circuit": str(row.get("Location") or "Unknown circuit"),
                "date_race": race_date.isoformat(),
                "race_start_utc": race_start.astimezone(timezone.utc).isoformat(),
                "start_time_known": int(race_start_known),
                "qualifying_start_utc": (
                    qualifying_start.astimezone(timezone.utc).isoformat()
                    if qualifying_known
                    else None
                ),
                "data_tier": _data_tier(year),
                "format": "Sprint" if "sprint" in event_format.lower() else "Standard",
                "session_schedule": session_schedule,
                "status": _initial_status(race_start, now),
                "notes": None if race_start_known else "FastF1 did not supply the race start time.",
                "source": "FastF1",
            }
        )
    return events


def fetch_all_races(
    http_get: Callable = requests.get,
    page_size: int = PAGE_SIZE,
) -> list[dict]:
    """Fetch every paginated race row; fail rather than syncing a partial calendar."""
    events: list[dict] = []
    offset = 0
    expected_total: int | None = None

    while expected_total is None or offset < expected_total:
        response = http_get(
            JOLPICA_RACES_URL,
            params={"limit": page_size, "offset": offset},
            timeout=(5, 30),
            headers={"User-Agent": "F1-Strategy-Lab/1.0"},
        )
        response.raise_for_status()
        payload = response.json()
        metadata = payload.get("MRData", {})
        try:
            total = int(metadata["total"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Jolpica returned an invalid race pagination response.") from error
        current_page = metadata.get("RaceTable", {}).get("Races", [])
        if expected_total is None:
            expected_total = total
        elif total != expected_total:
            raise ValueError("The Jolpica race registry changed during pagination; retry the sync.")
        if not current_page and offset < expected_total:
            raise ValueError("Jolpica returned an incomplete race registry page.")
        events.extend(current_page)
        offset += len(current_page)

    if not events:
        raise ValueError("Jolpica returned no race calendar data.")
    return events


def sync_registry(
    path: Path | str | None = None,
    current_year: int | None = None,
    now: datetime | None = None,
    http_get: Callable = requests.get,
    schedule_loader: Callable | None = _fastf1_current_schedule,
) -> dict:
    now_utc = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    through_year = current_year or now_utc.year
    fetched = fetch_all_races(http_get=http_get)
    normalized = [
        _normalize_event(event, now_utc)
        for event in fetched
        if 1950 <= int(event["season"]) <= through_year
    ]
    if not normalized:
        raise ValueError("The race calendar contained no records for 1950 through the current season.")
    calendar_warning = None
    if schedule_loader is not None:
        try:
            jolpica_current = {
                event["race_id"]: event
                for event in normalized
                if event["year"] == through_year
            }
            current_events = schedule_loader(through_year, now_utc)
            for event in current_events:
                if event["race_id"] in jolpica_current:
                    event["circuit"] = jolpica_current[event["race_id"]]["circuit"]
                    if not event.get("session_schedule"):
                        event["session_schedule"] = jolpica_current[event["race_id"]].get(
                            "session_schedule", []
                        )
            normalized = [event for event in normalized if event["year"] != through_year]
            normalized.extend(current_events)
        except (
            ImportError,
            OSError,
            RuntimeError,
            ValueError,
            KeyError,
            requests.RequestException,
        ) as error:
            calendar_warning = (
                f"FastF1 current-season calendar unavailable; Jolpica-F1 calendar retained: {error}"
            )

    updated_at = now_utc.isoformat()
    connection = connect_registry(path)
    try:
        with connection:
            for event in normalized:
                previous = connection.execute(
                    """
                    SELECT race_start_utc, date_race, grand_prix
                    FROM races WHERE race_id = ?
                    """,
                    (event["race_id"],),
                ).fetchone()
                if previous is not None and (
                    previous["race_start_utc"] != event["race_start_utc"]
                    or previous["date_race"] != event["date_race"]
                ):
                    connection.execute(
                        """
                        INSERT INTO calendar_changes (
                            race_id, year, grand_prix, change_type,
                            old_value, new_value, detected_at
                        ) VALUES (?, ?, ?, 'moved', ?, ?, ?)
                        """,
                        (
                            event["race_id"],
                            event["year"],
                            event["grand_prix"],
                            previous["race_start_utc"],
                            event["race_start_utc"],
                            updated_at,
                        ),
                    )
                connection.execute(
                    """
                    INSERT INTO races (
                        race_id, year, round, grand_prix, circuit, date_race,
                        race_start_utc, start_time_known, qualifying_start_utc,
                        data_tier, format, session_schedule_json,
                        status, notes, source, updated_at
                    ) VALUES (
                        :race_id, :year, :round, :grand_prix, :circuit, :date_race,
                        :race_start_utc, :start_time_known, :qualifying_start_utc,
                        :data_tier, :format, :session_schedule_json,
                        :status, :notes, :source, :updated_at
                    )
                    ON CONFLICT(race_id) DO UPDATE SET
                        grand_prix = excluded.grand_prix,
                        circuit = excluded.circuit,
                        date_race = excluded.date_race,
                        race_start_utc = excluded.race_start_utc,
                        start_time_known = excluded.start_time_known,
                        qualifying_start_utc = excluded.qualifying_start_utc,
                        data_tier = excluded.data_tier,
                        format = excluded.format,
                        session_schedule_json = excluded.session_schedule_json,
                        status = excluded.status,
                        notes = excluded.notes,
                        source = excluded.source,
                        updated_at = excluded.updated_at
                    """,
                    {
                        **event,
                        "session_schedule_json": json.dumps(
                            event.get("session_schedule", [])
                        ),
                        "updated_at": updated_at,
                    },
                )

            present_ids = {event["race_id"] for event in normalized}
            current_scheduled = connection.execute(
                """
                SELECT race_id, notes FROM races
                WHERE year = ? AND date_race >= ? AND status IN ('scheduled', 'predicted')
                """,
                (through_year, now_utc.date().isoformat()),
            ).fetchall()
            for row in current_scheduled:
                if row["race_id"] not in present_ids:
                    old_notes = row["notes"] or ""
                    cancellation_note = "Removed from the upstream season calendar during sync."
                    notes = f"{old_notes} {cancellation_note}".strip()
                    connection.execute(
                        """
                        UPDATE races SET status = 'cancelled', notes = ?, updated_at = ?
                        WHERE race_id = ?
                        """,
                        (notes, updated_at, row["race_id"]),
                    )
                    connection.execute(
                        """
                        INSERT INTO calendar_changes (
                            race_id, year, grand_prix, change_type,
                            old_value, new_value, detected_at
                        ) SELECT race_id, year, grand_prix, 'cancelled',
                                 race_start_utc, NULL, ?
                          FROM races WHERE race_id = ?
                        """,
                        (updated_at, row["race_id"]),
                    )

            connection.execute(
                """
                INSERT INTO registry_meta(key, value) VALUES ('last_sync_at', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (updated_at,),
            )
            source = (
                "Jolpica-F1 + FastF1 current season"
                if schedule_loader is not None and calendar_warning is None
                else "Jolpica-F1"
            )
            connection.execute(
                """
                INSERT INTO registry_meta(key, value) VALUES ('source', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (source,),
            )
            if calendar_warning is not None:
                connection.execute(
                    """
                    INSERT INTO registry_meta(key, value) VALUES ('calendar_warning', ?)
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value
                    """,
                    (calendar_warning,),
                )
            else:
                connection.execute(
                    "DELETE FROM registry_meta WHERE key = 'calendar_warning'"
                )
    finally:
        connection.close()

    return {
        "source": source,
        "updatedAt": updated_at,
        "raceCount": len(normalized),
        "firstYear": min(event["year"] for event in normalized),
        "lastYear": max(event["year"] for event in normalized),
        "warning": calendar_warning,
    }


def registry_is_empty(path: Path | str | None = None) -> bool:
    connection = connect_registry(path)
    try:
        return connection.execute("SELECT COUNT(*) FROM races").fetchone()[0] == 0
    finally:
        connection.close()


def list_calendar(
    year: int,
    path: Path | str | None = None,
    now: datetime | None = None,
) -> dict:
    now_utc = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    connection = connect_registry(path)
    try:
        events = connection.execute(
            """
            SELECT r.race_id, r.year, r.round, r.grand_prix, r.circuit,
                   r.date_race, r.race_start_utc, r.start_time_known,
                   r.qualifying_start_utc,
                   r.data_tier, r.format, r.status, r.notes,
                   r.session_schedule_json,
                   p0.locked_at AS early_prediction_at,
                   p1.locked_at AS updated_prediction_at,
                   CASE WHEN e.prediction_id IS NULL THEN 0 ELSE 1 END AS evaluated
            FROM races r
            LEFT JOIN (
                SELECT race_id, MAX(locked_at) AS locked_at
                FROM predictions WHERE prediction_type = 'P1'
                GROUP BY race_id
            ) p1 ON p1.race_id = r.race_id
            LEFT JOIN (
                SELECT race_id, MAX(locked_at) AS locked_at
                FROM predictions WHERE prediction_type = 'P0'
                GROUP BY race_id
            ) p0 ON p0.race_id = r.race_id
            LEFT JOIN evaluations e ON e.prediction_id = (
                SELECT prediction_id FROM predictions
                WHERE race_id = r.race_id AND prediction_type = 'P1'
                ORDER BY version DESC LIMIT 1
            )
            WHERE r.year = ?
            ORDER BY r.round
            """,
            (year,),
        ).fetchall()
        meta = dict(
            connection.execute("SELECT key, value FROM registry_meta").fetchall()
        )
        changes = connection.execute(
            """
            SELECT race_id, grand_prix, change_type, old_value, new_value, detected_at
            FROM calendar_changes
            WHERE year = ?
            ORDER BY detected_at DESC, change_id DESC
            LIMIT 20
            """,
            (year,),
        ).fetchall()
    finally:
        connection.close()

    serialized = []
    for row in events:
        start = datetime.fromisoformat(row["race_start_utc"])
        status = row["status"]
        if status != "cancelled":
            if row["evaluated"]:
                status = "evaluated"
            elif now_utc.date() > date.fromisoformat(row["date_race"]):
                status = "finished"
            elif now_utc < start:
                if row["updated_prediction_at"]:
                    status = "updated_forecast"
                elif row["early_prediction_at"]:
                    status = "early_forecast"
                else:
                    status = "scheduled"
            elif now_utc <= start + timedelta(hours=5):
                status = "live"
            else:
                status = "awaiting_timing"
        schedule = json.loads(row["session_schedule_json"] or "[]")
        for session in schedule:
            session_start = datetime.fromisoformat(session["startUtc"])
            session["completed"] = now_utc >= session_start + timedelta(hours=2)
        serialized.append(
            {
                "raceId": row["race_id"],
                "year": row["year"],
                "round": row["round"],
                "grandPrix": row["grand_prix"],
                "circuit": row["circuit"],
                "date": row["date_race"],
                "raceStartUtc": row["race_start_utc"],
                "startTimeKnown": bool(row["start_time_known"]),
                "qualifyingStartUtc": row["qualifying_start_utc"],
                "sessions": schedule,
                "dataTier": row["data_tier"],
                "format": row["format"],
                "status": status,
                "forecastStage": (
                    "Updated" if row["updated_prediction_at"]
                    else "Early" if row["early_prediction_at"]
                    else None
                ),
                "predictionLockedAt": (
                    row["updated_prediction_at"] or row["early_prediction_at"]
                ),
                "notes": row["notes"],
            }
        )

    future = [
        event for event in serialized
        if event["status"] in {
            "scheduled", "early_forecast", "updated_forecast", "predicted"
        }
    ]
    finished = [event for event in serialized if event["status"] in {"finished", "evaluated"}]
    default_event = (future or finished or serialized)
    return {
        "year": year,
        "currentYear": now_utc.year,
        "years": list(range(now_utc.year, 1949, -1)),
        "events": serialized,
        "defaultRaceId": default_event[0]["raceId"] if future and default_event else (
            finished[-1]["raceId"] if finished else None
        ),
        "source": meta.get("source", "Jolpica-F1"),
        "lastUpdated": meta.get("last_sync_at"),
        "warning": meta.get("calendar_warning"),
        "changes": [
            {
                "raceId": row["race_id"],
                "grandPrix": row["grand_prix"],
                "type": row["change_type"],
                "oldValue": row["old_value"],
                "newValue": row["new_value"],
                "detectedAt": row["detected_at"],
            }
            for row in changes
        ],
    }


def list_predictions(
    year: int | None = None,
    path: Path | str | None = None,
) -> list[dict]:
    connection = connect_registry(path)
    try:
        query = """
            SELECT p.prediction_id, p.race_id, p.prediction_type, p.version,
                   p.locked_at, p.model_version, p.training_data_cutoff,
                   p.payload_json, p.supersedes_id, p.correction_reason,
                   r.year, r.grand_prix, r.circuit,
                   e.metrics_json, e.cause_tag
            FROM predictions p
            JOIN races r ON r.race_id = p.race_id
            LEFT JOIN evaluations e ON e.prediction_id = p.prediction_id
        """
        params: tuple = ()
        if year is not None:
            query += " WHERE r.year = ?"
            params = (year,)
        query += " ORDER BY p.locked_at DESC, p.version DESC"
        rows = connection.execute(query, params).fetchall()
    finally:
        connection.close()

    return [
        {
            "predictionId": row["prediction_id"],
            "raceId": row["race_id"],
            "predictionType": row["prediction_type"],
            "version": row["version"],
            "lockedAt": row["locked_at"],
            "modelVersion": row["model_version"],
            "trainingDataCutoff": row["training_data_cutoff"],
            "supersedesId": row["supersedes_id"],
            "correctionReason": row["correction_reason"],
            "payload": json.loads(row["payload_json"]),
            "year": row["year"],
            "grandPrix": row["grand_prix"],
            "circuit": row["circuit"],
            "metrics": json.loads(row["metrics_json"]) if row["metrics_json"] else None,
            "causeTag": row["cause_tag"],
        }
        for row in rows
    ]


def save_user_pick(
    owner_id: str,
    race_id: str,
    strategy: dict,
    path: Path | str | None = None,
    now: datetime | None = None,
) -> dict:
    if not owner_id or len(owner_id) > 80:
        raise ValueError("A valid local user identifier is required.")
    compounds = strategy.get("compounds")
    pit_laps = strategy.get("pitLaps")
    if (
        not isinstance(compounds, list)
        or not 2 <= len(compounds) <= 3
        or len(set(compounds)) < 2
        or any(compound not in {"SOFT", "MEDIUM", "HARD"} for compound in compounds)
        or not isinstance(pit_laps, list)
        or len(pit_laps) != len(compounds) - 1
        or any(not isinstance(lap, int) or lap < 1 for lap in pit_laps)
        or sorted(pit_laps) != pit_laps
        or len(set(pit_laps)) != len(pit_laps)
    ):
        raise ValueError("The submitted strategy has invalid compounds or pit laps.")
    timestamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    connection = connect_registry(path)
    try:
        with connection:
            race = connection.execute(
                "SELECT race_start_utc, status FROM races WHERE race_id = ?", (race_id,)
            ).fetchone()
            if race is None:
                raise ValueError(f"Race {race_id} is not present in the registry.")
            if race["status"] == "cancelled":
                raise ValueError("Pre-race picks are unavailable for a cancelled race.")
            if timestamp >= datetime.fromisoformat(race["race_start_utc"]):
                raise ValueError("Pre-race picks close when the race starts.")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS user_picks (
                    pick_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    race_id TEXT NOT NULL REFERENCES races(race_id),
                    strategy_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    actual_match INTEGER,
                    UNIQUE(owner_id, race_id)
                )
                """
            )
            pick_id = str(uuid.uuid4())
            connection.execute(
                """
                INSERT INTO user_picks (
                    pick_id, owner_id, race_id, strategy_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(owner_id, race_id) DO UPDATE SET
                    strategy_json = excluded.strategy_json,
                    created_at = excluded.created_at,
                    actual_match = NULL
                """,
                (
                    pick_id,
                    owner_id,
                    race_id,
                    json.dumps(strategy, sort_keys=True, allow_nan=False),
                    timestamp.isoformat(),
                ),
            )
            row = connection.execute(
                """
                SELECT pick_id, race_id, strategy_json, created_at
                FROM user_picks WHERE owner_id = ? AND race_id = ?
                """,
                (owner_id, race_id),
            ).fetchone()
    finally:
        connection.close()
    return {
        "pickId": row["pick_id"],
        "raceId": row["race_id"],
        "strategy": json.loads(row["strategy_json"]),
        "createdAt": row["created_at"],
    }


def list_user_pick_scoreboard(
    owner_id: str,
    path: Path | str | None = None,
) -> dict:
    connection = connect_registry(path)
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS user_picks (
                pick_id TEXT PRIMARY KEY,
                owner_id TEXT NOT NULL,
                race_id TEXT NOT NULL REFERENCES races(race_id),
                strategy_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                actual_match INTEGER,
                UNIQUE(owner_id, race_id)
            )
            """
        )
        picks = connection.execute(
            """
            SELECT p.pick_id, p.race_id, p.strategy_json, p.created_at,
                   a.actual_result_json
            FROM user_picks p
            LEFT JOIN actuals a ON a.race_id = p.race_id
            WHERE p.owner_id = ?
            ORDER BY p.created_at DESC
            """,
            (owner_id,),
        ).fetchall()
    finally:
        connection.close()

    scored = []
    for row in picks:
        strategy = json.loads(row["strategy_json"])
        actual = (
            json.loads(row["actual_result_json"]).get("winnerStrategy")
            if row["actual_result_json"]
            else None
        )
        match = (
            actual is not None
            and strategy.get("compounds") == actual.get("compoundOrder")
            and strategy.get("pitLaps") == actual.get("pitLaps")
        )
        scored.append(
            {
                "pickId": row["pick_id"],
                "raceId": row["race_id"],
                "strategy": strategy,
                "createdAt": row["created_at"],
                "evaluated": actual is not None,
                "exactMatch": match if actual is not None else None,
            }
        )
    evaluated = [row for row in scored if row["evaluated"]]
    return {
        "picks": scored,
        "evaluated": len(evaluated),
        "exactMatches": sum(row["exactMatch"] is True for row in evaluated),
        "hitRate": (
            sum(row["exactMatch"] is True for row in evaluated) / len(evaluated)
            if evaluated
            else None
        ),
    }


def list_backtest_runs(
    year: int | None = None,
    path: Path | str | None = None,
) -> list[dict]:
    connection = connect_registry(path)
    try:
        query = """
            SELECT b.run_id, b.race_id, b.evaluation_json, b.created_at, r.year
            FROM backtests b JOIN races r ON r.race_id = b.race_id
        """
        params: tuple = ()
        if year is not None:
            query += " WHERE r.year = ?"
            params = (year,)
        query += " ORDER BY b.created_at DESC, b.race_id"
        rows = connection.execute(query, params).fetchall()
    finally:
        connection.close()

    runs: dict[str, dict] = {}
    for row in rows:
        evaluation = json.loads(row["evaluation_json"])
        run = runs.setdefault(
            row["run_id"],
            {
                "runId": row["run_id"],
                "createdAt": row["created_at"],
                "years": [],
                "races": 0,
                "stopMatches": [],
                "pitErrors": [],
                "lapTimeErrors": [],
            },
        )
        run["races"] += 1
        run["years"].append(row["year"])
        metrics = evaluation["metrics"]
        if metrics.get("stopCountMatch") is not None:
            run["stopMatches"].append(bool(metrics["stopCountMatch"]))
        if metrics.get("pitLapError") is not None:
            run["pitErrors"].append(float(metrics["pitLapError"]))
        if metrics.get("lapTimeMae") is not None:
            run["lapTimeErrors"].append(float(metrics["lapTimeMae"]))

    results = []
    for run in runs.values():
        results.append(
            {
                "runId": run["runId"],
                "createdAt": run["createdAt"],
                "startYear": min(run["years"]),
                "endYear": max(run["years"]),
                "evaluatedRaces": run["races"],
                "stopCountHitRate": (
                    sum(run["stopMatches"]) / len(run["stopMatches"])
                    if run["stopMatches"]
                    else None
                ),
                "meanPitLapError": (
                    sum(run["pitErrors"]) / len(run["pitErrors"])
                    if run["pitErrors"]
                    else None
                ),
                "meanLapTimeMae": (
                    sum(run["lapTimeErrors"]) / len(run["lapTimeErrors"])
                    if run["lapTimeErrors"]
                    else None
                ),
            }
        )
    return results
