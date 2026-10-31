from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.race_registry import (
    connect_registry,
    database_path,
    list_calendar,
    list_backtest_runs,
    list_user_pick_scoreboard,
    lock_prediction,
    save_actuals,
    save_evaluation,
    save_user_pick,
    sync_registry,
)


class FakeResponse:
    def __init__(self, events, total):
        self.payload = {
            "MRData": {
                "total": str(total),
                "RaceTable": {"Races": events},
            }
        }

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def event(year, round_number, race_name, race_date, race_time=None, sprint=False):
    row = {
        "season": str(year),
        "round": str(round_number),
        "raceName": race_name,
        "date": race_date,
        "Circuit": {"circuitName": "Test Circuit"},
    }
    if race_time:
        row["time"] = race_time
        row["Qualifying"] = {
            "date": (date.fromisoformat(race_date) - timedelta(days=2)).isoformat(),
            "time": "19:00:00Z",
        }
    if sprint:
        row["Sprint"] = {"date": race_date, "time": "10:00:00Z"}
    return row


def paged_get(events):
    def get(_url, params, **_kwargs):
        offset = params["offset"]
        return FakeResponse(
            events[offset : offset + params["limit"]],
            total=len(events),
        )

    return get


def test_registry_respects_deployment_storage_environment(monkeypatch, tmp_path):
    monkeypatch.delenv("RACE_DB_PATH", raising=False)
    monkeypatch.setenv("RACE_STORAGE_DIR", str(tmp_path / "persistent"))

    assert database_path() == tmp_path / "persistent" / "processed" / "race_registry.sqlite"

    monkeypatch.setenv("RACE_DB_PATH", "custom/registry.sqlite")
    assert database_path() == (
        Path(__file__).resolve().parents[1] / "custom" / "registry.sqlite"
    )


def test_registry_generates_tiers_statuses_and_sprint_format(tmp_path):
    races = [
        event(1950, 1, "Historic Grand Prix", "1950-05-13"),
        event(2026, 19, "Mexico City Grand Prix", "2026-11-01", "20:00:00Z", sprint=True),
    ]
    now = datetime(2026, 10, 31, 12, tzinfo=timezone.utc)

    result = sync_registry(
        path=tmp_path / "races.sqlite",
        current_year=2026,
        now=now,
        http_get=paged_get(races),
        schedule_loader=None,
    )

    calendar = list_calendar(2026, path=tmp_path / "races.sqlite", now=now)
    historic = list_calendar(1950, path=tmp_path / "races.sqlite", now=now)
    assert result["raceCount"] == 2
    assert calendar["defaultRaceId"] == "2026_19"
    assert calendar["events"][0]["dataTier"] == "T3"
    assert calendar["events"][0]["status"] == "scheduled"
    assert calendar["events"][0]["format"] == "Sprint"
    assert historic["events"][0]["dataTier"] == "T0"
    assert historic["events"][0]["startTimeKnown"] is False
    assert historic["events"][0]["status"] == "finished"


def test_sync_marks_removed_current_season_races_cancelled(tmp_path):
    db_path = tmp_path / "races.sqlite"
    now = datetime(2026, 10, 31, 12, tzinfo=timezone.utc)
    races = [
        event(1950, 1, "Historic Grand Prix", "1950-05-13"),
        event(2026, 19, "Mexico City Grand Prix", "2026-11-01", "20:00:00Z"),
    ]
    sync_registry(
        path=db_path,
        current_year=2026,
        now=now,
        http_get=paged_get(races),
        schedule_loader=None,
    )

    updated = [
        event(1950, 1, "Historic Grand Prix", "1950-05-13"),
        event(2026, 20, "São Paulo Grand Prix", "2026-11-08", "17:00:00Z"),
    ]
    sync_registry(
        path=db_path,
        current_year=2026,
        now=now,
        http_get=paged_get(updated),
        schedule_loader=None,
    )

    calendar = list_calendar(2026, path=db_path, now=now)
    cancelled = next(row for row in calendar["events"] if row["raceId"] == "2026_19")
    assert cancelled["status"] == "cancelled"
    assert "Removed from the upstream season calendar" in cancelled["notes"]


def test_sync_records_moved_calendar_event(tmp_path):
    db_path = tmp_path / "races.sqlite"
    now = datetime(2026, 10, 20, tzinfo=timezone.utc)
    original = event(2026, 19, "Mexico City Grand Prix", "2026-11-01", "20:00:00Z")
    sync_registry(
        path=db_path,
        current_year=2026,
        now=now,
        http_get=paged_get([original]),
        schedule_loader=None,
    )
    moved = event(2026, 19, "Mexico City Grand Prix", "2026-11-02", "20:00:00Z")
    sync_registry(
        path=db_path,
        current_year=2026,
        now=now,
        http_get=paged_get([moved]),
        schedule_loader=None,
    )

    calendar = list_calendar(2026, path=db_path, now=now)

    assert calendar["changes"][0]["type"] == "moved"
    assert calendar["changes"][0]["oldValue"].startswith("2026-11-01")
    assert calendar["changes"][0]["newValue"].startswith("2026-11-02")


def test_locked_prediction_is_immutable_and_must_precede_race_start(tmp_path):
    db_path = tmp_path / "races.sqlite"
    now = datetime(2026, 10, 31, 12, tzinfo=timezone.utc)
    races = [
        event(1950, 1, "Historic Grand Prix", "1950-05-13"),
        event(2026, 19, "Mexico City Grand Prix", "2026-11-01", "20:00:00Z"),
    ]
    sync_registry(
        path=db_path,
        current_year=2026,
        now=now,
        http_get=paged_get(races),
        schedule_loader=None,
    )
    payload = {"bestStrategy": {"compounds": ["MEDIUM", "HARD"], "pitLaps": [24]}}

    locked = lock_prediction(
        "2026_19",
        payload,
        "source:v1",
        "2026-10-25",
        model_hash="model-hash",
        path=db_path,
        now=now,
    )
    saved_file = tmp_path / "predictions" / "ledger" / "2026_19-P1-v1.json"
    assert saved_file.exists()
    assert saved_file.read_text(encoding="utf-8").find('"version": 1') >= 0
    with pytest.raises(ValueError, match="next version must be 2"):
        lock_prediction(
            "2026_19",
            payload,
            "source:v1",
            "2026-10-25",
            model_hash="model-hash",
            path=db_path,
            now=now,
        )
    with pytest.raises(ValueError, match="before race start"):
        lock_prediction(
            "2026_19",
            payload,
            "source:v1",
            "2026-10-25",
            model_hash="model-hash",
            path=db_path,
            now=datetime(2026, 11, 1, 21, tzinfo=timezone.utc),
        )

    corrected = lock_prediction(
        "2026_19",
        payload,
        "source:v2",
        "2026-10-25",
        model_hash="model-hash-v2",
        version=2,
        supersedes_id=locked["predictionId"],
        correction_reason="Corrected the weather scenario before lights out.",
        path=db_path,
        now=datetime(2026, 11, 1, 19, tzinfo=timezone.utc),
    )
    assert corrected["version"] == 2


def test_early_forecast_rejects_target_event_training_cutoff(tmp_path):
    db_path = tmp_path / "races.sqlite"
    now = datetime(2026, 10, 31, 12, tzinfo=timezone.utc)
    sync_registry(
        path=db_path,
        current_year=2026,
        now=now,
        http_get=paged_get(
            [event(2026, 19, "Mexico City Grand Prix", "2026-11-01", "20:00:00Z")]
        ),
        schedule_loader=None,
    )
    payload = {"bestStrategy": {"compounds": ["MEDIUM", "HARD"], "pitLaps": [24]}}

    with pytest.raises(ValueError, match="P0 training data must precede"):
        lock_prediction(
            "2026_19",
            payload,
            "source:v1",
            "2026-11-01",
            model_hash="model-hash",
            prediction_type="P0",
            path=db_path,
            now=now,
        )

    locked = lock_prediction(
        "2026_19",
        payload,
        "source:v1",
        "2026-10-25",
        model_hash="model-hash",
        prediction_type="P0",
        path=db_path,
        now=now,
    )
    assert locked["predictionType"] == "P0"


def test_user_picks_require_valid_dry_compounds_and_scoreboard_is_unscored(tmp_path):
    db_path = tmp_path / "races.sqlite"
    now = datetime(2026, 10, 31, 12, tzinfo=timezone.utc)
    sync_registry(
        path=db_path,
        current_year=2026,
        now=now,
        http_get=paged_get(
            [event(2026, 19, "Mexico City Grand Prix", "2026-11-01", "20:00:00Z")]
        ),
        schedule_loader=None,
    )

    with pytest.raises(ValueError, match="invalid compounds or pit laps"):
        save_user_pick(
            "local-user-123",
            "2026_19",
            {"compounds": ["MEDIUM", "MEDIUM"], "pitLaps": [24]},
            path=db_path,
            now=now,
        )
    saved = save_user_pick(
        "local-user-123",
        "2026_19",
        {"compounds": ["MEDIUM", "HARD"], "pitLaps": [24]},
        path=db_path,
        now=now,
    )
    scoreboard = list_user_pick_scoreboard("local-user-123", path=db_path)

    assert saved["raceId"] == "2026_19"
    assert scoreboard["evaluated"] == 0
    assert scoreboard["hitRate"] is None
    assert scoreboard["picks"][0]["exactMatch"] is None


def test_demo_actuals_and_unbacked_evaluations_are_rejected(tmp_path):
    db_path = tmp_path / "races.sqlite"
    now = datetime(2026, 10, 31, 12, tzinfo=timezone.utc)
    races = [
        event(1950, 1, "Historic Grand Prix", "1950-05-13"),
        event(2026, 19, "Mexico City Grand Prix", "2026-11-01", "20:00:00Z"),
    ]
    sync_registry(
        path=db_path,
        current_year=2026,
        now=now,
        http_get=paged_get(races),
        schedule_loader=None,
    )

    with pytest.raises(ValueError, match="Synthetic or demo"):
        save_actuals("2026_19", {}, "Demo data", path=db_path)
    with pytest.raises(ValueError, match="requires a saved prediction"):
        save_evaluation("missing", {}, "model_error", path=db_path)


def test_backtest_runs_aggregate_stored_race_metrics(tmp_path):
    db_path = tmp_path / "races.sqlite"
    sync_registry(
        path=db_path,
        current_year=2026,
        now=datetime(2026, 1, 1, tzinfo=timezone.utc),
        http_get=paged_get([event(2024, 1, "Test Grand Prix", "2024-03-01")]),
        schedule_loader=None,
    )
    connection = connect_registry(db_path)
    with connection:
        connection.execute(
            """
            INSERT INTO backtests(
                run_id, race_id, prediction_json, evaluation_json, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                "run-1",
                "2024_01",
                "{}",
                '{"metrics":{"stopCountMatch":true,"pitLapError":2.5,"lapTimeMae":0.8}}',
                "2026-01-02T00:00:00+00:00",
            ),
        )
    connection.close()

    runs = list_backtest_runs(2024, path=db_path)

    assert runs == [
        {
            "runId": "run-1",
            "createdAt": "2026-01-02T00:00:00+00:00",
            "startYear": 2024,
            "endYear": 2024,
            "evaluatedRaces": 1,
            "stopCountHitRate": 1.0,
            "meanPitLapError": 2.5,
            "meanLapTimeMae": 0.8,
        }
    ]
