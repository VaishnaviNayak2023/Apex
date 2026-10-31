from datetime import date
import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pandas as pd

from api.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["season"] == date.today().year


def test_event_schedule_creates_fastf1_cache_directory(monkeypatch, tmp_path):
    from api import main

    cache_dir = tmp_path / "raw"
    configured_dirs = []
    fastf1 = SimpleNamespace(
        Cache=SimpleNamespace(
            configure=lambda cache_dir: configured_dirs.append(cache_dir)
        ),
        get_event_schedule=lambda _year, include_testing: (
            pd.DataFrame({"EventName": ["Test Grand Prix"]})
            if include_testing is False
            else pd.DataFrame()
        ),
    )
    monkeypatch.setenv("RACE_STORAGE_DIR", str(tmp_path))
    monkeypatch.setitem(sys.modules, "fastf1", fastf1)
    main._event_schedule.cache_clear()

    try:
        schedule = main._event_schedule(2026)
    finally:
        main._event_schedule.cache_clear()

    assert not schedule.empty
    assert cache_dir.is_dir()
    assert configured_dirs == [str(cache_dir)]


def test_dashboard_token_protects_api_but_not_health(monkeypatch):
    from api import main

    monkeypatch.setenv("DASHBOARD_API_TOKEN", "test-dashboard-token")
    monkeypatch.setattr(main, "list_registry_predictions", lambda _year: [])

    assert client.get("/api/health").status_code == 200
    assert client.get("/api/predictions").status_code == 401
    authorized = client.get(
        "/api/predictions",
        headers={"x-dashboard-api-token": "test-dashboard-token"},
    )
    assert authorized.status_code == 200
    assert authorized.json()["predictions"] == []


def test_strategy_request_rejects_invalid_stop_count():
    response = client.post("/api/strategies", json={"stops": 3})
    assert response.status_code == 422


def test_race_route_returns_service_unavailable_without_real_timing(monkeypatch):
    from api import main
    from src.data_loader import RealDataUnavailable

    def unavailable(*_args, **_kwargs):
        raise RealDataUnavailable("No real timing data is available.")

    monkeypatch.setattr(main, "_load_race_data", unavailable)

    response = client.get(
        f"/api/race?year={date.today().year}&race=Test%20Grand%20Prix"
    )

    assert response.status_code == 503
    assert "No real timing data" in response.json()["detail"]


def test_events_route_returns_current_season_and_default_event(monkeypatch):
    from api import main

    monkeypatch.setattr(
        main,
        "_schedule_rows",
        lambda year, refresh=False: [
            {"name": "Earlier GP", "round": 1, "date": "2026-03-01", "completed": True},
            {"name": "Latest GP", "round": 2, "date": "2026-09-01", "completed": True},
        ],
    )
    response = client.get(f"/api/events?year={date.today().year}")
    assert response.status_code == 200
    assert response.json()["defaultEvent"] == "Latest GP"
    assert date.today().year in response.json()["years"]


def test_calendar_route_returns_generated_registry(monkeypatch):
    from api import main

    registry = {
        "year": 2026,
        "events": [{"raceId": "2026_19", "status": "scheduled"}],
        "defaultRaceId": "2026_19",
    }
    monkeypatch.setattr(main, "registry_is_empty", lambda: False)
    monkeypatch.setattr(main, "list_registry_calendar", lambda year: {**registry, "year": year})

    response = client.get("/api/calendar?year=2026")

    assert response.status_code == 200
    assert response.json()["defaultRaceId"] == "2026_19"
    assert response.json()["events"][0]["status"] == "scheduled"


def test_predictions_endpoint_returns_ledger_and_filter(monkeypatch):
    from api import main

    monkeypatch.setattr(
        main,
        "list_registry_predictions",
        lambda year: [{"year": year, "predictionType": "P1"}],
    )
    response = client.get("/api/predictions?year=2026")
    assert response.status_code == 200
    assert response.json()["predictions"] == [{"year": 2026, "predictionType": "P1"}]


def test_evaluations_endpoint_summarizes_only_scored_predictions(monkeypatch):
    from api import main

    monkeypatch.setattr(
        main,
        "list_registry_predictions",
        lambda _year: [
            {"metrics": {"stopCountMatch": True, "pitLapError": 2, "lapTimeMae": 0.6}},
            {"metrics": None},
        ],
    )
    response = client.get("/api/evaluations?year=2026")
    assert response.status_code == 200
    assert response.json()["summary"] == {
        "evaluatedRaces": 1,
        "stopCountHitRate": 1.0,
        "meanPitLapError": 2.0,
        "meanLapTimeMae": 0.6,
    }


def test_backtests_endpoint_returns_runs_for_selected_year(monkeypatch):
    from api import main

    run = {"runId": "run-1", "startYear": 2024, "endYear": 2024, "evaluatedRaces": 3}
    monkeypatch.setattr(main, "list_backtest_runs", lambda year: [run] if year == 2024 else [])

    response = client.get("/api/backtests?year=2024")

    assert response.status_code == 200
    assert response.json() == {"year": 2024, "runs": [run]}


def test_race_endpoint_selects_lowest_mae_model(monkeypatch):
    import numpy as np
    import pandas as pd

    from api import main
    from src.train import ModelBundle

    class ApiModel:
        def predict(self, features):
            return np.full(len(features), 90.0)

    laps = pd.DataFrame(
        [
            {
                "Driver": "VER",
                "Team": "Red Bull",
                "LapNumber": 1,
                "LapTime": 90.0,
                "Compound": "MEDIUM",
                "TyreLife": 1,
                "Stint": 1,
                "TrackTemp": 30.0,
                "AirTemp": 20.0,
                "TrackStatus": "1",
                "TrafficPressure": 0.0,
                "TrafficGapSeconds": 1.25,
                "Position": 2,
                "PositionChange": 1,
                "Overtakes": 0.0,
                "IsPitIn": False,
                "IsPitOut": False,
                "team": "Red Bull",
                "tyre_age": 1.0,
                "traffic_pressure": 0.0,
                "overtakes": 0.0,
            }
        ]
    )
    models = {
        "Linear": ModelBundle(
            model=ApiModel(),
            metrics={"MAE": 0.2, "RMSE": 0.3, "R2": 0.8},
            feature_columns=[],
        ),
        "XGBoost": ModelBundle(
            model=ApiModel(),
            metrics={"MAE": 0.9, "RMSE": 1.0, "R2": 0.4},
            feature_columns=[],
        ),
    }
    monkeypatch.setattr(
        main,
        "get_race_data",
        lambda year, race: (laps, models, None, "OpenF1", 30, 0),
    )

    response = client.get(
        f"/api/race?year={date.today().year}&race=Test%20Grand%20Prix"
    )

    assert response.status_code == 200
    assert response.json()["bestModel"] == "Linear"
    assert response.json()["bestModelMae"] == 0.2
    lap = response.json()["laps"][0]
    assert lap["trackStatus"] == "1"
    assert lap["trafficGapSeconds"] == 1.25
    assert lap["position"] == 2
    assert lap["positionChange"] == 1


def test_strategy_endpoint_returns_monte_carlo_ranges(monkeypatch):
    import pandas as pd

    from api import main
    from src.train import ModelBundle

    class ApiModel:
        def predict(self, features):
            return (
                90 + features["tyre_age"] * 0.1
                + features["traffic_pressure"] * 0.5
                - features["overtakes"] * 0.1
            )

    laps = pd.DataFrame()
    model = ModelBundle(
        model=ApiModel(),
        metrics={"MAE": 0.3, "RMSE": 0.4, "R2": 0.8},
        feature_columns=[],
    )
    monkeypatch.setattr(
        main,
        "get_race_data",
        lambda year, race: (laps, {"Linear": model}, None, "Test data", 30, 0),
    )
    response = client.post(
        "/api/strategies",
        json={
            "year": date.today().year,
            "race": "Test Grand Prix",
            "total_laps": 30,
            "stops": 1,
        },
    )
    assert response.status_code == 200
    strategy = response.json()["strategies"][0]
    assert strategy["p10"] <= strategy["p90"]
    assert 0 <= strategy["winProbability"] <= 1
