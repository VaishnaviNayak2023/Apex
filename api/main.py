"""FastAPI endpoints used by the Next.js dashboard."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from threading import Lock
import os
from pathlib import Path

import joblib
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.cleaning import clean_laps
from src.data_loader import RealDataUnavailable, load_race
from src.features import add_features
from src.race_registry import (
    connect_registry,
    get_race,
    list_calendar as list_registry_calendar,
    list_backtest_runs,
    list_predictions as list_registry_predictions,
    list_user_pick_scoreboard,
    registry_is_empty,
    save_user_pick,
    sync_registry,
)
from src.pipeline import MODEL_DIR, create_p0_prediction, create_p1_prediction
from src.simulate import Strategy, StrategyScenario, rank_strategies, simulate_strategy
from src.train import train_models

app = FastAPI(title="F1 Strategy Lab API", version="1.0.0")
_race_data_lock = Lock()
_registry_sync_lock = Lock()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@lru_cache(maxsize=12)
def _event_schedule(year: int) -> pd.DataFrame:
    import fastf1

    fastf1.Cache.configure(cache_dir="data/raw")
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    if schedule.empty:
        raise ValueError(f"No Formula 1 Grand Prix schedule found for {year}.")
    return schedule.copy()


def _schedule_rows(year: int, refresh: bool = False) -> list[dict]:
    if refresh:
        _event_schedule.cache_clear()
    try:
        schedule = _event_schedule(year)
    except (OSError, RuntimeError, ValueError, KeyError, requests.RequestException) as error:
        raise HTTPException(
            status_code=503,
            detail=f"Could not retrieve the {year} Formula 1 calendar: {error}",
        ) from error

    rows = []
    for _, event in schedule.iterrows():
        event_date = pd.Timestamp(event["EventDate"]).date()
        rows.append(
            {
                "name": str(event["EventName"]),
                "round": int(event["RoundNumber"]),
                "date": event_date.isoformat(),
                "completed": event_date < date.today(),
            }
        )
    return rows


def _ensure_registry() -> None:
    if not registry_is_empty():
        return
    with _registry_sync_lock:
        if registry_is_empty():
            sync_registry()


@app.get("/api/calendar")
def calendar(
    year: int = Query(default=date.today().year, ge=1950, le=2100),
) -> dict:
    try:
        _ensure_registry()
        return list_registry_calendar(year)
    except (
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
        sqlite3.Error,
        requests.RequestException,
    ) as error:
        raise HTTPException(
            status_code=503,
            detail=f"Could not load the race registry: {error}",
        ) from error


@app.post("/api/registry/sync")
def sync_calendar_registry() -> dict:
    try:
        with _registry_sync_lock:
            result = sync_registry()
        return result
    except (
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
        sqlite3.Error,
        requests.RequestException,
    ) as error:
        raise HTTPException(
            status_code=503,
            detail=f"Could not refresh the race registry: {error}",
        ) from error


@app.get("/api/predictions")
def predictions(
    year: int | None = Query(default=None, ge=1950, le=2100),
) -> dict:
    try:
        return {
            "year": year,
            "predictions": list_registry_predictions(year),
        }
    except (OSError, sqlite3.Error, ValueError) as error:
        raise HTTPException(
            status_code=503,
            detail=f"Could not load the prediction ledger: {error}",
        ) from error


@app.get("/api/evaluations")
def evaluations(
    year: int | None = Query(default=None, ge=1950, le=2100),
) -> dict:
    try:
        rows = [
            prediction
            for prediction in list_registry_predictions(year)
            if prediction["metrics"] is not None
        ]
        stop_matches = [
            row["metrics"]["stopCountMatch"]
            for row in rows
            if row["metrics"].get("stopCountMatch") is not None
        ]
        pit_errors = [
            row["metrics"]["pitLapError"]
            for row in rows
            if row["metrics"].get("pitLapError") is not None
        ]
        lap_errors = [
            row["metrics"]["lapTimeMae"]
            for row in rows
            if row["metrics"].get("lapTimeMae") is not None
        ]
        return {
            "year": year,
            "evaluations": rows,
            "summary": {
                "evaluatedRaces": len(rows),
                "stopCountHitRate": (
                    sum(stop_matches) / len(stop_matches) if stop_matches else None
                ),
                "meanPitLapError": (
                    sum(pit_errors) / len(pit_errors) if pit_errors else None
                ),
                "meanLapTimeMae": (
                    sum(lap_errors) / len(lap_errors) if lap_errors else None
                ),
            },
        }
    except (OSError, sqlite3.Error, ValueError) as error:
        raise HTTPException(
            status_code=503,
            detail=f"Could not load race evaluations: {error}",
        ) from error


@app.get("/api/backtests")
def backtests(
    year: int | None = Query(default=None, ge=2018, le=2100),
) -> dict:
    try:
        return {
            "year": year,
            "runs": list_backtest_runs(year),
        }
    except (OSError, sqlite3.Error, ValueError) as error:
        raise HTTPException(
            status_code=503,
            detail=f"Could not load walk-forward backtest results: {error}",
        ) from error


@app.get("/api/forecasts/{race_id}")
def forecast_detail(race_id: str) -> dict:
    target = get_race(race_id)
    if target is None:
        raise HTTPException(status_code=404, detail=f"Race {race_id} is not in the calendar.")
    if target["status"] == "cancelled":
        raise HTTPException(status_code=410, detail="A forecast is unavailable for a cancelled race.")

    now = datetime.now(timezone.utc)
    race_start = datetime.fromisoformat(target["race_start_utc"])
    forecasts = [
        row for row in list_registry_predictions(target["year"])
        if row["raceId"] == race_id
    ]
    has_early = any(row["predictionType"] == "P0" for row in forecasts)
    forecast_error = None
    if not has_early and now < race_start:
        try:
            create_p0_prediction(race_id, now=now)
            forecasts = [
                row for row in list_registry_predictions(target["year"])
                if row["raceId"] == race_id
            ]
        except (ImportError, OSError, RuntimeError, ValueError, KeyError) as error:
            forecast_error = str(error)

    early = max(
        (row for row in forecasts if row["predictionType"] == "P0"),
        key=lambda row: row["version"],
        default=None,
    )
    updated = max(
        (row for row in forecasts if row["predictionType"] == "P1"),
        key=lambda row: row["version"],
        default=None,
    )
    if now >= race_start and now <= race_start + timedelta(hours=5):
        stage = "Live"
    elif now > race_start + timedelta(hours=5):
        stage = "Evaluated" if any(row["metrics"] for row in forecasts) else "Awaiting timing"
    elif updated is not None:
        stage = "Updated"
    elif early is not None:
        stage = "Early"
    else:
        stage = "Awaiting forecast"

    session_schedule = []
    try:
        session_schedule = __import__("json").loads(target.get("session_schedule_json") or "[]")
    except (TypeError, ValueError):
        session_schedule = []
    for session in session_schedule:
        session_start = datetime.fromisoformat(session["startUtc"])
        session["elapsed"] = now >= session_start + timedelta(hours=2)
        session["dataStatus"] = (
            "Awaiting timing" if session["elapsed"] else "Scheduled"
        )

    calendar = list_registry_calendar(target["year"])
    changes = [
        change for change in calendar["changes"]
        if change["raceId"] == race_id
    ]
    circuit_profile = {
        "altitudeMeters": 2238 if "Hermanos" in target["circuit"] or "Mexico" in target["grand_prix"] else None,
        "lapLengthKm": 4.304 if "Hermanos" in target["circuit"] or "Mexico" in target["grand_prix"] else None,
        "tyreWear": "Medium (educational rating)" if "Mexico" in target["grand_prix"] else None,
        "pitLaneLossSeconds": None,
        "typicalStops": None,
        "profileSource": "Published circuit reference; tyre-wear rating is qualitative",
    }
    if "Mexico" in target["grand_prix"]:
        circuit_profile["mexicoContext"] = (
            "Mexico City's high altitude reduces air density, affecting aerodynamic "
            "load and cooling. The current model has no direct altitude feature, so "
            "this is context only and is not represented as a learned adjustment."
        )

    latest = updated or early
    payload = latest["payload"] if latest else None
    data_used = payload.get("assumptions", {}) if payload else {}
    training_count = int(data_used.get("trainingRaceCount", 0))
    confidence = (
        "High" if training_count >= 12
        else "Medium" if training_count >= 5
        else "Low" if training_count
        else "Unavailable"
    )
    freshness = {
        "calendarLastSynced": calendar["lastUpdated"],
        "forecastGeneratedAt": latest["lockedAt"] if latest else None,
        "fastf1CredentialsConfigured": bool(
            os.environ.get("F1_USERNAME") and os.environ.get("F1_PASSWORD")
        ),
        "openf1Historical": "Public historical provider",
    }
    circuit_history = []
    connection = connect_registry()
    try:
        rows = connection.execute(
            """
            SELECT r.race_id, r.year, r.grand_prix, r.date_race,
                   a.actual_result_json, a.data_source
            FROM races r LEFT JOIN actuals a ON a.race_id = r.race_id
            WHERE r.circuit = ? AND r.date_race < ?
            ORDER BY r.date_race DESC LIMIT 5
            """,
            (target["circuit"], target["date_race"]),
        ).fetchall()
        for row in rows:
            result = __import__("json").loads(row["actual_result_json"]) if row["actual_result_json"] else None
            circuit_history.append(
                {
                    "raceId": row["race_id"],
                    "year": row["year"],
                    "grandPrix": row["grand_prix"],
                    "date": row["date_race"],
                    "winner": result.get("winner") if result else None,
                    "winnerStrategy": result.get("winnerStrategy") if result else None,
                    "source": row["data_source"],
                    "actualsAvailable": result is not None,
                }
            )
    finally:
        connection.close()

    return {
        "race": {
            "raceId": race_id,
            "year": target["year"],
            "grandPrix": target["grand_prix"],
            "circuit": target["circuit"],
            "raceStartUtc": target["race_start_utc"],
            "qualifyingStartUtc": target["qualifying_start_utc"],
            "sessions": session_schedule,
            "status": stage,
            "format": target["format"],
        },
        "stage": stage,
        "forecastError": forecast_error,
        "earlyForecast": early,
        "updatedForecast": updated,
        "history": forecasts,
        "confidence": confidence,
        "circuitProfile": circuit_profile,
        "pastEditions": circuit_history,
        "calendarChanges": changes,
        "freshness": freshness,
        "dataUsed": data_used,
        "inputAvailability": {
            "tyreNominations": "Historical assumption; current event nomination feed unavailable",
            "weatherForecast": "Unavailable; using historical circuit averages where present",
            "practiceLongRuns": "Not ingested by the current forecast model",
            "qualifyingGrid": "Not ingested by the current forecast model; P1 is time-gated after qualifying",
        },
        "scenarioNotes": {
            "rain": "No rain-trained model or race-weekend rain forecast is available; no rain pick is claimed.",
        },
    }


@app.post("/api/forecasts/{race_id}/update")
def update_forecast(race_id: str) -> dict:
    try:
        return create_p1_prediction(race_id)
    except (ImportError, OSError, RuntimeError, ValueError, KeyError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/api/forecasts/{race_id}/what-if")
def forecast_what_if(race_id: str, request: ForecastWhatIfRequest) -> dict:
    detail = forecast_detail(race_id)
    forecast = detail["updatedForecast"] or detail["earlyForecast"]
    if forecast is None:
        raise HTTPException(
            status_code=409,
            detail=detail["forecastError"] or "Generate a historical forecast before comparing strategies.",
        )
    model_info = forecast["payload"].get("model", {})
    artifact = model_info.get("artifact")
    if not artifact or Path(artifact).name != artifact:
        raise HTTPException(status_code=409, detail="The forecast model artifact is unavailable.")
    model_path = MODEL_DIR / artifact
    if not model_path.is_file():
        raise HTTPException(status_code=409, detail="The saved forecast model artifact is missing.")
    strategy = Strategy(tuple(request.compounds), tuple(request.pit_laps))
    if (
        len(strategy.compounds) - 1 != len(strategy.pit_laps)
        or len(set(strategy.compounds)) < 2
        or any(compound not in {"SOFT", "MEDIUM", "HARD"} for compound in strategy.compounds)
        or any(lap < 1 or lap >= forecast["payload"]["totalLaps"] for lap in strategy.pit_laps)
        or list(strategy.pit_laps) != sorted(strategy.pit_laps)
    ):
        raise HTTPException(status_code=422, detail="Select a valid dry strategy with at least two compounds.")

    import json

    model = joblib.load(model_path)
    assumptions = forecast["payload"]["assumptions"]
    scenario = StrategyScenario(
        track_temp=max(
            0.0, float(assumptions.get("trackTemp", 35.0) or 35.0)
            - (5.0 if request.scenario == "coolerTrack" else 0.0)
        ),
        air_temp=float(assumptions.get("airTemp", 20.0) or 20.0),
        safety_car_probability=0.65 if request.scenario == "safetyCar" else 0.1,
        model_error=float(forecast["payload"]["confidence"]["validationMaeSeconds"]),
    )
    total_laps = int(forecast["payload"]["totalLaps"])
    pit_loss = float(assumptions["pitLossSeconds"])
    predicted_time, lap_frame = simulate_strategy(
        model, strategy, total_laps, pit_loss, scenario=scenario
    )
    best_strategy = forecast["payload"]["bestStrategy"]
    baseline = Strategy(
        tuple(best_strategy["compounds"]), tuple(best_strategy["pitLaps"])
    )
    baseline_time, baseline_lap_frame = simulate_strategy(
        model,
        baseline,
        total_laps,
        pit_loss,
        scenario=scenario,
    )
    return {
        "scenario": request.scenario,
        "predictedTime": predicted_time,
        "modelPickTime": baseline_time,
        "deltaToModelPick": predicted_time - baseline_time,
        "lapTimes": [
            {"lap": int(row.lap_number), "lapTime": float(row.PredictedLapTime)}
            for row in lap_frame.itertuples()
        ],
        "modelPickLapTimes": [
            {"lap": int(row.lap_number), "lapTime": float(row.PredictedLapTime)}
            for row in baseline_lap_frame.itertuples()
        ],
        "modelPick": best_strategy,
        "strategy": {"compounds": list(strategy.compounds), "pitLaps": list(strategy.pit_laps)},
    }


@app.post("/api/picks/{race_id}")
def submit_user_pick(race_id: str, request: UserPickRequest) -> dict:
    try:
        return save_user_pick(
            request.owner_id,
            race_id,
            {"compounds": request.compounds, "pitLaps": request.pit_laps},
        )
    except (OSError, sqlite3.Error, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/api/picks/scoreboard")
def user_pick_scoreboard(owner_id: str = Query(min_length=8, max_length=80)) -> dict:
    try:
        return list_user_pick_scoreboard(owner_id)
    except (OSError, sqlite3.Error, ValueError) as error:
        raise HTTPException(
            status_code=503, detail=f"Could not load the prediction scoreboard: {error}"
        ) from error


@lru_cache(maxsize=6)
def _load_race_data(
    year: int, race: str
) -> tuple[pd.DataFrame, dict, str | None, str, int, int]:
    loaded_laps = load_race(year, race)
    warning = loaded_laps.attrs.get("load_warning")
    source = loaded_laps.attrs.get("source", "Unknown")
    total_laps = int(loaded_laps.attrs.get("totalLaps", loaded_laps["LapNumber"].max()))
    safety_car_episodes = int(loaded_laps.attrs.get("safetyCarEpisodes", 0))
    laps = add_features(clean_laps(loaded_laps))
    models = train_models(laps)
    return laps, models, warning, source, total_laps, safety_car_episodes


def get_race_data(
    year: int, race: str
) -> tuple[pd.DataFrame, dict, str | None, str, int, int]:
    with _race_data_lock:
        return _load_race_data(year, race)


class StrategyRequest(BaseModel):
    year: int = Field(default_factory=lambda: date.today().year, ge=2018, le=2100)
    race: str = Field(min_length=1, max_length=80)
    total_laps: int = Field(default=57, ge=20, le=100)
    pit_loss: float = Field(default=22, ge=0, le=60)
    stops: int = Field(default=1, ge=1, le=2)
    track_temp: float = Field(default=38, ge=0, le=60)
    air_temp: float = Field(default=27, ge=-10, le=50)
    weather_change: float = Field(default=0, ge=-20, le=20)
    traffic_intensity: float = Field(default=0.15, ge=0, le=1)
    overtake_rate: float = Field(default=0.25, ge=0, le=1)
    safety_car_probability: float = Field(default=0.2, ge=0, le=1)
    safety_car_duration: int = Field(default=4, ge=1, le=20)
    safety_car_pit_saving: float = Field(default=8, ge=0, le=30)
    safety_car_lap_delta: float = Field(default=28, ge=0, le=60)
    available_sets: dict[str, int] = Field(
        default_factory=lambda: {"SOFT": 2, "MEDIUM": 3, "HARD": 2}
    )
    team: str = Field(default="Average", min_length=1, max_length=80)


class UserPickRequest(BaseModel):
    owner_id: str = Field(min_length=8, max_length=80)
    compounds: list[str] = Field(min_length=2, max_length=3)
    pit_laps: list[int] = Field(max_length=2)


class ForecastWhatIfRequest(BaseModel):
    compounds: list[str] = Field(min_length=1, max_length=3)
    pit_laps: list[int] = Field(max_length=2)
    scenario: str = Field(default="baseline", pattern="^(baseline|safetyCar|coolerTrack)$")


@app.get("/api/health")
def health() -> dict[str, str | int]:
    return {"status": "ok", "season": date.today().year}


@app.get("/api/events")
def events(
    year: int = Query(default=date.today().year, ge=2018, le=2100),
    refresh: bool = False,
) -> dict:
    rows = _schedule_rows(year, refresh)
    completed = [event for event in rows if event["completed"]]
    return {
        "year": year,
        "currentYear": date.today().year,
        "events": rows,
        "defaultEvent": (completed or rows)[-1]["name"] if rows else None,
        "years": list(range(date.today().year, 2017, -1)),
    }


@app.get("/api/race")
def race_data(
    year: int = Query(default=date.today().year, ge=2018, le=2100),
    race: str = Query(min_length=1, max_length=80),
    refresh: bool = False,
) -> dict:
    try:
        if refresh:
            _load_race_data.cache_clear()
        laps, models, warning, source, total_laps, safety_car_episodes = get_race_data(year, race)
        best_model_name, best_model = min(
            models.items(),
            key=lambda item: item[1].metrics["MAE"],
        )
        teams = sorted(laps["team"].unique().tolist())
        default_team = str(laps["team"].mode().iloc[0])
        traffic_intensity = float(laps["traffic_pressure"].mean())
        overtake_rate = float(
            laps["overtakes"].sum() / max(laps["traffic_pressure"].sum() * 2, 1)
        )
        degradation = (
            laps.groupby(["Compound", "tyre_age"], as_index=False)["LapTime"]
            .mean()
            .rename(columns={"tyre_age": "tyreAge", "LapTime": "lapTime"})
        )
        lap_columns = {
            "Driver": "driver",
            "LapNumber": "lapNumber",
            "LapTime": "lapTime",
            "Compound": "compound",
            "TyreLife": "tyreLife",
            "TrackStatus": "trackStatus",
            "TrafficGapSeconds": "trafficGapSeconds",
            "Position": "position",
            "PositionChange": "positionChange",
        }
        lap_records = laps[
            [column for column in lap_columns if column in laps]
        ].rename(columns=lap_columns)
        lap_records = lap_records.astype(object).where(pd.notna(lap_records), None)
        response = {
            "year": year,
            "race": race,
            "source": source,
            "totalLaps": total_laps,
            "lapCount": len(laps),
            "driverCount": int(laps["Driver"].nunique()),
            "drivers": sorted(laps["Driver"].unique().tolist()),
            "teams": teams,
            "defaultTeam": default_team,
            "loadWarning": warning,
            "scenarioDefaults": {
                "trackTemp": float(laps["TrackTemp"].median()),
                "airTemp": float(laps["AirTemp"].median()),
                "trafficIntensity": traffic_intensity,
                "overtakeRate": min(1.0, max(0.0, overtake_rate)),
                "safetyCarProbability": min(
                    0.8, 0.1 + safety_car_episodes * 0.15
                ),
                "modelError": best_model.metrics["MAE"],
            },
            "metrics": {name: bundle.metrics for name, bundle in models.items()},
            "bestModel": best_model_name,
            "bestModelMae": best_model.metrics["MAE"],
            "laps": lap_records.to_dict(orient="records"),
            "degradation": degradation.to_dict(orient="records"),
        }
        defaults = response["scenarioDefaults"]
        default_scenario = StrategyScenario(
            team=default_team,
            track_temp=defaults["trackTemp"],
            air_temp=defaults["airTemp"],
            traffic_intensity=defaults["trafficIntensity"],
            overtake_rate=defaults["overtakeRate"],
            safety_car_probability=defaults["safetyCarProbability"],
            safety_car_lap_delta=28,
            model_error=best_model.metrics["MAE"],
        )
        ranked = rank_strategies(
            best_model.model,
            total_laps,
            22,
            scenario=default_scenario,
        )
        response["strategies"] = ranked.head(50)[
            [
                "Strategy",
                "Total time (s)",
                "Delta (s)",
                "P10 (s)",
                "P90 (s)",
                "Win probability",
            ]
        ].rename(
            columns={
                "Strategy": "strategy",
                "Total time (s)": "totalTime",
                "Delta (s)": "delta",
                "P10 (s)": "p10",
                "P90 (s)": "p90",
                "Win probability": "winProbability",
            }
        ).to_dict(orient="records")
        return response
    except RealDataUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except (ValueError, KeyError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/api/strategies")
def strategies(request: StrategyRequest) -> dict:
    try:
        _, models, warning, source, _, _ = get_race_data(request.year, request.race)
        model = min(
            models.values(),
            key=lambda bundle: bundle.metrics["MAE"],
        )
        scenario = StrategyScenario(
            team=request.team,
            track_temp=request.track_temp,
            air_temp=request.air_temp,
            weather_change=request.weather_change,
            traffic_intensity=request.traffic_intensity,
            overtake_rate=request.overtake_rate,
            safety_car_probability=request.safety_car_probability,
            safety_car_duration=request.safety_car_duration,
            safety_car_pit_saving=request.safety_car_pit_saving,
            safety_car_lap_delta=request.safety_car_lap_delta,
            available_sets=request.available_sets,
            model_error=model.metrics["MAE"],
        )
        ranked = rank_strategies(
            model.model,
            request.total_laps,
            request.pit_loss,
            request.stops,
            scenario=scenario,
        )
        return {
            "source": source,
            "loadWarning": warning,
            "strategies": ranked.head(50)[
                [
                    "Strategy",
                    "Total time (s)",
                    "Delta (s)",
                    "P10 (s)",
                    "P90 (s)",
                    "Win probability",
                ]
            ].rename(
                columns={
                    "Strategy": "strategy",
                    "Total time (s)": "totalTime",
                    "Delta (s)": "delta",
                    "P10 (s)": "p10",
                    "P90 (s)": "p90",
                    "Win probability": "winProbability",
                }
            ).to_dict(orient="records"),
        }
    except RealDataUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except (ValueError, KeyError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
