"""Reproducible race prediction, ingestion, evaluation, and backtest commands."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sqlite3
import subprocess
import sys
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.cleaning import clean_laps
from src.data_loader import REAL_DATA_SOURCES, load_race
from src.evaluate import chronological_backtest, score_prediction
from src.features import add_features
from src.race_registry import (
    connect_registry,
    database_path,
    get_race,
    lock_prediction,
    races_before,
    save_actuals,
    save_race_context,
    save_evaluation,
    sync_registry,
)
from src.simulate import StrategyScenario, rank_strategies, simulate_strategy
from src.train import train_walk_forward_models

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STORAGE_DIR = Path(os.environ.get("RACE_STORAGE_DIR", PROJECT_ROOT / "data"))
FEATURE_CACHE = Path(
    os.environ.get("RACE_FEATURE_CACHE_DIR", STORAGE_DIR / "processed" / "laps")
)
MODEL_DIR = Path(os.environ.get("RACE_MODEL_DIR", PROJECT_ROOT / "models"))
PIT_LOSS_ASSUMPTION = 22.0
F1_SOURCE_FILES = (
    "src/cleaning.py",
    "src/data_loader.py",
    "src/openf1_loader.py",
    "src/features.py",
    "src/simulate.py",
    "src/train.py",
    "src/pipeline.py",
)


def _utc(value: datetime | None = None) -> datetime:
    current = value or datetime.now(timezone.utc)
    if current.tzinfo is None:
        return current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc)


def _project_git_commit() -> str | None:
    try:
        root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if Path(root).resolve() != PROJECT_ROOT.resolve():
            return None
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _source_hash() -> str:
    digest = hashlib.sha256()
    for relative_path in F1_SOURCE_FILES:
        path = PROJECT_ROOT / relative_path
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _race_context_for_circuit(
    circuit: str,
    before_date: str,
    path: Path | str | None,
) -> dict | None:
    connection = connect_registry(path)
    try:
        row = connection.execute(
            """
            SELECT c.*, r.date_race
            FROM race_context c
            JOIN races r ON r.race_id = c.race_id
            WHERE r.circuit = ? AND r.date_race < ?
              AND c.source IN ('FastF1', 'OpenF1')
            ORDER BY r.date_race DESC, r.round DESC
            LIMIT 1
            """,
            (circuit, before_date),
        ).fetchone()
        return dict(row) if row is not None else None
    finally:
        connection.close()


def _load_training_features(
    target: dict,
    path: Path | str | None,
) -> tuple[pd.DataFrame, list[str]]:
    historical_races = races_before(target["date_race"], "T3", path)
    if not historical_races:
        raise ValueError("No earlier T3 races exist in the registry for model training.")

    FEATURE_CACHE.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    skipped: list[str] = []
    for race in historical_races:
        cache_file = FEATURE_CACHE / f"{race['race_id']}.parquet"
        if cache_file.exists():
            features = pd.read_parquet(cache_file)
            if (
                "data_source" not in features
                or not features["data_source"].isin(REAL_DATA_SOURCES).all()
            ):
                skipped.append(race["race_id"])
                continue
            if (features["race_date"] >= target["date_race"]).any():
                raise ValueError(
                    f"Cached training data for {race['race_id']} is not earlier than the target race."
                )
            frames.append(features)
            continue

        laps = load_race(race["year"], race["grand_prix"])
        source = laps.attrs.get("source")
        if source not in REAL_DATA_SOURCES:
            skipped.append(race["race_id"])
            continue
        total_laps = int(laps.attrs.get("totalLaps", laps["LapNumber"].max()))
        cleaned = clean_laps(laps)
        if cleaned.empty:
            skipped.append(race["race_id"])
            continue
        features = add_features(cleaned, total_laps)
        features["race_id"] = race["race_id"]
        features["race_date"] = race["date_race"]
        features["data_source"] = source
        features.to_parquet(cache_file, index=False)
        save_race_context(
            race["race_id"],
            total_laps,
            float(cleaned["TrackTemp"].median()),
            float(cleaned["AirTemp"].median()),
            None,
            source,
            path,
        )
        frames.append(features)

    if not frames:
        detail = (
            f" Could not load usable real FastF1/OpenF1 data for: {', '.join(skipped[:8])}."
            if skipped
            else ""
        )
        raise ValueError(
            "No real historical FastF1 or OpenF1 laps are cached or available. "
            "OpenF1 provides public historical timing from 2023 onward; F1 credentials "
            f"are optional and enable FastF1.{detail}"
        )
    training = pd.concat(frames, ignore_index=True)
    race_dates = training.groupby("race_id")["race_date"].min()
    training = training[training["race_date"] < target["date_race"]]
    if training["race_id"].nunique() < 2:
        detail = f" Skipped races: {', '.join(skipped[:8])}." if skipped else ""
        raise ValueError(
            "At least two earlier real races are required for walk-forward training."
            f"{detail}"
        )
    if (training["race_date"] >= target["date_race"]).any():
        raise ValueError("Training data includes the target race or a later race.")
    if race_dates.empty:
        raise ValueError("The training cutoff could not be established.")
    return training, skipped


def _model_hash(model: object, artifact_path: Path | None = None) -> str:
    if artifact_path is not None:
        return hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    return hashlib.sha256(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()


def _strategy_json(row: pd.Series) -> dict:
    strategy = row["strategy"]
    return {
        "compounds": list(strategy.compounds),
        "pitLaps": list(strategy.pit_laps),
        "totalTime": float(row["Total time (s)"]),
        "p10": float(row["P10 (s)"]),
        "p90": float(row["P90 (s)"]),
    }


def build_prediction(
    race_id: str,
    path: Path | str | None = None,
    now: datetime | None = None,
    save_model: bool = False,
    prediction_type: str = "P1",
) -> dict:
    target = get_race(race_id, path)
    if target is None:
        raise ValueError(f"Race {race_id} is not present in the registry.")
    if target["status"] == "cancelled":
        raise ValueError("Predictions are not generated for cancelled races.")
    current = _utc(now)
    race_start = datetime.fromisoformat(target["race_start_utc"])
    if prediction_type not in {"P0", "P1"}:
        raise ValueError("Forecasts must use the P0 early or P1 updated stage.")
    if current >= race_start:
        raise ValueError(f"{prediction_type} forecasts must be generated before race start.")
    if prediction_type == "P1":
        qualifying_start = target["qualifying_start_utc"]
        if not qualifying_start:
            raise ValueError("The calendar has no qualifying start time; a P1 prediction cannot be timed.")
        qualifying_deadline = datetime.fromisoformat(qualifying_start) + timedelta(hours=2)
        if current < qualifying_deadline:
            raise ValueError("P1 predictions are only generated 60 minutes after qualifying ends.")

    training, skipped = _load_training_features(target, path)
    bundles = train_walk_forward_models(training)
    model_name, bundle = min(
        bundles.items(),
        key=lambda item: item[1].metrics["MAE"],
    )
    circuit_context = _race_context_for_circuit(
        target["circuit"], target["date_race"], path
    )
    if circuit_context is None:
        raise ValueError(
            "No earlier real-data event at this circuit is available to estimate race distance and conditions."
        )
    total_laps = int(circuit_context["total_laps"])
    if total_laps < 20:
        raise ValueError("Historical race-distance estimate is invalid.")

    team = str(training["team"].mode().iloc[0])
    track_temp = circuit_context["median_track_temp"]
    air_temp = circuit_context["median_air_temp"]
    if pd.isna(track_temp):
        track_temp = float(training["track_temp"].median())
    if pd.isna(air_temp):
        air_temp = float(training["air_temp"].median())
    traffic = float(np.clip(training["traffic_pressure"].mean(), 0, 1))
    overtakes = float(
        np.clip(
            training["overtakes"].sum() / max(training["traffic_pressure"].sum() * 2, 1),
            0,
            1,
        )
    )
    scenario = StrategyScenario(
        team=team,
        track_temp=float(track_temp),
        air_temp=float(air_temp),
        traffic_intensity=traffic,
        overtake_rate=overtakes,
        safety_car_probability=0.1,
        model_error=max(0.1, bundle.metrics["MAE"]),
    )
    pit_loss = float(circuit_context["pit_loss_estimate"] or PIT_LOSS_ASSUMPTION)
    rankings = pd.concat(
        [
            rank_strategies(bundle.model, total_laps, pit_loss, stops=stops, scenario=scenario)
            for stops in (1, 2)
        ],
        ignore_index=True,
    ).sort_values("Total time (s)", ignore_index=True)
    if rankings.empty:
        raise ValueError("No valid 1-stop or 2-stop strategies were generated.")

    scale = max(8.0, bundle.metrics["MAE"] * np.sqrt(total_laps))
    best_by_stop = rankings.groupby(
        rankings["strategy"].map(lambda strategy: len(strategy.pit_laps))
    )["Total time (s)"].min()
    stop_weights = np.exp(-(best_by_stop - best_by_stop.min()) / scale)
    stop_probabilities = (stop_weights / stop_weights.sum()).to_dict()
    top_strategies = [_strategy_json(row) for _, row in rankings.head(50).iterrows()]
    for strategy in top_strategies:
        strategy["stops"] = len(strategy["pitLaps"])
        strategy["confidence"] = (
            "High" if len(training["race_id"].unique()) >= 12
            else "Medium" if len(training["race_id"].unique()) >= 5
            else "Low"
        )
    pit_candidates = [
        strategy
        for strategy in top_strategies
        if strategy["compounds"] == list(rankings.iloc[0]["strategy"].compounds)
    ]
    best = rankings.iloc[0]
    best_strategy = best["strategy"]
    _, predicted_laps = simulate_strategy(
        bundle.model,
        best_strategy,
        total_laps,
        pit_loss,
        scenario=scenario,
    )
    predicted_lap_times = [
        {"lap": int(row.lap_number), "lapTime": float(row.PredictedLapTime)}
        for row in predicted_laps.itertuples()
    ]
    latest_training_date = str(training["race_date"].max())
    pit_windows = [strategy["pitLaps"] for strategy in pit_candidates if strategy["pitLaps"]]
    pit_window = {
        "earliest": min(laps[0] for laps in pit_windows),
        "latest": max(laps[0] for laps in pit_windows),
    } if any(pit_windows) else {
        "earliest": best_strategy.pit_laps[0] if best_strategy.pit_laps else None,
        "latest": best_strategy.pit_laps[0] if best_strategy.pit_laps else None,
    }

    code_hash = _source_hash()
    artifact_path = None
    if save_model:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        stable_model_hash = _model_hash(bundle.model)
        artifact_path = MODEL_DIR / f"{race_id}-{stable_model_hash[:16]}.joblib"
        joblib.dump(bundle.model, artifact_path)
    model_hash = _model_hash(bundle.model, artifact_path)
    git_commit = _project_git_commit()
    model_version = (
        f"commit:{git_commit or 'uncommitted'};source:{code_hash[:12]};model:{model_hash[:12]}"
    )
    scenarios = {}
    for name, changes in {
        "safetyCar": {"safety_car_probability": 0.65},
        "coolerTrack": {"track_temp": max(0.0, float(track_temp) - 5.0)},
    }.items():
        scenario_rankings = pd.concat(
            [
                rank_strategies(
                    bundle.model,
                    total_laps,
                    pit_loss,
                    stops=stops,
                    scenario=StrategyScenario(
                        team=team,
                        track_temp=float(changes.get("track_temp", track_temp)),
                        air_temp=float(air_temp),
                        traffic_intensity=traffic,
                        overtake_rate=overtakes,
                        model_error=max(0.1, bundle.metrics["MAE"]),
                        safety_car_probability=float(
                            changes.get("safety_car_probability", 0.1)
                        ),
                    ),
                )
                for stops in (1, 2)
            ],
            ignore_index=True,
        ).sort_values("Total time (s)", ignore_index=True)
        scenarios[name] = {
            "bestStrategy": _strategy_json(scenario_rankings.iloc[0]),
            "note": (
                "Scenario stress test only; no session-specific incident or "
                "weather observation is included."
            ),
        }
    scenarios["rain"] = {
        "bestStrategy": None,
        "note": "Rain is not a trained model input; no rain-specific pick is claimed.",
    }

    prediction = {
        "raceId": race_id,
        "race": target["grand_prix"],
        "circuit": target["circuit"],
        "predictionType": prediction_type,
        "stage": "Early" if prediction_type == "P0" else "Updated",
        "generatedAt": current.isoformat(),
        "bestStrategy": {
            "compounds": list(best_strategy.compounds),
            "pitLaps": list(best_strategy.pit_laps),
        },
        "predictedTotalTime": float(best["Total time (s)"]),
        "topStrategies": top_strategies[:5],
        "strategyRanking": top_strategies,
        "stopCountProbabilities": {
            str(stops): float(probability)
            for stops, probability in stop_probabilities.items()
        },
        "scenarios": scenarios,
        "pitWindow": pit_window,
        "predictedStops": len(best_strategy.pit_laps),
        "predictedCompoundOrder": list(best_strategy.compounds),
        "predictedLapTimes": predicted_lap_times,
        "totalLaps": total_laps,
        "assumptions": {
            "dataSource": " + ".join(sorted(training["data_source"].unique())),
            "trainingDataCutoff": latest_training_date,
            "pitLossSeconds": pit_loss,
            "pitLossSource": "historical circuit context" if circuit_context["pit_loss_estimate"] else "22-second educational assumption",
            "trackTemp": float(track_temp),
            "airTemp": float(air_temp),
            "weather": "historical circuit median; no race-day forecast supplied",
            "safetyCarProbability": scenario.safety_car_probability,
            "trafficExposure": traffic,
            "overtakeRate": overtakes,
            "tyreSets": {"SOFT": 2, "MEDIUM": 3, "HARD": 2},
            "skippedHistoricalRaces": skipped,
            "trainingRaceCount": int(training["race_id"].nunique()),
            "trainingSources": sorted(training["data_source"].unique().tolist()),
            "inputFallbacks": [
                name
                for name, missing in (
                    ("same-circuit historical temperature", pd.isna(circuit_context["median_track_temp"])),
                    ("same-circuit historical pit loss", not circuit_context["pit_loss_estimate"]),
                )
                if missing
            ],
            "targetRaceExcluded": bool((training["race_id"] == race_id).any()),
        },
        "confidence": {
            "label": (
                "High" if training["race_id"].nunique() >= 12
                else "Medium" if training["race_id"].nunique() >= 5
                else "Low"
            ),
            "gapToSecondSeconds": (
                float(rankings.iloc[1]["Total time (s)"] - best["Total time (s)"])
                if len(rankings) > 1
                else None
            ),
            "p10Seconds": float(best["P10 (s)"]),
            "p90Seconds": float(best["P90 (s)"]),
            "validationMaeSeconds": float(bundle.metrics["MAE"]),
            "validationRaceCount": int(training["race_id"].nunique()),
        },
        "model": {
            "name": model_name,
            "version": model_version,
            "hash": model_hash,
            "sourceHash": code_hash,
            "gitCommit": git_commit,
            "artifact": artifact_path.name if artifact_path else None,
        },
    }
    return {
        "payload": prediction,
        "modelVersion": model_version,
        "modelHash": model_hash,
        "trainingDataCutoff": latest_training_date,
    }


def create_p0_prediction(
    race_id: str,
    path: Path | str | None = None,
    now: datetime | None = None,
) -> dict:
    current = _utc(now)
    connection = connect_registry(path)
    try:
        existing = connection.execute(
            """
            SELECT prediction_id FROM predictions
            WHERE race_id = ? AND prediction_type = 'P0'
            ORDER BY version DESC LIMIT 1
            """,
            (race_id,),
        ).fetchone()
    finally:
        connection.close()
    if existing is not None:
        raise ValueError(
            f"An early P0 forecast already exists for {race_id}; revisions must be explicit."
        )
    built = build_prediction(
        race_id,
        path=path,
        now=current,
        save_model=True,
        prediction_type="P0",
    )
    return lock_prediction(
        race_id,
        built["payload"],
        built["modelVersion"],
        built["trainingDataCutoff"],
        built["modelHash"],
        prediction_type="P0",
        path=path,
        now=current,
    )


def create_p1_prediction(
    race_id: str,
    path: Path | str | None = None,
    now: datetime | None = None,
) -> dict:
    current = _utc(now)
    connection = connect_registry(path)
    try:
        existing = connection.execute(
            """
            SELECT prediction_id FROM predictions
            WHERE race_id = ? AND prediction_type = 'P1'
            ORDER BY version DESC LIMIT 1
            """,
            (race_id,),
        ).fetchone()
    finally:
        connection.close()
    if existing is not None:
        raise ValueError(
            f"A P1 prediction already exists for {race_id}; corrections must be explicit new versions."
        )
    built = build_prediction(race_id, path=path, now=current, save_model=True)
    return lock_prediction(
        race_id,
        built["payload"],
        built["modelVersion"],
        built["trainingDataCutoff"],
        built["modelHash"],
        path=path,
        now=current,
    )


def _official_lap_time(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timedelta):
        return float(value.total_seconds())
    return float(value)


def ingest_actuals(
    race_id: str,
    path: Path | str | None = None,
    now: datetime | None = None,
) -> dict:
    if not os.environ.get("F1_USERNAME") or not os.environ.get("F1_PASSWORD"):
        raise ValueError("Official actuals require F1_USERNAME and F1_PASSWORD in the pipeline environment.")
    race = get_race(race_id, path)
    if race is None:
        raise ValueError(f"Race {race_id} is not present in the registry.")
    if race["status"] == "cancelled":
        raise ValueError("Actual race data is not ingested for cancelled events.")

    import fastf1

    fastf1.Cache.configure(cache_dir=str(PROJECT_ROOT / "data" / "raw"))
    session = fastf1.get_session(race["year"], race["grand_prix"], "R")
    session.load(laps=True, telemetry=False, weather=True, messages=True)
    laps = session.laps.copy()
    results = session.results.copy()
    if laps.empty or results.empty:
        raise ValueError(f"FastF1 has not published complete race data for {race['grand_prix']}.")

    if "Position" not in results:
        raise ValueError("FastF1 race results do not contain finishing positions.")
    positions = pd.to_numeric(results["Position"], errors="coerce")
    finishers = results.loc[positions.notna()].copy()
    finishers["Position"] = positions.loc[positions.notna()]
    winner = finishers.sort_values("Position").iloc[0]
    driver = str(winner.get("Abbreviation") or winner.get("DriverNumber") or "")
    if not driver:
        raise ValueError("FastF1 race results do not identify the winning driver.")
    winner_laps = laps.pick_driver(driver).copy()
    if winner_laps.empty:
        raise ValueError("FastF1 returned no laps for the classified race winner.")

    stints = winner_laps.dropna(subset=["Stint"]).sort_values("LapNumber").groupby("Stint")
    compound_order = [
        str(stint["Compound"].dropna().iloc[0]).upper()
        for _, stint in stints
        if not stint["Compound"].dropna().empty
    ]
    pit_laps = []
    if "PitInTime" in winner_laps:
        pit_laps = [
            int(lap)
            for lap in winner_laps.loc[winner_laps["PitInTime"].notna(), "LapNumber"]
        ]
    if not pit_laps and stints.ngroups > 1:
        pit_laps = [
            int(stint["LapNumber"].max())
            for _, stint in list(stints)[:-1]
        ]

    classified_time = winner.get("Time")
    actual_total = _official_lap_time(classified_time)
    if actual_total is None:
        actual_total = float(winner_laps["LapTime"].dropna().dt.total_seconds().sum())
    lap_status = winner_laps.get("TrackStatus", pd.Series("1", index=winner_laps.index)).astype(str)
    pit_in = winner_laps.get("PitInTime", pd.Series(index=winner_laps.index, dtype=object)).notna()
    pit_out = winner_laps.get("PitOutTime", pd.Series(index=winner_laps.index, dtype=object)).notna()
    comparable = winner_laps.loc[lap_status.isin(["1", "2"]) & ~pit_in & ~pit_out]
    actual_lap_times = [
        {"lap": int(row.LapNumber), "lapTime": float(row.LapTime.total_seconds())}
        for row in comparable.itertuples()
        if pd.notna(row.LapTime)
    ]
    cause_tags = []
    if lap_status.isin(["4", "6", "7"]).any():
        cause_tags.append("safety_car")
    weather = session.weather_data
    if weather is not None and not weather.empty and "Rainfall" in weather:
        if weather["Rainfall"].fillna(False).astype(bool).any():
            cause_tags.append("weather")

    total_laps = int(session.total_laps or laps["LapNumber"].max())
    actual_result = {
        "winner": driver,
        "winnerStrategy": {
            "compoundOrder": compound_order,
            "pitLaps": pit_laps,
        },
        "actualTotalTime": actual_total,
        "totalLaps": total_laps,
        "classifiedDrivers": int(len(finishers)),
        "causeTags": cause_tags,
    }
    official_time = _utc(now)
    save_actuals(
        race_id,
        actual_result,
        "FastF1",
        winner_strategy=actual_result["winnerStrategy"],
        actual_lap_times=actual_lap_times,
        path=path,
        ingested_at=official_time,
    )
    track_temp = laps["TrackTemp"].median() if "TrackTemp" in laps else None
    air_temp = laps["AirTemp"].median() if "AirTemp" in laps else None
    save_race_context(
        race_id,
        total_laps,
        float(track_temp) if pd.notna(track_temp) else None,
        float(air_temp) if pd.notna(air_temp) else None,
        None,
        "FastF1",
        path,
        updated_at=official_time,
    )
    return actual_result


def evaluate_race(race_id: str, path: Path | str | None = None) -> dict:
    connection = connect_registry(path)
    try:
        prediction_row = connection.execute(
            """
            SELECT prediction_id, version, locked_at, payload_json FROM predictions
            WHERE race_id = ? AND prediction_type = 'P1'
            ORDER BY version DESC LIMIT 1
            """,
            (race_id,),
        ).fetchone()
        actual_row = connection.execute(
            """
            SELECT actual_result_json, actual_lap_times_json
            FROM actuals WHERE race_id = ?
            """,
            (race_id,),
        ).fetchone()
    finally:
        connection.close()
    if prediction_row is None:
        raise ValueError(f"No locked P1 prediction exists for {race_id}.")
    if actual_row is None:
        raise ValueError(f"No official actuals have been ingested for {race_id}.")
    actual = json.loads(actual_row["actual_result_json"])
    actual["actualLapTimes"] = (
        json.loads(actual_row["actual_lap_times_json"])
        if actual_row["actual_lap_times_json"]
        else None
    )
    evaluation = score_prediction(
        json.loads(prediction_row["payload_json"]),
        actual,
    )
    save_evaluation(
        prediction_row["prediction_id"],
        evaluation["metrics"],
        evaluation["causeTag"],
        path,
    )
    result = {
        "predictionId": prediction_row["prediction_id"],
        "raceId": race_id,
        **evaluation,
    }
    if path is None:
        report_root = PROJECT_ROOT / "predictions" / "reports"
    else:
        report_root = Path(path).parent / "predictions" / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    prediction = json.loads(prediction_row["payload_json"])
    report = {
        **result,
        "race": prediction.get("race"),
        "modelVersion": prediction.get("model", {}).get("version"),
        "actualsSource": "FastF1",
        "evaluatedAt": _utc().isoformat(),
        "prediction": {
            "lockedAt": prediction_row["locked_at"],
            "bestStrategy": prediction.get("bestStrategy"),
            "predictedTotalTime": prediction.get("predictedTotalTime"),
        },
        "actual": actual,
    }
    report_path = report_root / f"{race_id}-P1-v{prediction_row['version']}.json"
    if report_path.exists():
        existing_report = json.loads(report_path.read_text(encoding="utf-8"))
        if existing_report.get("metrics") != report["metrics"]:
            raise ValueError(
                f"An existing scorecard differs at {report_path}; do not overwrite a published report."
            )
    else:
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    result["reportPath"] = str(report_path)
    return result


def predict_due_races(path: Path | str | None = None, now: datetime | None = None) -> list[dict]:
    current = _utc(now)
    connection = connect_registry(path)
    try:
        races = connection.execute(
            """
            SELECT r.race_id, r.race_start_utc, r.qualifying_start_utc
            FROM races r
            WHERE r.status != 'cancelled'
              AND r.race_start_utc > ?
              AND r.qualifying_start_utc IS NOT NULL
              AND datetime(r.qualifying_start_utc, '+2 hours') <= datetime(?)
              AND NOT EXISTS (
                  SELECT 1 FROM predictions p
                  WHERE p.race_id = r.race_id AND p.prediction_type = 'P1'
              )
            ORDER BY r.race_start_utc
            """,
            (current.isoformat(), current.isoformat()),
        ).fetchall()
    finally:
        connection.close()
    return [
        create_p1_prediction(row["race_id"], path=path, now=current)
        for row in races
    ]


def ingest_due_races(path: Path | str | None = None, now: datetime | None = None) -> list[dict]:
    current = _utc(now)
    connection = connect_registry(path)
    try:
        rows = connection.execute(
            """
            SELECT r.race_id FROM races r
            WHERE r.status != 'cancelled'
              AND datetime(r.race_start_utc, '+5 hours') <= datetime(?)
              AND NOT EXISTS (SELECT 1 FROM actuals a WHERE a.race_id = r.race_id)
            ORDER BY r.race_start_utc
            """,
            (current.isoformat(),),
        ).fetchall()
    finally:
        connection.close()
    return [
        {"raceId": row["race_id"], "actuals": ingest_actuals(row["race_id"], path, current)}
        for row in rows
    ]


def evaluate_due_races(path: Path | str | None = None) -> list[dict]:
    connection = connect_registry(path)
    try:
        rows = connection.execute(
            """
            SELECT DISTINCT p.race_id FROM predictions p
            JOIN actuals a ON a.race_id = p.race_id
            LEFT JOIN evaluations e ON e.prediction_id = p.prediction_id
            WHERE p.prediction_type = 'P1' AND e.prediction_id IS NULL
            ORDER BY p.race_id
            """
        ).fetchall()
    finally:
        connection.close()
    return [evaluate_race(row["race_id"], path) for row in rows]


def run_walk_forward_backtest(
    start_year: int = 2018,
    end_year: int | None = None,
    path: Path | str | None = None,
    now: datetime | None = None,
) -> dict:
    current = _utc(now)
    final_year = end_year or current.year
    connection = connect_registry(path)
    try:
        races = [
            dict(row)
            for row in connection.execute(
                """
                SELECT * FROM races
                WHERE year BETWEEN ? AND ? AND data_tier = 'T3'
                  AND status != 'cancelled' AND date_race < ?
                ORDER BY date_race, year, round
                """,
                (start_year, final_year, current.date().isoformat()),
            ).fetchall()
        ]
    finally:
        connection.close()
    if not races:
        raise ValueError("No completed T3 races are available for the requested backtest range.")

    def make_prediction(race: dict, _completed: list[dict]) -> dict:
        race_start = datetime.fromisoformat(race["race_start_utc"])
        qualifying = race.get("qualifying_start_utc")
        if not qualifying:
            raise ValueError(f"{race['race_id']} has no qualifying timestamp for a pre-race backtest.")
        pseudo_lock = min(
            datetime.fromisoformat(qualifying) + timedelta(hours=2),
            race_start - timedelta(seconds=1),
        )
        built = build_prediction(race["race_id"], path=path, now=pseudo_lock)
        return built["payload"]

    def load_actual(race: dict) -> dict | None:
        connection = connect_registry(path)
        try:
            cached = connection.execute(
                """
                SELECT actual_result_json, actual_lap_times_json
                FROM actuals WHERE race_id = ?
                """,
                (race["race_id"],),
            ).fetchone()
        finally:
            connection.close()
        if cached is None:
            ingest_actuals(race["race_id"], path=path, now=current)
            connection = connect_registry(path)
            try:
                cached = connection.execute(
                    """
                    SELECT actual_result_json, actual_lap_times_json
                    FROM actuals WHERE race_id = ?
                    """,
                    (race["race_id"],),
                ).fetchone()
            finally:
                connection.close()
        if cached is None:
            return None
        actual = json.loads(cached["actual_result_json"])
        actual["actualLapTimes"] = (
            json.loads(cached["actual_lap_times_json"])
            if cached["actual_lap_times_json"]
            else None
        )
        return actual

    reports = chronological_backtest(
        races,
        make_prediction,
        load_actual,
        minimum_prior_races=2,
    )
    run_id = str(uuid.uuid4())
    connection = connect_registry(path)
    try:
        with connection:
            for report in reports:
                connection.execute(
                    """
                    INSERT INTO backtests(
                        run_id, race_id, prediction_json, evaluation_json, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        report["raceId"],
                        json.dumps(report["prediction"], sort_keys=True, allow_nan=False),
                        json.dumps(report["evaluation"], sort_keys=True, allow_nan=False),
                        current.isoformat(),
                    ),
                )
    finally:
        connection.close()

    stop_matches = [
        row["evaluation"]["metrics"]["stopCountMatch"]
        for row in reports
        if row["evaluation"]["metrics"]["stopCountMatch"] is not None
    ]
    return {
        "runId": run_id,
        "startYear": start_year,
        "endYear": final_year,
        "evaluatedRaces": len(reports),
        "stopCountHitRate": sum(stop_matches) / len(stop_matches) if stop_matches else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="F1 race registry and prediction pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("sync-calendar")
    predict = commands.add_parser("predict")
    predict.add_argument("--race-id", required=True)
    commands.add_parser("predict-due")
    commands.add_parser("ingest-due")
    commands.add_parser("evaluate-due")
    ingest = commands.add_parser("ingest")
    ingest.add_argument("--race-id", required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--race-id", required=True)
    backtest = commands.add_parser("backtest")
    backtest.add_argument("--start-year", type=int, default=2018)
    backtest.add_argument("--end-year", type=int, default=None)
    args = parser.parse_args(argv)

    if args.command == "sync-calendar":
        result = sync_registry()
    elif args.command == "predict":
        result = create_p1_prediction(args.race_id)
    elif args.command == "predict-due":
        result = predict_due_races()
    elif args.command == "ingest-due":
        result = ingest_due_races()
    elif args.command == "evaluate-due":
        result = evaluate_due_races()
    elif args.command == "ingest":
        result = ingest_actuals(args.race_id)
    elif args.command == "evaluate":
        result = evaluate_race(args.race_id)
    else:
        result = run_walk_forward_backtest(args.start_year, args.end_year)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
