"""Leakage-aware lap-time model training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover
    XGBRegressor = None


@dataclass
class ModelBundle:
    model: Pipeline
    metrics: dict[str, float]
    feature_columns: list[str]


def _pipeline(regressor: object) -> Pipeline:
    numeric = [
        "tyre_age", "lap_number", "fuel_proxy", "track_temp", "air_temp",
        "traffic_pressure", "overtakes",
    ]
    categorical = ["compound", "team"]
    preprocessor = ColumnTransformer(
        [("numeric", SimpleImputer(strategy="median"), numeric),
         ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical)]
    )
    return Pipeline([("preprocessor", preprocessor), ("model", regressor)])


def train_models(features: pd.DataFrame, test_driver: str | None = None) -> dict[str, ModelBundle]:
    """Train baseline and main models using a driver holdout, never random laps."""
    if features["Driver"].nunique() < 2:
        raise ValueError("At least two drivers are required for driver-held-out validation.")
    holdout = test_driver or sorted(features["Driver"].unique())[-1]
    train = features[features["Driver"] != holdout]
    test = features[features["Driver"] == holdout]
    feature_columns = [
        "tyre_age", "lap_number", "fuel_proxy", "track_temp", "air_temp",
        "traffic_pressure", "overtakes", "team", "compound",
    ]
    x_train = train[feature_columns]
    x_test = test[x_train.columns]
    y_train, y_test = train["LapTime"], test["LapTime"]
    regressors: dict[str, object] = {
        "Linear": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1, min_samples_leaf=3),
    }
    if XGBRegressor is not None:
        regressors["XGBoost"] = XGBRegressor(
            n_estimators=180, max_depth=4, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.9, objective="reg:squarederror", random_state=42, n_jobs=2,
        )
    bundles: dict[str, ModelBundle] = {}
    for name, regressor in regressors.items():
        model = _pipeline(regressor)
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        bundles[name] = ModelBundle(
            model=model,
            metrics={
                "MAE": float(mean_absolute_error(y_test, predictions)),
                "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
                "R2": float(r2_score(y_test, predictions)),
            },
            feature_columns=feature_columns,
        )
    return bundles


def train_walk_forward_models(
    features: pd.DataFrame,
    race_column: str = "race_id",
    date_column: str = "race_date",
) -> dict[str, ModelBundle]:
    """Validate on the latest prior race, then fit final models on all prior races."""
    if race_column not in features or date_column not in features:
        raise ValueError("Walk-forward training requires race_id and race_date columns.")
    race_dates = features.groupby(race_column)[date_column].min().sort_values()
    if len(race_dates) < 2:
        raise ValueError("At least two earlier races are required for walk-forward training.")

    validation_race = race_dates.index[-1]
    training = features[features[race_column] != validation_race]
    validation = features[features[race_column] == validation_race]
    if training["Driver"].nunique() < 2 or validation.empty:
        raise ValueError("The chronological validation split has insufficient driver data.")

    feature_columns = [
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
    x_train = training[feature_columns]
    x_validation = validation[feature_columns]
    y_train = training["LapTime"]
    y_validation = validation["LapTime"]
    regressors: dict[str, object] = {
        "Linear": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=120,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=3,
        ),
    }
    if XGBRegressor is not None:
        regressors["XGBoost"] = XGBRegressor(
            n_estimators=180,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=2,
        )

    bundles: dict[str, ModelBundle] = {}
    for name, regressor in regressors.items():
        validation_model = _pipeline(regressor)
        validation_model.fit(x_train, y_train)
        predictions = validation_model.predict(x_validation)
        final_model = _pipeline(regressor)
        final_model.fit(features[feature_columns], features["LapTime"])
        bundles[name] = ModelBundle(
            model=final_model,
            metrics={
                "MAE": float(mean_absolute_error(y_validation, predictions)),
                "RMSE": float(np.sqrt(mean_squared_error(y_validation, predictions))),
                "R2": float(r2_score(y_validation, predictions)),
            },
            feature_columns=feature_columns,
        )
    return bundles
