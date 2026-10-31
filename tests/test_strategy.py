import pandas as pd

from src.data_loader import _demo_laps
from src.cleaning import clean_laps
from src.features import add_features
from src.simulate import StrategyScenario, generate_strategies, rank_strategies
from src.train import train_models, train_walk_forward_models


class ScenarioModel:
    def predict(self, frame: pd.DataFrame) -> pd.Series:
        compound_pace = frame["compound"].map({"SOFT": 0.0, "MEDIUM": 0.3, "HARD": 0.6})
        team_pace = frame["team"].map({"Team 1": -0.2, "Team 2": 0.2}).fillna(0)
        return (
            90 + compound_pace + frame["tyre_age"] * 0.08
            + frame["track_temp"] * 0.01
            + frame["traffic_pressure"] * 0.7
            - frame["overtakes"] * 0.2
            + team_pace
        )


def test_cleaning_removes_pit_and_invalid_laps():
    laps = _demo_laps(20, 2)
    laps.loc[0, "IsPitIn"] = True
    laps.loc[1, "TrackStatus"] = "4"
    cleaned = clean_laps(laps)
    assert len(cleaned) == len(laps) - 2


def test_strategy_generation_uses_two_compounds_and_valid_stints():
    strategies = generate_strategies(57, stops=1)
    assert strategies
    assert all(len(set(strategy.compounds)) >= 2 for strategy in strategies)
    assert all(strategy.pit_laps[0] >= 10 for strategy in strategies)


def test_tyre_allocation_limits_candidate_strategies():
    strategies = generate_strategies(
        57, stops=2, available_sets={"SOFT": 0, "MEDIUM": 2, "HARD": 2}
    )
    assert strategies
    assert all("SOFT" not in strategy.compounds for strategy in strategies)
    assert all(strategy.compounds.count("MEDIUM") <= 2 for strategy in strategies)
    assert all(strategy.compounds.count("HARD") <= 2 for strategy in strategies)


def test_monte_carlo_rankings_include_uncertainty_and_win_probability():
    scenario = StrategyScenario(
        team="Team 1",
        traffic_intensity=0.4,
        overtake_rate=0.5,
        safety_car_probability=0.4,
        draws=100,
    )
    result = rank_strategies(ScenarioModel(), 57, 22, stops=1, scenario=scenario)
    assert {"P10 (s)", "P90 (s)", "Win probability"} <= set(result.columns)
    assert (result["P90 (s)"] >= result["P10 (s)"]).all()
    assert abs(result["Win probability"].sum() - 1) < 1e-9
    assert result.iloc[0]["Total time (s)"] <= result.iloc[-1]["Total time (s)"]


def test_models_train_on_feature_frame():
    features = add_features(clean_laps(_demo_laps(30, 4)))
    models = train_models(features)
    assert "Linear" in models
    assert {"team", "traffic_pressure", "overtakes"} <= set(
        models["Linear"].feature_columns
    )
    assert all(bundle.metrics["MAE"] >= 0 for bundle in models.values())


def test_walk_forward_models_validate_on_latest_prior_race():
    first = add_features(clean_laps(_demo_laps(30, 4)), 30)
    first["race_id"] = "2024_01"
    first["race_date"] = "2024-03-01"
    second = add_features(clean_laps(_demo_laps(30, 4)), 30)
    second["race_id"] = "2024_02"
    second["race_date"] = "2024-03-10"
    second["LapTime"] += 0.4
    bundles = train_walk_forward_models(pd.concat([first, second], ignore_index=True))
    assert "XGBoost" in bundles
    assert bundles["XGBoost"].metrics["MAE"] >= 0
