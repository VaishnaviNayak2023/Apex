"""Scenario-aware race strategy generation, simulation, and ranking."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, product

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Strategy:
    compounds: tuple[str, ...]
    pit_laps: tuple[int, ...]

    @property
    def label(self) -> str:
        return " → ".join(c.title() for c in self.compounds) + (
            f" (pit {', '.join(map(str, self.pit_laps))})" if self.pit_laps else ""
        )


@dataclass(frozen=True)
class StrategyScenario:
    team: str = "Average"
    track_temp: float = 38.0
    air_temp: float = 27.0
    weather_change: float = 0.0
    traffic_intensity: float = 0.15
    overtake_rate: float = 0.25
    safety_car_probability: float = 0.2
    safety_car_duration: int = 4
    safety_car_pit_saving: float = 8.0
    safety_car_lap_delta: float = 28.0
    available_sets: dict[str, int] = field(
        default_factory=lambda: {"SOFT": 2, "MEDIUM": 3, "HARD": 2}
    )
    model_error: float = 0.5
    draws: int = 200
    seed: int = 42


def generate_strategies(
    total_laps: int,
    stops: int = 1,
    step: int = 2,
    available_sets: dict[str, int] | None = None,
) -> list[Strategy]:
    sets = available_sets or {"SOFT": 2, "MEDIUM": 3, "HARD": 2}
    strategies: list[Strategy] = []
    for compounds in product(("SOFT", "MEDIUM", "HARD"), repeat=stops + 1):
        if len(set(compounds)) < 2:
            continue
        if any(compounds.count(compound) > sets.get(compound, 0) for compound in set(compounds)):
            continue
        for pit_laps in combinations(range(10, total_laps - 9, step), stops):
            if all(b - a >= 8 for a, b in zip((0, *pit_laps), (*pit_laps, total_laps))):
                strategies.append(Strategy(compounds, pit_laps))
    return strategies


def _strategy_frame(
    strategy: Strategy,
    total_laps: int,
    scenario: StrategyScenario,
) -> pd.DataFrame:
    rows = []
    boundaries = (0, *strategy.pit_laps, total_laps)
    for stint, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        for lap in range(start + 1, end + 1):
            progress = (lap - 1) / max(total_laps - 1, 1)
            rows.append(
                {
                    "tyre_age": lap - start,
                    "lap_number": lap,
                    "fuel_proxy": total_laps - lap,
                    "track_temp": float(
                        np.clip(scenario.track_temp + scenario.weather_change * progress, 0, 60)
                    ),
                    "air_temp": scenario.air_temp,
                    "traffic_pressure": scenario.traffic_intensity,
                    "overtakes": scenario.traffic_intensity * scenario.overtake_rate,
                    "team": scenario.team,
                    "compound": strategy.compounds[stint],
                }
            )
    return pd.DataFrame(rows)


def simulate_strategy(
    model: object,
    strategy: Strategy,
    total_laps: int,
    pit_loss: float,
    track_temp: float = 38.0,
    air_temp: float = 27.0,
    scenario: StrategyScenario | None = None,
) -> tuple[float, pd.DataFrame]:
    scenario = scenario or StrategyScenario(track_temp=track_temp, air_temp=air_temp)
    frame = _strategy_frame(strategy, total_laps, scenario)
    frame["PredictedLapTime"] = model.predict(frame)
    return float(frame["PredictedLapTime"].sum() + len(strategy.pit_laps) * pit_loss), frame


def rank_strategies(
    model: object,
    total_laps: int,
    pit_loss: float,
    stops: int = 1,
    track_temp: float = 38.0,
    air_temp: float = 27.0,
    scenario: StrategyScenario | None = None,
) -> pd.DataFrame:
    scenario = scenario or StrategyScenario(track_temp=track_temp, air_temp=air_temp)
    strategies = generate_strategies(
        total_laps, stops, available_sets=scenario.available_sets
    )
    if not strategies:
        raise ValueError(
            "No valid strategies fit the race length and available tyre-set limits."
        )

    rng = np.random.default_rng(scenario.seed)
    draws = max(50, min(scenario.draws, 1000))
    traffic_count = rng.binomial(total_laps, scenario.traffic_intensity, size=draws)
    successful_overtakes = rng.binomial(
        traffic_count, scenario.overtake_rate, size=draws
    )
    weather_shock = rng.normal(0, 1.5, size=draws)
    safety_car = rng.random(draws) < scenario.safety_car_probability
    safety_car_start = rng.integers(1, total_laps + 1, size=draws)
    outcomes = np.empty((len(strategies), draws), dtype=float)
    rows: list[dict[str, object]] = []

    frames = [_strategy_frame(strategy, total_laps, scenario) for strategy in strategies]
    all_features = pd.concat(frames, ignore_index=True)
    one_degree_warmer = all_features.copy()
    one_degree_warmer["track_temp"] = np.clip(
        one_degree_warmer["track_temp"] + 1, 0, 60
    )
    clear_air = all_features.copy()
    clear_air["traffic_pressure"] = 0.0
    clear_air["overtakes"] = 0.0
    no_passes = all_features.copy()
    no_passes["overtakes"] = 0.0
    batch_predictions = np.asarray(
        model.predict(
            pd.concat(
                [all_features, one_degree_warmer, clear_air, no_passes],
                ignore_index=True,
            )
        )
    ).reshape(4, len(strategies), total_laps)

    for index, strategy in enumerate(strategies):
        expected_lap_time = float(batch_predictions[0, index].sum())
        weather_sensitivity = float(
            batch_predictions[1, index].sum() - expected_lap_time
        )
        traffic_effect = float(
            expected_lap_time - batch_predictions[2, index].sum()
        )
        passing_effect = float(
            expected_lap_time - batch_predictions[3, index].sum()
        )
        model_shock = rng.normal(
            0, scenario.model_error * np.sqrt(total_laps), size=draws
        )

        samples = (
            expected_lap_time
            + len(strategy.pit_laps) * pit_loss
            + weather_sensitivity * weather_shock
            + traffic_effect / total_laps * (traffic_count - total_laps * scenario.traffic_intensity)
            + passing_effect / total_laps
            * (successful_overtakes - total_laps * scenario.traffic_intensity * scenario.overtake_rate)
            + model_shock
        )

        for pit_lap in strategy.pit_laps:
            under_safety_car = (
                safety_car
                & (pit_lap >= safety_car_start)
                & (pit_lap < safety_car_start + scenario.safety_car_duration)
            )
            samples -= under_safety_car * scenario.safety_car_pit_saving
        safety_car_laps = np.minimum(
            scenario.safety_car_duration,
            np.maximum(0, total_laps - safety_car_start + 1),
        )
        samples += safety_car * safety_car_laps * scenario.safety_car_lap_delta

        outcomes[index] = samples
        rows.append(
            {
                "strategy": strategy,
                "Strategy": strategy.label,
                "Total time (s)": float(samples.mean()),
                "P10 (s)": float(np.percentile(samples, 10)),
                "P90 (s)": float(np.percentile(samples, 90)),
            }
        )

    winner_indexes = np.argmin(outcomes, axis=0)
    win_probabilities = np.bincount(winner_indexes, minlength=len(strategies)) / draws
    for row, win_probability in zip(rows, win_probabilities):
        row["Win probability"] = float(win_probability)

    result = pd.DataFrame(rows).sort_values("Total time (s)").reset_index(drop=True)
    result["Delta (s)"] = result["Total time (s)"] - result["Total time (s)"].iloc[0]
    return result
