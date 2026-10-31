"""Comparison metrics for pre-race strategy predictions."""

from __future__ import annotations

from math import fsum


CAUSE_TAGS = {
    "safety_car",
    "weather",
    "traffic",
    "reliability",
    "tyre_surprise",
    "model_error",
    "insufficient_data",
}


def _compound_order(strategy: dict | None) -> list[str] | None:
    if not strategy:
        return None
    compounds = strategy.get("compoundOrder") or strategy.get("compounds")
    if not compounds:
        return None
    return [str(compound).upper() for compound in compounds]


def _pit_laps(strategy: dict | None) -> list[int] | None:
    if not strategy:
        return None
    pit_laps = strategy.get("pitLaps") or strategy.get("pit_laps")
    if pit_laps is None:
        return None
    return [int(lap) for lap in pit_laps]


def _strategy_key(strategy: dict | None) -> tuple | None:
    compounds = _compound_order(strategy)
    pit_laps = _pit_laps(strategy)
    if compounds is None or pit_laps is None:
        return None
    return tuple(compounds), tuple(pit_laps)


def _lap_time_mae(predicted: list[dict] | None, actual: list[dict] | None) -> float | None:
    if not predicted or not actual:
        return None
    actual_by_lap = {
        int(row["lap"]): float(row["lapTime"])
        for row in actual
        if row.get("lap") is not None and row.get("lapTime") is not None
    }
    errors = [
        abs(float(row["lapTime"]) - actual_by_lap[int(row["lap"])])
        for row in predicted
        if row.get("lap") is not None
        and row.get("lapTime") is not None
        and int(row["lap"]) in actual_by_lap
    ]
    return fsum(errors) / len(errors) if errors else None


def _cause_tag(actual: dict, metrics: dict) -> str:
    explicit_tags = actual.get("causeTags") or []
    for tag in (
        "safety_car",
        "weather",
        "traffic",
        "reliability",
        "tyre_surprise",
        "model_error",
    ):
        if tag in explicit_tags:
            return tag
    if (
        metrics["stopCountMatch"] is None
        or metrics["compoundOrderMatch"] is None
        or metrics["totalTimeError"] is None
    ):
        return "insufficient_data"
    return "model_error"


def score_prediction(prediction: dict, actual: dict) -> dict:
    """Score only fields backed by both the saved prediction and official actuals."""
    predicted_strategy = prediction.get("bestStrategy")
    actual_strategy = actual.get("winnerStrategy")
    predicted_stops = _pit_laps(predicted_strategy)
    actual_stops = _pit_laps(actual_strategy)
    predicted_order = _compound_order(predicted_strategy)
    actual_order = _compound_order(actual_strategy)
    predicted_key = _strategy_key(predicted_strategy)
    actual_key = _strategy_key(actual_strategy)

    pit_lap_error = None
    if predicted_stops and actual_stops:
        pit_lap_error = abs(predicted_stops[0] - actual_stops[0])

    predicted_total = prediction.get("predictedTotalTime")
    actual_total = actual.get("actualTotalTime")
    total_time_error = (
        float(predicted_total) - float(actual_total)
        if predicted_total is not None and actual_total is not None
        else None
    )

    ranked = prediction.get("strategyRanking") or prediction.get("topStrategies") or []
    actual_rank = None
    actual_rank_exact = None
    if actual_key is not None:
        matching_compounds = [
            (index, _pit_laps(row))
            for index, row in enumerate(ranked, start=1)
            if tuple(_compound_order(row) or []) == actual_key[0]
        ]
        if matching_compounds:
            actual_rank, candidate_laps = min(
                matching_compounds,
                key=lambda item: (
                    sum(
                        abs(left - right)
                        for left, right in zip(item[1] or [], actual_key[1])
                    ),
                    item[0],
                ),
            )
            actual_rank_exact = candidate_laps == list(actual_key[1])

    metrics = {
        "stopCountMatch": (
            len(predicted_stops) == len(actual_stops)
            if predicted_stops is not None and actual_stops is not None
            else None
        ),
        "compoundOrderMatch": (
            predicted_order == actual_order
            if predicted_order is not None and actual_order is not None
            else None
        ),
        "pitLapError": pit_lap_error,
        "lapTimeMae": _lap_time_mae(
            prediction.get("predictedLapTimes"),
            actual.get("actualLapTimes"),
        ),
        "totalTimeError": total_time_error,
        "actualStrategyRank": actual_rank,
        "actualStrategyRankExact": actual_rank_exact,
        "strategyCoverageTop5": (
            actual_rank <= 5 if actual_rank is not None else None
        ),
        "degradationError": actual.get("degradationError"),
    }
    cause_tag = _cause_tag(actual, metrics)
    if cause_tag not in CAUSE_TAGS:
        raise ValueError(f"Unsupported cause tag: {cause_tag}.")
    return {"metrics": metrics, "causeTag": cause_tag}


def chronological_backtest(
    races: list[dict],
    make_prediction,
    load_actual,
    minimum_prior_races: int = 0,
) -> list[dict]:
    """Run a deterministic walk-forward loop; callbacks own data loading and fitting."""
    ordered = sorted(races, key=lambda race: (race["date"], race["raceId"]))
    completed: list[dict] = []
    reports: list[dict] = []
    for race in ordered:
        actual = load_actual(race)
        if actual is None:
            continue
        if len(completed) < minimum_prior_races:
            completed.append(race)
            continue
        prediction = make_prediction(race, list(completed))
        reports.append(
            {
                "raceId": race["raceId"],
                "prediction": prediction,
                "evaluation": score_prediction(prediction, actual),
            }
        )
        completed.append(race)
    return reports
