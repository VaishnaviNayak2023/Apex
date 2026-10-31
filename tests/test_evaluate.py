import pytest

from src.evaluate import chronological_backtest, score_prediction


def test_score_prediction_reports_strategy_time_and_lap_metrics():
    prediction = {
        "bestStrategy": {"compounds": ["MEDIUM", "HARD"], "pitLaps": [24]},
        "predictedTotalTime": 5402.0,
        "predictedLapTimes": [
            {"lap": 1, "lapTime": 91.0},
            {"lap": 2, "lapTime": 92.0},
        ],
        "topStrategies": [
            {"compounds": ["MEDIUM", "SOFT"], "pitLaps": [20]},
            {"compounds": ["MEDIUM", "HARD"], "pitLaps": [24]},
        ],
    }
    actual = {
        "winnerStrategy": {"compoundOrder": ["MEDIUM", "HARD"], "pit_laps": [27]},
        "actualTotalTime": 5418.0,
        "actualLapTimes": [
            {"lap": 1, "lapTime": 91.5},
            {"lap": 2, "lapTime": 91.5},
        ],
    }

    scored = score_prediction(prediction, actual)

    assert scored["metrics"]["stopCountMatch"] is True
    assert scored["metrics"]["compoundOrderMatch"] is True
    assert scored["metrics"]["pitLapError"] == 3
    assert scored["metrics"]["lapTimeMae"] == pytest.approx(0.5)
    assert scored["metrics"]["totalTimeError"] == -16
    assert scored["metrics"]["actualStrategyRank"] == 2
    assert scored["metrics"]["actualStrategyRankExact"] is False
    assert scored["metrics"]["strategyCoverageTop5"] is True
    assert scored["causeTag"] == "model_error"


def test_score_prediction_marks_missing_comparison_data():
    scored = score_prediction({}, {})
    assert scored["causeTag"] == "insufficient_data"
    assert scored["metrics"]["stopCountMatch"] is None
    assert scored["metrics"]["totalTimeError"] is None


def test_walk_forward_backtest_only_uses_prior_completed_races():
    races = [
        {"raceId": "2022_02", "date": "2022-03-20"},
        {"raceId": "2022_01", "date": "2022-03-10"},
        {"raceId": "2022_03", "date": "2022-04-10"},
    ]
    training_cutoffs = []

    def make_prediction(race, completed):
        training_cutoffs.append([row["raceId"] for row in completed])
        return {
            "bestStrategy": {"compounds": ["MEDIUM", "HARD"], "pitLaps": [20]},
            "predictedTotalTime": 5000,
            "topStrategies": [],
        }

    def load_actual(_race):
        return {
            "winnerStrategy": {"compounds": ["MEDIUM", "HARD"], "pitLaps": [20]},
            "actualTotalTime": 5001,
        }

    reports = chronological_backtest(races, make_prediction, load_actual)
    assert [row["raceId"] for row in reports] == ["2022_01", "2022_02", "2022_03"]
    assert training_cutoffs == [[], ["2022_01"], ["2022_01", "2022_02"]]
