from __future__ import annotations

import sys
from types import SimpleNamespace

import pandas as pd
import pytest
import requests

from src import data_loader
from src.openf1_loader import load_openf1_race


class FakeResponse:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


def fixture_responses():
    return {
        "meetings": [
            {"meeting_key": 1210, "meeting_name": "Monaco Grand Prix"},
        ],
        "sessions": [
            {
                "meeting_key": 1210,
                "session_key": 9094,
                "session_name": "Race",
                "date_end": "2023-05-28T15:00:00+00:00",
                "is_cancelled": False,
            }
        ],
        "laps": [
            {
                "driver_number": 1,
                "lap_number": 1,
                "date_start": "2023-05-28T13:03:00+00:00",
                "lap_duration": 80.0,
                "is_pit_out_lap": False,
            },
            {
                "driver_number": 1,
                "lap_number": 2,
                "date_start": "2023-05-28T13:04:20+00:00",
                "lap_duration": 79.0,
                "is_pit_out_lap": False,
            },
            {
                "driver_number": 16,
                "lap_number": 1,
                "date_start": "2023-05-28T13:03:02+00:00",
                "lap_duration": 82.0,
                "is_pit_out_lap": False,
            },
            {
                "driver_number": 16,
                "lap_number": 2,
                "date_start": None,
                "lap_duration": 90.0,
                "is_pit_out_lap": False,
            },
        ],
        "stints": [
            {
                "driver_number": 1,
                "stint_number": 1,
                "lap_start": 1,
                "lap_end": 1,
                "compound": "MEDIUM",
                "tyre_age_at_start": 0,
            },
            {
                "driver_number": 1,
                "stint_number": 2,
                "lap_start": 2,
                "lap_end": 2,
                "compound": "HARD",
                "tyre_age_at_start": 0,
            },
            {
                "driver_number": 16,
                "stint_number": 1,
                "lap_start": 1,
                "lap_end": 1,
                "compound": "SOFT",
                "tyre_age_at_start": 3,
            },
        ],
        "drivers": [
            {"driver_number": 1, "name_acronym": "VER", "team_name": "Red Bull"},
            {"driver_number": 16, "name_acronym": "LEC", "team_name": "Ferrari"},
        ],
        "weather": [
            {
                "date": "2023-05-28T13:03:00+00:00",
                "track_temperature": 45.0,
                "air_temperature": 24.0,
            }
        ],
        "race_control": [
            {
                "date": "2023-05-28T13:02:00+00:00",
                "flag": "GREEN",
                "message": "GREEN FLAG",
            },
            {
                "date": "2023-05-28T13:04:00+00:00",
                "flag": "YELLOW",
                "message": "YELLOW FLAG",
            },
            {
                "date": "2023-05-28T13:04:10+00:00",
                "flag": None,
                "message": "VIRTUAL SAFETY CAR DEPLOYED",
            },
        ],
        "intervals": [
            {
                "driver_number": 1,
                "date": "2023-05-28T13:02:00+00:00",
                "interval": 0.6,
            },
            {
                "driver_number": 1,
                "date": "2023-05-28T13:04:00+00:00",
                "interval": 1.2,
            },
            {
                "driver_number": 16,
                "date": "2023-05-28T13:02:00+00:00",
                "interval": 1.4,
            },
        ],
        "position": [
            {
                "driver_number": 1,
                "date": "2023-05-28T13:02:00+00:00",
                "position": 1,
            },
            {
                "driver_number": 1,
                "date": "2023-05-28T13:04:00+00:00",
                "position": 2,
            },
            {
                "driver_number": 16,
                "date": "2023-05-28T13:02:00+00:00",
                "position": 2,
            },
        ],
    }


def test_openf1_loader_merges_laps_stints_drivers_and_weather(tmp_path):
    endpoint_data = fixture_responses()
    calls = []

    def get(url, params, **_kwargs):
        endpoint = url.rsplit("/", 1)[-1]
        calls.append(endpoint)
        return FakeResponse(endpoint_data[endpoint])

    laps, total_laps = load_openf1_race(
        2023, "Monaco Grand Prix", cache_dir=tmp_path, getter=get
    )

    verstappen_lap = laps[(laps["Driver"] == "VER") & (laps["LapNumber"] == 1)].iloc[0]
    leclerc_lap = laps[(laps["Driver"] == "LEC") & (laps["LapNumber"] == 1)].iloc[0]
    assert laps.attrs["source"] == "OpenF1"
    assert total_laps == 2
    assert verstappen_lap["Team"] == "Red Bull"
    assert verstappen_lap["Compound"] == "MEDIUM"
    assert verstappen_lap["IsPitIn"]
    verstappen_lap_2 = laps[
        (laps["Driver"] == "VER") & (laps["LapNumber"] == 2)
    ].iloc[0]
    assert verstappen_lap["TrackStatus"] == "1"
    assert verstappen_lap["TrafficGapSeconds"] == pytest.approx(0.6)
    assert verstappen_lap["Position"] == 1
    assert pd.isna(verstappen_lap["PositionChange"])
    assert verstappen_lap_2["TrackStatus"] == "6"
    assert verstappen_lap_2["TrafficGapSeconds"] == pytest.approx(1.2)
    assert verstappen_lap_2["Position"] == 2
    assert verstappen_lap_2["PositionChange"] == -1
    assert verstappen_lap_2["Overtakes"] == 0
    assert leclerc_lap["Compound"] == "SOFT"
    assert leclerc_lap["TyreLife"] == 4
    assert leclerc_lap["TrafficGapSeconds"] == pytest.approx(1.4)
    assert leclerc_lap["Position"] == 2
    assert laps["TrackTemp"].notna().sum() == 3
    assert laps["AirTemp"].notna().sum() == 3
    assert set(calls) == {
        "meetings",
        "sessions",
        "laps",
        "stints",
        "drivers",
        "weather",
        "race_control",
        "intervals",
        "position",
    }
    assert laps.attrs["dataAvailability"]["positionChange"] == 1
    assert laps.attrs["safetyCarEpisodes"] == 1


def test_openf1_missing_race_control_is_not_assumed_green(tmp_path):
    endpoint_data = fixture_responses()
    endpoint_data["race_control"] = []

    def get(url, params, **_kwargs):
        endpoint = url.rsplit("/", 1)[-1]
        return FakeResponse(endpoint_data[endpoint])

    laps, _ = load_openf1_race(
        2023, "Monaco Grand Prix", cache_dir=tmp_path, getter=get
    )

    assert laps["TrackStatus"].isna().all()
    assert laps.attrs["dataAvailability"]["trackStatus"] == 0


def test_load_race_uses_openf1_before_synthetic_demo_when_credentials_absent(
    monkeypatch, tmp_path
):
    expected = data_loader._demo_laps(10, 2)
    expected.attrs["source"] = "OpenF1"
    expected.attrs["totalLaps"] = 10
    monkeypatch.delenv("F1_USERNAME", raising=False)
    monkeypatch.delenv("F1_PASSWORD", raising=False)
    monkeypatch.setattr(
        data_loader,
        "load_openf1_race",
        lambda _year, _race, _cache: (expected, 10),
    )

    result = data_loader.load_race(2023, "Monaco Grand Prix", cache_dir=tmp_path)

    assert result.attrs["source"] == "OpenF1"
    assert result.attrs["totalLaps"] == 10


def test_load_race_uses_fastf1_first_when_credentials_exist(monkeypatch, tmp_path):
    class FakeLaps(pd.DataFrame):
        @property
        def _constructor(self):
            return FakeLaps

        def get_weather_data(self):
            return pd.DataFrame()

    laps = FakeLaps(
        {
            "Driver": ["VER"],
            "Team": ["Red Bull"],
            "LapNumber": [1],
            "LapTime": [pd.Timedelta(seconds=80)],
            "Compound": ["MEDIUM"],
            "TyreLife": [1],
            "Stint": [1],
            "TrackStatus": ["1"],
            "PitInTime": [pd.NaT],
            "PitOutTime": [pd.NaT],
        }
    )
    fake_session = SimpleNamespace(
        laps=laps,
        total_laps=1,
        load=lambda **_kwargs: None,
    )
    fake_fastf1 = SimpleNamespace(
        Cache=SimpleNamespace(configure=lambda **_kwargs: None),
        get_session=lambda *_args, **_kwargs: fake_session,
    )
    monkeypatch.setenv("F1_USERNAME", "example")
    monkeypatch.setenv("F1_PASSWORD", "test-only")
    monkeypatch.setitem(sys.modules, "fastf1", fake_fastf1)
    monkeypatch.setattr(
        data_loader,
        "load_openf1_race",
        lambda *_args, **_kwargs: pytest.fail("OpenF1 should be the fallback"),
    )

    result = data_loader.load_race(2023, "Monaco Grand Prix", cache_dir=tmp_path)

    assert result.attrs["source"] == "FastF1"
    assert result["LapTime"].iloc[0] == 80


def test_load_race_aligns_fastf1_weather_by_row_with_duplicate_indexes(
    monkeypatch, tmp_path
):
    class FakeLaps(pd.DataFrame):
        @property
        def _constructor(self):
            return FakeLaps

        def get_weather_data(self):
            return pd.DataFrame(
                {"TrackTemp": [30.0, 31.0], "AirTemp": [20.0, 21.0]},
                index=[0, 0],
            )

    laps = FakeLaps(
        {
            "Driver": ["VER", "VER"],
            "Team": ["Red Bull", "Red Bull"],
            "LapNumber": [1, 2],
            "LapTime": [pd.Timedelta(seconds=80), pd.Timedelta(seconds=81)],
            "Compound": ["MEDIUM", "MEDIUM"],
            "TyreLife": [1, 2],
            "Stint": [1, 1],
            "TrackStatus": ["1", "1"],
            "PitInTime": [pd.NaT, pd.NaT],
            "PitOutTime": [pd.NaT, pd.NaT],
        },
        index=[0, 1],
    )
    fake_session = SimpleNamespace(
        laps=laps,
        total_laps=2,
        load=lambda **_kwargs: None,
    )
    fake_fastf1 = SimpleNamespace(
        Cache=SimpleNamespace(configure=lambda **_kwargs: None),
        get_session=lambda *_args, **_kwargs: fake_session,
    )
    monkeypatch.setenv("F1_USERNAME", "example")
    monkeypatch.setenv("F1_PASSWORD", "test-only")
    monkeypatch.setitem(sys.modules, "fastf1", fake_fastf1)
    monkeypatch.setattr(
        data_loader,
        "load_openf1_race",
        lambda *_args, **_kwargs: pytest.fail("OpenF1 should not be used"),
    )

    result = data_loader.load_race(2023, "Monaco Grand Prix", cache_dir=tmp_path)

    assert result.attrs["source"] == "FastF1"
    assert result["TrackTemp"].tolist() == [30.0, 31.0]
    assert result["AirTemp"].tolist() == [20.0, 21.0]


def test_load_race_fails_closed_after_both_real_sources_fail(monkeypatch, tmp_path):
    monkeypatch.delenv("F1_USERNAME", raising=False)
    monkeypatch.delenv("F1_PASSWORD", raising=False)
    monkeypatch.delenv("ALLOW_DEMO_DATA", raising=False)
    monkeypatch.setattr(
        data_loader,
        "load_openf1_race",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            requests.RequestException("offline")
        ),
    )

    with pytest.raises(data_loader.RealDataUnavailable, match="OpenF1 could not load"):
        data_loader.load_race(2022, "Monaco Grand Prix", cache_dir=tmp_path)


def test_demo_data_requires_explicit_opt_in(monkeypatch, tmp_path):
    monkeypatch.delenv("F1_USERNAME", raising=False)
    monkeypatch.delenv("F1_PASSWORD", raising=False)
    monkeypatch.setenv("ALLOW_DEMO_DATA", "true")
    monkeypatch.setattr(
        data_loader,
        "load_openf1_race",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            requests.RequestException("offline")
        ),
    )

    result = data_loader.load_race(2022, "Monaco Grand Prix", cache_dir=tmp_path)

    assert result.attrs["source"] == "Demo data"


def test_openf1_cache_avoids_repeated_requests(tmp_path):
    endpoint_data = fixture_responses()
    calls = []

    def get(url, params, **_kwargs):
        endpoint = url.rsplit("/", 1)[-1]
        calls.append(endpoint)
        return FakeResponse(endpoint_data[endpoint])

    load_openf1_race(2023, "Monaco Grand Prix", cache_dir=tmp_path, getter=get)
    first_call_count = len(calls)
    load_openf1_race(2023, "Monaco Grand Prix", cache_dir=tmp_path, getter=get)

    assert first_call_count == 9
    assert len(calls) == first_call_count
