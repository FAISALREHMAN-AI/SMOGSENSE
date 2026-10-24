"""Run with:  python tests/test_pipeline.py   (no pytest needed)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import FEATURES, add_features, aqi_level
from forecast import load_model, predict_row, row_from_inputs
from collect_daily import TZ, daily_from_payloads


def make_daily(n=40, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-11-01", periods=n)
    return pd.DataFrame({
        "aqi": rng.uniform(80, 300, n), "temp_c": rng.uniform(8, 25, n),
        "humidity": rng.uniform(40, 90, n), "dew_point_c": rng.uniform(2, 15, n),
        "wind_kmh": rng.uniform(0, 15, n), "pressure_in": rng.uniform(28.8, 29.4, n)}, index=idx)


def test_no_future_leak():
    """Changing a future day's AQI must not change any feature on an earlier day."""
    a = make_daily()
    b = a.copy()
    b.loc[b.index[20:], "aqi"] += 500
    fa, fb = add_features(a), add_features(b)
    pd.testing.assert_frame_equal(fa.loc[fa.index[:20], FEATURES], fb.loc[fb.index[:20], FEATURES])


def test_target_is_tomorrow():
    a = make_daily()
    f = add_features(a)
    assert f["target"].iloc[5] == a["aqi"].iloc[6]
    assert np.isnan(f["target"].iloc[-1])


def test_row_matches_pipeline_features():
    a = make_daily()
    f = add_features(a)
    t = 15
    r = f.iloc[t]
    row = row_from_inputs(f.index[t], r["aqi"], f["aqi"].iloc[t - 1], f["aqi"].iloc[t - 2],
                          r["aqi_mean7"], r["temp_c"], r["humidity"], r["dew_point_c"],
                          r["wind_kmh"], r["pressure_in"])
    np.testing.assert_allclose(row.iloc[0].values.astype(float), r[FEATURES].values.astype(float), rtol=1e-9)


def test_levels():
    assert aqi_level(50) == 0 and aqi_level(51) == 1 and aqi_level(301) == 5 and aqi_level(500) == 5


def test_model_predicts_sane_values():
    m = load_model()
    row = row_from_inputs("2023-12-05", 300, 280, 260, 270, 14, 70, 9, 2, 29.1)
    v = predict_row(m, row)
    assert 0 <= v <= 500
    low = predict_row(m, row_from_inputs("2023-07-05", 60, 70, 65, 70, 32, 70, 24, 14, 28.9))
    assert low < v, "a clean July day should forecast lower than a smoggy December day"


def test_collector_parsing():
    times = pd.date_range("2025-11-01", periods=48, freq="h").strftime("%Y-%m-%dT%H:%M").tolist() + ["2025-11-03T00:00", "2025-11-03T01:00"]
    n = len(times)
    weather = {"hourly": {"time": times, "temperature_2m": [20.0] * n, "relative_humidity_2m": [60.0] * n,
                          "dew_point_2m": [12.0] * n, "wind_speed_10m": [5.0] * n, "surface_pressure": [990.0] * n}}
    air = {"hourly": {"time": times, "us_aqi_pm2_5": [150.0] * n}}
    d = daily_from_payloads(weather, air)
    assert len(d) == 2, "the partial third day must be dropped"
    assert abs(d["pressure_in"].iloc[0] - 990 * 0.02953) < 1e-6


def test_collector_excludes_today_and_future():
    times = pd.date_range("2025-11-01", periods=72, freq="h").strftime("%Y-%m-%dT%H:%M").tolist()
    n = len(times)
    weather = {"hourly": {"time": times, "temperature_2m": [20.0] * n, "relative_humidity_2m": [60.0] * n,
                          "dew_point_2m": [12.0] * n, "wind_speed_10m": [5.0] * n, "surface_pressure": [990.0] * n}}
    air = {"hourly": {"time": times, "us_aqi_pm2_5": [150.0] * n}}
    now = pd.Timestamp("2025-11-02 00:30", tz=TZ)
    d = daily_from_payloads(weather, air, now=now)
    assert d.index.tolist() == [pd.Timestamp("2025-11-01")]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
