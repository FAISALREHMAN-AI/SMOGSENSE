"""Shared forecast helpers used by the app, the daily logger and the tests."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import FEATURES, LEVEL_NAMES, aqi_level

ROOT = Path(__file__).resolve().parents[1]

ADVICE = [
    "Air is fine. Normal outdoor activity.",
    "Acceptable. Very sensitive people should watch for symptoms.",
    "Children, older adults and people with lung or heart problems should limit time outside.",
    "Everyone should cut down on outdoor exercise. Wear a well-fitted N95 mask outside.",
    "Avoid outdoor activity. Keep windows closed. N95 mask if you must go out.",
    "Stay indoors where possible. Avoid all outdoor exertion. Seek medical help if you feel unwell.",
]


def load_model(name="best_model.joblib"):
    return joblib.load(ROOT / "models" / name)


def row_from_inputs(date, aqi_today, aqi_yesterday, aqi_2days_ago, aqi_mean7,
                    temp_c, humidity, dew_point_c, wind_kmh, pressure_in):
    """Build one feature row exactly like features.add_features does."""
    mean3 = np.mean([aqi_today, aqi_yesterday, aqi_2days_ago])
    month = pd.Timestamp(date).month
    row = {
        "aqi": aqi_today, "aqi_lag1": aqi_yesterday, "aqi_lag2": aqi_2days_ago,
        "aqi_mean3": mean3, "aqi_mean7": aqi_mean7, "aqi_change": aqi_today - aqi_yesterday,
        "temp_c": temp_c, "humidity": humidity, "dew_point_c": dew_point_c,
        "wind_kmh": wind_kmh, "pressure_in": pressure_in,
        "temp_dew_gap": temp_c - dew_point_c,
        "month_sin": np.sin(2 * np.pi * month / 12), "month_cos": np.cos(2 * np.pi * month / 12),
        "is_winter": int(month in (11, 12, 1, 2)),
    }
    return pd.DataFrame([row])[FEATURES]


def predict_row(model, row) -> float:
    return float(np.clip(model.predict(row)[0], 0, 500))


def describe(value):
    lvl = aqi_level(value)
    return lvl, LEVEL_NAMES[lvl], ADVICE[lvl]
