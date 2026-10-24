"""Step 2: build features. Every feature uses only information known at the
END of day t. The target is the AQI of day t+1.
"""
import numpy as np
import pandas as pd

FEATURES = [
    "aqi", "aqi_lag1", "aqi_lag2", "aqi_mean3", "aqi_mean7", "aqi_change",
    "temp_c", "humidity", "dew_point_c", "wind_kmh", "pressure_in",
    "temp_dew_gap", "month_sin", "month_cos", "is_winter",
]

LEVELS = [(50, "Good"), (100, "Moderate"), (150, "Unhealthy for sensitive groups"),
          (200, "Unhealthy"), (300, "Very unhealthy"), (10_000, "Hazardous")]
LEVEL_NAMES = [n for _, n in LEVELS]


def aqi_level(value: float) -> int:
    """Index of the alert level (0 = Good ... 5 = Hazardous)."""
    for i, (limit, _) in enumerate(LEVELS):
        if value <= limit:
            return i
    return len(LEVELS) - 1


def add_features(daily: pd.DataFrame) -> pd.DataFrame:
    """daily: index = date (every calendar day), column 'aqi' plus weather."""
    d = daily.copy()
    a = d["aqi"]
    d["aqi_lag1"] = a.shift(1)
    d["aqi_lag2"] = a.shift(2)
    d["aqi_mean3"] = a.rolling(3, min_periods=2).mean()
    d["aqi_mean7"] = a.rolling(7, min_periods=3).mean()
    d["aqi_change"] = a - d["aqi_lag1"]
    d["temp_dew_gap"] = d["temp_c"] - d["dew_point_c"]
    m = d.index.month
    d["month_sin"] = np.sin(2 * np.pi * m / 12)
    d["month_cos"] = np.cos(2 * np.pi * m / 12)
    d["is_winter"] = np.isin(m, [11, 12, 1, 2]).astype(int)
    d["target"] = a.shift(-1)  # tomorrow's AQI
    return d
