"""Step 1: clean the raw Lahore daily file and save a tidy table.

Rules (every one is reported in reports/cleaning_report.json):
- dates are dd-mm-yy
- the daily table is reindexed so every calendar day exists
- impossible sensor values become missing (not guessed)
- weather gaps of up to 3 days are filled by interpolation
- the AQI target is NEVER filled; days without AQI are kept as missing
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "lahore_aqi_2019_to_2023.csv"
OUT = ROOT / "data" / "processed" / "lahore_daily_clean.csv"
REPORT = ROOT / "reports" / "cleaning_report.json"

# Limits used to flag impossible readings for Lahore
MAX_TEMP_F = 125
MAX_WIND_MPH = 80


def clean() -> pd.DataFrame:
    df = pd.read_csv(RAW)
    report = {"raw_rows": int(len(df))}

    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%y")
    df = df.sort_values("date").set_index("date")
    full = pd.date_range(df.index.min(), df.index.max(), freq="D")
    report["missing_calendar_days_added"] = int(len(full) - len(df))
    df = df.reindex(full)
    df.index.name = "date"

    df = df.rename(columns={"aqi_pm2.5": "aqi"})

    # impossible values -> missing
    flags = {}
    for col in ["max_temp_f", "avg_temp_f", "min_temp_f"]:
        bad = df[col] > MAX_TEMP_F
        flags[col] = int(bad.sum())
        df.loc[bad, col] = np.nan
    for col in ["max_wind_speed_mph", "avg_wind_speed_mph"]:
        bad = df[col] > MAX_WIND_MPH
        flags[col] = int(bad.sum())
        df.loc[bad, col] = np.nan
    # a min pressure of 0.0 is a sensor fault, and it also pulls the daily
    # average down, so the average pressure is not trusted on those days
    bad = df["min_pressure_in"] <= 0
    flags["min_pressure_in_zero"] = int(bad.sum())
    df.loc[bad, ["min_pressure_in", "avg_pressure_in"]] = np.nan
    report["impossible_values_set_to_missing"] = flags

    # keep a small, meaningful weather set and convert units
    out = pd.DataFrame(index=df.index)
    out["aqi"] = df["aqi"]
    out["temp_c"] = (df["avg_temp_f"] - 32) * 5 / 9
    out["humidity"] = df["avg_humidity_percent"]
    out["dew_point_c"] = (df["avg_dew_point_f"] - 32) * 5 / 9
    out["wind_kmh"] = df["avg_wind_speed_mph"] * 1.609
    out["pressure_in"] = df["avg_pressure_in"]

    weather = ["temp_c", "humidity", "dew_point_c", "wind_kmh", "pressure_in"]
    report["weather_missing_before_fill"] = {c: int(out[c].isna().sum()) for c in weather}
    out[weather] = out[weather].interpolate(limit=3, limit_area="inside")
    report["weather_missing_after_fill"] = {c: int(out[c].isna().sum()) for c in weather}

    report["aqi_missing_days"] = int(out["aqi"].isna().sum())
    report["aqi_missing_by_year"] = {
        str(y): int(v) for y, v in out["aqi"].isna().groupby(out.index.year).sum().items()
    }
    report["date_range"] = [str(out.index.min().date()), str(out.index.max().date())]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT)
    REPORT.write_text(json.dumps(report, indent=2))
    return out


if __name__ == "__main__":
    d = clean()
    print(d.describe().round(1))
    print(json.dumps(json.loads(REPORT.read_text()), indent=2))
