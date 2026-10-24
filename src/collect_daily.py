"""Daily collector (runs from GitHub Actions). Appends yesterday's daily values
to data/live/daily_log.csv, and writes tomorrow's forecast to
data/live/forecast_log.csv so it can be scored against the real value later.

IMPORTANT: live AQI here comes from Open-Meteo's air-quality API, which is a
computer-model estimate (CAMS), not the monitor used in the training file.
The two sources can differ, so treat live scores as a separate test.
"""
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

LAT, LON, TZ = 31.5204, 74.3587, "Asia/Karachi"
ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "data" / "live"
HPA_TO_INHG = 0.02953


def _get(url, params):
    with urllib.request.urlopen(url + "?" + urllib.parse.urlencode(params), timeout=30) as r:
        return json.load(r)


def daily_from_payloads(weather: dict, air: dict, now=None) -> pd.DataFrame:
    """Return complete daily rows strictly before today's date in Asia/Karachi."""
    w = pd.DataFrame(weather["hourly"]).assign(time=lambda d: pd.to_datetime(d["time"])).set_index("time")
    a = pd.DataFrame(air["hourly"]).assign(time=lambda d: pd.to_datetime(d["time"])).set_index("time")
    h = w.join(a, how="inner")
    daily = pd.DataFrame({
        "aqi": h["us_aqi_pm2_5"].resample("D").mean(),
        "temp_c": h["temperature_2m"].resample("D").mean(),
        "humidity": h["relative_humidity_2m"].resample("D").mean(),
        "dew_point_c": h["dew_point_2m"].resample("D").mean(),
        "wind_kmh": h["wind_speed_10m"].resample("D").mean(),
        "pressure_in": h["surface_pressure"].resample("D").mean() * HPA_TO_INHG,
        "hours": h["us_aqi_pm2_5"].resample("D").count(),
    })
    current_time = pd.Timestamp.now(tz=TZ) if now is None else pd.Timestamp(now)
    if current_time.tzinfo is None:
        current_time = current_time.tz_localize(TZ)
    else:
        current_time = current_time.tz_convert(TZ)
    complete_days = (daily["hours"] == 24) & (daily.index.date < current_time.date())
    return daily[complete_days].drop(columns="hours")


def main():
    now = pd.Timestamp.now(tz=TZ)
    weather = _get("https://api.open-meteo.com/v1/forecast", {
        "latitude": LAT, "longitude": LON, "timezone": TZ, "past_days": 8, "forecast_days": 1,
        "hourly": "temperature_2m,relative_humidity_2m,dew_point_2m,wind_speed_10m,surface_pressure"})
    air = _get("https://air-quality-api.open-meteo.com/v1/air-quality", {
        "latitude": LAT, "longitude": LON, "timezone": TZ, "past_days": 8, "forecast_days": 1,
        "hourly": "us_aqi_pm2_5"})
    new = daily_from_payloads(weather, air, now=now)
    new.index.name = "date"

    LIVE.mkdir(parents=True, exist_ok=True)
    log_path = LIVE / "daily_log.csv"
    if log_path.exists():
        old = pd.read_csv(log_path, index_col="date", parse_dates=True)
        log = pd.concat([old, new[~new.index.isin(old.index)]]).sort_index()
    else:
        log = new
    log = log[log.index.date < now.date()]
    log.round(2).to_csv(log_path)
    print(f"daily log now has {len(log)} days, last = {log.index.max().date()}")

    # tomorrow's forecast from the latest full day
    sys.path.insert(0, str(Path(__file__).parent))
    from features import add_features
    from forecast import load_model, FEATURES, predict_row
    feats = add_features(log).dropna(subset=["aqi"])
    last = feats.iloc[[-1]]
    fc = predict_row(load_model(), last[FEATURES])
    target_day = (last.index[0] + pd.Timedelta(days=1)).date()
    f_path = LIVE / "forecast_log.csv"
    row = pd.DataFrame([{"made_on": last.index[0].date(), "for_date": target_day, "forecast": round(fc, 1)}])
    if f_path.exists():
        prev = pd.read_csv(f_path)
        prev = prev[prev["made_on"] < str(now.date())]
        prev = prev[prev["for_date"] != str(target_day)]
        row = pd.concat([prev, row])
    row.to_csv(f_path, index=False)
    print(f"forecast for {target_day}: {fc:.0f}")


if __name__ == "__main__":
    main()
