"""SmogSense: explainable next-day AQI forecasting for Lahore."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from features import FEATURES, LEVEL_NAMES, add_features, aqi_level
from forecast import describe, load_model, predict_row, row_from_inputs

CLEAN_PATH = ROOT / "data" / "processed" / "lahore_daily_clean.csv"
BEST_MODEL_PATH = ROOT / "models" / "best_model.joblib"
BACKTEST_MODEL_PATH = ROOT / "models" / "backtest_model.joblib"
RESULTS_PATH = ROOT / "reports" / "results.json"
PREDICTIONS_PATH = ROOT / "reports" / "test_predictions.csv"
IMPORTANCE_PATH = ROOT / "reports" / "importance.png"
TEST_PLOT_PATH = ROOT / "reports" / "test_forecast.png"
LIVE_DAILY_PATH = ROOT / "data" / "live" / "daily_log.csv"
LIVE_FORECAST_PATH = ROOT / "data" / "live" / "forecast_log.csv"

DATA_COLUMNS = {"aqi", "temp_c", "humidity", "dew_point_c", "wind_kmh", "pressure_in"}
PREDICTION_COLUMNS = {
    "date", "actual_tomorrow", "Tomorrow = today", "Ridge",
    "RandomForest", "HistGradientBoosting",
}
FEATURE_LABELS = {
    "aqi": "AQI on the input day",
    "aqi_lag1": "AQI one day earlier",
    "aqi_lag2": "AQI two days earlier",
    "aqi_mean3": "Three-day average AQI",
    "aqi_mean7": "Seven-day average AQI",
    "aqi_change": "Change from the previous day",
    "temp_c": "Temperature",
    "humidity": "Relative humidity",
    "dew_point_c": "Dew point",
    "wind_kmh": "Wind speed",
    "pressure_in": "Air pressure",
    "temp_dew_gap": "Temperature minus dew point",
    "month_sin": "Seasonal month pattern (sine)",
    "month_cos": "Seasonal month pattern (cosine)",
    "is_winter": "Winter-month indicator",
}
AQI_COLORS = ("#28D7A0", "#D9DD62", "#F6A64A", "#F05D62", "#B578F2", "#8F3157")
AQI_BADGE_COLORS = ("green", "yellow", "orange", "red", "violet", "red")
PAGE_OPTIONS = (
    (":material/dashboard:  Overview", "Overview"),
    (":material/near_me:  AQI forecast", "AQI forecast"),
    (":material/show_chart:  Historical trends", "Historical trends"),
    (":material/tune:  What-if simulator", "What-if simulator"),
    (":material/leaderboard:  Model performance", "Model performance"),
    (":material/insights:  Model insights", "Model insights"),
    (":material/database:  Collected data", "Collected data"),
    (":material/info:  About SmogSense", "About SmogSense"),
)
PAGE_COPY = {
    "Overview": (
        "A clear view of Lahore's air-quality history",
        "Explore measured history, collected estimates, and model outputs—kept distinct.",
    ),
    "AQI forecast": (
        "Next-day estimates, with their dates and sources",
        "A forecast is an estimate for a stated target date, never a live measurement.",
    ),
    "Historical trends": (
        "Explore the measured AQI record",
        "Historical observations from 2019–2023, with gaps left visible and unfilled.",
    ),
    "What-if simulator": (
        "Explore a model-based scenario",
        "Adjust the inputs and see how the saved model responds to your example.",
    ),
    "Model performance": (
        "How the models performed on held-out data",
        "Compare the preserved 2023 test results without changing the evaluation.",
    ),
    "Model insights": (
        "What the model relied on",
        "Feature importance describes predictive patterns; it does not prove causes.",
    ),
    "Collected data": (
        "Collected Open-Meteo estimates",
        "These modelled readings are separate from the historical monitor dataset.",
    ),
    "About SmogSense": (
        "An educational forecast for Lahore",
        "Understand the data, model, limitations, and ideas discussed by the project.",
    ),
}

st.set_page_config(
    page_title="SmogSense | Lahore air-quality forecast",
    page_icon=":material/air:",
    layout="wide",
)

st.html("""
<style>
:root {
  --smog-bg: #07111F;
  --smog-surface: #101D30;
  --smog-raised: #13243A;
  --smog-line: rgba(167, 198, 220, .14);
  --smog-cyan: #35D6E8;
  --smog-teal: #28D7A0;
  --smog-muted: #95A8BB;
}
.stApp {
  background:
    radial-gradient(ellipse at 74% 2%, rgba(24, 102, 131, .17), transparent 37%),
    linear-gradient(145deg, #07111F 0%, #081525 52%, #07111F 100%);
}
[data-testid="stHeader"] { background: rgba(7, 17, 31, .75); }
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0C192A 0%, #091525 100%);
  border-right: 1px solid var(--smog-line);
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #A8B8C9; }
[data-testid="stSidebar"] [role="radiogroup"] { gap: .38rem; }
[data-testid="stSidebar"] [role="radiogroup"] label {
  min-height: 2.8rem;
  padding: .42rem .72rem;
  border: 1px solid transparent;
  border-radius: 12px;
  transition: background-color .16s ease, border-color .16s ease, transform .16s ease;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {
  background: rgba(53, 214, 232, .08);
  border-color: rgba(53, 214, 232, .2);
  transform: translateX(2px);
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
  background: linear-gradient(100deg, rgba(53, 214, 232, .16), rgba(40, 215, 160, .06));
  border-color: rgba(53, 214, 232, .3);
  box-shadow: inset 3px 0 #35D6E8;
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {
  color: #F4FBFF;
  font-weight: 650;
}
[data-testid="stVerticalBlockBorderWrapper"] {
  background: linear-gradient(150deg, rgba(16, 29, 48, .94), rgba(13, 27, 44, .92));
  border-color: var(--smog-line) !important;
  border-radius: 18px !important;
  box-shadow: 0 12px 32px rgba(0, 0, 0, .12);
  transition: border-color .18s ease, box-shadow .18s ease, transform .18s ease;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
  border-color: rgba(53, 214, 232, .3) !important;
  box-shadow: 0 16px 36px rgba(0, 0, 0, .2);
  transform: translateY(-2px);
}
[data-testid="stMetric"] {
  background: linear-gradient(145deg, rgba(19, 36, 58, .88), rgba(13, 27, 44, .88));
  border: 1px solid var(--smog-line);
  border-radius: 16px;
  padding: 1rem 1.1rem;
}
[data-testid="stMetricLabel"] p { color: #A8B8C9; font-size: .84rem; }
[data-testid="stMetricValue"] { color: #F3FAFF; }
[data-testid="stPlotlyChart"], [data-testid="stVegaLiteChart"] {
  border-radius: 14px;
}
[data-testid="stSlider"] [data-baseweb="slider"] { padding-top: .4rem; }
[data-testid="stDataFrame"] { border: 1px solid var(--smog-line); border-radius: 14px; }
.smog-header {
  display: flex; justify-content: space-between; align-items: center; gap: 1rem;
  padding: .55rem 0 1.25rem; margin-bottom: .55rem;
  border-bottom: 1px solid rgba(167,198,220,.1);
}
.smog-header-title { color: #F4FAFF; font-size: clamp(1.5rem, 3vw, 2.05rem); font-weight: 730; letter-spacing: -.04em; }
.smog-header-brand { color: #35D6E8; font-size: .69rem; font-weight: 730; letter-spacing: .14em; margin-bottom: .25rem; }
.smog-header-sub { color: #95A8BB; font-size: .93rem; margin-top: .3rem; }
.smog-chip-row { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: .45rem; }
.smog-chip {
  display: inline-flex; align-items: center; gap: .4rem; padding: .42rem .68rem;
  border: 1px solid rgba(167,198,220,.16); border-radius: 999px;
  background: rgba(16,29,48,.8); color: #B7C7D7; font-size: .76rem; white-space: nowrap;
}
.smog-chip-dot { width: 7px; height: 7px; border-radius: 100%; background: #35D6E8; }
.smog-chip-dot.muted { background: #8395A8; }
.smog-hero {
  position: relative; overflow: hidden; min-height: 260px; height: 100%;
  display: flex; flex-direction: column; justify-content: space-between;
  padding: clamp(1.3rem, 3vw, 2.15rem);
  border: 1px solid rgba(53,214,232,.2); border-radius: 24px;
  background:
    radial-gradient(ellipse at 90% 2%, rgba(53,214,232,.17), transparent 44%),
    linear-gradient(135deg, rgba(16,29,48,.98), rgba(10,28,45,.96));
  box-shadow: 0 22px 55px rgba(0,0,0,.2);
}
.smog-hero-kicker { color: #35D6E8; font-size: .75rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
.smog-hero-title { max-width: 680px; margin: .65rem 0; color: #F4FAFF; font-size: clamp(1.8rem, 4vw, 3.15rem); line-height: 1.08; font-weight: 760; letter-spacing: -.055em; }
.smog-hero-copy { max-width: 630px; color: #A7B9CB; font-size: .98rem; line-height: 1.65; }
.smog-hero-foot { display: flex; flex-wrap: wrap; gap: .55rem; margin-top: 1.4rem; }
.smog-hero-tag { padding: .42rem .68rem; border-radius: 999px; border: 1px solid rgba(167,198,220,.13); color: #B9CAD8; background: rgba(7,17,31,.36); font-size: .75rem; }
.smog-gauge {
  display: flex; height: 100%; min-height: 260px; flex-direction: column; align-items: center; justify-content: center;
  padding: 1.2rem; border: 1px solid var(--smog-line); border-radius: 24px;
  background: linear-gradient(150deg, rgba(16,29,48,.96), rgba(11,24,40,.98));
}
.smog-gauge-title { color: #A8B8C9; font-size: .84rem; text-transform: uppercase; letter-spacing: .08em; }
.smog-gauge-wrap { position: relative; width: 176px; height: 176px; margin: .35rem 0; }
.smog-gauge-wrap {
  display: grid; place-items: center; border-radius: 50%;
  background: conic-gradient(var(--aqi-color) 0deg var(--aqi-progress), rgba(167,198,220,.13) var(--aqi-progress) 360deg);
}
.smog-gauge-ring { position: absolute; inset: 12px; border-radius: 50%; background: #101D30; }
.smog-gauge-center { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; }
.smog-gauge-number { color: #F4FAFF; font-size: 2.75rem; font-weight: 760; line-height: 1; letter-spacing: -.06em; }
.smog-gauge-unit { color: #8297AB; font-size: .73rem; margin-top: .32rem; }
.smog-gauge-category { color: #F4FAFF; font-weight: 700; font-size: 1rem; text-align: center; }
.smog-gauge-date { color: #95A8BB; font-size: .76rem; text-align: center; margin-top: .4rem; }
.smog-section-kicker { color: #35D6E8; font-size: .72rem; text-transform: uppercase; letter-spacing: .12em; font-weight: 720; }
.smog-health {
  padding: 1.15rem 1.3rem; border-radius: 16px;
  background: linear-gradient(110deg, rgba(16,29,48,.96), rgba(16,29,48,.76));
  border: 1px solid var(--smog-line); border-left: 3px solid var(--aqi-color, #35D6E8);
}
.smog-health-title { color: #F3FAFF; font-size: 1.04rem; font-weight: 700; }
.smog-health-copy { color: #A8B8C9; line-height: 1.6; margin-top: .35rem; }
@media (max-width: 760px) {
  .smog-header { align-items: flex-start; flex-direction: column; }
  .smog-chip-row { justify-content: flex-start; }
  .smog-hero, .smog-gauge { min-height: 230px; }
  [data-testid="stVerticalBlockBorderWrapper"]:hover { transform: none; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { scroll-behavior: auto !important; transition-duration: .01ms !important; animation-duration: .01ms !important; }
}
</style>
""")

@st.cache_data(show_spinner=False)
def load_historical_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col="date", parse_dates=True)


@st.cache_data(show_spinner=False)
def load_results(path: str) -> dict:
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


@st.cache_data(show_spinner=False)
def load_test_predictions(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


@st.cache_resource(show_spinner=False)
def load_models():
    return load_model(), load_model("backtest_model.joblib")


def file_signature(path: Path) -> tuple[str, int, int]:
    if not path.exists():
        return str(path), 0, 0
    stat = path.stat()
    return str(path), stat.st_mtime_ns, stat.st_size


@st.cache_data(ttl=60, max_entries=8, show_spinner=False)
def read_live_daily(path: str, modified_ns: int, size: int) -> pd.DataFrame:
    del modified_ns
    if size == 0:
        return pd.DataFrame()
    frame = pd.read_csv(path)
    required = {"date", *DATA_COLUMNS}
    if not required.issubset(frame.columns):
        missing = ", ".join(sorted(required - set(frame.columns)))
        raise ValueError(f"{path} is missing required columns: {missing}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any():
        raise ValueError(f"{path} contains a missing or invalid date.")
    return frame.set_index("date").sort_index()


@st.cache_data(ttl=60, max_entries=8, show_spinner=False)
def read_live_forecasts(path: str, modified_ns: int, size: int) -> pd.DataFrame:
    del modified_ns
    if size == 0:
        return pd.DataFrame()
    frame = pd.read_csv(path)
    required = {"made_on", "for_date", "forecast"}
    if not required.issubset(frame.columns):
        missing = ", ".join(sorted(required - set(frame.columns)))
        raise ValueError(f"{path} is missing required columns: {missing}")
    for column in ("made_on", "for_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
        if frame[column].isna().any():
            raise ValueError(f"{path} contains a missing or invalid {column} date.")
    if frame["forecast"].isna().any():
        raise ValueError(f"{path} contains a missing forecast value.")
    return frame.sort_values(["made_on", "for_date"])


def render_aqi_badge(value: float) -> None:
    level = aqi_level(float(value))
    color = AQI_COLORS[level]
    st.html(
        f'<span style="display:inline-flex;align-items:center;gap:.45rem;'
        f'padding:.34rem .7rem;border:1px solid {color}55;border-radius:999px;'
        f'background:{color}18;color:{color};font-size:.8rem;font-weight:700">'
        f'<span style="width:7px;height:7px;border-radius:50%;background:{color}"></span>'
        f'{LEVEL_NAMES[level]}</span>'
    )


def render_page_header(page: str, live_daily: pd.DataFrame, daily: pd.DataFrame) -> None:
    heading, description = PAGE_COPY[page]
    if not live_daily.empty and live_daily["aqi"].notna().any():
        observation_date = live_daily["aqi"].dropna().index[-1]
        age = (pd.Timestamp.now(tz="Asia/Karachi").date() - observation_date.date()).days
        freshness = (
            f"Collected estimate · {observation_date:%d %b %Y}"
            if age <= 1
            else f"Last collected {age} days ago · {observation_date:%d %b %Y}"
        )
        dot_class = ""
    else:
        observation_date = daily["aqi"].dropna().index[-1]
        freshness = f"Historical only · through {observation_date:%d %b %Y}"
        dot_class = " muted"
    st.html(
        f'<header class="smog-header"><div>'
        f'<div class="smog-header-brand">SMOGSENSE · AIR INTELLIGENCE</div>'
        f'<div class="smog-header-title">{heading}</div>'
        f'<div class="smog-header-sub">{description}</div>'
        f'</div><div class="smog-chip-row">'
        f'<span class="smog-chip"><span class="smog-chip-dot"></span>'
        f'Lahore, Pakistan</span>'
        f'<span class="smog-chip"><span class="smog-chip-dot{dot_class}"></span>'
        f'{freshness}</span></div></header>'
    )


def render_gauge(value: float, date: pd.Timestamp) -> None:
    level = aqi_level(value)
    color = AQI_COLORS[level]
    progress = min(max(float(value) / 500, 0.0), 1.0)
    st.html(
        f'<section class="smog-gauge" role="img" aria-label="Historical AQI '
        f'{value:.0f}, {LEVEL_NAMES[level]}, observed {date:%d %b %Y}">'
        '<div class="smog-gauge-title">Historical air quality</div>'
        f'<div class="smog-gauge-wrap" style="--aqi-color:{color};--aqi-progress:{progress * 360:.1f}deg">'
        '<div class="smog-gauge-ring"></div><div class="smog-gauge-center">'
        f'<div class="smog-gauge-number">{value:.0f}</div>'
        '<div class="smog-gauge-unit">AQI points</div></div></div>'
        f'<div class="smog-gauge-category" style="color:{color}">'
        f'{LEVEL_NAMES[level]}</div>'
        f'<div class="smog-gauge-date">Historical observation · {date:%d %b %Y}</div>'
        '</section>'
    )


def render_health_advisory(
    value: float,
    source: str,
    date: pd.Timestamp | None,
) -> None:
    level, category, advice = describe(float(value))
    color = AQI_COLORS[level]
    date_label = f" · {date:%d %b %Y}" if date is not None else ""
    st.html(
        f'<section class="smog-health" style="--aqi-color:{color}">'
        f'<div class="smog-section-kicker">Health guidance · {source}</div>'
        f'<div class="smog-health-title" style="margin-top:.4rem">{category}'
        f' <span style="color:#95A8BB;font-size:.82rem;font-weight:500">'
        f'{date_label}</span></div>'
        f'<div class="smog-health-copy">{advice}</div>'
        '</section>'
    )


def render_aqi_card(container, label: str, value: float, detail: str) -> None:
    with container.container(border=True):
        st.metric(label, f"{value:.0f}")
        render_aqi_badge(value)
        st.caption(detail)


def render_missing_report(path: Path, label: str) -> None:
    if not path.exists():
        st.info(f"{label} is unavailable because `{path.relative_to(ROOT)}` was not found.")


def check_required_files() -> None:
    required = (CLEAN_PATH, BEST_MODEL_PATH, BACKTEST_MODEL_PATH)
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        st.error("Required project files are missing: " + ", ".join(missing))
        st.stop()


def show_overview(
    daily: pd.DataFrame,
    latest_aqi: pd.Series,
    live_daily: pd.DataFrame,
    live_forecasts: pd.DataFrame,
) -> None:
    latest_date = latest_aqi.index[-1]
    latest_value = float(latest_aqi.iloc[-1])
    last_seven = latest_aqi.tail(7)
    hero_copy, gauge = st.columns([1.55, 1], vertical_alignment="top")
    with hero_copy:
        st.html(
            '<section class="smog-hero">'
            '<div><div class="smog-hero-kicker">Air intelligence · Lahore</div>'
            '<div class="smog-hero-title">A clearer view of the air we breathe.</div>'
            '<div class="smog-hero-copy">SmogSense turns Lahore’s historical air-quality '
            'record into an understandable next-day forecasting experiment. Explore the '
            'observations, model estimates, and their limits—all in one place.</div></div>'
            '<div class="smog-hero-foot">'
            '<span class="smog-hero-tag">Historical coverage · '
            f'{daily.index.min():%b %Y} — {daily.index.max():%b %Y}</span>'
            '<span class="smog-hero-tag">Missing AQI days remain missing</span>'
            '</div></section>'
        )
    with gauge:
        render_gauge(latest_value, latest_date)

    st.markdown("#### At a glance")
    card_a, card_b, card_c, card_d = st.columns(4)
    render_aqi_card(
        card_a,
        ":material/air:  Latest measured AQI",
        latest_value,
        f"Historical observation · {latest_date:%d %b %Y}",
    )
    with card_b:
        st.metric(
            ":material/avg_time:  7-reading average",
            f"{last_seven.mean():.1f}",
            border=True,
            help="Average of the seven latest days that have a measured historical AQI.",
        )
        st.caption("Historical observations · AQI points")
    with card_c:
        st.metric(
            ":material/dataset:  Historical AQI records",
            f"{int(latest_aqi.size):,}",
            border=True,
        )
        st.caption("Measured records · historical dataset")
    with card_d:
        st.metric(
            ":material/event_busy:  Missing AQI days",
            f"{int(daily['aqi'].isna().sum()):,}",
            border=True,
        )
        st.caption("Kept missing; AQI is never filled.")

    today = pd.Timestamp.now(tz="Asia/Karachi").date()
    tomorrow = (pd.Timestamp(today) + pd.Timedelta(days=1)).date()
    tomorrow_rows = (
        live_forecasts.loc[
            (live_forecasts["for_date"].dt.date == tomorrow)
            & (live_forecasts["made_on"].dt.date == today)
        ]
        if not live_forecasts.empty
        else pd.DataFrame()
    )
    st.markdown("#### Tomorrow’s forecast")
    if tomorrow_rows.empty:
        st.info(
            "A forecast for tomorrow is not available in the saved data. "
            "SmogSense does not extend a one-day model into unsupported multi-day predictions."
        )
        if not live_forecasts.empty:
            recent_forecast = live_forecasts.iloc[-1]
            st.caption(
                f"Most recent saved forecast: {recent_forecast['forecast']:.0f} AQI "
                f"for {recent_forecast['for_date']:%d %b %Y}, made from "
                f"{recent_forecast['made_on']:%d %b %Y} collected input."
            )
        else:
            st.caption("There are no saved forecasts in the collected-data log.")
    else:
        forecast = tomorrow_rows.iloc[-1]
        forecast_col, context_col = st.columns([1, 2])
        render_aqi_card(
            forecast_col,
            f"Model estimate · {tomorrow:%d %b %Y}",
            float(forecast["forecast"]),
            f"Generated from collected input dated {forecast['made_on']:%d %b %Y}.",
        )
        with context_col:
            st.caption(
                "This is a one-day-ahead Open-Meteo based model estimate—not a "
                "monitor reading or official air-quality alert."
            )

    st.markdown("#### Historical AQI trend")
    periods = {"Last 90 days": 90, "Last 365 days": 365, "Full dataset": None}
    window = st.selectbox(
        "Choose the historical time window",
        options=list(periods),
        index=1,
        key="overview_trend_window",
    )
    trend = daily.loc[:, ["aqi"]].rename(columns={"aqi": "Historical measured AQI"})
    if periods[window] is not None:
        trend = trend.tail(periods[window])
    chart_data = trend.rename_axis("Observation date").reset_index()
    chart = (
        alt.Chart(chart_data)
        .mark_line(color="#35D6E8", strokeWidth=2, point=alt.OverlayMarkDef(size=14))
        .encode(
            x=alt.X("Observation date:T", title=None, axis=alt.Axis(format="%b %Y", labelColor="#91A6BA", grid=False)),
            y=alt.Y("Historical measured AQI:Q", title="AQI points", scale=alt.Scale(zero=False), axis=alt.Axis(labelColor="#91A6BA", gridColor="#23364B")),
            tooltip=[
                alt.Tooltip("Observation date:T", title="Observation date", format="%d %b %Y"),
                alt.Tooltip("Historical measured AQI:Q", title="Historical AQI", format=".1f"),
            ],
        )
        .properties(height=290, background="transparent")
        .configure_view(strokeOpacity=0)
        .configure_axis(domainColor="#31455B", tickColor="#31455B", titleColor="#A8B8C9")
    )
    st.altair_chart(
        chart,
        alt="Historical measured AQI trend for Lahore. Missing AQI days remain visible as gaps.",
    )
    st.caption(
        f"Historical dataset · {daily.index.min():%d %b %Y} to "
        f"{daily.index.max():%d %b %Y}. Missing AQI days stay missing; collected "
        "estimates and model forecasts are not plotted as observations."
    )

    if not live_daily.empty and live_daily["aqi"].notna().any():
        advisory_date = live_daily["aqi"].dropna().index[-1]
        advisory_value = float(live_daily.loc[advisory_date, "aqi"])
        advisory_source = "collected Open-Meteo estimate"
    else:
        advisory_date = latest_date
        advisory_value = latest_value
        advisory_source = "historical observation"
    st.markdown("#### Health context")
    render_health_advisory(advisory_value, advisory_source, advisory_date)
    st.caption("General educational guidance only—not an official alert or medical advice.")


def show_historical_backtest(
    daily: pd.DataFrame,
    features: pd.DataFrame,
    backtest_model,
    predictions: pd.DataFrame,
    results: dict,
) -> None:
    st.markdown("#### Held-out day explorer")
    st.write(
        "Choose a feature date in the held-out 2023 test period. The model uses "
        "information available on that date to estimate the following day; the "
        "recorded AQI is shown separately for comparison."
    )
    if predictions.empty:
        st.info("Historical backtest unavailable: the saved test predictions file is empty.")
        render_missing_report(PREDICTIONS_PATH, "Saved test predictions")
        return

    missing = PREDICTION_COLUMNS - set(predictions.columns)
    if missing:
        st.error(
            "The saved test predictions file is missing required columns: "
            + ", ".join(sorted(missing))
        )
        return

    chosen_model = results.get("chosen_model", "RandomForest")
    if chosen_model not in predictions.columns:
        st.error(
            f"The results report names `{chosen_model}`, but that prediction column "
            "is missing from the saved test predictions."
        )
        return

    dates = predictions["date"].dropna().sort_values().tolist()
    if not dates:
        st.info("Historical backtest unavailable: the predictions file has no dated rows.")
        return
    feature_day = st.select_slider(
        "Select a 2023 test feature date",
        options=dates,
        value=dates[-1],
        format_func=lambda day: pd.Timestamp(day).strftime("%d %b %Y"),
    )
    record = predictions.loc[predictions["date"] == feature_day].iloc[0]
    feature_day = pd.Timestamp(feature_day)
    if feature_day not in features.index:
        st.error(f"No historical features are available for {feature_day:%d %b %Y}.")
        return

    model_output = predict_row(backtest_model, features.loc[[feature_day], FEATURES])
    saved_output = float(record[chosen_model])
    if not np.isclose(model_output, saved_output, rtol=1e-5, atol=0.05):
        st.warning(
            "The loaded backtest model does not exactly match the saved test report. "
            "The displayed prediction uses the preserved report value."
        )
        model_output = saved_output
    actual = float(record["actual_tomorrow"])
    target_day = feature_day + pd.Timedelta(days=1)
    forecast_col, actual_col, baseline_col = st.columns(3)
    render_aqi_card(
        forecast_col,
        f"Model estimate for {target_day:%d %b %Y}",
        model_output,
        f"One-day-ahead prediction from the held-out test model ({chosen_model}).",
    )
    render_aqi_card(
        actual_col,
        f"Recorded AQI for {target_day:%d %b %Y}",
        actual,
        "Historical measured value from the test dataset; not a prediction.",
    )
    render_aqi_card(
        baseline_col,
        "Simple baseline estimate",
        float(record["Tomorrow = today"]),
        "Baseline assumes tomorrow's AQI equals the input day's AQI.",
    )
    st.metric(
        "Absolute error on this selected day",
        f"{abs(model_output - actual):.1f} AQI points",
        border=True,
    )
    st.caption(
        "A single selected day is an example, not a measure of overall performance. "
        "Use Model performance for the full held-out test results."
    )


def show_forecast(live_forecasts: pd.DataFrame) -> None:
    today = pd.Timestamp.now(tz="Asia/Karachi").date()
    tomorrow = (pd.Timestamp(today) + pd.Timedelta(days=1)).date()
    st.markdown("#### Forecast availability")
    tomorrow_rows = (
        live_forecasts.loc[
            (live_forecasts["for_date"].dt.date == tomorrow)
            & (live_forecasts["made_on"].dt.date == today)
        ]
        if not live_forecasts.empty
        else pd.DataFrame()
    )
    if tomorrow_rows.empty:
        st.info(
            f"No saved one-day-ahead forecast is available for tomorrow "
            f"({tomorrow:%d %b %Y}). The model needs a current complete input "
            "day and the forecast log does not contain one."
        )
        st.caption(
            "The project supports a one-day-ahead prediction only. It does not "
            "forecast five days, and this dashboard will not fill that gap with "
            "extrapolated model outputs."
        )
    else:
        forecast = tomorrow_rows.iloc[-1]
        col, context = st.columns([1, 1.5])
        render_aqi_card(
            col,
            f"Model estimate · {tomorrow:%d %b %Y}",
            float(forecast["forecast"]),
            f"Generated from collected input dated {forecast['made_on']:%d %b %Y}.",
        )
        with context:
            st.markdown("**What this estimate means**")
            st.write(
                "The saved model estimates next-day AQI from the previous complete "
                "day's collected AQI and weather features. It is not a measurement "
                "and is not guaranteed to match actual conditions."
            )

    st.markdown("#### Latest saved forecast record")
    if live_forecasts.empty:
        st.info(
            "No forecasts have been saved yet. The daily collector has not "
            "written a forecast record to `data/live/forecast_log.csv`."
        )
        return
    latest = live_forecasts.iloc[-1]
    latest_target = latest["for_date"].date()
    latest_made = latest["made_on"].date()
    previous_col, note_col = st.columns([1, 1.5])
    render_aqi_card(
        previous_col,
        f"Saved estimate · {latest_target:%d %b %Y}",
        float(latest["forecast"]),
        f"Made on {latest_made:%d %b %Y} · Open-Meteo based input",
    )
    with note_col:
        if latest_target < today:
            st.warning(
                f"This saved estimate is for {latest_target:%d %b %Y}; it is a "
                "historical forecast, not a current prediction."
            )
        elif latest_target == today:
            st.info(
                "This saved estimate targets today. It is not a forecast for tomorrow."
            )
        elif latest_target == tomorrow:
            st.success("This saved estimate targets tomorrow.")
        else:
            st.info(
                "This saved estimate has a different target date; it is not "
                "presented as a forecast for tomorrow."
            )
        st.caption(
            "Source: Open-Meteo computer-model estimates. These are separate "
            "from historical monitor records."
        )


def show_historical_trends(
    daily: pd.DataFrame,
    features: pd.DataFrame,
    backtest_model,
    predictions: pd.DataFrame,
    results: dict,
) -> None:
    measured = daily["aqi"].dropna()
    latest = measured.iloc[-1]
    latest_date = measured.index[-1]
    last_month = measured.loc[measured.index >= measured.index.max() - pd.Timedelta(days=30)]
    average_col, latest_col, coverage_col = st.columns(3)
    average_col.metric(
        "Latest 30-day measured average",
        f"{last_month.mean():.1f}",
        border=True,
        help="Average over available measured AQI values in the last 30 calendar days.",
    )
    latest_col.metric(
        "Latest measured historical AQI",
        f"{latest:.0f}",
        border=True,
        help=f"Historical observation from {latest_date:%d %b %Y}; not a current reading.",
    )
    coverage_col.metric(
        "Historical coverage",
        f"{daily.index.min():%Y}–{daily.index.max():%Y}",
        border=True,
    )

    periods = {"90 days": 90, "1 year": 365, "Full record": None}
    window = st.selectbox(
        "Historical period",
        list(periods),
        index=1,
        key="historical_trend_window",
    )
    trend = daily[["aqi"]].rename(columns={"aqi": "Measured AQI"})
    if periods[window] is not None:
        trend = trend.tail(periods[window])
    data = trend.rename_axis("Observation date").reset_index()
    chart = (
        alt.Chart(data)
        .mark_line(color="#35D6E8", strokeWidth=2)
        .encode(
            x=alt.X("Observation date:T", title="Observation date", axis=alt.Axis(format="%b %Y", labelColor="#91A6BA", grid=False)),
            y=alt.Y("Measured AQI:Q", title="Measured AQI (points)", scale=alt.Scale(zero=False), axis=alt.Axis(labelColor="#91A6BA", gridColor="#23364B")),
            tooltip=[
                alt.Tooltip("Observation date:T", title="Date", format="%d %b %Y"),
                alt.Tooltip("Measured AQI:Q", title="Historical AQI", format=".1f"),
            ],
        )
        .properties(height=350, background="transparent")
        .configure_view(strokeOpacity=0)
        .configure_axis(domainColor="#31455B", tickColor="#31455B", titleColor="#A8B8C9")
    )
    st.altair_chart(
        chart,
        alt="Historical measured AQI over time; missing AQI days appear as gaps.",
    )
    st.caption(
        f"Source: historical project dataset · {daily.index.min():%d %b %Y} to "
        f"{daily.index.max():%d %b %Y}. Missing values are not interpolated for AQI."
    )
    show_historical_backtest(daily, features, backtest_model, predictions, results)


def show_what_if(model) -> None:
    st.markdown("#### Build a scenario")
    st.write(
        "Change the example inputs to see how the trained model responds. These "
        "values are user-entered scenarios, not live readings or a physical "
        "simulation of emissions or weather."
    )
    left, right = st.columns(2)
    with left:
        st.markdown("**:material/air:  Recent AQI conditions**")
        month = st.selectbox(
            "Example input month",
            options=list(range(1, 13)),
            index=11,
            format_func=lambda value: pd.Timestamp(2023, value, 1).strftime("%B"),
            help="The month sets the seasonal feature. All scenario input dates use the same example year.",
        )
        aqi_today = st.slider("AQI on the input day", 0, 500, 250, help="Example AQI value for the day used as model input.")
        aqi_yesterday = st.slider("AQI one day earlier", 0, 500, 240, help="Previous day's AQI scenario.")
        aqi_two_days_ago = st.slider("AQI two days earlier", 0, 500, 230, help="AQI scenario from two days before the input day.")
        aqi_mean7 = st.slider("Average AQI over the previous 7 days", 0.0, 500.0, 240.0, help="Enter the seven-day average independently of the three recent AQI inputs.")
    with right:
        st.markdown("**:material/device_thermostat:  Weather conditions**")
        temp_c = st.slider("Temperature (°C)", -5.0, 50.0, 20.0, help="Scenario temperature in degrees Celsius.")
        humidity = st.slider("Relative humidity (%)", 0, 100, 70, help="Scenario relative humidity.")
        dew_point_c = st.slider("Dew point (°C)", -10.0, 35.0, 12.0, help="Scenario dew-point temperature.")
        wind_kmh = st.slider("Wind speed (km/h)", 0.0, 80.0, 5.0, help="Scenario daily wind speed.")
        pressure_in = st.slider("Air pressure (inHg)", 27.0, 32.0, 29.0, help="Scenario atmospheric pressure.")

    row = row_from_inputs(
        f"2023-{month:02d}-15",
        aqi_today,
        aqi_yesterday,
        aqi_two_days_ago,
        aqi_mean7,
        temp_c,
        humidity,
        dew_point_c,
        wind_kmh,
        pressure_in,
    )
    prediction = predict_row(model, row)
    st.markdown("#### Scenario result · next-day estimate")
    render_aqi_card(
        st,
        ":material/auto_awesome:  What-if next-day AQI estimate",
        prediction,
        "This output uses only the scenario values above; it is not a measured AQI.",
    )
    _, category, advice = describe(prediction)
    render_health_advisory(prediction, "modelled scenario", None)
    st.caption(
        f"Scenario category: {category}. {advice} This is general context, not "
        "medical advice. The scenario is not a physical simulation and its result "
        "is not guaranteed to match the actual AQI."
    )


def show_performance(results: dict) -> None:
    if not results:
        st.info("Historical model results are unavailable.")
        render_missing_report(RESULTS_PATH, "Model results report")
        return

    train_period = results["train_period"]
    test_period = results["test_period"]
    st.write(
        f"Training feature dates: **{train_period[0]} to {train_period[1]}**. "
        f"Held-out test dates: **{test_period[0]} to {test_period[1]}** "
        f"({results['rows']['test']} evaluated days). The model was selected on "
        "validation data, not on the test set."
    )
    st.info(
        "**MAE** means mean absolute error: the average distance between a "
        "prediction and the recorded AQI, measured in AQI points. Lower is closer "
        "on average; it is not a percentage accuracy score."
    )
    model_names = (
        "Tomorrow = today",
        "Ridge",
        "RandomForest",
        "HistGradientBoosting",
    )
    comparison = pd.DataFrame(
        {
            "All test days — MAE (AQI points)": [
                results["test_results"][name]["all_test_days"]["mae"]
                for name in model_names
            ],
            "Winter test days — MAE (AQI points)": [
                results["test_results"][name]["winter_days"]["mae"]
                for name in model_names
            ],
            "Exact AQI category (%)": [
                100 * results["test_results"][name]["all_test_days"]["exact_level"]
                for name in model_names
            ],
            "Within one category (%)": [
                100 * results["test_results"][name]["all_test_days"]["within_one_level"]
                for name in model_names
            ],
        },
        index=model_names,
    ).round(1)
    chart_data = (
        comparison[["All test days — MAE (AQI points)", "Winter test days — MAE (AQI points)"]]
        .rename_axis("Model")
        .reset_index()
        .melt(id_vars="Model", var_name="Test group", value_name="MAE")
    )
    comparison_chart = (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("MAE:Q", title="Mean absolute error (AQI points)", axis=alt.Axis(labelColor="#91A6BA", gridColor="#23364B")),
            y=alt.Y("Model:N", sort=model_names, title=None, axis=alt.Axis(labelColor="#B8C8D8")),
            yOffset="Test group:N",
            color=alt.Color(
                "Test group:N",
                scale=alt.Scale(
                    domain=["All test days — MAE (AQI points)", "Winter test days — MAE (AQI points)"],
                    range=["#35D6E8", "#28D7A0"],
                ),
                legend=alt.Legend(title="Held-out group", labelColor="#A8B8C9", titleColor="#A8B8C9"),
            ),
            tooltip=[
                alt.Tooltip("Model:N", title="Model"),
                alt.Tooltip("Test group:N", title="Test group"),
                alt.Tooltip("MAE:Q", title="MAE (AQI points)", format=".1f"),
            ],
        )
        .properties(height=230, background="transparent")
        .configure_view(strokeOpacity=0)
        .configure_axis(domainColor="#31455B", tickColor="#31455B")
    )
    st.altair_chart(
        comparison_chart,
        alt="Mean absolute error for four approaches on all and winter 2023 test days.",
    )
    st.dataframe(comparison, alt="Held-out 2023 MAE and AQI category comparison by model.")
    st.caption(
        f"Winter results cover {results['rows']['test_winter']} test days in "
        "January, February, and November 2023. The historical dataset ends "
        "30 November 2023, so December is not included."
    )

    chosen = results["chosen_model"]
    confidence_interval = results["improvement_over_baseline_95ci_mae_points"][chosen]
    st.subheader(f"Selected model: {chosen}")
    st.write(
        "The selected model was chosen on the 2022 validation year and evaluated "
        "once on the 2023 test period. The interval below compares its MAE with "
        "the simple 'tomorrow equals today' baseline."
    )
    st.dataframe(
        pd.DataFrame(
            {
                "95% interval for baseline MAE minus model MAE (AQI points)": [
                    f"{confidence_interval['all'][0]:.1f} to "
                    f"{confidence_interval['all'][1]:.1f}",
                    f"{confidence_interval['winter'][0]:.1f} to "
                    f"{confidence_interval['winter'][1]:.1f}",
                ]
            },
            index=["All test days", "Winter test days"],
        ),
        alt="Bootstrap intervals for the selected model's MAE difference from baseline.",
    )
    st.caption(
        "Positive values favor the model; if an interval includes zero, these "
        "results do not clearly establish an improvement for that group. This "
        "evaluation is limited to the available historical test period."
    )
    if TEST_PLOT_PATH.is_file():
        st.image(
            str(TEST_PLOT_PATH),
            caption="Recorded AQI and model estimates during the held-out 2023 test period.",
            alt="Line chart comparing actual AQI, the tomorrow-equals-today baseline, and the selected model in 2023.",
        )
    else:
        st.info("The saved test forecast chart is unavailable.")


def show_insights(results: dict) -> None:
    st.write(
        "Feature importance describes how much the model's test error changed "
        "when an input column was shuffled. It can summarize patterns the model "
        "used, but it cannot show that an input caused pollution or a forecast."
    )
    if not results:
        st.info("Feature-importance results are unavailable without the model report.")
        return

    importance = results.get("permutation_importance_mae_increase", {})
    if importance:
        importance_table = pd.DataFrame(
            [
                {
                    "Input": FEATURE_LABELS.get(name, name),
                    "Test MAE increase when shuffled (AQI points)": value,
                    "Technical feature": name,
                }
                for name, value in importance.items()
            ]
        ).sort_values(
            "Test MAE increase when shuffled (AQI points)",
            ascending=False,
        )
        importance_chart = (
            alt.Chart(importance_table)
            .mark_bar(cornerRadiusEnd=4, color="#35D6E8")
            .encode(
                x=alt.X(
                    "Test MAE increase when shuffled (AQI points):Q",
                    title="Change in test MAE (AQI points)",
                    axis=alt.Axis(labelColor="#91A6BA", gridColor="#23364B"),
                ),
                y=alt.Y("Input:N", sort="-x", title=None, axis=alt.Axis(labelColor="#B8C8D8")),
                tooltip=[
                    alt.Tooltip("Input:N", title="Input"),
                    alt.Tooltip(
                        "Test MAE increase when shuffled (AQI points):Q",
                        title="MAE change (AQI points)",
                        format=".3f",
                    ),
                ],
            )
            .properties(height=350, background="transparent")
            .configure_view(strokeOpacity=0)
            .configure_axis(domainColor="#31455B", tickColor="#31455B")
        )
        st.altair_chart(
            importance_chart,
            alt="Permutation importance showing test-error change for each input feature.",
        )
        st.dataframe(
            importance_table.round(3),
            hide_index=True,
            alt="Permutation feature importance values for the held-out test period.",
        )
    render_missing_report(IMPORTANCE_PATH, "Original feature-importance report image")

    st.subheader("How to interpret the inputs")
    st.markdown(
        "- Recent AQI values summarize recent pollution patterns and are model "
        "inputs, not pollution sources.\n"
        "- Weather variables describe the recorded conditions associated with "
        "the input day.\n"
        "- A small or negative importance value can mean the input added little "
        "measurable value in this particular test run.\n"
        "- The dataset does not directly measure traffic, factories, brick kilns, "
        "or crop burning. Importance is not proof of cause."
    )
    with st.expander("Technical details"):
        st.write(f"Selected model: {results.get('chosen_model', 'not reported')}")
        st.write("Importance method: permutation importance using test-set MAE.")
        st.write("Model inputs:")
        st.code(", ".join(results.get("features", FEATURES)))


def show_collected_data(
    live_daily: pd.DataFrame,
    live_forecasts: pd.DataFrame,
    today: pd.Timestamp,
) -> None:
    st.write(
        "This section shows records written by the daily collector. Its AQI comes "
        "from Open-Meteo's CAMS-based computer model, not the historical monitor "
        "source. Collected records and historical dataset records are not merged."
    )
    if live_daily.empty:
        st.info(
            "Live data unavailable — no collected daily readings are present in "
            "`data/live/daily_log.csv`."
        )
    else:
        latest_date = live_daily.index[-1]
        age_days = (today.normalize() - latest_date.normalize()).days
        latest_row = live_daily.iloc[-1]
        st.warning(
            f"Latest complete collected day: **{latest_date:%d %b %Y}** "
            f"({age_days} day(s) before today's Lahore date). These are modelled "
            "air-quality estimates, not official monitoring-station readings."
        )
        if pd.notna(latest_row["aqi"]):
            render_aqi_card(
                st,
                f"Latest collected modeled AQI — {latest_date:%d %b %Y}",
                float(latest_row["aqi"]),
                "Open-Meteo estimate; separate from the historical training dataset.",
            )
        collected_chart_data = (
            live_daily.loc[:, ["aqi"]]
            .rename(columns={"aqi": "Collected model estimate"})
            .rename_axis("Collected observation date")
            .reset_index()
        )
        collected_chart = (
            alt.Chart(collected_chart_data)
            .mark_line(color="#28D7A0", strokeWidth=2, point=True)
            .encode(
                x=alt.X("Collected observation date:T", title=None, axis=alt.Axis(format="%d %b", labelColor="#91A6BA", grid=False)),
                y=alt.Y("Collected model estimate:Q", title="Modelled AQI (points)", scale=alt.Scale(zero=False), axis=alt.Axis(labelColor="#91A6BA", gridColor="#23364B")),
                tooltip=[
                    alt.Tooltip("Collected observation date:T", title="Collected date", format="%d %b %Y"),
                    alt.Tooltip("Collected model estimate:Q", title="Open-Meteo AQI estimate", format=".1f"),
                ],
            )
            .properties(height=270, background="transparent")
            .configure_view(strokeOpacity=0)
            .configure_axis(domainColor="#31455B", tickColor="#31455B")
        )
        st.altair_chart(
            collected_chart,
            alt="Collected Open-Meteo daily modelled AQI estimates, separate from historical monitor records.",
        )
        display = live_daily.tail(10).sort_index(ascending=False).round(2)
        display.insert(
            1,
            "AQI category",
            [
                describe(float(value))[1] if pd.notna(value) else "Unavailable"
                for value in display["aqi"]
            ],
        )
        st.dataframe(
            display,
            alt="Recent collected daily weather and modelled AQI records.",
        )

    st.subheader("Saved forecasts")
    if live_forecasts.empty:
        st.info(
            "Live forecast unavailable — no saved forecasts are present in "
            "`data/live/forecast_log.csv`."
        )
    else:
        latest_forecast = live_forecasts.iloc[-1]
        target_date = latest_forecast["for_date"].date()
        if target_date < today.date():
            st.info(
                f"The latest saved one-day-ahead estimate was for "
                f"{target_date:%d %b %Y}; it is now a historical forecast, not a "
                "forecast for tomorrow."
            )
        elif target_date == today.date():
            st.info(
                "The latest saved estimate is for today. A forecast for tomorrow "
                "is not available in this saved log."
            )
        st.dataframe(
            live_forecasts.sort_values("made_on", ascending=False).head(10),
            hide_index=True,
            alt="Recent saved one-day-ahead forecasts, with input and target dates.",
        )
        render_aqi_card(
            st,
            f"Latest saved estimate for {target_date:%d %b %Y}",
            float(latest_forecast["forecast"]),
            f"Generated on {latest_forecast['made_on']:%d %b %Y}.",
        )

        if not live_daily.empty:
            actual_values = live_daily["aqi"].rename("actual")
            scored = live_forecasts.merge(
                actual_values,
                left_on="for_date",
                right_index=True,
                how="inner",
            )
            scored["baseline"] = scored["made_on"].map(actual_values)
            scored = scored.dropna(subset=["forecast", "actual", "baseline"])
            if scored.empty:
                st.caption(
                    "No saved forecast target has a matching collected AQI and "
                    "baseline input value yet, so live MAE is unavailable."
                )
            else:
                st.subheader("Separate collected-data score")
                score_a, score_b, score_c = st.columns(3)
                score_a.metric("Matched forecast days", f"{len(scored)}")
                score_b.metric(
                    "Forecast MAE (AQI points)",
                    f"{(scored['forecast'] - scored['actual']).abs().mean():.1f}",
                )
                score_c.metric(
                    "Tomorrow-equals-today MAE",
                    f"{(scored['baseline'] - scored['actual']).abs().mean():.1f}",
                )
                st.caption(
                    "These scores use the collected Open-Meteo series only and "
                    "must not be compared as if they were the historical monitor "
                    "test results."
                )


def show_ideas_and_about(daily: pd.DataFrame, results: dict) -> None:
    st.markdown("#### Project overview")
    st.write(
        "SmogSense is an educational project that forecasts next-day Lahore AQI "
        "from recent AQI and daily weather features. Its historical records come "
        "from the project's Kaggle Lahore daily AQI-and-weather dataset and cover "
        f"**{daily.index.min():%d %B %Y} to {daily.index.max():%d %B %Y}**; "
        "missing AQI values remain missing rather than being invented."
    )
    st.write(
        "The separate daily collector records Open-Meteo weather and CAMS-based "
        "computer-model AQI estimates. These estimates are not measurements from "
        "the historical dataset's monitoring source."
    )
    with st.container(border=True):
        st.markdown(
            "**What the model predicts**\n\n"
            "One day ahead, using the input day's AQI and weather plus recent AQI "
            "history and calendar features. The model does not produce a supported "
            "five-day forecast, and the what-if simulator is not a physical simulation."
        )
    if results:
        st.write(
            f"The model was selected using {results['chosen_on']} and evaluated "
            f"on the held-out period {results['test_period'][0]} to "
            f"{results['test_period'][1]}."
        )
    st.markdown(
        "- AQI values in the historical section come from the project dataset.\n"
        "- A forecast is a model estimate for a target date, not a measured value "
        "or an official alert.\n"
        "- Predictions are not guaranteed to match actual air quality. The "
        "training history is limited, contains missing AQI days, and omits direct "
        "emissions measurements and other important factors.\n"
        "- Open-Meteo collected data is a different, modelled source and is "
        "displayed separately."
    )

    st.subheader("Ideas discussed for reducing smog")
    st.write(
        "The following are ideas described in the project's cited reporting. "
        "They are not findings produced by the forecasting model."
    )
    with st.container(border=True):
        st.markdown(
            "**Potential sources discussed**\n\n"
            "Vehicles, industry, brick kilns, waste and crop-residue burning, and "
            "power plants. Published estimates differ, and the Punjab EPA has "
            "disputed at least one estimate."
        )
    with st.container(border=True):
        st.markdown(
            "**Measures discussed**\n\n"
            "- Cleaner fuels and electric two- and three-wheelers\n"
            "- Modernizing brick kilns with zigzag technology\n"
            "- Enforcing rules for smoky vehicles\n"
            "- Improving air-quality monitoring, including in rural areas\n"
            "- Better crop-residue practices and preventing waste burning"
        )
    st.caption(
        "Sources listed in the original project: Dawn (2019 and 2023 reports on "
        "Punjab emission shares), Pakistan Air Quality Initiative study as "
        "reported by Profit, Foreign Policy (2021), and Aaj News (December 2022)."
    )
    st.caption(
        f"Historical dataset coverage: {daily.index.min():%d %b %Y} to "
        f"{daily.index.max():%d %b %Y}. SmogSense is educational software, not "
        "an official air-quality service or a substitute for medical advice."
    )


check_required_files()

try:
    daily = load_historical_data(str(CLEAN_PATH))
    model, backtest_model = load_models()
    live_daily = read_live_daily(*file_signature(LIVE_DAILY_PATH))
    live_forecasts = read_live_forecasts(*file_signature(LIVE_FORECAST_PATH))
except (OSError, ValueError, KeyError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
    st.error(f"SmogSense could not load a project data or model file: {error}")
    st.stop()

if daily.empty:
    st.error(f"The historical dataset is empty: {CLEAN_PATH.relative_to(ROOT)}.")
    st.stop()
missing_data_columns = DATA_COLUMNS - set(daily.columns)
if missing_data_columns:
    st.error(
        "The historical dataset is missing required columns: "
        + ", ".join(sorted(missing_data_columns))
    )
    st.stop()
if daily.index.isna().any():
    st.error("The historical dataset contains missing or invalid dates.")
    st.stop()
daily = daily.sort_index()
if daily["aqi"].dropna().empty:
    st.error("The historical dataset has no measured AQI values.")
    st.stop()

if not live_daily.empty and live_daily.index.isna().any():
    st.error("The collected daily log contains missing or invalid dates.")
    st.stop()
if not live_forecasts.empty and not np.isfinite(live_forecasts["forecast"]).all():
    st.error("The collected forecast log contains a non-finite forecast value.")
    st.stop()

features = add_features(daily)
historical_aqi = daily["aqi"].dropna()
today = pd.Timestamp.now(tz="Asia/Karachi").normalize().tz_localize(None)

with st.sidebar:
    st.markdown("### :material/eco: SmogSense")
    st.caption("Air intelligence · Lahore")
    st.markdown("**Explore**")
    selected = st.radio(
        "Dashboard navigation",
        options=[label for label, _ in PAGE_OPTIONS],
        format_func=lambda label: label,
        label_visibility="collapsed",
        key="dashboard_nav",
    )
    page = dict((label, name) for label, name in PAGE_OPTIONS)[selected]
    st.space("small")
    st.caption("HISTORICAL DATA COVERAGE")
    st.markdown(
        f"**{daily.index.min():%d %b %Y}**  \nthrough  \n"
        f"**{daily.index.max():%d %b %Y}**"
    )
    st.caption(
        f"{int(historical_aqi.size):,} measured AQI days · "
        f"{int(daily['aqi'].isna().sum()):,} missing AQI days"
    )
    st.caption("Historical monitor records are not live readings.")

render_page_header(page, live_daily, daily)

if page == "Overview":
    show_overview(daily, historical_aqi, live_daily, live_forecasts)
elif page == "AQI forecast":
    show_forecast(live_forecasts)
elif page == "Historical trends":
    predictions = (
        load_test_predictions(str(PREDICTIONS_PATH))
        if PREDICTIONS_PATH.is_file()
        else pd.DataFrame()
    )
    results = load_results(str(RESULTS_PATH)) if RESULTS_PATH.is_file() else {}
    show_historical_trends(daily, features, backtest_model, predictions, results)
elif page == "What-if simulator":
    show_what_if(model)
elif page == "Model performance":
    results = load_results(str(RESULTS_PATH)) if RESULTS_PATH.is_file() else {}
    show_performance(results)
elif page == "Model insights":
    results = load_results(str(RESULTS_PATH)) if RESULTS_PATH.is_file() else {}
    show_insights(results)
elif page == "Collected data":
    show_collected_data(live_daily, live_forecasts, today)
elif page == "About SmogSense":
    results = load_results(str(RESULTS_PATH)) if RESULTS_PATH.is_file() else {}
    show_ideas_and_about(daily, results)

st.divider()
st.caption(
    "SmogSense is an educational forecasting project. Predictions are estimates, "
    "not official air-quality alerts or medical advice."
)
