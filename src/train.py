"""Step 3: compare models against the 'tomorrow = today' baseline.

- time-based split (never random)
- the best model is chosen on a validation year (2022), NOT on the test set
- the test set (2023) is touched once, at the end
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURES, LEVEL_NAMES, add_features, aqi_level

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "processed" / "lahore_daily_clean.csv"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
SEED = 42

TRAIN_END = "2022-12-30"   # last feature-date used for training
TEST_START = "2023-01-01"
VAL_START, VAL_END = "2022-01-01", "2022-12-30"
TRAIN_FOR_VAL_END = "2021-12-30"
WINTER_MONTHS = [11, 12, 1, 2]


def build_models():
    return {
        "Ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0)),
        "RandomForest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=SEED, n_jobs=-1)),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=250, min_samples_leaf=20, random_state=SEED),
    }


def level_scores(y_true, y_pred):
    t = np.array([aqi_level(v) for v in y_true])
    p = np.array([aqi_level(v) for v in y_pred])
    return {"exact_level": float((t == p).mean()), "within_one_level": float((abs(t - p) <= 1).mean())}


def score(y_true, y_pred):
    return {"mae": float(mean_absolute_error(y_true, y_pred)), **level_scores(y_true, y_pred)}


def paired_bootstrap(err_base, err_model, n=2000):
    """95% interval for (baseline MAE - model MAE). > 0 means model is better."""
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(err_base), (n, len(err_base)))
    diffs = err_base[idx].mean(axis=1) - err_model[idx].mean(axis=1)
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def main():
    daily = pd.read_csv(CLEAN, index_col="date", parse_dates=True)
    df = add_features(daily)
    df = df.dropna(subset=["aqi", "target"])          # need today's AQI and tomorrow's AQI
    X, y = df[FEATURES], df["target"]

    train = df.index <= TRAIN_END
    test = df.index >= TEST_START
    print(f"train rows {train.sum()}  test rows {test.sum()}")

    # ---- choose the model on the validation year ----
    sub_tr = df.index <= TRAIN_FOR_VAL_END
    val = (df.index >= VAL_START) & (df.index <= VAL_END)
    val_mae = {}
    for name, model in build_models().items():
        model.fit(X[sub_tr], y[sub_tr])
        val_mae[name] = float(mean_absolute_error(y[val], model.predict(X[val])))
    val_base = float(mean_absolute_error(y[val], X.loc[val, "aqi"]))
    best_name = min(val_mae, key=val_mae.get)
    print("validation MAE:", {k: round(v, 1) for k, v in val_mae.items()}, "baseline", round(val_base, 1))
    print("chosen on validation:", best_name)

    # ---- final fit on all training years, evaluate once on test ----
    results, preds = {}, {}
    base_pred = X.loc[test, "aqi"].values
    preds["Tomorrow = today"] = base_pred
    models = build_models()
    for name, model in models.items():
        model.fit(X[train], y[train])
        preds[name] = model.predict(X[test])

    y_test = y[test].values
    winter = np.isin(df.index[test].month, WINTER_MONTHS)
    for name, p in preds.items():
        results[name] = {"all_test_days": score(y_test, p),
                         "winter_days": score(y_test[winter], p[winter])}

    err_base = np.abs(y_test - base_pred)
    ci = {}
    for name in models:
        err_m = np.abs(y_test - preds[name])
        ci[name] = {"all": paired_bootstrap(err_base, err_m),
                    "winter": paired_bootstrap(err_base[winter], err_m[winter])}

    # ---- what the chosen model relied on (permutation importance on test) ----
    best = models[best_name]
    imp = permutation_importance(best, X[test], y[test], scoring="neg_mean_absolute_error",
                                 n_repeats=20, random_state=SEED)
    importance = pd.Series(imp.importances_mean, index=FEATURES).sort_values(ascending=False)

    meta = {
        "chosen_model": best_name,
        "chosen_on": "validation year 2022 (not on the test set)",
        "features": FEATURES,
        "train_period": [str(df.index[train].min().date()), str(df.index[train].max().date())],
        "test_period": [str(df.index[test].min().date()), str(df.index[test].max().date())],
        "rows": {"train": int(train.sum()), "test": int(test.sum()), "test_winter": int(winter.sum())},
        "validation_mae": val_mae, "validation_baseline_mae": val_base,
        "test_results": results,
        "improvement_over_baseline_95ci_mae_points": ci,
        "permutation_importance_mae_increase": importance.round(3).to_dict(),
        "level_names": LEVEL_NAMES,
        "note_winter": "Test winter days are Jan, Feb, Nov 2023 only (data ends 30 Nov 2023).",
    }
    MODELS.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True)
    # backtest model: trained on the training years only, so the app can show
    # honest 2023 backtests. final model: refit on everything for live use.
    joblib.dump(models[best_name], MODELS / "backtest_model.joblib")
    final = build_models()[best_name].fit(X, y)
    joblib.dump(final, MODELS / "best_model.joblib")
    (REPORTS / "results.json").write_text(json.dumps(meta, indent=2))

    out = pd.DataFrame({"actual_tomorrow": y_test}, index=df.index[test])
    for k, v in preds.items():
        out[k] = v
    out.to_csv(REPORTS / "test_predictions.csv")

    # ---- figures ----
    fig, ax = plt.subplots(figsize=(11, 4))
    out = out.reindex(pd.date_range(out.index.min(), out.index.max()))  # gaps stay gaps in the plot
    out["actual_tomorrow"].plot(ax=ax, color="black", lw=1.2, label="Actual")
    out["Tomorrow = today"].plot(ax=ax, color="grey", lw=0.8, alpha=0.7, label="Tomorrow = today")
    out[best_name].plot(ax=ax, color="tab:red", lw=1.2, label=best_name)
    ax.set_ylabel("AQI"); ax.set_title("2023 test period: actual vs forecast"); ax.legend()
    fig.tight_layout(); fig.savefig(REPORTS / "test_forecast.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    importance.head(8)[::-1].plot.barh(ax=ax, color="tab:blue")
    ax.set_xlabel("Error increase when the input is shuffled (AQI points)")
    ax.set_title("What the model relied on\n(patterns, not proof of cause)")
    fig.tight_layout(); fig.savefig(REPORTS / "importance.png", dpi=130); plt.close(fig)

    print(json.dumps({k: {kk: {m: round(x, 3) for m, x in vv.items()} for kk, vv in v.items()}
                      for k, v in results.items()}, indent=1))
    print("95% CI of improvement over baseline (MAE points):", json.dumps(ci))
    print("importance:\n", importance.round(2).head(8))


if __name__ == "__main__":
    main()
