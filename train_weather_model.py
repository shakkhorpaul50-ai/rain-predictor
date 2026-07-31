import os, json, time
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import lightgbm as lgb

SEED = 42
CSV_PATH = os.path.join("datasets", "weather", "bangladesh_hourly_weather.csv")
MODEL_PATH = os.path.join("backend", "models", "weather_model.joblib")
META_PATH = os.path.join("backend", "models", "weather_meta.json")

NUMERIC_COLS = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "surface_pressure"]
CATEGORICAL_COLS = ["district"]
LABEL_COL = "rain_next_3h"


def cyclical(series, period):
    x = np.asarray(series, dtype=np.float64)
    return np.stack([np.sin(2 * np.pi * x / period), np.cos(2 * np.pi * x / period)], axis=1)


def build_features(df):
    feats = []
    feats.append(df[NUMERIC_COLS].astype(np.float64).values)
    feats.append(cyclical(df["hour"].values, 24))
    feats.append(cyclical(df["month"].values, 12))
    district_codes = df["district"].map(DISTRICT_MAP).values
    feats.append(district_codes.reshape(-1, 1))
    return np.concatenate(feats, axis=1)


def main():
    print("Loading weather CSV...", flush=True)
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=NUMERIC_COLS).reset_index(drop=True)
    print(f"Rows: {len(df)}, rain rate: {df[LABEL_COL].mean():.3f}", flush=True)

    global DISTRICT_MAP
    districts = sorted(df["district"].unique())
    DISTRICT_MAP = {d: i for i, d in enumerate(districts)}
    df["district"] = df["district"].map(DISTRICT_MAP)

    means = df[NUMERIC_COLS].mean()
    stds = df[NUMERIC_COLS].std().replace(0, 1)
    df[NUMERIC_COLS] = (df[NUMERIC_COLS] - means) / stds

    X = build_features(df)
    y = df[LABEL_COL].values.astype(np.int64)
    print(f"Feature matrix: {X.shape}, classes: {np.bincount(y)}", flush=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    print(f"Train: {len(X_train)}, Test: {len(X_test)}", flush=True)

    print("5-fold CV...", flush=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]
        m = lgb.LGBMClassifier(
            n_estimators=3000, learning_rate=0.03, num_leaves=31, max_depth=6,
            feature_fraction=0.7, min_child_samples=50,
            reg_lambda=5, reg_alpha=5, class_weight="balanced",
            random_state=SEED, n_jobs=-1, verbosity=-1)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="logloss",
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        p = m.predict(X_va)
        acc = accuracy_score(y_va, p)
        cv_scores.append(acc)
        print(f"  Fold {fold + 1}: {acc:.4f} (best iter: {m.best_iteration_})", flush=True)
    print(f"CV mean: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}", flush=True)

    print("Training final model on full train set...", flush=True)
    final = lgb.LGBMClassifier(
        n_estimators=3000, learning_rate=0.03, num_leaves=31, max_depth=6,
        feature_fraction=0.7, min_child_samples=50,
        reg_lambda=5, reg_alpha=5, class_weight="balanced",
        random_state=SEED, n_jobs=-1, verbosity=-1)
    final.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric="logloss",
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])

    p = final.predict(X_test)
    pb = final.predict_proba(X_test)[:, 1]
    print(f"Hold-out test — Acc: {accuracy_score(y_test, p):.4f}, "
          f"F1: {f1_score(y_test, p):.4f}, AUC: {roc_auc_score(y_test, pb):.4f}", flush=True)
    print(f"Best iteration: {final.best_iteration_}", flush=True)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(final, MODEL_PATH)
    meta = {
        "features": NUMERIC_COLS + ["hour_sin", "hour_cos", "month_sin", "month_cos", "district_code"],
        "numeric_means": means.to_dict(),
        "numeric_stds": stds.to_dict(),
        "district_map": DISTRICT_MAP,
        "label": "rain_next_3h",
        "trained_on": "Open-Meteo hourly, 64 Bangladesh districts, 2023-2024",
        "test_metrics": {
            "accuracy": float(accuracy_score(y_test, p)),
            "f1": float(f1_score(y_test, p)),
            "roc_auc": float(roc_auc_score(y_test, pb)),
        },
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("Saved weather_model.joblib + weather_meta.json.", flush=True)


if __name__ == "__main__":
    main()
