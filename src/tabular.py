import numpy as np
import pandas as pd


def cyclical_encode(hour, max_val=24):
    return np.array([np.sin(2 * np.pi * hour / max_val), np.cos(2 * np.pi * hour / max_val)])


def cyclical_encode_month(month, max_val=12):
    return np.array([np.sin(2 * np.pi * month / max_val), np.cos(2 * np.pi * month / max_val)])


def normalize_numeric(df, columns, means=None, stds=None):
    if means is None or stds is None:
        means = df[columns].mean()
        stds = df[columns].std().replace(0, 1)
        fitted = True
    else:
        fitted = False

    normalized = (df[columns] - means) / stds
    return normalized, means, stds, fitted


def encode_location(lat, lon, num_bins=10):
    lat_bin = np.digitize(lat, np.linspace(-90, 90, num_bins)) - 1
    lon_bin = np.digitize(lon, np.linspace(-180, 180, num_bins)) - 1
    loc_encoded = np.zeros(num_bins * 2)
    loc_encoded[lat_bin] = 1
    loc_encoded[num_bins + lon_bin] = 1
    return loc_encoded


def process_weather_data(df, hour_col="hour", month_col="month",
                         temp_col="temperature", humidity_col="humidity",
                         wind_col="wind_speed", lat_col=None, lon_col=None,
                         normalize_params=None):
    features = []

    if hour_col in df.columns:
        hour_feats = np.array([cyclical_encode(h, 24) for h in df[hour_col].values])
        features.append(hour_feats)

    if month_col in df.columns:
        month_feats = np.array([cyclical_encode_month(m, 12) for m in df[month_col].values])
        features.append(month_feats)

    numeric_cols = [c for c in [temp_col, humidity_col, wind_col] if c in df.columns]
    if numeric_cols:
        means = normalize_params.get("means") if normalize_params else None
        stds = normalize_params.get("stds") if normalize_params else None
        normalized, means, stds, _ = normalize_numeric(df, numeric_cols, means, stds)
        features.append(normalized.values)
        saved_params = {"means": means, "stds": stds}
    else:
        saved_params = {}

    if lat_col and lon_col and lat_col in df.columns and lon_col in df.columns:
        loc_feats = np.array([
            encode_location(float(lat), float(lon))
            for lat, lon in zip(df[lat_col].values, df[lon_col].values)
        ])
        features.append(loc_feats)

    return np.concatenate(features, axis=1), saved_params


def simulate_weather_data(n_samples=1000):
    np.random.seed(42)
    data = {
        "hour": np.random.randint(0, 24, n_samples),
        "month": np.random.randint(1, 13, n_samples),
        "temperature": np.random.normal(20, 8, n_samples),
        "humidity": np.random.uniform(30, 100, n_samples),
        "wind_speed": np.random.exponential(5, n_samples),
    }
    return pd.DataFrame(data)
