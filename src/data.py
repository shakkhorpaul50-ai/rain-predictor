import numpy as np
import os


def simulate_cloudcast_data(n_samples=500, img_size=(128, 128), num_classes=11, seed=42):
    rng = np.random.default_rng(seed)
    images = rng.integers(0, 255, size=(n_samples, *img_size), dtype=np.uint8)
    cloud_masks = rng.integers(0, num_classes, size=(n_samples, *img_size), dtype=np.uint8)
    labels = rng.integers(0, 2, size=n_samples)
    return images, cloud_masks, labels


def simulate_dataset(n_samples=500, img_size=(128, 128), seed=42):
    rng = np.random.default_rng(seed)
    images = rng.integers(0, 255, size=(n_samples, *img_size), dtype=np.uint8)
    cloud_masks = rng.integers(0, 11, size=(n_samples, *img_size), dtype=np.uint8)

    rain_prob = 0.3 + 0.5 * (cloud_masks.mean(axis=(1, 2)) / 10.0)
    rain_prob = np.clip(rain_prob, 0, 1)
    labels = (rng.random(n_samples) < rain_prob).astype(int)

    weather = {
        "temperature": rng.normal(20, 8, n_samples),
        "humidity": rng.uniform(30, 100, n_samples),
        "wind_speed": rng.exponential(5, n_samples),
        "hour": rng.integers(0, 24, n_samples),
        "month": rng.integers(1, 13, n_samples),
    }

    return images, cloud_masks, weather, labels


def save_simulated_data(save_dir, images, cloud_masks, weather, labels):
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "images.npy"), images)
    np.save(os.path.join(save_dir, "cloud_masks.npy"), cloud_masks)
    np.save(os.path.join(save_dir, "labels.npy"), labels)
    import pandas as pd
    df = pd.DataFrame(weather)
    df.to_csv(os.path.join(save_dir, "weather.csv"), index=False)


def load_simulated_data(data_dir):
    images = np.load(os.path.join(data_dir, "images.npy"))
    cloud_masks = np.load(os.path.join(data_dir, "cloud_masks.npy"))
    labels = np.load(os.path.join(data_dir, "labels.npy"))
    import pandas as pd
    weather = pd.read_csv(os.path.join(data_dir, "weather.csv"))
    return images, cloud_masks, weather, labels
