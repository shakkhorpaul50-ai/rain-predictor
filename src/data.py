import numpy as np
import os

from src.synthetic_clouds import generate_synthetic_dataset as _gen


def simulate_dataset(n_samples=500, img_size=(128, 128), seed=42):
    return _gen(n_samples=n_samples, size=img_size, seed=seed)


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
