import numpy as np
from scipy.ndimage import gaussian_filter


def perlin_noise_2d(size, scale=10, octaves=4, seed=None):
    rng = np.random.default_rng(seed)
    noise = np.zeros(size)
    for o in range(octaves):
        freq = 2 ** o
        s = max(1, size[0] // (scale * freq))
        small = rng.uniform(-1, 1, (s, s))
        large = gaussian_filter(small, sigma=0.5, mode="wrap", truncate=2)
        zoomed = np.zeros(size)
        for i in range(size[0]):
            for j in range(size[1]):
                si = min(i * s // size[0], s - 1)
                sj = min(j * s // size[1], s - 1)
                zoomed[i, j] = large[si, sj]
        noise += zoomed / (o + 1)
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    return noise


def make_cloud_image(size=(128, 128), coverage=0.5, thickness=0.5, seed=None):
    rng = np.random.default_rng(seed)

    bg = rng.uniform(40, 90)
    sky = np.full(size, bg, dtype=np.float32)

    n_blobs = rng.integers(3, 12)
    for _ in range(n_blobs):
        cx = rng.uniform(0, size[1])
        cy = rng.uniform(0, size[0])
        rx = rng.uniform(15, 50) * (0.5 + coverage)
        ry = rng.uniform(10, 35) * (0.5 + coverage)
        intensity = rng.uniform(160, 240) * (0.3 + 0.7 * thickness)
        Y, X = np.ogrid[:size[0], :size[1]]
        mask = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2
        blob = intensity * np.exp(-0.5 * mask)
        sky = np.maximum(sky, blob)

    pn = perlin_noise_2d(size, scale=8 + int(8 * coverage), octaves=4, seed=rng)
    sky = sky + (pn - 0.5) * 30 * thickness
    sky = np.clip(sky, 0, 255).astype(np.uint8)

    cloud_mask = (sky > bg + 15).astype(np.uint8)

    return sky, cloud_mask


def generate_synthetic_dataset(n_samples=500, size=(128, 128), seed=42):
    rng = np.random.default_rng(seed)
    images = []
    cloud_masks = []
    weather_list = {
        "temperature": [],
        "humidity": [],
        "wind_speed": [],
        "hour": [],
        "month": [],
    }
    labels = []

    for i in range(n_samples):
        s = seed + i if seed else None
        coverage = rng.uniform(0.05, 0.95)
        thickness = rng.uniform(0.2, 1.0)
        img, mask = make_cloud_image(size, coverage, thickness, seed=s)
        images.append(img)
        cloud_masks.append(mask)

        temp = rng.normal(25 - 15 * coverage, 3)
        humid = rng.uniform(30, 100)
        wind = rng.exponential(3 + 7 * coverage)
        hour = rng.integers(0, 24)
        month = rng.integers(1, 13)

        weather_list["temperature"].append(temp)
        weather_list["humidity"].append(humid)
        weather_list["wind_speed"].append(wind)
        weather_list["hour"].append(hour)
        weather_list["month"].append(month)

        rain_prob = 0.1 + 0.6 * coverage + 0.15 * (humid / 100) + 0.15 * (1 - temp / 40)
        rain_prob = np.clip(rain_prob, 0, 1)
        labels.append(1 if rng.random() < rain_prob else 0)

    return np.array(images), np.array(cloud_masks), weather_list, np.array(labels)
