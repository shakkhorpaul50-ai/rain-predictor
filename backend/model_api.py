import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import cv2
from PIL import Image
import io
import joblib

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
IMAGE_MODEL_PATH = os.path.join(MODEL_DIR, "rain_model.joblib")
WEATHER_MODEL_PATH = os.path.join(MODEL_DIR, "weather_model.joblib")
WEATHER_META_PATH = os.path.join(MODEL_DIR, "weather_meta.json")

DEFAULT_WEATHER_WEIGHT = 0.4
DISAGREE_THRESHOLD = 0.25

IMAGE_FALLBACK_FEATURES = np.zeros(9, dtype=np.float64)


def _load_or_none(path):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            return None
    return None


def color_hist(hsv, bins=16):
    hists = []
    for i in range(3):
        hist = cv2.calcHist([hsv], [i], None, [bins], [0, 180 if i == 0 else 256])
        hists.append(hist.flatten() / float(hsv.shape[0] * hsv.shape[1]))
    return np.concatenate(hists)


def quadrant_stats(hsv):
    h, w = hsv.shape[:2]
    qs = []
    for yi in range(2):
        for xi in range(2):
            q = hsv[yi*h//2:(yi+1)*h//2, xi*w//2:(xi+1)*w//2]
            qs.extend([float(q[:,:,i].mean()) for i in range(3)])
            qs.extend([float(np.std(q[:,:,i])) for i in range(3)])
    return np.array(qs)


def region_stats(hsv, mask):
    cloud_px = hsv[mask > 0]
    sky_px = hsv[mask == 0]
    feats = []
    for px in [cloud_px, sky_px]:
        if len(px) == 0:
            feats.extend([0]*6)
        else:
            for i in range(3):
                feats.append(float(px[:, i].mean()))
                feats.append(float(np.std(px[:, i])))
    return np.array(feats)


def color_ratios(bgr):
    b = bgr[:,:,0].astype(np.float32)
    g = bgr[:,:,1].astype(np.float32)
    r = bgr[:,:,2].astype(np.float32)
    denom = b + g + r + 1e-8
    return np.array([
        float(np.mean(b / denom)), float(np.mean(g / denom)), float(np.mean(r / denom)),
        float(np.mean((b - r) / denom)),
        float(np.mean((g - r) / denom)),
        float(np.mean((b - g) / denom)),
    ])


def image_features(gray, hsv, bgr):
    _, cloud = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cloud_mask = (cloud > 0).astype(np.uint8)
    cov = float(np.mean(cloud_mask))
    raw = gray.flatten().astype(np.float32) / 255.0
    ch = color_hist(hsv)
    qs = quadrant_stats(hsv)
    rs = region_stats(hsv, cloud_mask)
    cr = color_ratios(bgr)
    cc = cv2.connectedComponentsWithStats(cloud_mask, connectivity=8)[2]
    if cc is not None and cc.shape[0] > 1:
        areas = cc[1:, cv2.CC_STAT_AREA]
        clust = np.array([len(areas), float(areas.mean()) if len(areas) > 0 else 0])
    else:
        clust = np.array([0, 0])
    return np.concatenate([raw, ch, qs, rs, cr,
        np.array([cov, float(gray.mean()), float(np.std(gray))]), clust])


def cyclical(value, period):
    x = float(value)
    return np.array([np.sin(2 * np.pi * x / period), np.cos(2 * np.pi * x / period)])


class MultiModelRainPredictor:
    def __init__(self):
        self.image_model = _load_or_none(IMAGE_MODEL_PATH)
        self.weather_model = _load_or_none(WEATHER_MODEL_PATH)
        self.weather_meta = None
        if os.path.exists(WEATHER_META_PATH):
            try:
                with open(WEATHER_META_PATH, "r", encoding="utf-8") as f:
                    self.weather_meta = json.load(f)
            except Exception:
                self.weather_meta = None
        if self.image_model is None:
            raise RuntimeError("No image model found. Train one first.")

    @property
    def has_weather_model(self):
        return self.weather_model is not None and self.weather_meta is not None

    def _weather_features(self, weather):
        meta = self.weather_meta
        means = meta["numeric_means"]
        stds = meta["numeric_stds"]
        district_map = meta["district_map"]
        district = str(weather.get("district", "")).strip()
        code = 0
        if district in district_map:
            code = district_map[district]
        else:
            d_lower = district.lower()
            for name, c in district_map.items():
                if name.lower() == d_lower:
                    code = c
                    break
        feats = [
            (float(weather.get("temperature", 25.0)) - means["temperature_2m"]) / stds["temperature_2m"],
            (float(weather.get("humidity", 60.0)) - means["relative_humidity_2m"]) / stds["relative_humidity_2m"],
            (float(weather.get("wind_speed", 10.0)) - means["wind_speed_10m"]) / stds["wind_speed_10m"],
            (float(weather.get("pressure", 1013.25)) - means["surface_pressure"]) / stds["surface_pressure"],
        ]
        hour = int(weather.get("hour", 12))
        month = int(weather.get("month", 6))
        return np.concatenate([feats, cyclical(hour, 24), cyclical(month, 12), [code]]).reshape(1, -1)

    def _predict_image(self, gray, hsv, bgr):
        X = image_features(gray, hsv, bgr).reshape(1, -1)
        return self.image_model.predict_proba(X)[0]

    def _predict_weather(self, weather):
        if not self.has_weather_model:
            return None
        X = self._weather_features(weather)
        return self.weather_model.predict_proba(X)[0]

    def predict(self, image_bytes: bytes, weather_data: dict = None) -> dict:
        weather_data = weather_data or {}

        np_img = np.frombuffer(image_bytes, np.uint8)
        bgr = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if bgr is None:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        bgr = cv2.resize(bgr, (32, 32))
        bgr = cv2.GaussianBlur(bgr, (5, 5), 1.0)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        p_image = self._predict_image(gray, hsv, bgr)
        weather_missing = any(
            weather_data.get(k) is None
            for k in ("temperature", "humidity", "wind_speed", "pressure", "hour", "month", "district")
        )
        p_weather = self._predict_weather(weather_data) if not weather_missing else None

        if p_weather is None:
            weight_weather = 0.0
            p_final = p_image
        else:
            weight_weather = DEFAULT_WEATHER_WEIGHT
            p_final = (1 - weight_weather) * p_image + weight_weather * p_weather

        pred = int(np.argmax(p_final))
        p_rain = float(p_final[1])
        p_img_rain = float(p_image[1])
        p_wea_rain = float(p_weather[1]) if p_weather is not None else None

        if p_wea_rain is not None and abs(p_img_rain - p_wea_rain) > DISAGREE_THRESHOLD:
            hint = "Models disagree - the image and weather signals point different ways. Treat this result with caution."
        elif p_wea_rain is None:
            hint = "Weather model unavailable \u2014 prediction based on the cloud image only."
        elif abs(p_rain - 0.5) < 0.1:
            hint = "Low confidence \u2014 neither signal is decisive."
        else:
            hint = "Both models agree - high confidence prediction."

        cloud_mask = (gray > 128).astype(np.uint8)
        cov = float(np.mean(cloud_mask))

        return {
            "prediction": pred,
            "probability_rain": p_rain,
            "probability_no_rain": 1.0 - p_rain,
            "image_probability": p_img_rain,
            "weather_probability": p_wea_rain,
            "ensemble_weight_image": 1.0 - weight_weather,
            "ensemble_weight_weather": weight_weather,
            "cloud_coverage": cov,
            "label": "Rain" if pred == 1 else "No Rain",
            "hint": hint,
        }
