import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import cv2
from PIL import Image
import io
import joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "rain_model.joblib")

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

class RainPredictor:
    def __init__(self):
        self.model = load_model()
        if self.model is None:
            raise RuntimeError("No model found. Train one first.")

    def predict(self, image_bytes: bytes, weather_data: dict = None) -> dict:
        np_img = np.frombuffer(image_bytes, np.uint8)
        bgr = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if bgr is None:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        bgr = cv2.resize(bgr, (64, 64))
        bgr = cv2.GaussianBlur(bgr, (5, 5), 1.0)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        cloud = (gray > 128).astype(np.uint8)
        cov = float(np.mean(cloud))
        color_m = np.array([float(hsv[:, :, i].mean()) for i in range(3)])
        color_s = np.array([float(np.std(hsv[:, :, i])) for i in range(3)])
        cc = cv2.connectedComponentsWithStats(cloud, connectivity=8)[2]
        if cc is not None and cc.shape[0] > 1:
            areas = cc[1:, cv2.CC_STAT_AREA]
            clust = np.array([len(areas), float(areas.mean()) if len(areas) > 0 else 0])
        else:
            clust = np.array([0, 0])
        raw = gray.flatten().astype(np.float32) / 255.0

        X = np.concatenate([raw, color_m, color_s, np.array([cov, float(gray.mean()), float(np.std(gray))]), clust]).reshape(1, -1)

        proba = self.model.predict_proba(X)[0]
        pred = self.model.predict(X)[0]

        return {
            "prediction": int(pred),
            "probability_rain": float(proba[1]),
            "probability_no_rain": float(proba[0]),
            "cloud_coverage": cov,
            "label": "Rain" if pred == 1 else "No Rain",
        }
