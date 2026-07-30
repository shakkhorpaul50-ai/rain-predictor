import sys, os
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

def features(gray, hsv, bgr):
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

        bgr = cv2.resize(bgr, (32, 32))
        bgr = cv2.GaussianBlur(bgr, (5, 5), 1.0)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        X = features(gray, hsv, bgr).reshape(1, -1)

        proba = self.model.predict_proba(X)[0]
        pred = self.model.predict(X)[0]

        cloud_mask = (gray > 128).astype(np.uint8)
        cov = float(np.mean(cloud_mask))

        return {
            "prediction": int(pred),
            "probability_rain": float(proba[1]),
            "probability_no_rain": float(proba[0]),
            "cloud_coverage": cov,
            "label": "Rain" if pred == 1 else "No Rain",
        }
