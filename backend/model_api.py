import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import cv2
from PIL import Image
import io

from src.preprocessing import cloud_segmentation
from src.features import extract_all_image_features
from src.tabular import process_weather_data
from src.model import train_pipeline
from src.data import simulate_dataset
from sklearn.model_selection import train_test_split
import joblib


MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "rain_model.joblib")
NORM_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "models", "norm_params.json")


def train_and_save_model():
    images, cloud_masks, weather, labels = simulate_dataset(n_samples=500)
    from src.features import extract_all_image_features
    all_feats = []
    for i in range(len(images)):
        all_feats.append(extract_all_image_features(images[i], cloud_masks[i]))
    image_features = np.array(all_feats)

    import pandas as pd
    tabular_features, norm_params = process_weather_data(
        pd.DataFrame(weather),
        hour_col="hour", month_col="month",
        temp_col="temperature", humidity_col="humidity",
        wind_col="wind_speed",
    )

    X = np.concatenate([image_features, tabular_features], axis=1)
    y = labels

    model, _ = train_pipeline(X, y, model_type="rf", tune=False)
    joblib.dump(model, MODEL_PATH)

    serializable_params = {}
    for k, v in norm_params.items():
        if hasattr(v, "to_dict"):
            serializable_params[k] = {str(kk): float(vv) for kk, vv in v.to_dict().items()}
        elif isinstance(v, dict):
            serializable_params[k] = {str(kk): float(vv) if hasattr(vv, "item") else vv for kk, vv in v.items()}
        else:
            serializable_params[k] = v
    with open(NORM_PARAMS_PATH, "w") as f:
        json.dump(serializable_params, f, indent=2)

    return model


def load_model():
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        norm_params = {}
        if os.path.exists(NORM_PARAMS_PATH):
            with open(NORM_PARAMS_PATH) as f:
                norm_params = json.load(f)
        return model, norm_params
    return None, None


class RainPredictor:
    def __init__(self):
        self.model, self.norm_params = load_model()
        if self.model is None:
            self.model = train_and_save_model()
            self.model, self.norm_params = load_model()

    def predict(self, image_bytes: bytes, weather_data: dict) -> dict:
        np_img = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)
        if img is None:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("L")
            img = np.array(pil_img)

        processed = cv2.resize(img, (128, 128))
        processed = cv2.GaussianBlur(processed, (5, 5), 1.0)
        processed = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(processed)

        cloud_mask = cloud_segmentation(processed)

        img_features = extract_all_image_features(processed, cloud_mask)

        import pandas as pd
        df = pd.DataFrame([weather_data])
        tab_features, _ = process_weather_data(
            df,
            hour_col="hour", month_col="month",
            temp_col="temperature", humidity_col="humidity",
            wind_col="wind_speed",
            normalize_params=self.norm_params if self.norm_params else None,
        )

        X = np.concatenate([img_features.reshape(1, -1), tab_features], axis=1)

        proba = self.model.predict_proba(X)[0]
        pred = self.model.predict(X)[0]

        cloud_cov = float(np.mean(cloud_mask > 0))

        return {
            "prediction": int(pred),
            "probability_rain": float(proba[1]),
            "probability_no_rain": float(proba[0]),
            "cloud_coverage": cloud_cov,
            "label": "Rain" if pred == 1 else "No Rain",
        }
