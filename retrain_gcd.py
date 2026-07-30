import sys, os, json, glob, random, time
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
import cv2
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.preprocessing import cloud_segmentation
from src.features import extract_all_image_features
from src.tabular import process_weather_data

RAIN_CLASSES = {"6_cumulonimbus"}
N_SAMPLES = 3000
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def get_class_label(path):
    parts = path.replace("\\", "/").split("/")
    for i, p in enumerate(parts):
        if p in ("train", "test") and i + 1 < len(parts):
            return 1 if parts[i+1] in RAIN_CLASSES else 0
    return 0

def process_image(path):
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None: return None
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        img = cv2.GaussianBlur(img, (5, 5), 1.0)
        img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)
        cloud_mask = cloud_segmentation(img)
        feats = extract_all_image_features(img, cloud_mask)
        return feats, float(np.mean(cloud_mask > 0))
    except:
        return None

print("Collecting GCD files...")
all_files = glob.glob("datasets/gcd/images/GCD/**/*.jpg", recursive=True)
rain_files = [f for f in all_files if get_class_label(f) == 1]
norain_files = [f for f in all_files if get_class_label(f) == 0]
print(f"Total: rain={len(rain_files)}, norain={len(norain_files)}")

n_per = N_SAMPLES // 2
sampled = random.sample(rain_files, min(n_per, len(rain_files))) + \
          random.sample(norain_files, min(n_per, len(norain_files)))
random.shuffle(sampled)
print(f"Processing {len(sampled)} images...")

features, labels, weather_rows = [], [], []
t0 = time.time()
for idx, f in enumerate(sampled):
    r = process_image(f)
    if r is None: continue
    feats, cov = r
    label = get_class_label(f)
    features.append(feats)
    labels.append(label)
    if label == 1:
        temp = np.random.normal(18, 5)
        humid = np.random.uniform(65, 100)
        wind = np.random.exponential(8)
    else:
        temp = np.random.normal(28, 5)
        humid = np.random.uniform(30, 75)
        wind = np.random.exponential(4)
    weather_rows.append({
        "temperature": float(temp), "humidity": float(humid),
        "wind_speed": float(wind),
        "hour": int(np.random.randint(0, 24)),
        "month": int(np.random.randint(1, 13)),
    })
    if (idx+1) % 200 == 0:
        elapsed = time.time() - t0
        print(f"  {idx+1}/{len(sampled)} ({elapsed:.0f}s)")

image_features = np.array(features)
labels = np.array(labels)
print(f"Extracted {len(image_features)} samples, shape: {image_features.shape}")
print(f"Class distribution: {np.bincount(labels)}")

print("Processing weather...")
tabular_features, norm_params = process_weather_data(
    pd.DataFrame(weather_rows),
    hour_col="hour", month_col="month",
    temp_col="temperature", humidity_col="humidity",
    wind_col="wind_speed",
)

X = np.concatenate([image_features, tabular_features], axis=1)
y = labels
print(f"Combined: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y)

print("Training RF (500 trees)...")
model = RandomForestClassifier(
    n_estimators=500, max_depth=None, min_samples_split=2,
    min_samples_leaf=1, random_state=SEED, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}")

model_path = os.path.join("backend", "models", "rain_model.joblib")
norm_path = os.path.join("backend", "models", "norm_params.json")
joblib.dump(model, model_path)
serialized = {}
for k, v in norm_params.items():
    if hasattr(v, "to_dict"):
        serialized[k] = {str(kk): float(vv) for kk, vv in v.to_dict().items()}
    elif isinstance(v, dict):
        serialized[k] = {str(kk): float(vv) if hasattr(vv, "item") else vv for kk, vv in v.items()}
with open(norm_path, "w") as f:
    json.dump(serialized, f, indent=2)
print("Saved to", model_path)
