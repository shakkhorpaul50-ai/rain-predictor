import sys, os, glob, random, time
sys.path.insert(0, os.getcwd())
import numpy as np, cv2, joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import lightgbm as lgb

RAIN_CLASSES = {"6_cumulonimbus"}; SEED = 42; SIZE = 64

def get_label(path):
    parts = path.replace("\\", "/").split("/")
    for i, p in enumerate(parts):
        if p in ("train", "test") and i + 1 < len(parts):
            return 1 if parts[i+1] in RAIN_CLASSES else 0
    return 0

def features(gray, hsv):
    cloud = (gray > 128).astype(np.uint8)
    cov = float(np.mean(cloud))
    color_m = np.array([float(hsv[:,:,i].mean()) for i in range(3)])
    color_s = np.array([float(np.std(hsv[:,:,i])) for i in range(3)])
    cc = cv2.connectedComponentsWithStats(cloud, connectivity=8)[2]
    if cc is not None and cc.shape[0] > 1:
        areas = cc[1:, cv2.CC_STAT_AREA]
        clust = np.array([len(areas), float(areas.mean()) if len(areas) > 0 else 0])
    else:
        clust = np.array([0, 0])
    raw = gray.flatten().astype(np.float32) / 255.0
    return np.concatenate([raw, color_m, color_s, np.array([cov, float(gray.mean()), float(np.std(gray))]), clust])

def main():
    random.seed(SEED); np.random.seed(SEED)
    files = glob.glob("datasets/gcd/images/GCD/**/*.jpg", recursive=True)
    rain = [f for f in files if get_label(f) == 1]
    norain = [f for f in files if get_label(f) == 0]
    print(f"R={len(rain)} N={len(norain)}", flush=True)
    r = random.sample(rain, min(5764, len(rain)))
    n = random.sample(norain, min(5764, len(norain)))
    sampled = r + n; random.shuffle(sampled)
    print(f"{len(sampled)}...", flush=True)
    X, y = [], []; t0 = time.time()
    for idx, f in enumerate(sampled):
        bgr = cv2.imread(f, cv2.IMREAD_COLOR)
        if bgr is None: continue
        bgr = cv2.resize(bgr, (SIZE, SIZE))
        bgr = cv2.GaussianBlur(bgr, (5,5), 1.0)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        X.append(features(gray, hsv))
        y.append(get_label(f))
        if (idx + 1) % 1000 == 0:
            print(f"  {idx+1}/{len(sampled)} ({time.time()-t0:.0f}s)", flush=True)
    X = np.array(X); y = np.array(y)
    print(f"S={X.shape} C={np.bincount(y)}", flush=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    print("LGB...", flush=True)
    m = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=31,
        max_depth=6, feature_fraction=0.8, min_child_samples=10,
        random_state=SEED, n_jobs=-1, verbosity=-1)
    m.fit(Xtr, ytr)
    p = m.predict(Xte); pb = m.predict_proba(Xte)[:, 1]
    print(f"Acc={accuracy_score(yte, p):.4f} F1={f1_score(yte, p):.4f} AUC={roc_auc_score(yte, pb):.4f}", flush=True)
    joblib.dump(m, os.path.join("backend", "models", "rain_model.joblib"))
    print("OK", flush=True)

if __name__ == "__main__":
    main()
