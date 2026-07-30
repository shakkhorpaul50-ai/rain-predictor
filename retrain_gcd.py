import sys, os, glob, random, time
sys.path.insert(0, os.getcwd())
import numpy as np, cv2, joblib
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import lightgbm as lgb

RAIN_CLASSES = {"6_cumulonimbus"}; SEED = 42; SIZE = 32

def get_label(path):
    parts = path.replace("\\", "/").split("/")
    for i, p in enumerate(parts):
        if p in ("train", "test") and i + 1 < len(parts):
            return 1 if parts[i+1] in RAIN_CLASSES else 0
    return 0

def color_hist(hsv, bins=16):
    hists = []
    for i in range(3):
        hist = cv2.calcHist([hsv], [i], None, [bins], [0, 180 if i == 0 else 256])
        hists.append(hist.flatten() / float(hsv.shape[0] * hsv.shape[1]))
    return np.concatenate(hists)

def quadrant_stats(hsv):
    h = hsv.shape[0]
    qs = []
    for yi in range(2):
        for xi in range(2):
            q = hsv[yi*h//2:(yi+1)*h//2, xi*h//2:(xi+1)*h//2]
            for i in range(3):
                qs.append(float(q[:,:,i].mean()))
                qs.append(float(np.std(q[:,:,i])))
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

def extract_features(gray, hsv, bgr):
    _, cloud = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cloud_mask = (cloud > 0).astype(np.uint8)
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
        np.array([float(np.mean(cloud_mask)), float(gray.mean()), float(np.std(gray))]), clust])

def main():
    random.seed(SEED); np.random.seed(SEED)
    files = glob.glob("datasets/gcd/images/GCD/**/*.jpg", recursive=True)
    rain = [f for f in files if get_label(f) == 1]
    norain = [f for f in files if get_label(f) == 0]
    print(f"R={len(rain)} N={len(norain)}", flush=True)
    r = random.sample(rain, min(5764, len(rain)))
    n = random.sample(norain, min(5764, len(norain)))
    sampled = r + n; random.shuffle(sampled)
    print(f"Extracting features for {len(sampled)} images...", flush=True)
    X, y = [], []; t0 = time.time()
    for idx, f in enumerate(sampled):
        bgr = cv2.imread(f, cv2.IMREAD_COLOR)
        if bgr is None: continue
        bgr = cv2.resize(bgr, (SIZE, SIZE))
        bgr = cv2.GaussianBlur(bgr, (5,5), 1.0)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        X.append(extract_features(gray, hsv, bgr))
        y.append(get_label(f))
        if (idx + 1) % 1000 == 0:
            print(f"  {idx+1}/{len(sampled)} ({time.time()-t0:.0f}s)", flush=True)
    X = np.array(X); y = np.array(y)
    print(f"Shape: {X.shape}, classes: {np.bincount(y)}", flush=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}", flush=True)

    print("5-fold CV...", flush=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]
        m = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.03, num_leaves=31,
            max_depth=6, feature_fraction=0.7, min_child_samples=20,
            reg_lambda=5, reg_alpha=5, class_weight='balanced',
            random_state=SEED, n_jobs=-1, verbosity=-1)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric='logloss',
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        p = m.predict(X_va)
        acc = accuracy_score(y_va, p)
        cv_scores.append(acc)
        print(f"  Fold {fold+1}: {acc:.4f} (best iter: {m.best_iteration_})", flush=True)
    print(f"CV mean: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}", flush=True)

    print("Training final model on full train set...", flush=True)
    final = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.03, num_leaves=31,
        max_depth=6, feature_fraction=0.7, min_child_samples=20,
        reg_lambda=5, reg_alpha=5, class_weight='balanced',
        random_state=SEED, n_jobs=-1, verbosity=-1)
    final.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='logloss',
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])

    p = final.predict(X_test)
    pb = final.predict_proba(X_test)[:, 1]
    print(f"Hold-out test — Acc: {accuracy_score(y_test, p):.4f}, F1: {f1_score(y_test, p):.4f}, AUC: {roc_auc_score(y_test, pb):.4f}", flush=True)
    print(f"Best iteration: {final.best_iteration_}", flush=True)

    joblib.dump(final, os.path.join("backend", "models", "rain_model.joblib"))
    print("Saved.", flush=True)

if __name__ == "__main__":
    main()
