import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib


def train_logistic_baseline(X_train, y_train, max_iter=5000):
    model = LogisticRegression(max_iter=max_iter, solver="saga", random_state=42)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None, random_state=42):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
    )
    model.fit(X_train, y_train)
    return model


def train_svm(X_train, y_train, kernel="rbf", random_state=42):
    model = SVC(kernel=kernel, probability=True, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def grid_search_rf(X_train, y_train, cv=3):
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
    }
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid = GridSearchCV(rf, param_grid, cv=cv, scoring="f1", n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


def train_pipeline(X_train, y_train, model_type="rf", tune=False):
    if model_type == "lr":
        model = train_logistic_baseline(X_train, y_train)
        best_params = None
    elif model_type == "rf":
        if tune:
            model, best_params = grid_search_rf(X_train, y_train)
        else:
            model = train_random_forest(X_train, y_train)
            best_params = {"n_estimators": 100, "max_depth": None}
    elif model_type == "svm":
        model = train_svm(X_train, y_train)
        best_params = None
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model, best_params


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    if y_prob is not None and len(np.unique(y_test)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_test, y_prob)

    metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred)
    return metrics, y_pred, y_prob


def get_feature_importance(model, feature_names=None):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return None

    if feature_names is not None:
        return list(zip(feature_names, importances))
    return importances
