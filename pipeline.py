import os
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split

from src.preprocessing import preprocess_pipeline, cloud_segmentation
from src.features import extract_all_image_features, extract_hog_features, extract_glcm_features
from src.tabular import process_weather_data, simulate_weather_data
from src.model import train_pipeline, evaluate_model, get_feature_importance
from src.evaluate import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance,
    plot_precision_recall,
)
from src.data import simulate_dataset, save_simulated_data, load_simulated_data


def extract_features_batch(images, cloud_masks, verbose=True):
    n = len(images)
    all_features = []
    for i in range(n):
        if verbose and (i + 1) % 100 == 0:
            print(f"  Extracting features: {i + 1}/{n}")
        feats = extract_all_image_features(images[i], cloud_masks[i] if cloud_masks is not None else None)
        all_features.append(feats)
    return np.array(all_features)


def run_pipeline(data_dir=None, model_type="rf", tune=False, test_size=0.2, seed=42):
    if data_dir and os.path.exists(data_dir):
        print(f"Loading data from {data_dir}...")
        images, cloud_masks, weather, labels = load_simulated_data(data_dir)
    else:
        print("Generating simulated data...")
        images, cloud_masks, weather, labels = simulate_dataset(n_samples=500, seed=seed)
        if data_dir:
            save_simulated_data(data_dir, images, cloud_masks, weather, labels)
            print(f"Saved simulated data to {data_dir}")

    print(f"Loaded {len(images)} samples, image shape: {images[0].shape}")

    print("Extracting image features...")
    image_features = extract_features_batch(images, cloud_masks)

    print("Processing weather data...")
    tabular_features, norm_params = process_weather_data(
        pd.DataFrame(weather),
        hour_col="hour",
        month_col="month",
        temp_col="temperature",
        humidity_col="humidity",
        wind_col="wind_speed",
    )

    print(f"Image features shape: {image_features.shape}")
    print(f"Tabular features shape: {tabular_features.shape}")

    X = np.concatenate([image_features, tabular_features], axis=1)
    y = labels
    print(f"Combined feature matrix: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    print(f"Training {model_type} model...")
    model, best_params = train_pipeline(X_train, y_train, model_type=model_type, tune=tune)
    if best_params:
        print(f"Best params: {best_params}")

    print("Evaluating...")
    metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test)

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    for key, val in metrics.items():
        if key != "confusion_matrix":
            print(f"  {key}: {val:.4f}")
    print(f"\nConfusion Matrix:\n{metrics['confusion_matrix']}")

    print("\nGenerating plots...")
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    plot_confusion_matrix(metrics["confusion_matrix"], save_path=os.path.join(output_dir, "confusion_matrix.png"))
    print(f"  Saved confusion_matrix.png")

    if y_prob is not None:
        plot_roc_curve(y_test, y_prob, save_path=os.path.join(output_dir, "roc_curve.png"))
        print(f"  Saved roc_curve.png")
        plot_precision_recall(y_test, y_prob, save_path=os.path.join(output_dir, "precision_recall.png"))
        print(f"  Saved precision_recall.png")

    importances = get_feature_importance(model)
    if importances is not None:
        plot_feature_importance(importances, top_n=20, save_path=os.path.join(output_dir, "feature_importance.png"))
        print(f"  Saved feature_importance.png")

    print(f"\nAll outputs saved to '{output_dir}/'")

    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rain Prediction Pipeline")
    parser.add_argument("--data-dir", default=None, help="Path to data directory")
    parser.add_argument("--model", default="rf", choices=["rf", "lr", "svm"], help="Model type")
    parser.add_argument("--tune", action="store_true", help="Perform hyperparameter tuning")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        model_type=args.model,
        tune=args.tune,
        test_size=args.test_size,
        seed=args.seed,
    )
