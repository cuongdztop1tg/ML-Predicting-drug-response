"""
Random Forest / KNN Baselines + Learning Curves
================================================

Trains RandomForestRegressor and KNeighborsRegressor on the GDSC2 drug response
dataset (with configurable sample cap for speed), evaluates on held-out data,
and plots learning curves.
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor

sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.max_columns", 120)


def safe_corr(y_true, y_pred):
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def compute_metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson": safe_corr(y_true, y_pred),
    }


def run_learning_curve(model_name, base_model, X_train, y_train, X_val, y_val,
                       train_pool_idx, val_eval_idx, train_fractions):
    """Train on increasing fractions of the train pool and collect metrics."""
    rows = []
    for fraction in train_fractions:
        sample_size = max(50, int(len(train_pool_idx) * fraction))
        sample_idx = train_pool_idx[:sample_size]

        model = clone(base_model)
        start = time.time()
        model.fit(X_train[sample_idx], y_train[sample_idx])
        fit_seconds = time.time() - start

        train_pred = model.predict(X_train[sample_idx])
        val_pred = model.predict(X_val[val_eval_idx])

        train_metrics = compute_metrics(y_train[sample_idx], train_pred)
        val_metrics = compute_metrics(y_val[val_eval_idx], val_pred)

        rows.append({
            "model": model_name,
            "train_fraction_of_cap": fraction,
            "train_size": sample_size,
            "val_eval_size": len(val_eval_idx),
            "fit_seconds": fit_seconds,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })
    return pd.DataFrame(rows)


def plot_learning_curves(lc_df, output_dir):
    """Plot RMSE and R2 learning curves."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    sns.lineplot(
        data=lc_df, x="train_size", y="train_rmse",
        hue="model", style="model", markers=True, dashes=False, ax=axes[0],
    )
    sns.lineplot(
        data=lc_df, x="train_size", y="val_rmse",
        hue="model", style="model", markers=True, dashes=True, ax=axes[0],
        legend=False,
    )
    axes[0].set_title("Learning Curve - RMSE")
    axes[0].set_xlabel("Training samples")
    axes[0].set_ylabel("RMSE")

    sns.lineplot(
        data=lc_df, x="train_size", y="train_r2",
        hue="model", style="model", markers=True, dashes=False, ax=axes[1],
    )
    sns.lineplot(
        data=lc_df, x="train_size", y="val_r2",
        hue="model", style="model", markers=True, dashes=True, ax=axes[1],
        legend=False,
    )
    axes[1].set_title("Learning Curve - R2")
    axes[1].set_xlabel("Training samples")
    axes[1].set_ylabel("R2")

    plt.tight_layout()
    fig.savefig(output_dir / "random_forest_knn_learning_curves.png",
                dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Random Forest / KNN baselines and learning curves on GDSC2 data."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="../processed_data/model_ready/gdsc2_lgbm_style_data.npz",
        help="Path to processed .npz data file",
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default="../processed_data/model_ready/artifacts/gdsc2_lgbm_style_metadata.json",
        help="Path to metadata JSON",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../models/random_forest_knn_baselines",
        help="Output directory for models and figures",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--max_train_size",
        type=int,
        default=20000,
        help="Number of training samples to use (subsampled for speed)",
    )
    parser.add_argument(
        "--eval_size",
        type=int,
        default=5000,
        help="Number of validation/test samples to evaluate on",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────
    loaded = np.load(args.data_path)
    X_train = loaded["X_train"]
    y_train = loaded["y_train"]
    X_val = loaded["X_val"]
    y_val = loaded["y_val"]
    X_test = loaded["X_test"]
    y_test = loaded["y_test"]

    with open(args.metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape}  y_val:   {y_val.shape}")
    print(f"X_test:  {X_test.shape}  y_test:  {y_test.shape}")
    print(metadata)

    # ── 2. Subsampling ────────────────────────────────────────────────────
    rng = np.random.default_rng(args.random_state)
    train_pool_idx = rng.permutation(len(y_train))[:args.max_train_size]
    val_eval_idx = rng.permutation(len(y_val))[: min(args.eval_size, len(y_val))]
    test_eval_idx = rng.permutation(len(y_test))[: min(args.eval_size, len(y_test))]

    print(f"\nTrain pool: {len(train_pool_idx)}")
    print(f"Validation eval: {len(val_eval_idx)}")
    print(f"Test eval: {len(test_eval_idx)}")

    # ── 3. Models ─────────────────────────────────────────────────────────
    models = {
        "random_forest": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=2, max_features="sqrt",
            n_jobs=-1, random_state=args.random_state, verbose=0,
        ),
        "knn_k15_distance": KNeighborsRegressor(
            n_neighbors=15, weights="distance", metric="minkowski",
            p=2, n_jobs=-1,
        ),
    }

    # ── 4. Fit baselines ──────────────────────────────────────────────────
    baseline_rows = []
    fitted_models = {}

    for model_name, model in models.items():
        print(f"\nFitting {model_name}")
        fitted = clone(model)
        start = time.time()
        fitted.fit(X_train[train_pool_idx], y_train[train_pool_idx])
        fit_seconds = time.time() - start

        val_pred = fitted.predict(X_val[val_eval_idx])
        test_pred = fitted.predict(X_test[test_eval_idx])

        row = {
            "model": model_name,
            "train_size": len(train_pool_idx),
            "val_eval_size": len(val_eval_idx),
            "test_eval_size": len(test_eval_idx),
            "fit_seconds": fit_seconds,
            **{f"val_{k}": v for k, v in compute_metrics(y_val[val_eval_idx], val_pred).items()},
            **{f"test_{k}": v for k, v in compute_metrics(y_test[test_eval_idx], test_pred).items()},
        }
        baseline_rows.append(row)
        fitted_models[model_name] = fitted
        joblib.dump(fitted, output_dir / f"{model_name}.joblib")
        np.save(output_dir / f"{model_name}_val_pred_sample.npy", val_pred)
        np.save(output_dir / f"{model_name}_test_pred_sample.npy", test_pred)

    baseline_df = pd.DataFrame(baseline_rows).sort_values("val_rmse")
    print("\nBaseline results:")
    print(baseline_df.round(4))
    baseline_df.to_csv(output_dir / "random_forest_knn_baseline_results.csv", index=False)

    # ── 5. Learning curves ────────────────────────────────────────────────
    train_fractions = [0.1, 0.2, 0.4, 0.7, 1.0]
    curve_parts = []

    for model_name, model in models.items():
        print(f"Learning curve for {model_name}")
        curve_parts.append(
            run_learning_curve(
                model_name, model, X_train, y_train, X_val, y_val,
                train_pool_idx, val_eval_idx, train_fractions,
            )
        )

    learning_curve_df = pd.concat(curve_parts, ignore_index=True)
    print("\nLearning curve results:")
    print(learning_curve_df.round(4))
    learning_curve_df.to_csv(output_dir / "random_forest_knn_learning_curves.csv", index=False)

    plot_learning_curves(learning_curve_df, output_dir)
    print(f"\nFigures saved to {output_dir}")


if __name__ == "__main__":
    main()
