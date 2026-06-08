"""
Linear Regression Baseline
==========================

Trains a plain LinearRegression on the processed GDSC2 drug-response dataset,
evaluates it on the validation and test sets, and plots learning curves
(train/validation RMSE and R2 across increasing training sample sizes).
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

plt.style.use("seaborn-v0_8-whitegrid")
pd.set_option("display.max_columns", 120)


def compute_metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson": float(pearsonr(y_true, y_pred)[0]),
        "spearman": float(spearmanr(y_true, y_pred)[0]),
    }


def run_learning_curve(base_model, X_train, y_train, X_val, y_val,
                       train_fractions, random_state=42):
    """Train the model on increasing fractions of the training set and collect metrics."""
    rng = np.random.default_rng(random_state)
    n_train = len(X_train)
    rows = []

    for fraction in train_fractions:
        sample_size = max(10, int(n_train * fraction))
        sample_idx = rng.choice(n_train, size=sample_size, replace=False)

        model = clone(base_model)
        start = time.time()
        model.fit(X_train[sample_idx], y_train[sample_idx])
        fit_seconds = time.time() - start

        train_pred = model.predict(X_train[sample_idx])
        val_pred = model.predict(X_val)

        train_metrics = compute_metrics(y_train[sample_idx], train_pred)
        val_metrics = compute_metrics(y_val, val_pred)

        rows.append({
            "train_fraction": fraction,
            "train_size": sample_size,
            "fit_seconds": fit_seconds,
            "train_rmse": train_metrics["rmse"],
            "val_rmse": val_metrics["rmse"],
            "train_mae": train_metrics["mae"],
            "val_mae": val_metrics["mae"],
            "train_r2": train_metrics["r2"],
            "val_r2": val_metrics["r2"],
            "train_pearson": train_metrics["pearson"],
            "val_pearson": val_metrics["pearson"],
        })

    return pd.DataFrame(rows)


def plot_learning_curve(lc_df, output_dir):
    """Plot RMSE and R2 learning curves."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True)

    axes[0].plot(
        lc_df["train_size"], lc_df["train_rmse"],
        marker="o", linewidth=2.2, color="#2563eb", label="Train curve",
    )
    axes[0].plot(
        lc_df["train_size"], lc_df["val_rmse"],
        marker="o", linewidth=2.2, color="#dc2626", label="Validation curve",
    )
    axes[0].set_title("Train vs Validation Curve - RMSE", fontsize=13, weight="bold")
    axes[0].set_xlabel("Training samples")
    axes[0].set_ylabel("RMSE")
    axes[0].legend()

    axes[1].plot(
        lc_df["train_size"], lc_df["train_r2"],
        marker="o", linewidth=2.2, color="#2563eb", label="Train curve",
    )
    axes[1].plot(
        lc_df["train_size"], lc_df["val_r2"],
        marker="o", linewidth=2.2, color="#dc2626", label="Validation curve",
    )
    axes[1].set_title("Train vs Validation Curve - R2", fontsize=13, weight="bold")
    axes[1].set_xlabel("Training samples")
    axes[1].set_ylabel("R2")
    axes[1].legend()

    for ax in axes:
        ax.grid(True, alpha=0.25)
    plt.tight_layout()

    fig.savefig(output_dir / "linear_regression_train_validation_curves.png",
                dpi=160, bbox_inches="tight")
    fig.savefig(output_dir / "linear_regression_learning_curve.png",
                dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Train Linear Regression baseline on GDSC2 processed data."
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
        default="../models/linear_regression_processed_data",
        help="Output directory for models and figures",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed",
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

    print("Loaded processed data")
    print(f"X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape}  y_val:   {y_val.shape}")
    print(f"X_test:  {X_test.shape}  y_test:  {y_test.shape}")
    print(metadata)

    # ── 2. Train and evaluate ─────────────────────────────────────────────
    final_model = LinearRegression()
    final_model.fit(X_train, y_train)

    val_pred = final_model.predict(X_val)
    test_pred = final_model.predict(X_test)

    print("\nValidation Metrics:")
    for k, v in compute_metrics(y_val, val_pred).items():
        print(f"  {k}: {v:.4f}")

    print("\nTest Metrics:")
    for k, v in compute_metrics(y_test, test_pred).items():
        print(f"  {k}: {v:.4f}")

    # ── 3. Learning curve ─────────────────────────────────────────────────
    train_fractions = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    lc_df = run_learning_curve(
        LinearRegression(), X_train, y_train, X_val, y_val,
        train_fractions, random_state=args.random_state,
    )
    print("\nLearning curve results:")
    print(lc_df.round(4))
    lc_df.to_csv(output_dir / "linear_regression_learning_curve.csv", index=False)

    plot_learning_curve(lc_df, output_dir)
    print(f"\nFigures saved to {output_dir}")


if __name__ == "__main__":
    main()
