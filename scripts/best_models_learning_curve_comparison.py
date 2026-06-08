"""
Best Fixed Models Learning Curve Comparison
============================================

Compares learning curves of several fixed-configuration models (Linear
Regression, Ridge, Lasso, Random Forest, KNN) on the GDSC2 drug response
dataset.  All models use the same training sample sizes at each curve point
and the full validation/test sets for evaluation.
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor

plt.style.use("seaborn-v0_8-whitegrid")
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
                       X_test, y_test, train_perm, train_fractions):
    """Train on increasing fractions and collect train/val/test metrics."""
    rows = []
    n_train = len(y_train)
    for fraction in train_fractions:
        sample_size = max(20, int(n_train * fraction))
        sample_idx = train_perm[:sample_size]

        model = clone(base_model)
        start = time.time()
        model.fit(X_train[sample_idx], y_train[sample_idx])
        fit_seconds = time.time() - start

        train_pred = model.predict(X_train[sample_idx])
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        rows.append({
            "model": model_name,
            "train_fraction": fraction,
            "train_size": sample_size,
            "val_size": len(y_val),
            "test_size": len(y_test),
            "fit_seconds": fit_seconds,
            **{f"train_{k}": v for k, v in compute_metrics(y_train[sample_idx], train_pred).items()},
            **{f"val_{k}": v for k, v in compute_metrics(y_val, val_pred).items()},
            **{f"test_{k}": v for k, v in compute_metrics(y_test, test_pred).items()},
        })
    return pd.DataFrame(rows)


def fit_full_eval(model_name, base_model, X_train, y_train, X_val, y_val,
                  X_test, y_test, output_dir):
    """Fit on full training set and evaluate on all splits."""
    model = clone(base_model)
    start = time.time()
    model.fit(X_train, y_train)
    fit_seconds = time.time() - start

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    joblib.dump(model, output_dir / f"{model_name}.joblib")
    np.save(output_dir / f"{model_name}_val_pred.npy", val_pred)
    np.save(output_dir / f"{model_name}_test_pred.npy", test_pred)

    return {
        "model": model_name,
        "train_size": len(y_train),
        "val_size": len(y_val),
        "test_size": len(y_test),
        "fit_seconds": fit_seconds,
        **{f"train_{k}": v for k, v in compute_metrics(y_train, train_pred).items()},
        **{f"val_{k}": v for k, v in compute_metrics(y_val, val_pred).items()},
        **{f"test_{k}": v for k, v in compute_metrics(y_test, test_pred).items()},
    }


def plot_big_comparison(lc_df, models_order, output_dir):
    """Six-panel learning curve comparison."""
    colors = plt.cm.tab10(np.linspace(0, 1, len(models_order)))
    color_map = dict(zip(models_order, colors))

    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    plot_specs = [
        (axes[0, 0], "train_rmse", "Train RMSE", "RMSE"),
        (axes[0, 1], "val_rmse", "Validation RMSE", "RMSE"),
        (axes[0, 2], "test_rmse", "Test RMSE", "RMSE"),
        (axes[1, 0], "train_r2", "Train R2", "R2"),
        (axes[1, 1], "val_r2", "Validation R2", "R2"),
        (axes[1, 2], "test_r2", "Test R2", "R2"),
    ]

    for model_name in models_order:
        model_df = lc_df[lc_df["model"] == model_name].sort_values("train_size")
        for ax, metric_col, _, _ in plot_specs:
            ax.plot(
                model_df["train_size"], model_df[metric_col],
                marker="o", linewidth=2, label=model_name,
                color=color_map[model_name],
            )

    for ax, _, title, ylabel in plot_specs:
        ax.set_title(title)
        ax.set_xlabel("Training samples")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=True)
    fig.suptitle("Learning Curves - Fixed Model Configs", fontsize=18, weight="bold")
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    fig.savefig(output_dir / "best_fixed_models_learning_curves_big_plot.png",
                dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_train_val_rmse(lc_df, models_order, output_dir):
    """Two-panel train/validation RMSE plot."""
    color_map = {
        "linear": "#1f77b4",
        "ridge": "#ff7f0e",
        "lasso": "#2ca02c",
        "random_forest": "#d62728",
        "knn": "#9467bd",
    }

    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5), sharey=False)

    for model_name in models_order:
        model_df = lc_df[lc_df["model"] == model_name].sort_values("train_size")
        axes[0].plot(
            model_df["train_size"], model_df["train_rmse"],
            marker="o", linewidth=3.0, markersize=7,
            label=model_name, color=color_map.get(model_name, "#333333"),
        )
        axes[1].plot(
            model_df["train_size"], model_df["val_rmse"],
            marker="o", linewidth=3.0, markersize=7,
            label=model_name, color=color_map.get(model_name, "#333333"),
        )

    axes[0].set_title("Train GDSC2")
    axes[1].set_title("Validation GDSC2")
    for ax in axes:
        ax.set_xlabel("Training samples")
        ax.set_ylabel("RMSE")
        ax.tick_params(axis="y", labelleft=True)
        ax.grid(True, alpha=0.25)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=True)
    fig.suptitle(
        "Train/Validation RMSE Learning Curves - Same Split Sizes For Every Model",
        y=1.03, fontsize=16, weight="bold",
    )
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    fig.savefig(output_dir / "train_validation_rmse_learning_curve.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compare learning curves of fixed-config models on GDSC2 data."
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
        default="../models/best_models_learning_curve_comparison",
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

    print(f"X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape}  y_val:   {y_val.shape}")
    print(f"X_test:  {X_test.shape}  y_test:  {y_test.shape}")
    print(metadata)

    # ── 2. Models ─────────────────────────────────────────────────────────
    models = {
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=100.0),
        "lasso": Lasso(
            alpha=0.001, max_iter=5000, tol=1e-3,
            selection="random", random_state=args.random_state,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, max_features="sqrt",
            n_jobs=-1, random_state=args.random_state,
        ),
        "knn": KNeighborsRegressor(
            n_neighbors=31, weights="distance", metric="minkowski",
            p=2, n_jobs=-1,
        ),
    }
    models_order = list(models.keys())
    rng = np.random.default_rng(args.random_state)
    train_perm = rng.permutation(len(y_train))

    # ── 3. Learning curves ────────────────────────────────────────────────
    train_fractions = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    curve_parts = []

    for model_name, model in models.items():
        print(f"Learning curve for {model_name}")
        curve_parts.append(
            run_learning_curve(
                model_name, model, X_train, y_train, X_val, y_val,
                X_test, y_test, train_perm, train_fractions,
            )
        )

    learning_curve_df = pd.concat(curve_parts, ignore_index=True)

    # Assert consistency
    size_check = (
        learning_curve_df.groupby("train_fraction")
        .agg(
            n_models=("model", "nunique"),
            n_train_sizes=("train_size", "nunique"),
            n_val_sizes=("val_size", "nunique"),
            n_test_sizes=("test_size", "nunique"),
            train_size=("train_size", "first"),
            val_size=("val_size", "first"),
            test_size=("test_size", "first"),
        )
    )
    assert (size_check["n_models"] == len(models)).all()
    assert (size_check["n_train_sizes"] == 1).all()
    assert (size_check["n_val_sizes"] == 1).all()
    assert (size_check["n_test_sizes"] == 1).all()

    print("\nSize consistency check:")
    print(size_check)

    print("\nLearning curve results:")
    print(learning_curve_df.round(4))
    learning_curve_df.to_csv(output_dir / "best_fixed_models_learning_curves.csv", index=False)

    # ── 4. Full train/test evaluation ─────────────────────────────────────
    full_eval_rows = []
    for model_name, model in models.items():
        print(f"Full train/test eval for {model_name}")
        full_eval_rows.append(
            fit_full_eval(model_name, model, X_train, y_train, X_val, y_val,
                          X_test, y_test, output_dir)
        )

    full_eval_df = pd.DataFrame(full_eval_rows).sort_values("val_rmse")

    assert full_eval_df["train_size"].nunique() == 1
    assert full_eval_df["val_size"].nunique() == 1
    assert full_eval_df["test_size"].nunique() == 1

    print("\nFull evaluation results:")
    print(full_eval_df.round(4))
    full_eval_df.to_csv(output_dir / "best_fixed_models_full_eval.csv", index=False)

    # ── 5. Plots ──────────────────────────────────────────────────────────
    plot_big_comparison(learning_curve_df, models_order, output_dir)
    plot_train_val_rmse(learning_curve_df, models_order, output_dir)
    print(f"\nFigures saved to {output_dir}")


if __name__ == "__main__":
    main()
