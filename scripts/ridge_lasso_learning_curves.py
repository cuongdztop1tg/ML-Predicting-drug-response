"""
Ridge / Lasso Baselines + Learning Curves
==========================================

Tests Ridge and Lasso regression with several alpha values on the GDSC2 drug
response dataset, selects the best model per family by validation RMSE, and
plots learning curves.
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
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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


def fit_eval_model(model_name, model, X_train, y_train, X_val, y_val, X_test, y_test):
    """Fit a model and return metrics + predictions."""
    start = time.time()
    model.fit(X_train, y_train)
    fit_seconds = time.time() - start

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    row = {
        "model": model_name,
        "fit_seconds": fit_seconds,
        **{f"val_{k}": v for k, v in compute_metrics(y_val, val_pred).items()},
        **{f"test_{k}": v for k, v in compute_metrics(y_test, test_pred).items()},
    }
    return row, model, val_pred, test_pred


def run_learning_curve(model_name, base_model, X_train, y_train, X_val, y_val,
                       train_fractions, random_state=42):
    """Train on increasing fractions with a fixed permutation order."""
    rng = np.random.default_rng(random_state)
    n_train = len(X_train)
    perm = rng.permutation(n_train)
    rows = []

    for fraction in train_fractions:
        sample_size = max(10, int(n_train * fraction))
        sample_idx = perm[:sample_size]

        model = clone(base_model)
        start = time.time()
        model.fit(X_train[sample_idx], y_train[sample_idx])
        fit_seconds = time.time() - start

        train_pred = model.predict(X_train[sample_idx])
        val_pred = model.predict(X_val)

        train_metrics = compute_metrics(y_train[sample_idx], train_pred)
        val_metrics = compute_metrics(y_val, val_pred)

        rows.append({
            "model": model_name,
            "train_fraction": fraction,
            "train_size": sample_size,
            "fit_seconds": fit_seconds,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

    return pd.DataFrame(rows)


def plot_learning_curves(lc_df, output_dir):
    """Plot RMSE and R2 learning curves for both models."""
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
    fig.savefig(output_dir / "ridge_lasso_learning_curves.png",
                dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Ridge/Lasso alpha sweep and learning curves on GDSC2 data."
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
        default="../models/ridge_lasso_baselines",
        help="Output directory for models and figures",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--ridge_alphas",
        type=float,
        nargs="+",
        default=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        help="Ridge alpha values to try",
    )
    parser.add_argument(
        "--lasso_alphas",
        type=float,
        nargs="+",
        default=[0.0001, 0.001, 0.01, 0.1, 1.0],
        help="Lasso alpha values to try",
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

    # ── 2. Alpha sweep ────────────────────────────────────────────────────
    candidate_models = []
    for alpha in args.ridge_alphas:
        candidate_models.append((f"ridge_alpha_{alpha:g}", Ridge(alpha=alpha)))
    for alpha in args.lasso_alphas:
        candidate_models.append((
            f"lasso_alpha_{alpha:g}",
            Lasso(alpha=alpha, max_iter=5000, tol=1e-3,
                  selection="random", random_state=args.random_state),
        ))

    rows = []
    fitted_models = {}
    predictions = {}

    for model_name, model in candidate_models:
        print(f"Fitting {model_name}")
        row, fitted, val_pred, test_pred = fit_eval_model(
            model_name, model, X_train, y_train, X_val, y_val, X_test, y_test,
        )
        rows.append(row)
        fitted_models[model_name] = fitted
        predictions[model_name] = {"val": val_pred, "test": test_pred}

    results_df = pd.DataFrame(rows).sort_values("val_rmse")
    print("\nAlpha sweep results:")
    print(results_df.round(4))
    results_df.to_csv(output_dir / "ridge_lasso_alpha_sweep.csv", index=False)

    # ── 3. Best models ────────────────────────────────────────────────────
    best_by_family = {}
    for family in ["ridge", "lasso"]:
        family_df = results_df[results_df["model"].str.startswith(family)]
        best_name = family_df.iloc[0]["model"]
        best_by_family[family] = best_name
        joblib.dump(fitted_models[best_name], output_dir / f"best_{family}.joblib")
        np.save(output_dir / f"best_{family}_val_pred.npy", predictions[best_name]["val"])
        np.save(output_dir / f"best_{family}_test_pred.npy", predictions[best_name]["test"])

    print(f"\nBest per family: {best_by_family}")

    # ── 4. Learning curves ────────────────────────────────────────────────
    train_fractions = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    curve_parts = []

    for family, best_name in best_by_family.items():
        print(f"Learning curve for {best_name}")
        curve_parts.append(
            run_learning_curve(
                best_name, fitted_models[best_name],
                X_train, y_train, X_val, y_val,
                train_fractions, random_state=args.random_state,
            )
        )

    learning_curve_df = pd.concat(curve_parts, ignore_index=True)
    print("\nLearning curve results:")
    print(learning_curve_df.round(4))
    learning_curve_df.to_csv(output_dir / "ridge_lasso_learning_curves.csv", index=False)

    plot_learning_curves(learning_curve_df, output_dir)
    print(f"\nFigures saved to {output_dir}")


if __name__ == "__main__":
    main()
