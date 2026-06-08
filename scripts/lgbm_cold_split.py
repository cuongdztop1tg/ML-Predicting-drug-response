"""
LGBM Cold Split — Hyperparameter Tuning, Learning Curves & Bootstrap Eval
==========================================================================

Trains a LightGBM regressor on the GDSC2 drug-response dataset after a cold
split by cell line.  Performs Optuna-based hyperparameter tuning, plots
learning curves, evaluates via bootstrap resampling, and saves the final
model.
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import optuna
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils import resample

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


def get_bootstrap_metrics(y_true, y_pred, n_iterations=100):
    """Bootstrap-resample (y_true, y_pred) pairs to estimate metric variance."""
    metrics = {"r2": [], "pearson": [], "spearman": [], "rmse": []}
    print(f"Dang chay Bootstrapping ({n_iterations} lan)...")
    for i in range(n_iterations):
        y_true_sample, y_pred_sample = resample(y_true, y_pred, random_state=i)
        metrics["r2"].append(r2_score(y_true_sample, y_pred_sample))
        metrics["pearson"].append(pearsonr(y_true_sample, y_pred_sample)[0])
        metrics["spearman"].append(spearmanr(y_true_sample, y_pred_sample)[0])
        metrics["rmse"].append(np.sqrt(mean_squared_error(y_true_sample, y_pred_sample)))
    return metrics


def run_learning_curve(base_model, X_train, y_train, X_val, y_val, X_test, y_test,
                       train_perm, train_fractions):
    """Train model on increasing training-set fractions and collect metrics."""
    rows = []
    n_train = len(y_train)

    for fraction in train_fractions:
        sample_size = max(20, int(n_train * fraction))
        sample_idx = train_perm[:sample_size]

        model = clone(base_model)
        start = time.time()
        model.fit(
            X_train.iloc[sample_idx], y_train[sample_idx],
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        )
        fit_seconds = time.time() - start

        train_pred = model.predict(X_train.iloc[sample_idx])
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        train_metrics = compute_metrics(y_train[sample_idx], train_pred)
        val_metrics = compute_metrics(y_val, val_pred)
        test_metrics = compute_metrics(y_test, test_pred)

        rows.append({
            "model": "LGBM",
            "train_fraction": fraction,
            "train_size": sample_size,
            "val_size": len(y_val),
            "test_size": len(y_test),
            "fit_seconds": fit_seconds,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
        })
    return pd.DataFrame(rows)


def plot_learning_curves(lc_df, output_dir):
    """Plot LGBM learning curves (train/val and train/val/test RMSE)."""
    # Two-panel train/val RMSE
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5), sharey=False)
    axes[0].plot(
        lc_df["train_size"], lc_df["train_rmse"],
        marker="o", linewidth=3.0, markersize=7,
        label="LGBM", color="#1f77b4",
    )
    axes[1].plot(
        lc_df["train_size"], lc_df["val_rmse"],
        marker="o", linewidth=3.0, markersize=7,
        label="LGBM", color="#ff7f0e",
    )
    axes[0].set_title("Train GDSC2")
    axes[1].set_title("Validation GDSC2")
    for ax in axes:
        ax.set_xlabel("Training samples")
        ax.set_ylabel("RMSE")
        ax.tick_params(axis="y", labelleft=True)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
    fig.suptitle("LGBM Train/Validation RMSE Learning Curve", y=1.03,
                 fontsize=16, weight="bold")
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    fig.savefig(output_dir / "lgbm_train_validation_rmse_learning_curve.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Three-panel RMSE
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    metrics = [
        ("train_rmse", "Train RMSE", "RMSE"),
        ("val_rmse", "Validation RMSE", "RMSE"),
        ("test_rmse", "Test RMSE", "RMSE"),
    ]
    for i, (metric_col, title, ylabel) in enumerate(metrics):
        axes[i].plot(
            lc_df["train_size"], lc_df[metric_col],
            marker="o", linewidth=2, label="LGBM", color="#1f77b4",
        )
        axes[i].set_title(title)
        axes[i].set_xlabel("Training samples")
        axes[i].set_ylabel(ylabel)
        axes[i].grid(True, alpha=0.25)
        axes[i].legend()
    fig.suptitle("LGBM Learning Curves - RMSE", fontsize=18, weight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / "lgbm_learning_curves_rmse_plot.png",
                dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_bootstrap_results(y_test, y_pred, bootstrapped_results, output_dir):
    """Actual-vs-predicted scatter, residuals, and R2 stability histograms."""
    r2_mean = np.mean(bootstrapped_results["r2"])
    r2_std = np.std(bootstrapped_results["r2"])
    rmse_mean = np.mean(bootstrapped_results["rmse"])
    rmse_std = np.std(bootstrapped_results["rmse"])

    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.scatter(y_test, y_pred, alpha=0.2, s=10)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             "r-", linewidth=1)
    plt.title(f"Actual vs Predicted\n$R^2$: {r2_mean:.3f} ± {r2_std:.3f}")
    plt.xlabel("Actual log(IC50)")
    plt.ylabel("Predicted log(IC50)")

    plt.subplot(1, 3, 2)
    residuals = y_test - y_pred
    sns.histplot(residuals, kde=True, color="purple")
    plt.axvline(x=0, color="black", linestyle="--")
    plt.title(f"Residuals Distribution\nRMSE: {rmse_mean:.3f} ± {rmse_std:.3f}")
    plt.xlabel("Error")

    plt.subplot(1, 3, 3)
    sns.histplot(bootstrapped_results["r2"], kde=True, color="green")
    plt.axvline(x=r2_mean, color="red", linestyle="-", label=f"Mean: {r2_mean:.3f}")
    plt.title(f"R2 Stability (Bootstrap)\nSTD: {r2_std:.4f}")
    plt.xlabel("R2 Value")
    plt.legend()

    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig(output_dir / "lgbm_bootstrapping_results.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="LGBM training with Optuna tuning, learning curves, and bootstrap eval."
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
        default="../lgbm_results/models",
        help="Output directory for models and figures",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="../lgbm_results/logs",
        help="Directory for Optuna log files",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=50,
        help="Number of Optuna hyperparameter trials",
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=100,
        help="Number of bootstrap iterations for evaluation",
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

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Logging setup ──────────────────────────────────────────────────
    import logging
    log_filename = time.strftime("lgbm_cold_split_%Y%m%d_%H%M%S.log")
    log_filepath = log_dir / log_filename

    optuna_logger = optuna.logging.get_logger("optuna")
    optuna_logger.handlers.clear()
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    optuna_logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    optuna_logger.addHandler(file_handler)
    optuna_logger.setLevel(logging.INFO)

    print(f"Log dang duoc luu tai: {log_filepath}")

    # ── 2. Load data ──────────────────────────────────────────────────────
    loaded = np.load(args.data_path)
    X_train_arr = loaded["X_train"]
    y_train = loaded["y_train"]
    X_val_arr = loaded["X_val"]
    y_val = loaded["y_val"]
    X_test_arr = loaded["X_test"]
    y_test = loaded["y_test"]

    num_features = X_train_arr.shape[1]
    feature_names = [f"feature_{i}" for i in range(num_features)]

    X_train = pd.DataFrame(X_train_arr, columns=feature_names)
    X_val = pd.DataFrame(X_val_arr, columns=feature_names)
    X_test = pd.DataFrame(X_test_arr, columns=feature_names)

    with open(args.metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape}  y_val:   {y_val.shape}")
    print(f"X_test:  {X_test.shape}  y_test:  {y_test.shape}")

    # ── 3. Optuna tuning ──────────────────────────────────────────────────
    def objective(trial):
        param = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "random_state": args.random_state,
            "n_jobs": -1,
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "max_depth": trial.suggest_int("max_depth", 5, 15),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 0.9),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 0.9),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        }
        model = lgb.LGBMRegressor(**param)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        del model, preds
        gc.collect()
        return rmse

    print("\nStarting Optuna hyperparameter tuning...")
    study = optuna.create_study(direction="minimize", study_name="DRP_LGBM_PCA")
    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)

    print("\nNumber of finished trials:", len(study.trials))
    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Value (RMSE): {best_trial.value}")
    print("  Params:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # ── 4. Learning curve preparation ─────────────────────────────────────
    best_params = study.best_params.copy()
    best_params["n_estimators"] = 5000

    base_lgbm = lgb.LGBMRegressor(
        **best_params, random_state=args.random_state, n_jobs=-1
    )
    train_fractions = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

    rng = np.random.default_rng(args.random_state)
    train_perm = rng.permutation(len(y_train))

    # ── 5. Execute learning curve ─────────────────────────────────────────
    print("\nLearning curve for LGBM")
    learning_curve_df = run_learning_curve(
        base_lgbm, X_train, y_train, X_val, y_val, X_test, y_test,
        train_perm, train_fractions,
    )
    print(learning_curve_df.round(4))
    learning_curve_df.to_csv(output_dir / "lgbm_learning_curves.csv", index=False)

    plot_learning_curves(learning_curve_df, output_dir)

    # ── 6. Final model ────────────────────────────────────────────────────
    print("\nTraining final model on full dataset...")
    final_model = clone(base_lgbm)
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=20),
        ],
    )
    y_pred = final_model.predict(X_test)

    # ── 7. Bootstrap evaluation ───────────────────────────────────────────
    bootstrapped_results = get_bootstrap_metrics(y_test, y_pred, args.n_bootstrap)

    r2_mean = np.mean(bootstrapped_results["r2"])
    r2_std = np.std(bootstrapped_results["r2"])
    p_mean = np.mean(bootstrapped_results["pearson"])
    p_std = np.std(bootstrapped_results["pearson"])
    s_mean = np.mean(bootstrapped_results["spearman"])
    s_std = np.std(bootstrapped_results["spearman"])
    rmse_mean = np.mean(bootstrapped_results["rmse"])
    rmse_std = np.std(bootstrapped_results["rmse"])

    print("\n" + "=" * 40)
    print(f"{'Metric':<12} | {'Mean':<10} | {'STD':<10}")
    print("-" * 40)
    print(f"{'R2':<12} | {r2_mean:<10.4f} | {r2_std:<10.4f}")
    print(f"{'Pearson':<12} | {p_mean:<10.4f} | {p_std:<10.4f}")
    print(f"{'Spearman':<12} | {s_mean:<10.4f} | {s_std:<10.4f}")
    print(f"{'RMSE':<12} | {rmse_mean:<10.4f} | {rmse_std:<10.4f}")
    print("=" * 40)

    plot_bootstrap_results(y_test, y_pred, bootstrapped_results, output_dir)

    # ── 8. Save model & feature importance ────────────────────────────────
    model_path = output_dir / "lgbm_cold_split_best.pkl"
    joblib.dump(final_model, model_path)
    print(f"\nDa luu mo hinh tai: {model_path}")

    feat_imp = pd.Series(final_model.feature_importances_).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    feat_imp.head(20).plot(kind="bar")
    plt.title("Top 20 Features Importance")
    plt.tight_layout()
    plt.savefig(output_dir / "lgbm_feature_importance.png", bbox_inches="tight")
    plt.close()

    print("\nAll done.")


if __name__ == "__main__":
    main()
