"""
Data Import, EDA, PCA and Dataset Export
==========================================

Loads GDSC2 drug response data via TDC, performs exploratory data analysis,
cold-splits by cell line, generates Morgan fingerprints from drug SMILES and
PCA-reduced cell line features, then saves the processed arrays for downstream
model training.

Output:
  - processed_data/model_ready/gdsc2_lgbm_style_data.npz  (X_train, y_train, X_val, y_val, X_test, y_test)
  - processed_data/model_ready/artifacts/                 (scaler, PCA, metadata)
"""

import argparse
import gc
import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tdc.multi_pred import DrugRes

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.max_columns", 120)


def fast_nunique(series, df):
    """Helper for nunique on columns containing numpy arrays (e.g., Cell Line)."""
    if series.name == "Cell Line" and "Cell Line_ID" in df.columns:
        return df["Cell Line_ID"].nunique()
    if series.name == "Drug" and "Drug_ID" in df.columns:
        return df["Drug_ID"].nunique()
    sample = series.dropna().head(5)
    has_vector_value = sample.map(lambda x: isinstance(x, (np.ndarray, list))).any()
    if has_vector_value:
        return "vector column"
    return series.nunique(dropna=True)


def smiles_to_fp(smiles, morgan_bits=1024):
    """Convert a SMILES string to a Morgan fingerprint bit vector."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(morgan_bits, dtype=np.int8)
        return np.array(
            AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=morgan_bits),
            dtype=np.int8,
        )
    except Exception:
        return np.zeros(morgan_bits, dtype=np.int8)


def build_dataset(target_df, pca_map, morgan_bits=1024):
    """Build feature matrix X and target y for a given dataframe split."""
    drug_feats = np.array(
        [smiles_to_fp(s, morgan_bits) for s in target_df["Drug"].values],
        dtype=np.int8,
    )
    cell_feats = np.array(
        [pca_map[cid] for cid in target_df["Cell Line_ID"].values],
        dtype=np.float32,
    )
    X = np.hstack([drug_feats, cell_feats])
    y = target_df["Y"].values.astype(np.float32)
    del drug_feats, cell_feats
    gc.collect()
    return X, y


def main():
    parser = argparse.ArgumentParser(
        description="Load GDSC2 data, perform EDA, PCA, and export processed datasets."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data",
        help="Directory for TDC data cache (default: ../data)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../processed_data/model_ready",
        help="Output directory for processed data (default: ../processed_data/model_ready)",
    )
    parser.add_argument(
        "--morgan_bits",
        type=int,
        default=1024,
        help="Number of Morgan fingerprint bits (default: 1024)",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--train_frac",
        type=float,
        default=0.7,
        help="Training fraction for cold split (default: 0.7)",
    )
    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.1,
        help="Validation fraction for cold split (default: 0.1)",
    )
    parser.add_argument(
        "--test_frac",
        type=float,
        default=0.2,
        help="Test fraction for cold split (default: 0.2)",
    )
    parser.add_argument(
        "--pca_variance",
        type=float,
        default=0.95,
        help="PCA variance to retain (default: 0.95)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    artifact_dir = output_dir / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load Data ──────────────────────────────────────────────────────
    print("Loading GDSC2 data...")
    data = DrugRes(name="GDSC2", path=args.data_dir)
    df = data.get_data()
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    df.info()

    # ── 2. EDA ────────────────────────────────────────────────────────────
    print("\n=== EDA ===")
    eda_summary = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "missing": df.isna().sum(),
            "missing_pct": (df.isna().mean() * 100).round(2),
            "unique": [fast_nunique(df[col], df) for col in df.columns],
        }
    ).sort_values("missing_pct", ascending=False)
    print(eda_summary)

    n_samples = len(df)
    n_drugs = df["Drug_ID"].nunique()
    n_cells = df["Cell Line_ID"].nunique()
    y_stats = df["Y"].describe()
    missing_total = int(df.isna().sum().sum())
    cell_counts = df["Cell Line_ID"].value_counts()
    drug_counts = df["Drug_ID"].value_counts()

    print(f"\nDATASET OVERVIEW")
    print(f"- Dataset co {n_samples:,} mau drug-cell response.")
    print(f"- Co {n_drugs:,} drugs va {n_cells:,} cell lines.")

    print(f"\nTARGET Y")
    print(f"- Mean: {y_stats['mean']:.3f}")
    print(f"- Std:  {y_stats['std']:.3f}")
    print(f"- Min:  {y_stats['min']:.3f}")
    print(f"- Max:  {y_stats['max']:.3f}")

    print(f"\nMISSING VALUES")
    if missing_total == 0:
        print("- Khong co missing value trong dataframe goc.")
    else:
        print(f"- Co {missing_total:,} missing values.")

    print(f"\nDATA IMBALANCE")
    print(
        f"- Cell line xuat hien nhieu nhat co {cell_counts.iloc[0]:,} mau; "
        f"median la {cell_counts.median():.0f} mau/cell line."
    )
    print(
        f"- Drug xuat hien nhieu nhat co {drug_counts.iloc[0]:,} mau; "
        f"median la {drug_counts.median():.0f} mau/drug."
    )

    # ── EDA Figures ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    sns.histplot(df["Y"], kde=True, bins=40, color="#2F6B8F", ax=axes[0, 0])
    axes[0, 0].set_title("Distribution of drug response Y", fontsize=14, weight="bold")
    axes[0, 0].set_xlabel("Y")
    axes[0, 0].set_ylabel("Sample count")

    sns.boxplot(x=df["Y"], color="#7AA95C", ax=axes[0, 1])
    axes[0, 1].set_title("Boxplot of Y", fontsize=14, weight="bold")
    axes[0, 1].set_xlabel("Y")

    cell_counts_top = df["Cell Line_ID"].value_counts().head(20).sort_values()
    cell_counts_top.plot(kind="barh", color="#8E5A9E", ax=axes[1, 0])
    axes[1, 0].set_title("Top 20 cell lines by sample count", fontsize=14, weight="bold")
    axes[1, 0].set_xlabel("Sample count")
    axes[1, 0].set_ylabel("Cell Line_ID")
    for container in axes[1, 0].containers:
        axes[1, 0].bar_label(container, padding=3, fontsize=9)

    drug_counts_top = df["Drug_ID"].value_counts().head(20).sort_values()
    drug_counts_top.plot(kind="barh", color="#C47A3A", ax=axes[1, 1])
    axes[1, 1].set_title("Top 20 drugs by sample count", fontsize=14, weight="bold")
    axes[1, 1].set_xlabel("Sample count")
    axes[1, 1].set_ylabel("Drug_ID")
    for container in axes[1, 1].containers:
        axes[1, 1].bar_label(container, padding=3, fontsize=9)

    for ax in axes.ravel():
        ax.grid(axis="x", alpha=0.25)
        ax.tick_params(labelsize=10)

    plt.tight_layout(pad=2.0)
    fig.savefig(output_dir / "eda_overview.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ── 3. Cold Split ─────────────────────────────────────────────────────
    print("\n=== Splitting data (cold split by Cell Line_ID) ===")
    split = data.get_split(
        method="cold_split",
        column_name="Cell Line_ID",
        seed=args.random_state,
        frac=[args.train_frac, args.val_frac, args.test_frac],
    )
    train_df = split["train"].reset_index(drop=True)
    val_df = split["valid"].reset_index(drop=True)
    test_df = split["test"].reset_index(drop=True)
    print(f"Train: {train_df.shape}")
    print(f"Val:   {val_df.shape}")
    print(f"Test:  {test_df.shape}")

    # ── 4. PCA on Cell Line Features ──────────────────────────────────────
    print("\n=== Fitting PCA on cell line features ===")
    train_unique_cells = train_df.drop_duplicates(subset=["Cell Line_ID"])
    train_matrix = np.array(
        [np.array(val, dtype=np.float32) for val in train_unique_cells["Cell Line"].values]
    )

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_matrix)

    pca = PCA(n_components=args.pca_variance, random_state=args.random_state)
    pca.fit(train_scaled)
    print(
        f"PCA hoan tat: giu lai {pca.n_components_} chieu, "
        f"giai thich {np.sum(pca.explained_variance_ratio_):.2%} phuong sai."
    )

    all_unique_df = df.drop_duplicates(subset=["Cell Line_ID"])
    all_ids = all_unique_df["Cell Line_ID"].values
    all_matrix = np.array(
        [np.array(val, dtype=np.float32) for val in all_unique_df["Cell Line"].values]
    )
    all_pca_feats = pca.transform(scaler.transform(all_matrix))
    cell_pca_map = {cid: feat for cid, feat in zip(all_ids, all_pca_feats)}

    joblib.dump(scaler, artifact_dir / "cell_line_scaler.joblib")
    joblib.dump(pca, artifact_dir / "cell_line_pca.joblib")

    print(f"Unique train cell lines: {len(train_unique_cells)}")
    print(f"All unique cell lines: {len(all_unique_df)}")
    print(f"Cell PCA feature size: {pca.n_components_}")

    del train_matrix, train_scaled, all_matrix, all_unique_df, all_pca_feats
    gc.collect()

    # ── 5. Build X, y ─────────────────────────────────────────────────────
    print("\n=== Building feature matrices ===")
    print("Dang tao ma tran X, y...")
    X_train, y_train = build_dataset(train_df, cell_pca_map, args.morgan_bits)
    X_val, y_val = build_dataset(val_df, cell_pca_map, args.morgan_bits)
    X_test, y_test = build_dataset(test_df, cell_pca_map, args.morgan_bits)

    print(f"X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape}  y_val:   {y_val.shape}")
    print(f"X_test:  {X_test.shape}  y_test:  {y_test.shape}")

    # ── 6. Save ───────────────────────────────────────────────────────────
    print("\n=== Saving processed data ===")
    np.savez_compressed(
        output_dir / "gdsc2_lgbm_style_data.npz",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )

    metadata = {
        "dataset": "GDSC2",
        "split_method": "cold_split by Cell Line_ID",
        "random_state": args.random_state,
        "drug_feature": f"Morgan fingerprint {args.morgan_bits} bits",
        "cell_feature": "StandardScaler + PCA fitted on train cell lines only",
        "cell_pca_components": int(pca.n_components_),
        "X_train_shape": list(X_train.shape),
        "X_val_shape": list(X_val.shape),
        "X_test_shape": list(X_test.shape),
        "saved_npz": str(output_dir / "gdsc2_lgbm_style_data.npz"),
    }

    with open(artifact_dir / "gdsc2_lgbm_style_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(json.dumps(metadata, indent=2))

    # ── 7. Smoke test ─────────────────────────────────────────────────────
    loaded = np.load(output_dir / "gdsc2_lgbm_style_data.npz")
    print(f"\nSmoke test: saved arrays: {loaded.files}")
    print(f"Loaded X_train: {loaded['X_train'].shape}")
    print(f"Loaded y_train: {loaded['y_train'].shape}")


if __name__ == "__main__":
    main()
