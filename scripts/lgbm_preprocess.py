"""
LGBM Preprocess — Add split_id to response.tsv
================================================

Reads the drug response TSV file and adds a 'split_id' column using the row
index, which is needed by the split file format used elsewhere.
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Add split_id column to response.tsv."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="../csa_data/raw_data/y_data/response.tsv",
        help="Path to response.tsv (default: ../csa_data/raw_data/y_data/response.tsv)",
    )
    args = parser.parse_args()

    file_path = Path(args.input)
    print(f"Dang doc file response tu {file_path}...")
    df = pd.read_csv(file_path, sep="\t", low_memory=False)

    # Add split_id using row index
    df["split_id"] = df.index

    # Save back
    df.to_csv(file_path, sep="\t", index=False)

    print("Da them cot 'split_id' thanh cong!")
    print(df[["split_id", "source", "improve_sample_id"]].head())


if __name__ == "__main__":
    main()
