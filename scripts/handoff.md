# Handoff — Notebook-to-Script Refactor

## What was done

All 7 Jupyter notebooks in `/notebooks` were converted to standalone Python scripts in `/scripts`. The folder structure is flat on both sides (no subdirectories).

## Logic-preservation verification

Each `.py` script was checked against the source `.ipynb` along these dimensions:

| Check | Method |
|---|---|
| Import equivalence | All imports in source notebook appear in output script; no missing dependencies |
| Function & class extraction | Inline notebook logic extracted into named functions where it was reused across cells (e.g. `compute_metrics`, `run_learning_curve`, `build_dataset`) |
| `display()` / `plt.show()` removal | Replaced with `plt.close(fig)` — all figures were already calling `savefig()` in the originals |
| Magic commands removed | `%matplotlib`, etc. stripped |
| Code paths preserved | All branches, loops, and conditionals from the notebook cells are present in the scripts |
| Hardcoded values → args | Paths, hyperparameters, random seeds, file sizes etc. moved to `argparse` with original-stated defaults |
| Entry point added | Every script has `if __name__ == "__main__": main()` |
| Module docstring | Brief English/Vietnamese docstring at the top of each file |

## Post-refactor checks

All 7 scripts pass `conda run -n ml_drug_predict python -m py_compile` with zero errors.

## File map

| Script | Source Notebook | argparse Configs |
|---|---|---|
| `data_eda_pca_preparation.py` | `notebooks/data_eda_pca_preparation.ipynb` | `--morgan_bits` (1024), `--pca_variance` (0.95), split fractions, paths |
| `linear_regression.py` | `notebooks/linear_regression.ipynb` | `--data_path`, `--output_dir`, `--random_state` |
| `ridge_lasso_learning_curves.py` | `notebooks/ridge_lasso_learning_curves.ipynb` | `--ridge_alphas`, `--lasso_alphas`, paths |
| `random_forest_knn_learning_curves.py` | `notebooks/random_forest_knn_learning_curves.ipynb` | `--max_train_size` (20000), `--eval_size` (5000), paths |
| `lgbm_preprocess.py` | `notebooks/lgbm_preprocess.ipynb` | `--input` TSV path |
| `lgbm_cold_split.py` | `notebooks/lgbm_cold_split.ipynb` | `--n_trials` (50), `--n_bootstrap` (100), paths |
| `best_models_learning_curve_comparison.py` | `notebooks/best_models_learning_curve_comparison.ipynb` | `--data_path`, `--output_dir`, `--random_state` |

## For the next person

- The conda environment is `ml_drug_predict`.
- Original notebooks remain in `/notebooks` — nothing was deleted.
- To test a script end-to-end (requires processed data): `conda run -n ml_drug_predict python scripts/linear_regression.py`
- The data-preparation pipeline must be run first (`data_eda_pca_preparation.py`) to generate the `.npz` files consumed by all other scripts.
- `lgbm_cold_split.py` runs Optuna (50 trials by default) and can take a long time.
