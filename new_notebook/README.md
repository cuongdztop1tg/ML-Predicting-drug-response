# New Benchmark Notebooks

This folder contains clean notebooks for reproducing the benchmark workflow from the IMPROVE cross-dataset DRP paper.

Benchmark fairness is defined by the paper-style contract:

- same CSA/IMPROVE response table
- same official train/validation/test split files
- same target (`auc`)
- same test metrics, with primary comparison by mean/std `R2` across folds

Preprocessing is model-specific, because the models expect different input representations. The tabular baselines and SimpleLinearNN use gene-expression plus Mordred drug-descriptor matrices, while GraphDRP uses drug molecular graphs from SMILES plus gene-expression features.

Start with:

- `within_dataset_benchmark.ipynb`: within-dataset benchmark using official IMPROVE train/val/test split files.
- `within_dataset_4models_benchmark.ipynb`: current five-model within-dataset benchmark with Ridge, Random Forest, LightGBM, GraphDRP, and SimpleLinearNN.
- `cross_dataset_benchmark.ipynb`: cross-dataset benchmark using source train/val splits and target `{dataset}_all.txt` files, matching the paper/IMPROVE CSA protocol.

The first notebook is intentionally configured for a small smoke run (`CCLE`, fold `0`). After it works, change:

```python
DATASETS = ["CCLE", "CTRPv2", "gCSI", "GDSCv1", "GDSCv2"]
FOLDS = list(range(10))
```

In Python, the equivalent full within-dataset config is:

```python
from within_dataset_4models import make_paper_config

cfg = make_paper_config()
```

Outputs are written to `new_notebook/results/`.
