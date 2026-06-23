# Revised Report Content for `within_dataset_4models_benchmark.ipynb`

This file contains replacement text for the report sections that are currently inconsistent with the implementation in `new_notebook/within_dataset_4models.py` and `new_notebook/within_dataset_4models_benchmark.ipynb`.

The current notebook evaluates five models, despite the historical filename `within_dataset_4models`:

- Ridge regression
- Random Forest
- LightGBM
- GraphDRP
- SimpleLinearNN

The current pipeline uses IMPROVE/CSA-style benchmark files, official split files, AUC as the target, LINCS/L1000 landmark gene expression features, Mordred drug descriptors for tabular models, and SMILES-derived molecular graphs for GraphDRP.

## Abstract

Machine learning and deep learning models have shown strong potential for cancer drug response prediction using pharmacogenomic data. However, reported performance can vary substantially across datasets, preprocessing choices, feature representations, and evaluation protocols. In this study, we benchmark representative tabular machine learning models and neural models for drug response prediction under a standardized within-dataset evaluation setting. We use five public cancer drug screening datasets, including gCSI, CCLE, GDSCv2, GDSCv1, and CTRPv2, with official train-validation-test split files from the benchmark data. Each sample is represented as a drug-cell line pair and the target response is the normalized area under the dose-response curve (AUC). For tabular models, cell lines are represented by gene expression features and drugs are represented by Mordred chemical descriptors. GraphDRP instead represents drugs as molecular graphs derived from SMILES strings and combines them with gene expression features. Model performance is evaluated using RMSE, MAE, Pearson correlation, and R2 score. This benchmark provides a controlled comparison of conventional machine learning and deep learning approaches for within-dataset drug response prediction.

## 2. Materials and Methods

The drug response prediction framework consists of four main stages: data loading, feature construction, model training, and performance evaluation. Instead of using a newly generated random split, this study follows the official split files provided with the benchmark data. For each dataset and fold, response records are loaded from the shared response table, and train, validation, and test subsets are selected using the corresponding split-index files.

The current implementation evaluates five models: Ridge regression, Random Forest, LightGBM, GraphDRP, and a simple fully connected neural network referred to as SimpleLinearNN. All models are trained to predict the AUC response value for a drug-cell line pair. The models share the same response table, split files, target column, and evaluation metrics, while feature construction is model-specific where required.

## 2.1 Problem Formulation

Drug response prediction is formulated as a supervised regression task. Each sample corresponds to a drug-cell line pair:

```text
(d_i, c_i, y_i)
```

where `d_i` denotes the drug representation, `c_i` denotes the cancer cell-line representation, and `y_i` is the observed drug response value. In this study, `y_i` is the AUC value from the response table. The goal is to learn a prediction function:

```text
y_hat_i = f(d_i, c_i)
```

For tabular models, the drug and cell-line representations are concatenated into a single numerical vector:

```text
x_i = [d_i; c_i]
```

GraphDRP uses a different representation: the drug is encoded as a molecular graph and the cell line is encoded as a gene expression vector. The model learns separate representations for the graph and the cell-line vector before combining them for final AUC prediction.

## 2.2 Benchmark Dataset

The benchmark data are stored in three main groups: response data, feature data, and split files. The response data are loaded from:

```text
data/csa_data/raw_data/y_data/response.tsv
```

The feature tables are loaded from:

```text
data/csa_data/raw_data/x_data/
```

The official split files are loaded from:

```text
data/csa_data/raw_data/splits/
```

The response table contains five public drug screening datasets: gCSI, CCLE, GDSCv2, GDSCv1, and CTRPv2. Each row corresponds to a measured drug response for a cancer sample and drug pair. The key identifiers are:

```text
improve_sample_id
improve_chem_id
source
auc
```

The column `improve_sample_id` links response records to cell-line molecular features, `improve_chem_id` links response records to drug features, `source` identifies the dataset, and `auc` is the prediction target.

The dataset sizes used by the current benchmark are:

| Dataset | Drugs | Cell lines | Responses |
|---|---:|---:|---:|
| gCSI | 16 | 312 | 4,941 |
| CCLE | 24 | 411 | 9,519 |
| GDSCv2 | 168 | 470 | 114,644 |
| GDSCv1 | 294 | 546 | 171,940 |
| CTRPv2 | 494 | 720 | 286,665 |

## 2.3 Data Partitions

The current implementation does not generate new TDC cold splits. Instead, it uses the official split files included with the benchmark data. For each dataset, ten folds are available. Each fold consists of three files:

```text
{dataset}_split_{fold}_train.txt
{dataset}_split_{fold}_val.txt
{dataset}_split_{fold}_test.txt
```

Each split file contains integer row indices into `response.tsv`. The implementation loads these indices and selects the corresponding response rows with `response.iloc[idx]`, then filters rows to keep only the requested source dataset. The train, validation, and test subsets are therefore fixed by the benchmark split files.

For fold 0, the split sizes are:

| Dataset | Train | Validation | Test |
|---|---:|---:|---:|
| gCSI | 3,953 | 494 | 494 |
| CCLE | 7,616 | 952 | 951 |
| GDSCv2 | 91,716 | 11,464 | 11,464 |
| GDSCv1 | 137,552 | 17,194 | 17,194 |
| CTRPv2 | 229,333 | 28,666 | 28,666 |

Validation data are used for early stopping in Random Forest, GraphDRP, and SimpleLinearNN. Test data are reserved for final performance reporting.

## 2.4 Feature Construction

Feature construction converts each response record into model-ready inputs. Each response row contains a dataset source, a cell-line identifier, a drug identifier, and the target AUC value:

```text
source, improve_sample_id, improve_chem_id, auc
```

The identifiers are used as keys to retrieve the corresponding cell-line and drug features. The current implementation uses three feature files:

```text
cancer_gene_expression.tsv  -> cell-line gene expression
drug_mordred.tsv            -> fixed-length drug descriptors for tabular models
drug_SMILES.tsv             -> SMILES strings for GraphDRP molecular graphs
```

Although the benchmark data directory contains additional omics features and drug fingerprints, the current within-dataset notebook does not use mutation, copy number, methylation, RPPA, miRNA, or ECFP features. This is important because the feature construction described here is narrower and more specific than the broader data resources available in the folder.

The feature construction procedure is split into two paths. Ridge, Random Forest, LightGBM, and SimpleLinearNN use a tabular representation formed by concatenating gene expression and Mordred descriptors. GraphDRP uses a graph-based representation in which each drug is converted from SMILES into a molecular graph and combined with a gene expression vector.

### 2.4.1 Cell-Line Gene Expression Features

Cell lines are represented using the gene expression table:

```text
data/csa_data/raw_data/x_data/cancer_gene_expression.tsv
```

The table is indexed by `improve_sample_id`, which is the same cell-line identifier used in the response table. The file contains metadata rows near the top, followed by numerical expression values. Therefore, the implementation skips the metadata rows when reading the numerical expression matrix and converts all expression values to numeric format. Non-numeric values are converted to missing values and handled later by imputation.

The expression columns in the data file are Ensembl gene identifiers, while the selected gene set is stored as gene symbols. To connect these two formats, the code also reads the gene-symbol metadata from the same expression file and builds a mapping:

```text
gene symbol -> Ensembl gene ID -> expression column
```

For the current benchmark, the selected gene set is the built-in LINCS/L1000 landmark gene list. The implementation stores 976 landmark gene symbols and maps them to the Ensembl gene identifiers present in the expression matrix. On the current CSA data, 958 landmark genes are successfully mapped and used. The notebook explicitly checks this number before running the benchmark. If only 512 genes are selected, this indicates that the old top-variance fallback is being used instead of the intended LINCS/L1000 feature set.

This gene selection step is performed before constructing the model-specific matrices. For a given dataset and fold, the same selected gene columns are used for train, validation, and test samples. The selected expression vector for a cell line can be written as:

```text
c_i = [g_1, g_2, ..., g_958]
```

where each `g_j` is the expression value of one mapped LINCS/L1000 landmark gene for cell line `i`.

Missing gene expression values are handled by median imputation. The imputer is fitted only on the training split and then applied to validation and test splits. Standardization is also fitted only on the training split and then applied to validation and test data. This gives the following preprocessing flow:

```text
train gene expression -> fit median imputer -> fit standard scaler
val/test gene expression -> transform using train-fitted imputer and scaler
```

This design prevents information from validation or test samples from leaking into the training preprocessing statistics.

### 2.4.2 Drug Descriptors for Tabular Models

Ridge, Random Forest, LightGBM, and SimpleLinearNN use Mordred descriptors as drug features. These descriptors are loaded from:

```text
data/csa_data/raw_data/x_data/drug_mordred.tsv
```

The table is indexed by `improve_chem_id`, which is the same drug identifier used in the response table. Mordred descriptors are fixed-length numerical chemical descriptors computed from molecular structure. They include many physicochemical and topological properties of each compound.

The implementation first removes duplicate drug rows, sets `improve_chem_id` as the index, and converts descriptor columns to numeric values. Missing or invalid descriptor values are kept as missing values at this stage and are later handled by median imputation in the tabular preprocessing pipeline.

Because the full Mordred table contains more descriptors than needed for the current experiment, the code selects the top-variance Mordred descriptors using only drugs present in the training split. In the current notebook configuration, the maximum number of Mordred descriptors is:

```text
top_mordred_features = 512
```

Using only the training split for feature selection avoids using validation or test distribution information to decide which drug descriptors should be retained. The selected Mordred vector for drug `i` can be written as:

```text
d_i = [m_1, m_2, ..., m_512]
```

where each `m_j` is one selected Mordred descriptor.

### 2.4.3 Tabular Feature Integration

For Ridge, Random Forest, LightGBM, and SimpleLinearNN, each response row is converted into a single tabular input vector by joining the response table with the gene expression table and the Mordred descriptor table:

```text
response row
    |
    |-- improve_sample_id -> LINCS/L1000 gene expression vector
    |-- improve_chem_id   -> selected Mordred descriptor vector
    |-- auc               -> target value
    v
concatenated tabular input vector
```

The final tabular feature vector is:

```text
x_i = [c_i; d_i]
```

In the current configuration, this produces approximately:

```text
958 gene expression features + 512 Mordred features = 1,470 tabular features
```

The implementation performs the joins using inner merges. Therefore, a response row is retained only if both required feature records are available:

```text
response.improve_sample_id must exist in cancer_gene_expression.tsv
response.improve_chem_id must exist in drug_mordred.tsv
```

If either the cell-line expression vector or the drug descriptor vector is missing, that response row is excluded from the tabular model input. This is why the raw split size and the final number of usable training rows may differ.

After the tabular matrices are built, missing feature values are median-imputed and standardized. For Ridge and LightGBM, this imputation is part of the scikit-learn pipeline. For Random Forest and SimpleLinearNN, the implementation explicitly fits the imputer and scaler on the training matrix and transforms the validation and test matrices with the same fitted objects.

The target vector is always the AUC column:

```text
y_i = auc_i
```

### 2.4.4 Drug Molecular Graphs for GraphDRP

GraphDRP does not use Mordred descriptors. Instead, it constructs molecular graphs from canonical SMILES strings loaded from:

```text
data/csa_data/raw_data/x_data/drug_SMILES.tsv
```

The SMILES table is linked to the response table through `improve_chem_id`. For each drug, the implementation reads the canonical SMILES string and parses it using RDKit. If RDKit cannot parse the SMILES string, or if the molecule has no atoms or no bonds, the corresponding sample is skipped for GraphDRP.

For valid molecules, atoms become graph nodes and chemical bonds become graph edges. Each atom is represented by a normalized feature vector consisting of:

- atom type
- atom degree
- total number of hydrogens
- implicit valence
- aromaticity indicator

The implementation uses predefined allowable sets for atom type, degree, number of hydrogens, and implicit valence. Unknown atom types are mapped to an `Unknown` category. Each bond is represented in both directions, producing a directed edge index suitable for PyTorch Geometric:

```text
atom u -- atom v  ->  edges (u, v) and (v, u)
```

The drug graph for compound `i` can be written as:

```text
G_i = (V_i, E_i, X_i)
```

where `V_i` is the set of atoms, `E_i` is the set of graph edges derived from bonds, and `X_i` is the atom feature matrix.

Each GraphDRP sample is represented as a PyTorch Geometric data object containing:

```text
x          -> atom feature matrix
edge_index -> molecular graph connectivity
target     -> cell-line gene expression vector
y          -> AUC target
```

GraphDRP uses the same selected LINCS/L1000 gene expression columns as the other models. However, instead of concatenating the cell-line vector with a drug descriptor vector before training, GraphDRP passes the drug graph through a graph neural network branch and passes the gene expression vector through a separate neural branch. The two learned representations are concatenated inside the model.

The GraphDRP feature construction flow is:

```text
response row
    |
    |-- improve_sample_id -> LINCS/L1000 gene expression vector -> impute/scale
    |-- improve_chem_id   -> SMILES -> RDKit molecule -> molecular graph
    |-- auc               -> target value
    v
PyTorch Geometric Data object
```

As with the tabular models, gene expression imputation and scaling for GraphDRP are fitted only on the training cell lines and then applied to validation and test cell lines. This keeps preprocessing consistent with the within-dataset train-validation-test protocol.

### 2.4.5 Summary of Feature Inputs by Model

The current feature usage can be summarized as follows:

| Model | Cell-line input | Drug input | Final representation |
|---|---|---|---|
| Ridge | LINCS/L1000 gene expression | Mordred descriptors | Concatenated tabular vector |
| Random Forest | LINCS/L1000 gene expression | Mordred descriptors | Concatenated tabular vector |
| LightGBM | LINCS/L1000 gene expression | Mordred descriptors | Concatenated tabular vector |
| SimpleLinearNN | LINCS/L1000 gene expression | Mordred descriptors | Concatenated tabular vector |
| GraphDRP | LINCS/L1000 gene expression | SMILES-derived molecular graph | Neural fusion of cell vector and drug graph |

Therefore, all models share the same response target and the same cell-line gene expression source, but they do not all use the same drug representation. The tabular models use Mordred descriptors, while GraphDRP uses molecular graphs constructed from SMILES strings.

## 2.5 Models

The benchmark compares five models with different levels of complexity.

### Ridge Regression

Ridge regression is used as a linear baseline for tabular drug response prediction. It receives the concatenated gene expression and Mordred descriptor vector. The pipeline applies median imputation, standardization, and Ridge regression with `alpha = 10.0`.

### Random Forest

Random Forest is used as a nonlinear tree-based baseline. It receives the same tabular input as Ridge regression. The current implementation follows an early-stopping style procedure by increasing the number of trees with warm start and monitoring validation mean squared error. The best validation model is retained and evaluated on the test set.

### LightGBM

LightGBM is used as a gradient boosting baseline. It receives the same tabular input as Ridge and Random Forest. The current configuration uses a regression objective, 800 estimators, learning rate 0.05, 31 leaves, and the fixed random seed used across the benchmark.

### GraphDRP

GraphDRP is a deep learning model that combines a drug molecular graph branch and a cell-line gene expression branch. The drug branch uses multiple GINConv graph convolution layers followed by global add pooling. The cell-line branch applies one-dimensional convolution layers to the gene expression vector. The learned drug and cell-line representations are concatenated and passed through fully connected layers to predict AUC. The model is trained with mean squared error loss and Adam optimization. Validation loss is used for early stopping.

### SimpleLinearNN

SimpleLinearNN is a fully connected PyTorch neural network for tabular inputs. It uses the same concatenated gene expression and Mordred descriptor vector as the tabular models. The default architecture contains two hidden layers whose widths are derived from the input dimension, LeakyReLU activations, dropout, and a final linear output layer. The model is trained with mean squared error loss and stochastic gradient descent. Validation loss is used for early stopping.

## 2.6 Experimental Workflow

For each dataset and fold, the workflow proceeds as follows:

1. Load response records from `response.tsv`.
2. Load train, validation, and test row indices from the official split files.
3. Filter response rows to the target dataset and remove rows with missing target or identifiers.
4. Select LINCS/L1000 gene expression features.
5. Select top-variance Mordred descriptors for tabular models.
6. Build tabular matrices for Ridge, Random Forest, LightGBM, and SimpleLinearNN.
7. Build SMILES-derived molecular graph datasets for GraphDRP.
8. Train each model on the training split.
9. Use validation data for early stopping or model selection where implemented.
10. Evaluate final performance on the test split.

The default notebook configuration runs a smoke-test experiment on CCLE folds 0, 1, and 2. The full paper-style within-dataset benchmark uses all five datasets and all ten folds:

```text
datasets = ["gCSI", "CCLE", "GDSCv2", "GDSCv1", "CTRPv2"]
folds = [0, 1, ..., 9]
```

## 2.7 Evaluation Metrics

Model performance is evaluated as a regression task using RMSE, MAE, Pearson correlation, and R2 score. For each model, metrics are computed separately for validation and test predictions. The primary test metric is R2, while RMSE, MAE, and Pearson correlation provide complementary views of prediction error and ranking agreement.

The metrics are computed after removing non-finite predictions or targets. Results are saved to:

```text
new_notebook/results/within_dataset_4models_results.csv
```

The benchmark contract describing the active datasets, folds, target, split files, metrics, and model-specific preprocessing is saved to:

```text
new_notebook/results/benchmark_contract.json
```

## Discussion Text to Replace Outdated Claims

The current benchmark should be interpreted as a within-dataset comparison under fixed official split files. Because each model is evaluated using the same response table, split definitions, target variable, and metrics, differences in performance primarily reflect model capacity and model-specific feature representations rather than differences in evaluation protocol.

However, preprocessing is not identical across all models. Ridge, Random Forest, LightGBM, and SimpleLinearNN use concatenated LINCS/L1000 gene expression features and Mordred drug descriptors, whereas GraphDRP uses LINCS/L1000 gene expression features and molecular graphs derived from SMILES strings. This design reflects the natural input requirements of each model but should be considered when interpreting model comparisons.

The current implementation does not use PCA-transformed gene expression features, Morgan fingerprints, TDC cold splits, mutation features, copy-number features, methylation features, RPPA, or miRNA expression. These data types exist in the broader benchmark data but are not part of the current `within_dataset_4models_benchmark.ipynb` experiment.

## Limitations

This study has several limitations. First, the current within-dataset setup evaluates models on fixed benchmark splits from the same source dataset, so it does not by itself measure generalization to completely unseen datasets or external experimental protocols. Second, model inputs are not identical across all methods: GraphDRP uses molecular graphs, while the other models use Mordred descriptors. Third, the hyperparameters are limited and are not the result of an exhaustive search. Fourth, the default notebook configuration is a smoke test on CCLE folds 0 to 2; full conclusions require running all five datasets and all ten folds. Finally, while additional omics and drug features are present in the benchmark data, the current experiment uses only gene expression, Mordred descriptors, and SMILES-derived molecular graphs.

## Items From the Old Report That Should Be Removed or Rewritten

The following old claims are inconsistent with the current notebook and should be removed or rewritten:

- The benchmark uses Therapeutics Data Commons cold-split generation.
- The split is 70% train, 10% validation, and 20% test with a newly fixed seed of 42.
- Drug features are Morgan fingerprints or ECFP vectors for all experiments.
- Gene expression is reduced using PCA retaining 95% variance.
- All models use the same concatenated fingerprint-plus-PCA representation.
- The selected models are DeepCDR, HiDRA, tCNNs, UNO, and LGBM.
- The current experiment compares six models consisting of five external deep learning models plus LGBM.
- Mutation, CNV, methylation, RPPA, or miRNA features are used in the current benchmark.
