# Giải thích benchmark: fold, luồng chạy, preprocessing và model

File này giải thích luồng hiện tại trong `new_notebook/within_dataset_4models.py` và hai notebook benchmark. Mục tiêu là làm rõ:

- `fold` là gì.
- Dữ liệu được đọc và chia như thế nào.
- Within-dataset và cross-dataset khác nhau ở đâu.
- Mỗi model xử lý dữ liệu đầu vào như thế nào.
- Kết quả R2/RMSE/MAE/Pearson được tính và lưu ra sao.

## 1. Fold là gì?

Trong paper, mỗi dataset được chia thành 10 bộ split độc lập theo kiểu random 10-fold cross-validation. Trong code, `fold` là chỉ số của một bộ chia dữ liệu như vậy.

Ví dụ với `CCLE`, fold 0 gồm ba file:

```text
data/csa_data/raw_data/splits/CCLE_split_0_train.txt
data/csa_data/raw_data/splits/CCLE_split_0_val.txt
data/csa_data/raw_data/splits/CCLE_split_0_test.txt
```

Fold 1 sẽ là:

```text
CCLE_split_1_train.txt
CCLE_split_1_val.txt
CCLE_split_1_test.txt
```

Và cứ thế đến fold 9.

Mỗi split file không chứa dữ liệu đầy đủ. Nó chỉ chứa các row index trỏ vào bảng response:

```text
data/csa_data/raw_data/y_data/response.tsv
```

Code dùng các index này để lấy dòng tương ứng trong `response.tsv`.

Tỉ lệ trong paper:

```text
train: 80%
val:   10%
test:  10%
```

Với mỗi dataset, một fold luôn có ba phần tách rời:

```text
train ∩ val  = rỗng
train ∩ test = rỗng
val ∩ test   = rỗng
train + val + test = toàn bộ dataset
```

Nói đơn giản:

- `fold=0` là lần chia dữ liệu thứ nhất.
- `fold=1` là lần chia dữ liệu thứ hai.
- ...
- `fold=9` là lần chia dữ liệu thứ mười.

Khi chạy full benchmark, model được train/evaluate trên cả 10 fold, sau đó lấy mean/std của metric.

## 2. Các dataset dùng trong benchmark

Trong code:

```python
BENCHMARK_DATASETS = ["gCSI", "CCLE", "GDSCv2", "GDSCv1", "CTRPv2"]
BENCHMARK_FOLDS = list(range(10))
```

Năm dataset là:

```text
gCSI
CCLE
GDSCv2
GDSCv1
CTRPv2
```

Target cần predict là:

```python
TARGET_COL = "auc"
```

Mỗi sample trong response về cơ bản là một cặp:

```text
cell line / cancer sample + drug -> AUC response
```

Các cột ID chính:

```python
CELL_ID_COL = "improve_sample_id"
DRUG_ID_COL = "improve_chem_id"
TARGET_COL = "auc"
```

## 3. Các file dữ liệu chính

Thư mục data chính:

```text
data/csa_data/raw_data/
```

Trong đó có ba nhóm quan trọng:

```text
y_data/response.tsv      # bảng target y
x_data/*.tsv             # bảng feature X
splits/*.txt             # index train/val/test/all
```

Response hiện tại có:

```text
587,709 rows
14 columns
785 unique cell/sample IDs
749 unique drug IDs
0 missing AUC
```

Số sample theo source dataset:

```text
CTRPv2    286,665
GDSCv1    171,940
GDSCv2    114,644
CCLE        9,519
gCSI        4,941
```

Dataset size lệch rất mạnh. Đây là một lý do cross-dataset generalization khó: model train trên dataset nhỏ như gCSI/CCLE thường không học được distribution đủ rộng để transfer tốt sang dataset lớn, và ngược lại target distribution cũng có thể lệch nhiều.

### Response

```text
data/csa_data/raw_data/y_data/response.tsv
```

Đây là bảng y, chứa response drug sensitivity. Code đọc bằng `read_response`.

Các cột chính trong file:

```text
source
improve_sample_id
improve_chem_id
study
auc
ic50
ec50
ec50se
r2fit
einf
hs
aac1
auc1
dss1
```

Trong benchmark hiện tại, target dùng để train/evaluate là:

```text
auc
```

Một dòng response nghĩa là:

```text
dataset source + cell/sample ID + drug ID -> drug response
```

Ví dụ logic, không phải toàn bộ dòng:

```text
source = CCLE
improve_sample_id = ACH-000956
improve_chem_id = Drug_749
auc = 0.7153
```

Nghĩa là drug `Drug_749` được thử trên cell/sample `ACH-000956` trong source `CCLE`, kết quả response là AUC `0.7153`.

Sau khi đọc, code thêm một cột:

```python
global_index
```

Cột này giúp đối chiếu với index gốc. Nhưng việc lấy split vẫn dùng `response.iloc[idx]`, tức là lấy theo vị trí dòng trong bảng response.

Sau khi lấy split, code còn lọc:

```python
df = df[df["source"].eq(dataset)]
df = df.dropna(subset=["auc", "improve_sample_id", "improve_chem_id"])
```

Nghĩa là:

- Chỉ giữ đúng source dataset đang chạy.
- Bỏ dòng thiếu target hoặc thiếu ID.

### Gene expression

```text
data/csa_data/raw_data/x_data/cancer_gene_expression.tsv
```

Đọc bằng `read_gene_expression`.

File này có metadata ở hai dòng đầu, nên code bỏ qua:

```python
skiprows=[1, 2]
```

Index của bảng là `improve_sample_id`.

File này có khoảng `30,806` columns khi đọc raw header. Các feature chính là gene expression theo Ensembl gene ID.

Cấu trúc đặc biệt của file:

```text
row 0: metadata / Entrez-like IDs
row 1: gene symbols
row 2 trở đi: dữ liệu expression theo improve_sample_id
```

Ví dụ header logic:

```text
improve_sample_id | ENSG00000000003 | ENSG00000000005 | ...
metadata row      | 7105            | 64102           | ...
symbol row        | TSPAN6          | TNMD            | ...
ACH-000016       | 4.1898          | 0.0             | ...
```

Vì vậy code dùng:

```python
pd.read_csv(path, sep="\t", skiprows=[1, 2], index_col=0)
```

Tức là bỏ hai dòng metadata/symbol khi lấy numeric matrix. Riêng `read_gene_symbols` đọc hai dòng đầu để map Ensembl ID sang gene symbol khi cần `LINCS_SYMBOL`.

Sau khi đọc, code ép toàn bộ gene expression sang numeric:

```python
ge.apply(pd.to_numeric, errors="coerce")
```

Nếu có giá trị không parse được thành số, nó thành `NaN` và về sau được impute bằng median.

### Mordred drug descriptors

```text
data/csa_data/raw_data/x_data/drug_mordred.tsv
```

Đọc bằng `read_mordred`.

Index của bảng là `improve_chem_id`.

Đây là feature dạng tabular cho drug.

File này có:

```text
improve_chem_id + 1613 Mordred descriptor columns
```

Ví dụ cột:

```text
improve_chem_id
mordred.ABC
mordred.ABCGG
mordred.nAcid
mordred.nBase
mordred.SpAbs_A
...
```

Một dòng nghĩa là:

```text
Drug_1 -> vector descriptor hóa học Mordred
```

Code xử lý:

```python
md = md.drop_duplicates("improve_chem_id").set_index("improve_chem_id")
md = md.apply(pd.to_numeric, errors="coerce")
```

Nếu descriptor bị thiếu hoặc lỗi parse, về sau tabular pipeline sẽ median-impute.

### SMILES

```text
data/csa_data/raw_data/x_data/drug_SMILES.tsv
```

Đọc bằng `read_smiles`.

GraphDRP dùng file này để biến drug thành molecular graph.

File này có hai cột chính:

```text
improve_chem_id
canSMILES
```

Ví dụ:

```text
Drug_10 -> O=P(O)(O)C(O)(Cn1ccnc1)P(=O)(O)O
```

Trong GraphDRP:

1. `improve_chem_id` nối response với SMILES.
2. `canSMILES` được parse bằng RDKit.
3. RDKit molecule được đổi thành graph atom/bond.
4. Graph này đi vào GINConv branch.

Nếu SMILES lỗi hoặc RDKit không parse được, sample tương ứng bị bỏ khỏi GraphDRP dataset.

### Split files

```text
data/csa_data/raw_data/splits/
```

Mỗi dataset có:

```text
{dataset}_all.txt
{dataset}_split_{fold}_train.txt
{dataset}_split_{fold}_val.txt
{dataset}_split_{fold}_test.txt
```

Ví dụ:

```text
CCLE_all.txt
CCLE_split_0_train.txt
CCLE_split_0_val.txt
CCLE_split_0_test.txt
...
CCLE_split_9_train.txt
CCLE_split_9_val.txt
CCLE_split_9_test.txt
```

Các file này chứa integer row index, không chứa `improve_sample_id` hay `improve_chem_id` trực tiếp.

Code đọc bằng:

```python
np.loadtxt(path, dtype=int)
```

rồi lấy dòng trong response:

```python
response.iloc[idx]
```

Fold 0 size thực tế:

```text
Dataset   all       train     val      test
gCSI        4,941     3,953     494      494
CCLE        9,519     7,616     952      951
GDSCv2    114,644    91,716  11,464   11,464
GDSCv1    171,940   137,552  17,194   17,194
CTRPv2    286,665   229,333  28,666   28,666
```

`{dataset}_all.txt` là toàn bộ row index của dataset đó. Với cross-dataset off-diagonal, target test dùng file này.

### Các file X khác chưa dùng trong code hiện tại

Trong `x_data/` còn có nhiều feature khác:

```text
cancer_discretized_copy_number.tsv
cancer_copy_number.tsv
cancer_mutation_count.tsv
cancer_mutation.parquet
cancer_mutation_long_format.tsv
cancer_RPPA.tsv
cancer_miRNA_expression.tsv
cancer_DNA_methylation.tsv
drug_ecfp4_nbits512.tsv
drug_info.tsv
```

Code hiện tại chỉ dùng:

```text
cancer_gene_expression.tsv
drug_mordred.tsv
drug_SMILES.tsv
```

Các file mutation/CNV/RPPA/miRNA/methylation/ECFP4 hiện chưa đi vào benchmark này. Chúng có thể dùng cho model khác hoặc mở rộng sau, nhưng không ảnh hưởng đến kết quả hiện tại.

## 3.1. Data flow từ raw files đến một sample train

Một sample train bắt đầu từ một dòng trong `response.tsv`:

```text
source, improve_sample_id, improve_chem_id, auc
```

Ví dụ logic:

```text
CCLE, ACH-000956, Drug_749, 0.7153
```

Sau đó code tạo feature tùy model.

### Với tabular models

Tabular models gồm:

```text
Ridge
RandomForest
LightGBM
SimpleLinearNN
```

Data flow:

```text
response row
    |
    |-- improve_sample_id -> cancer_gene_expression.tsv -> gene expression vector
    |
    |-- improve_chem_id   -> drug_mordred.tsv           -> Mordred vector
    |
    |-- auc               -> y target
    |
    v
concat(gene expression vector, Mordred vector) -> X row
auc -> y row
```

Nếu một response row không match được gene expression hoặc Mordred feature, nó sẽ bị mất trong inner merge:

```python
data = ids.merge(ge_part, on="improve_sample_id", how="inner")
data = data.merge(md_part, on="improve_chem_id", how="inner")
```

Vì vậy `split_train_rows` có thể lớn hơn `n_train` nếu có sample không đủ feature.

### Với GraphDRP

GraphDRP data flow:

```text
response row
    |
    |-- improve_sample_id -> cancer_gene_expression.tsv -> scaled gene expression vector
    |
    |-- improve_chem_id   -> drug_SMILES.tsv            -> RDKit molecule -> graph
    |
    |-- auc               -> y target
    |
    v
PyTorch Geometric Data(x=edge/node features, edge_index, target=cell vector, y=auc)
```

Nếu drug không có SMILES hợp lệ, hoặc cell không có gene expression, sample bị bỏ.

## 3.2. Vì sao scale/impute chỉ fit trên train?

Trong ML benchmark, mọi transform học từ dữ liệu phải fit trên train, sau đó apply sang val/test/target.

Code làm như vậy:

```python
imputer.fit_transform(X_train)
scaler.fit_transform(train_imputed)
imputer.transform(X_val)
scaler.transform(val_imputed)
imputer.transform(X_test)
scaler.transform(test_imputed)
```

Lý do:

- Không để validation/test leak thông tin vào training.
- Cross-dataset phản ánh đúng tình huống model chỉ biết source train.
- Paper cũng nói target feature có thể thay đổi across folds vì preprocessing phụ thuộc train split.

Với cross-dataset:

```text
source train fit imputer/scaler
source val transform bằng imputer/scaler đó
target all transform bằng imputer/scaler đó
```

Đây là lý do cùng một target dataset có thể có feature values hơi khác giữa fold 0 và fold 1: scaler được fit từ source train của fold khác nhau.

## 4. Within-dataset benchmark là gì?

Within-dataset nghĩa là train, validation và test đều lấy từ cùng một dataset.

Ví dụ:

```text
Dataset = CCLE
Fold = 0

Train: CCLE_split_0_train.txt
Val:   CCLE_split_0_val.txt
Test:  CCLE_split_0_test.txt
```

Luồng trong `run_within_dataset_benchmark`:

```text
for dataset in cfg.datasets:
    for fold in cfg.folds:
        load train_df
        load val_df
        load test_df
        build features
        train each model
        evaluate val and test
        append rows to results
```

Kết quả lưu ra:

```text
new_notebook/results/within_dataset_4models_results.csv
new_notebook/results/within_dataset_4models_summary.csv
```

Summary group theo:

```text
dataset, model
```

và lấy mean/std across folds cho test metric.

## 5. Cross-dataset benchmark là gì?

Cross-dataset nghĩa là model được train trên source dataset, nhưng test trên target dataset khác.

Ví dụ:

```text
Source = CCLE
Target = gCSI
Fold = 0

Train: CCLE_split_0_train.txt
Val:   CCLE_split_0_val.txt
Test:  gCSI_all.txt
```

Nếu source giống target, tức diagonal của matrix G:

```text
Source = CCLE
Target = CCLE

Train: CCLE_split_0_train.txt
Val:   CCLE_split_0_val.txt
Test:  CCLE_split_0_test.txt
```

Nếu source khác target, tức off-diagonal:

```text
Source = CCLE
Target = gCSI

Train: CCLE_split_0_train.txt
Val:   CCLE_split_0_val.txt
Test:  gCSI_all.txt
```

Điều này khớp với paper và IMPROVE CSA workflow.

Luồng trong `run_cross_dataset_benchmark`:

```text
for source_dataset in source_datasets:
    for fold in cfg.folds:
        load source train
        load source val
        select feature columns from source train
        build source train/val features

        for target_dataset in target_datasets:
            if source == target:
                load source split test
            else:
                load target_all.txt

            build target features
            train model
            evaluate on target
```

Kết quả lưu ra:

```text
new_notebook/results/cross_dataset_results.csv
new_notebook/results/cross_dataset_summary.csv
new_notebook/results/cross_dataset_G_r2_mean_{model}.csv
new_notebook/results/cross_dataset_G_r2_std_{model}.csv
new_notebook/results/cross_dataset_Gn_r2_{model}.csv
new_notebook/results/cross_dataset_aggregates_Ga_Gna.csv
```

Heatmap matrix lưu ra:

```text
new_notebook/results/cross_dataset_G_matrix_{model}.png
```

## 6. G matrix, Ga, Gn, Gna là gì?

### G matrix

`G[source, target]` là R2 trung bình across folds khi:

- train trên `source`
- evaluate trên `target`

Ví dụ:

```text
G[CCLE, gCSI]
```

nghĩa là:

```text
train/val từ CCLE
test trên toàn bộ gCSI
mean R2 across folds
```

Diagonal:

```text
G[CCLE, CCLE]
```

là within-dataset performance.

Off-diagonal:

```text
G[CCLE, gCSI]
```

là cross-dataset performance.

### Ga

`Ga[source]` là mean R2 của source đó trên tất cả target khác source.

Ví dụ:

```text
Ga[CCLE] = mean(
    G[CCLE, gCSI],
    G[CCLE, GDSCv2],
    G[CCLE, GDSCv1],
    G[CCLE, CTRPv2]
)
```

Nó đo source dataset đó generalize tốt đến đâu.

### Gn

`Gn[source, target]` là G matrix đã normalize theo within-dataset performance của source:

```text
Gn[source, target] = G[source, target] / G[source, source]
```

Vì vậy muốn `Gn` có ý nghĩa, cần chạy cả diagonal entry, ví dụ `CCLE -> CCLE`.

### Gna

`Gna[source]` là mean của các normalized cross entries:

```text
Gna[source] = mean(Gn[source, target khác source])
```

## 7. Cách chọn feature chung

Trước khi train tabular models, code chọn hai nhóm feature:

```text
gene expression features
Mordred drug descriptor features
```

### Gene expression columns

Hàm:

```python
select_gene_columns(...)
```

Logic:

1. Nếu `cfg.use_lincs_symbol_genes=True`, code cố import:

```python
from improvelib.statics import LINCS_SYMBOL
```

2. Nếu import được, code map gene symbol sang Ensembl column trong gene expression file.

3. Nếu không import được hoặc không map được, code fallback sang top-variance genes từ train cell lines:

```python
top_variance_columns(gene_expression, train_cell_ids, cfg.top_ge_features)
```

Mặc định:

```python
top_ge_features = 512
```

Điểm cần nhớ: nếu không có `improvelib`/`LINCS_SYMBOL`, preprocessing không còn bit-for-bit official nữa. Nó vẫn fair trong benchmark nội bộ vì các model tabular dùng cùng train split để chọn feature, nhưng không phải official reproduction tuyệt đối.

### Mordred columns

Hàm:

```python
top_variance_columns(mordred, train_drug_ids, cfg.top_mordred_features)
```

Mặc định:

```python
top_mordred_features = 512
```

Code chọn các Mordred descriptor có variance cao nhất trên drug xuất hiện trong train split.

### Build tabular matrix

Hàm:

```python
build_tabular_matrix(...)
```

Input:

```text
split_df
gene_expression
mordred
ge_cols
md_cols
```

Quy trình:

1. Lấy các cột ID và target:

```text
improve_sample_id
improve_chem_id
auc
```

2. Merge gene expression theo `improve_sample_id`.

3. Merge Mordred descriptors theo `improve_chem_id`.

4. Sort lại theo `row_id` để giữ thứ tự ban đầu.

5. Trả về:

```text
X: feature matrix
y: auc array
meta: sample/drug/auc metadata
```

Feature column được đặt prefix:

```text
ge.{gene_column}
mordred.{descriptor_column}
```

## 8. Model 1: Ridge

Tên trong config:

```python
"ridge"
```

Input:

```text
gene expression + Mordred descriptors
```

Pipeline:

```python
SimpleImputer(strategy="median")
StandardScaler()
Ridge(alpha=10.0)
```

Luồng:

1. Build `X_train`, `X_val`, `X_test`.
2. Median imputation fit trên train.
3. StandardScaler fit trên train.
4. Ridge fit trên train.
5. Predict trên val và test.
6. Tính metric.

Ridge là linear model có regularization L2. Nó thường ổn trong cross-dataset vì ít overfit hơn tree/deep model.

## 9. Model 2: Lasso

Trong yêu cầu ban đầu có Lasso, nhưng trong file hiện tại `DEFAULT_MODELS` không còn Lasso. Các model hiện tại là:

```python
["ridge", "random_forest", "lightgbm", "graphdrp", "simple_linear_nn"]
```

Nếu muốn đưa Lasso lại, cần thêm vào `get_tabular_models`:

```python
from sklearn.linear_model import Lasso

"lasso": make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    Lasso(alpha=...)
)
```

Hiện tại file `.py` không train Lasso.

## 10. Model 3: KNN

Tương tự Lasso, KNN có trong yêu cầu ban đầu nhưng không còn trong `DEFAULT_MODELS` hiện tại.

Nếu muốn đưa KNN lại, cần thêm:

```python
from sklearn.neighbors import KNeighborsRegressor
```

và pipeline có imputer + scaler + KNN.

Hiện tại file `.py` không train KNN.

## 11. Model 4: Random Forest official-style

Tên trong config:

```python
"random_forest"
```

Input:

```text
gene expression + Mordred descriptors
```

Preprocessing:

```python
fit_transform_tabular_features(...)
```

Gồm:

```python
SimpleImputer(strategy="median")
StandardScaler()
```

Fit imputer/scaler trên train, rồi transform val/test.

Model:

```python
RandomForestRegressor(
    max_depth=None,
    n_estimators=1,
    warm_start=True,
    n_jobs=-1,
    random_state=cfg.random_state,
)
```

Training loop:

```text
round 1: n_estimators = 1
round 2: n_estimators = 2
...
round N: n_estimators = N
```

Mặc định:

```python
random_forest_epochs = 100
random_forest_patience = 50
```

Sau mỗi round:

1. Fit RF.
2. Predict validation.
3. Tính validation MSE.
4. Nếu validation MSE tốt hơn best, lưu `best_model`.
5. Nếu không tốt hơn, tăng early-stop counter.
6. Dừng nếu hết epoch hoặc patience.

Output dùng `best_model`, không nhất thiết là model cuối cùng.

Điểm quan trọng: trong cross-dataset, RF có thể within-val rất tốt nhưng target R2 âm. Điều này không tự động là bug. R2 âm nghĩa là model tệ hơn baseline đoán mean của target dataset.

## 12. Model 5: LightGBM

Tên trong config:

```python
"lightgbm"
```

Input:

```text
gene expression + Mordred descriptors
```

Pipeline:

```python
SimpleImputer(strategy="median")
LGBMRegressor(
    objective="regression",
    n_estimators=800,
    learning_rate=0.05,
    num_leaves=31,
    n_jobs=-1,
    random_state=cfg.random_state,
    verbose=-1,
)
```

Lưu ý:

- Pipeline hiện tại không có `StandardScaler` cho LightGBM.
- Điều này thường ổn vì tree-based model không cần scale feature như linear/NN.
- Nếu `lightgbm` chưa được cài, model bị skip với status `skipped_missing_dependency`.

## 13. Model 6: SimpleLinearNN official-style

Tên trong config:

```python
"simple_linear_nn"
```

Input:

```text
gene expression + Mordred descriptors
```

Preprocessing:

```python
SimpleImputer(strategy="median")
StandardScaler()
```

Sau đó convert sang PyTorch tensor.

Mặc định:

```python
simple_nn_epochs = 300
simple_nn_batch_size = 64
simple_nn_patience = 50
simple_nn_learning_rate = 0.01
simple_nn_dropout = 0.01
simple_nn_model = "default"
```

Optimizer:

```python
torch.optim.SGD
```

Loss:

```python
nn.MSELoss()
```

Default architecture:

```text
input_dim
-> Linear(input_dim, input_dim / 2)
-> LeakyReLU
-> Dropout
-> Linear(input_dim / 2, input_dim / 4)
-> LeakyReLU
-> Linear(input_dim / 4, 1)
```

Có thêm các variant:

```text
small
tiny
large
```

Early stopping:

1. Train một epoch.
2. Predict validation.
3. Tính validation MSE.
4. Nếu tốt hơn best, lưu `state_dict`.
5. Nếu không tốt hơn `simple_nn_patience` lần liên tiếp thì dừng.
6. Load lại best state.
7. Evaluate val/test.

Nếu thiếu `torch`, model bị skip.

## 14. Model 7: GraphDRP official-style

Tên trong config:

```python
"graphdrp"
```

Input:

```text
drug SMILES -> molecular graph
gene expression -> cell feature vector
```

GraphDRP không dùng Mordred. Nó dùng:

```text
drug_SMILES.tsv
cancer_gene_expression.tsv
```

### Drug graph

Hàm nội bộ:

```python
mol_to_graph(drug_id)
```

Quy trình:

1. Lấy SMILES từ `drug_SMILES.tsv`.
2. Parse bằng RDKit:

```python
Chem.MolFromSmiles(smi)
```

3. Mỗi atom thành feature vector.
4. Mỗi bond thành hai directed edges:

```text
i -> j
j -> i
```

5. Trả về PyTorch Geometric `Data` object.

Atom feature gồm:

- atom symbol one-hot
- degree one-hot
- total hydrogens one-hot
- implicit valence one-hot
- aromatic flag

Sau đó normalize atom feature bằng tổng feature.

### Cell feature

GraphDRP dùng gene expression columns `ge_cols`, giống phần chọn gene chung.

Preprocessing:

```python
SimpleImputer(strategy="median")
StandardScaler()
```

Fit trên train cell lines, transform val/test/target cell lines.

### GraphDRP architecture trong code hiện tại

Drug branch:

```text
5 x GINConv
BatchNorm sau mỗi GINConv
global_add_pool
Linear -> 128
Dropout
```

Cell branch:

```text
Conv1d(1 -> 32, kernel=8)
MaxPool1d(3)
Conv1d(32 -> 64, kernel=8)
MaxPool1d(3)
Conv1d(64 -> 128, kernel=8)
MaxPool1d(3)
Linear -> 128
```

Fusion:

```text
concat(drug_128, cell_128)
-> Linear(256 -> 1024)
-> ReLU
-> Dropout
-> Linear(1024 -> 128)
-> ReLU
-> Dropout
-> Linear(128 -> 1)
-> sigmoid
```

Sigmoid được dùng vì target AUC nằm gần khoảng 0 đến 1.

Training:

```python
optimizer = Adam(lr=cfg.graphdrp_learning_rate)
loss = MSELoss()
```

Mặc định:

```python
graphdrp_epochs = 150
graphdrp_batch_size = 256
graphdrp_patience = 20
graphdrp_learning_rate = 1e-4
```

Early stopping dựa trên validation RMSE.

Nếu thiếu `torch`, `torch-geometric`, hoặc `rdkit`, model bị skip.

## 15. Sampling để debug

Config có:

```python
max_train_rows
max_eval_rows
```

Nếu là `None`, dùng toàn bộ data.

Nếu set số cụ thể, ví dụ:

```python
cfg.max_train_rows = 2000
cfg.max_eval_rows = 1000
```

thì code sample ngẫu nhiên để chạy nhanh. Chỉ nên dùng khi debug. Khi chạy kết quả thật cho paper-style benchmark, nên để:

```python
cfg.max_train_rows = None
cfg.max_eval_rows = None
```

## 16. Metrics

Hàm:

```python
regression_metrics(y_true, y_pred)
```

Tính:

```text
n
RMSE
MAE
R2
Pearson correlation
```

R2:

```text
R2 = 1 - MSE(model) / MSE(mean baseline)
```

Vì vậy R2 có thể âm.

R2 âm nghĩa là:

```text
model tệ hơn việc đoán trung bình y_true
```

Trong cross-dataset drug response, R2 âm là chuyện bình thường vì source và target có distribution khác nhau.

## 17. Output rows trong results

Mỗi model trả về các row dạng:

```text
analysis
dataset hoặc source_dataset/target_dataset
fold
stage
model
train_seconds
n_train
n_features
status
n
rmse
mae
r2
pearson
```

`stage` có thể là:

```text
val
test
```

Summary chỉ dùng:

```text
stage == "test"
status == "ok"
```

## 18. Luồng chạy within-dataset đầy đủ

Ví dụ config:

```python
cfg.datasets = ["gCSI", "CCLE", "GDSCv2", "GDSCv1", "CTRPv2"]
cfg.folds = list(range(10))
cfg.models = ["ridge", "random_forest", "lightgbm", "graphdrp", "simple_linear_nn"]
```

Luồng:

```text
1. Read response.tsv
2. Read cancer_gene_expression.tsv
3. Read drug_mordred.tsv
4. Read drug_SMILES.tsv
5. For each dataset:
6.   For each fold:
7.     Load train/val/test split của cùng dataset
8.     Select GE columns từ train cell IDs
9.     Select Mordred columns từ train drug IDs
10.    Build tabular X/y cho train/val/test
11.    Train/evaluate Ridge
12.    Train/evaluate RandomForest
13.    Train/evaluate LightGBM nếu có dependency
14.    Train/evaluate SimpleLinearNN nếu có torch
15.    Train/evaluate GraphDRP nếu có torch-geometric + rdkit
16. Save raw results CSV
17. Group test rows -> summary mean/std
```

## 19. Luồng chạy cross-dataset đầy đủ

Ví dụ config:

```python
source_datasets = BENCHMARK_DATASETS
target_datasets = BENCHMARK_DATASETS
cfg.folds = list(range(10))
```

Luồng:

```text
1. Read response/features
2. For each source dataset:
3.   For each fold:
4.     Load source train
5.     Load source val
6.     Select features dựa trên source train
7.     For each target dataset:
8.       If source == target:
9.         target data = source split test
10.      Else:
11.        target data = target_all.txt
12.      Build target features
13.      Train model using source train/val
14.      Evaluate target
15. Save raw results
16. Summarize into G matrix
17. Compute Ga, Gn, Gna
18. Plot heatmap
```

## 20. Caveat quan trọng trong code hiện tại

### Caveat 1: Cross runner đang train bên trong vòng target

Trong official IMPROVE workflow, model directory là:

```text
model_dir = {source_dataset}/split_{fold}
```

Nghĩa là một model được train cho một cặp:

```text
source dataset + fold
```

rồi infer trên nhiều target.

Trong code hiện tại, training nằm bên trong vòng:

```text
for target_dataset in target_datasets:
```

Với Ridge/RF/LightGBM deterministic, kết quả thường không đổi nhiều vì train data giống nhau. Nhưng với SimpleLinearNN/GraphDRP, nếu muốn benchmark nghiêm và sát official hơn, nên refactor để:

```text
train once per source/fold/model
predict many targets
```

### Caveat 2: Feature selection có thể chưa official tuyệt đối

Official RF/SimpleLinearNN/GraphDRP dùng `LINCS_SYMBOL` subset cho gene expression.

Code hiện tại cố dùng `improvelib.statics.LINCS_SYMBOL`. Nếu môi trường không có `improvelib`, code fallback sang top-variance genes.

Vì vậy:

- Benchmark vẫn chạy được.
- Các model vẫn fair theo cùng split/metric.
- Nhưng kết quả chưa phải official reproduction tuyệt đối.

### Caveat 3: Gn/Gna cần diagonal

Nếu chỉ chạy:

```text
CCLE -> gCSI
```

thì có `G[CCLE, gCSI]`, nhưng chưa có:

```text
G[CCLE, CCLE]
```

Do đó `Gn` và `Gna` chưa có mẫu số within-dataset để normalize. Muốn `Gn/Gna` đúng, cần chạy cả diagonal entries.

## 21. Cách đọc kết quả âm

Ví dụ:

```text
Source = CCLE
Target = gCSI
Model = random_forest
R2 = -0.284
```

Điều này nghĩa là:

```text
MSE(random_forest predictions on gCSI) > MSE(always predict mean AUC of gCSI)
```

Không nhất thiết là bug.

Trong paper, nhiều off-diagonal entries của G matrix có R2 âm. Điều này phản ánh cross-dataset generalization rất khó vì dataset khác nhau về:

- assay protocol
- drug set
- cell line distribution
- response distribution
- feature distribution

## 22. Cách chạy để ra matrix giống paper

Trong `cross_dataset_benchmark.ipynb`, dùng:

```python
source_datasets = BENCHMARK_DATASETS
target_datasets = BENCHMARK_DATASETS
cfg.folds = list(range(10))
cfg.models = ["ridge", "random_forest", "lightgbm", "graphdrp", "simple_linear_nn"]
cfg.max_train_rows = None
cfg.max_eval_rows = None
only_cross_dataset = False
```

Sau đó:

```python
results = run_cross_dataset_benchmark(...)
summary, aggregates = summarize_cross_dataset_results(results, cfg.out_dir)
figures = plot_all_cross_dataset_g_matrices(summary, dataset_order=BENCHMARK_DATASETS, out_dir=cfg.out_dir)
```

Khi chạy đủ source x target x folds, heatmap sẽ đầy ô như hình paper.
