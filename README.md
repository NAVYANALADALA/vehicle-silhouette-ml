# Vehicle Silhouette Classification with Five ML Models

M.Tech (AIML) – Machine Learning, Assignment 2

## a. Problem statement

Predict which of four vehicles (double-decker bus, Chevrolet van, Opel Manta 400, Saab 9000) a 2-D silhouette belongs to, using 18 numeric shape measurements. This is a 4-class classification problem.

Bus and van have distinctive outlines. Opel and Saab are similar-sized saloon cars whose silhouettes overlap, so most errors are expected between those two.

Five classifiers are trained and compared on the same data and split:

- Logistic Regression
- Decision Tree
- K-Nearest Neighbours
- Gaussian Naive Bayes
- Support Vector Machine

Each is scored with Accuracy, AUC, Precision, Recall, F1 and MCC. A Streamlit app loads the saved models and evaluates them on an uploaded test CSV, without retraining.

## b. Dataset description

| Item | Value |
|---|---|
| Dataset | Statlog (Vehicle Silhouettes), UCI ML Repository id 149 – <https://archive.ics.uci.edu/dataset/149/statlog+vehicle+silhouettes> |
| Origin | J. P. Siebert, Turing Institute (1987); CC BY 4.0 |
| Instances | 846 |
| Features | 18 integer shape measurements: compactness, circularity, radius ratio, aspect ratios, scatter ratio, elongatedness, rectangularity, scaled variances, radius of gyration, skewness, kurtosis, hollows ratio |
| Target | `vehicle_class`: bus 218, opel 212, saab 217, van 199 (near-balanced) |
| Missing values / duplicates | 0 / 0 |

The silhouettes come from scale models photographed at many angles, so one vehicle can produce very different shapes.

`prepare_dataset.py` builds `data/vehicle_silhouettes.csv` from either of two sources and checks the shape and the published UCI class counts:

- the original UCI files, placed in `data/raw/`;
- the identical 846 records redistributed by the PMLB project (`data/vehicle.tsv.gz`).

**EDA findings** (`python eda.py`, details in `outputs/eda/eda_report.md`):

- **Strong correlation.** 14 feature pairs have |r| ≥ 0.90. For example, scatter_ratio vs scaled_variance_minor has r = 0.996.
- **Skew and outliers.** Some features are skewed: max_length_aspect_ratio has skew 6.8. There are 55 IQR outliers across 8 features. They are kept as genuine viewing angles.
- **Very different scales.** Values range from about 0–22 to 184–1018, so LR, KNN and SVM need scaling.
- **Opel vs Saab is hard.** The best single-feature ANOVA F is 98 across all classes but only 7 for Opel vs Saab.

## c. GitHub repository link

- Repository: `<<add the public GitHub URL>>`
- Live app: `<<add the Streamlit Community Cloud URL>>`

```
app.py               Streamlit app (loads saved models, no retraining)
training.py          split, tune (5-fold CV), evaluate, save models
evaluation.py        the six metrics, shared by training and the app
validation.py        checks for uploaded CSV files
project_config.py    feature list, labels, seed, paths
prepare_dataset.py   builds the clean CSV from the UCI data
eda.py               exploratory analysis -> outputs/eda/
naive_bayes_ablation.py  extra diagnostic for Naive Bayes
test_data.csv        170-row hold-out test set (features + label)
model/               5 saved pipelines + model_metadata.json
data/                dataset files and training split
outputs/             comparison table, evaluation figures, logs
tests/               pytest suite
requirements.txt
```

Each `.joblib` file is a full scikit-learn Pipeline (imputer → scaler where needed → classifier). The preprocessing fitted on the training data travels with the model.

## d. Models used

**Setup**

- **Split.** One stratified 80:20 split (`random_state=2026`): 676 training rows, 170 test rows. It is shared by all five models.
- **Preprocessing.** The median imputer and `StandardScaler` (used for LR, KNN and SVM) sit inside each Pipeline, so they are fitted on training data only. Trees and Gaussian NB do not depend on feature scale, so they get no scaler.
- **Tuning.** `GridSearchCV`, 5-fold stratified, on the training split only, scoring macro-F1. The test set is used once, at the end.
- **Naive Bayes variant.** GaussianNB, because the features are continuous measurements, not counts.
- **SVM probabilities.** The SVC is wrapped in `CalibratedClassifierCV(method="sigmoid", ensemble=False)` to get probabilities for AUC. This replaces `SVC(probability=True)`, which is deprecated in scikit-learn 1.9.

| Model | Selected hyper-parameters |
|---|---|
| Logistic Regression | C = 10 (L2, lbfgs) |
| Decision Tree | entropy, max_depth = 8, min_samples_leaf = 5, ccp_alpha = 0.02 (pruned to 14 leaves) |
| KNN | k = 9, Manhattan distance, uniform weights |
| Naive Bayes | GaussianNB, var_smoothing = 1e-5 |
| SVM | RBF, C = 1000, γ = 0.03 |

**Metrics.** Precision, recall and F1 are macro averages, so each class counts equally. AUC is one-vs-rest macro, computed from predicted probabilities. MCC is the multiclass Matthews coefficient.

### Comparison table (test set, n = 170)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8176 | 0.9525 | 0.8210 | 0.8187 | 0.8170 | 0.7584 |
| Decision Tree | 0.6588 | 0.8583 | 0.6732 | 0.6584 | 0.6259 | 0.5620 |
| KNN | 0.7235 | 0.9124 | 0.7111 | 0.7248 | 0.7099 | 0.6356 |
| Naive Bayes | 0.4235 | 0.7533 | 0.5031 | 0.4316 | 0.3950 | 0.2577 |
| Support Vector Machine | **0.8412** | **0.9612** | **0.8443** | **0.8417** | **0.8403** | **0.7900** |

These values come from `outputs/model_comparison.csv`. The app shows the same values when `test_data.csv` is uploaded.

**Generalisation check** (macro-F1):

| Model | 5-fold CV (train) | Train | Test |
|---|---:|---:|---:|
| Logistic Regression | 0.797 ± 0.016 | 0.822 | 0.817 |
| Decision Tree | 0.733 ± 0.008 | 0.725 | 0.626 |
| KNN | 0.718 ± 0.027 | 0.792 | 0.710 |
| Naive Bayes | 0.449 ± 0.057 | 0.437 | 0.395 |
| SVM | 0.845 ± 0.036 | 0.997 | 0.840 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Second best on all six metrics (F1 0.817, MCC 0.758). Bus and van are almost perfect (F1 0.93 and 0.99). Most errors are Opel ↔ Saab: 17 Opels are predicted as Saab and 7 Saabs as Opel. Train, CV and test scores are close, so it shows the least overfitting of the five. |
| Decision Tree | Weak (F1 0.626, MCC 0.562). Cross-validation preferred a heavily pruned tree (14 leaves). It recognises only 8 of 42 Opels and sends 25 of them to Saab. Axis-parallel splits suit these correlated features poorly. Its test score is clearly below its CV score (0.733), which points to high variance. |
| KNN | Middle of the five (F1 0.710, MCC 0.636). Scaling is required because feature ranges differ widely. Bus and van recall is high (0.95, 0.93), but Opel recall is only 0.43: car neighbours are often the other car. |
| Naive Bayes | Worst (F1 0.395, MCC 0.258), though still above chance. It predicts "van" for 86 of 170 rows, including 28 of 44 buses. The independence assumption is broken by the strongly correlated features. In `naive_bayes_ablation.py` (CV on training data), decorrelating with PCA-whitening raises F1 from 0.45 to 0.76, while a Gaussianising transform only reaches 0.47. Train and test scores are equally low, which indicates under-fitting rather than over-fitting. |
| Support Vector Machine | Best on all six metrics (Accuracy 0.841, AUC 0.961, F1 0.840, MCC 0.790) and best in CV (0.845). The RBF kernel separates Saab best (recall 0.82). Bus precision is 1.00. The large C fits the training data almost perfectly (train F1 0.997), so its variance is the highest. |
| **Overall winner** | **SVM.** It is best on every metric, including MCC and macro-F1, which account for all classes, and it also ranks first in cross-validation. Its lead over Logistic Regression is small (143 vs 139 correct out of 170), within normal split-to-split variation. Logistic Regression is a simpler, more stable alternative. |

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt # Python 3.11–3.14

python prepare_dataset.py       # optional: rebuild the clean CSV
python eda.py                   # EDA figures + report
python training.py              # trains, evaluates, saves models and outputs/
streamlit run app.py            # then upload test_data.csv in the sidebar
```

Run the tests with `pip install -r requirements-dev.txt` followed by `python -m pytest -q`. The suite checks:

- data and split integrity;
- that the six metrics match an independent implementation;
- upload validation;
- a full app run with all scikit-learn `fit` methods disabled, which proves the app never retrains.

Re-running `training.py` with the pinned library versions reproduces the table above.

## App features

- CSV upload, with a bundled sample file and a schema description.
- Model dropdown for the five classifiers.
- Six metric cards, a confusion matrix and a classification report.
- A "Compare all models" tab, computed on the uploaded file.
- Downloadable predictions with per-class probabilities.
- Clear errors for malformed or incomplete files: missing columns, non-numeric values, unknown labels, empty, binary or oversized files.

## Limitations

- With only 170 test rows, differences of about 0.02 between models are within noise.
- The data covers four specific scale models from 1987, so the results do not transfer to real vehicle recognition.
- The `.joblib` files need the scikit-learn version they were saved with (1.9.1, pinned in `requirements.txt`).

## References

- Siebert, J. P. (1987). *Vehicle Recognition Using Rule Based Methods*. Turing Institute. UCI Statlog (Vehicle Silhouettes).
- Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR* 12.
- Romano, J. D. et al. (2021). PMLB v1.0. *Bioinformatics*.
