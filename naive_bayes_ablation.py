"""
Optional diagnostic - WHY does Gaussian Naive Bayes perform poorly here?

Hypothesis A: features are not Gaussian (skewed, heavy tails).
Hypothesis B: features are strongly correlated, so the "naive" conditional
              independence assumption counts the same evidence many times.

Test each by fixing only that problem and measuring 5-fold CV macro-F1 on
the TRAINING split (the test set is not touched):
  * Yeo-Johnson power transform -> makes features more Gaussian (A)
  * PCA with whitening          -> makes features uncorrelated (B)

This is analysis only; the submitted Naive Bayes model is unchanged.
Run:  python naive_bayes_ablation.py
"""
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler

from project_config import (CV_FOLDS, EVAL_DIR, FEATURES, RANDOM_STATE, TARGET,
                            TRAIN_DATA_FILE)

train = pd.read_csv(TRAIN_DATA_FILE)
X, y = train[FEATURES], train[TARGET]
cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

variants = {
    "GaussianNB on raw features (submitted)": GaussianNB(),
    "Yeo-Johnson -> GaussianNB (fixes skew)": make_pipeline(PowerTransformer(), GaussianNB()),
    "PCA whitening -> GaussianNB (removes correlation)":
        make_pipeline(StandardScaler(), PCA(whiten=True, random_state=RANDOM_STATE), GaussianNB()),
}
rows = []
for name, model in variants.items():
    s = cross_val_score(model, X, y, cv=cv, scoring="f1_macro")
    rows.append({"Variant": name, "CV macro-F1 mean": s.mean(), "CV macro-F1 std": s.std()})
res = pd.DataFrame(rows).set_index("Variant").round(4)
print(res.to_string())
res.to_csv(EVAL_DIR / "naive_bayes_ablation.csv")
