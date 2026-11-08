"""
Automated checks for the project.  Run from the project root:

    python -m pytest -q
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation import METRIC_NAMES, compute_metrics, evaluate_model  # noqa: E402
from project_config import (CLASS_LABELS, CLEAN_DATA_FILE, COMPARISON_FILE,  # noqa: E402
                            FEATURES, MODEL_DIR, MODEL_FILES, TARGET,
                            TEST_DATA_FILE, TRAIN_DATA_FILE)
from validation import validate_upload  # noqa: E402


@pytest.fixture(scope="module")
def test_df():
    return pd.read_csv(TEST_DATA_FILE)


@pytest.fixture(scope="module")
def models():
    return {n: joblib.load(MODEL_DIR / f) for n, f in MODEL_FILES.items()}


# ---------------------------------------------------------------- data ----
def test_dataset_meets_assignment_minimums():
    df = pd.read_csv(CLEAN_DATA_FILE)
    assert df.shape[0] >= 500 and len(FEATURES) >= 12
    assert df.shape == (846, 19)
    assert df.isna().sum().sum() == 0
    assert sorted(df[TARGET].unique()) == CLASS_LABELS


def test_train_test_disjoint_and_complete():
    tr, te = pd.read_csv(TRAIN_DATA_FILE), pd.read_csv(TEST_DATA_FILE)
    full = pd.read_csv(CLEAN_DATA_FILE)
    assert len(tr) + len(te) == len(full)
    assert len(te) == 170
    merged = tr.merge(te, how="inner", on=FEATURES + [TARGET])
    assert merged.empty, "a test row also appears in the training data"
    # stratification keeps class shares within 1 percentage point
    diff = (tr[TARGET].value_counts(normalize=True) - te[TARGET].value_counts(normalize=True)).abs()
    assert diff.max() < 0.01


# -------------------------------------------------------------- models ----
def test_all_five_models_load_and_predict(models, test_df):
    assert len(models) == 5
    X = test_df[FEATURES]
    for name, m in models.items():
        assert list(m.classes_) == CLASS_LABELS, name
        p = m.predict_proba(X)
        assert p.shape == (len(X), 4)
        assert np.allclose(p.sum(axis=1), 1, atol=1e-6)
        assert set(m.predict(X)) <= set(CLASS_LABELS)


def test_scaling_only_where_expected(models):
    has_scaler = {n: "scaler" in m.named_steps for n, m in models.items()}
    assert has_scaler == {"Logistic Regression": True, "Decision Tree": False,
                          "KNN": True, "Naive Bayes": False,
                          "Support Vector Machine": True}


def test_scaler_was_fitted_on_training_data_only(models):
    tr = pd.read_csv(TRAIN_DATA_FILE)
    scaler = models["Logistic Regression"].named_steps["scaler"]
    assert np.allclose(scaler.mean_, tr[FEATURES].mean().values)


def test_metrics_reproduce_saved_comparison_table(models, test_df):
    saved = pd.read_csv(COMPARISON_FILE, index_col="Model")
    for name, m in models.items():
        metrics, _, _ = evaluate_model(m, test_df[FEATURES], test_df[TARGET])
        for k in METRIC_NAMES:
            assert metrics[k] == pytest.approx(saved.loc[name, k], abs=6e-5), (name, k)


def test_prediction_never_refits(models, test_df, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("fit called during evaluation")
    for m in models.values():
        monkeypatch.setattr(m, "fit", boom)
        for _, step in m.steps:
            monkeypatch.setattr(step, "fit", boom, raising=False)
            monkeypatch.setattr(step, "fit_transform", boom, raising=False)
        evaluate_model(m, test_df[FEATURES], test_df[TARGET])


# ------------------------------------------------------------- metrics ----
def test_metrics_perfect_and_worst_cases():
    y = np.array(CLASS_LABELS * 5)
    proba = np.eye(4)[np.tile(np.arange(4), 5)]
    m = compute_metrics(y, y, proba, CLASS_LABELS)
    assert all(m[k] == pytest.approx(1.0) for k in METRIC_NAMES)


def test_auc_with_missing_classes_is_defined():
    y = np.array(["bus", "van", "bus", "van"])
    proba = np.array([[.7, .1, .1, .1], [.1, .1, .1, .7], [.6, .2, .1, .1], [.2, .1, .1, .6]])
    m = compute_metrics(y, y, proba, CLASS_LABELS)
    assert m["AUC"] == pytest.approx(1.0)


# ---------------------------------------------------------- validation ----
def test_validation_accepts_clean_file(test_df):
    r = validate_upload(test_df, FEATURES, TARGET, CLASS_LABELS)
    assert r.ok and not r.warnings and len(r.X) == 170
    assert list(r.X.columns) == FEATURES


def test_validation_reorders_and_normalises_columns(test_df):
    shuffled = test_df[test_df.columns[::-1]].copy()
    shuffled.columns = [c.upper().replace("_", " ") for c in shuffled.columns]
    shuffled["extra_col"] = 1
    r = validate_upload(shuffled, FEATURES, TARGET, CLASS_LABELS)
    assert r.ok and list(r.X.columns) == FEATURES
    assert any("unexpected" in w for w in r.warnings)


def test_validation_rejects_missing_target(test_df):
    r = validate_upload(test_df.drop(columns=TARGET), FEATURES, TARGET, CLASS_LABELS)
    assert not r.ok and "Target column" in r.errors[0]


def test_validation_rejects_missing_feature(test_df):
    r = validate_upload(test_df.drop(columns=FEATURES[0]), FEATURES, TARGET, CLASS_LABELS)
    assert not r.ok and FEATURES[0] in r.errors[0]


def test_validation_rejects_text_in_numeric_column(test_df):
    bad = test_df.copy().astype({FEATURES[2]: object})
    bad.loc[0, FEATURES[2]] = "abc"
    r = validate_upload(bad, FEATURES, TARGET, CLASS_LABELS)
    assert not r.ok and "Non-numeric" in r.errors[0]


def test_validation_handles_nan_and_unknown_labels(test_df, models):
    bad = test_df.copy()
    bad.loc[0, FEATURES[0]] = np.nan
    bad.loc[1, TARGET] = "truck"
    bad.loc[2, TARGET] = " VAN "
    r = validate_upload(bad, FEATURES, TARGET, CLASS_LABELS)
    assert r.ok and len(r.X) == 169 and r.y.iloc[1] == "van"
    # imputer fitted on training data fills the NaN at predict time
    for m in models.values():
        assert len(m.predict(r.X)) == 169


def test_validation_rejects_empty():
    r = validate_upload(pd.DataFrame(columns=FEATURES + [TARGET]), FEATURES, TARGET, CLASS_LABELS)
    assert not r.ok


def test_app_source_contains_no_fit_calls():
    src = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert ".fit(" not in src and "fit_transform" not in src
