"""
Extra checks: upload parsing, independent metric re-implementation,
comparison on uploaded data, and a full Streamlit run with every training
method disabled (proves the app never retrains).

Run from the project root:  python -m pytest -q
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import METRIC_NAMES, compare_models, compute_metrics  # noqa: E402
from project_config import (CLASS_LABELS, COMPARISON_FILE, FEATURES, MODEL_DIR,  # noqa: E402
                            MODEL_FILES, TARGET, TEST_DATA_FILE, TRAIN_DATA_FILE)
from validation import read_uploaded_csv  # noqa: E402


@pytest.fixture(scope="module")
def test_df():
    return pd.read_csv(TEST_DATA_FILE)


@pytest.fixture(scope="module")
def models():
    return {n: joblib.load(MODEL_DIR / f) for n, f in MODEL_FILES.items()}


# ------------------------------------------------------- upload parsing ----
def test_read_valid_csv_bytes():
    df, err = read_uploaded_csv(TEST_DATA_FILE.read_bytes(), "test_data.csv")
    assert err is None and df.shape == (170, 19)


@pytest.mark.parametrize("data,name,needle", [
    (b"", "x.csv", "empty"),
    (b"a,b\n1,2\n", "x.xlsx", ".csv extension"),
    (b"PK\x03\x04\x00\x00binary", "x.csv", "binary"),
    (b"a;b;c\n1;2;3\n", "x.csv", "Only one column"),
    (b'a,b\n"1,2\n3,4\n', "x.csv", "malformed"),
    (b"\n\n", "x.csv", "no columns"),
])
def test_read_rejects_bad_files(data, name, needle):
    df, err = read_uploaded_csv(data, name)
    assert df is None and needle in err


def test_read_rejects_oversized_file():
    df, err = read_uploaded_csv(b"a,b\n" + b"1,2\n" * 10, "x.csv", max_bytes=20)
    assert df is None and "limit" in err


def test_read_accepts_latin1_and_bom():
    df, err = read_uploaded_csv("﻿col_é,b\n1,2\n".encode("utf-8"), "x.csv")
    assert err is None and list(df.columns) == ["col_é", "b"]
    df, err = read_uploaded_csv("col_é,b\n1,2\n".encode("latin-1"), "x.csv")
    assert err is None and df.shape == (1, 2)


# ------------------------------- dataset / split sanity -----------------
def test_no_duplicate_rows_and_valid_test_schema(test_df):
    full = pd.concat([pd.read_csv(TRAIN_DATA_FILE), test_df])
    assert not full.duplicated().any()
    assert list(test_df.columns) == FEATURES + [TARGET]
    assert test_df[FEATURES].notna().all().all()
    assert set(test_df[TARGET]) == set(CLASS_LABELS)


# ------------------ independent re-implementation of the metrics ----------
def _manual_metrics(y, y_pred, proba, labels):
    k = len(labels)
    idx = {c: i for i, c in enumerate(labels)}
    C = np.zeros((k, k))
    for t, p in zip(y, y_pred):
        C[idx[t], idx[p]] += 1
    tp = np.diag(C)
    prec = np.divide(tp, C.sum(0), out=np.zeros(k), where=C.sum(0) > 0)
    rec = tp / C.sum(1)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(k), where=(prec + rec) > 0)
    # Gorodkin multiclass MCC
    t_k, p_k, c, s = C.sum(1), C.sum(0), tp.sum(), C.sum()
    mcc = (c * s - t_k @ p_k) / np.sqrt((s**2 - p_k @ p_k) * (s**2 - t_k @ t_k))
    # OvR AUC via Mann-Whitney U (probability a positive outranks a negative)
    aucs = []
    for i, cls in enumerate(labels):
        pos, neg = proba[np.asarray(y) == cls, i], proba[np.asarray(y) != cls, i]
        diff = pos[:, None] - neg[None, :]
        aucs.append(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)
    return {"Accuracy": c / s, "AUC": np.mean(aucs), "Precision": prec.mean(),
            "Recall": rec.mean(), "F1": f1.mean(), "MCC": mcc}


@pytest.mark.parametrize("name", list(MODEL_FILES))
def test_metrics_match_independent_implementation(models, test_df, name):
    m = models[name]
    X, y = test_df[FEATURES], test_df[TARGET].values
    assert list(m.classes_) == CLASS_LABELS          # probability column order
    proba, pred = m.predict_proba(X), m.predict(X)
    ours = compute_metrics(y, pred, proba, CLASS_LABELS)
    ref = _manual_metrics(y, pred, proba, CLASS_LABELS)
    for k in METRIC_NAMES:
        assert ours[k] == pytest.approx(ref[k], abs=1e-9), k


def test_auc_undefined_for_single_class_is_nan_not_crash():
    import warnings
    warnings.filterwarnings("ignore", message="A single label")
    y = np.array(["bus"] * 5)
    proba = np.tile([.7, .1, .1, .1], (5, 1))
    m = compute_metrics(y, y, proba, CLASS_LABELS)
    assert np.isnan(m["AUC"]) and m["Accuracy"] == 1.0


# ------------------------- comparison is computed on the uploaded data ----
def test_compare_models_uses_the_given_data(models, test_df):
    saved = pd.read_csv(COMPARISON_FILE, index_col="Model")
    full, cms = compare_models(models, test_df[FEATURES], test_df[TARGET], CLASS_LABELS)
    assert np.allclose(full.values, saved[METRIC_NAMES].values, atol=6e-5)
    sub = test_df.iloc[:60]
    part, cms = compare_models(models, sub[FEATURES], sub[TARGET], CLASS_LABELS)
    assert list(part.index) == list(MODEL_FILES) and list(part.columns) == METRIC_NAMES
    assert not np.allclose(part.values, full.values)     # really recomputed
    assert all(cm.values.sum() == 60 for cm in cms.values())


# ------------------- full Streamlit run with training disabled ------------
def test_streamlit_app_all_models_without_any_fit(monkeypatch):
    from sklearn.base import BaseEstimator
    from streamlit.testing.v1 import AppTest

    calls = []

    def forbidden(self, *a, **k):
        calls.append(type(self).__name__)
        raise AssertionError(f"{type(self).__name__}.fit called inside the app")

    # disable fit / fit_transform / partial_fit on every sklearn estimator class
    def subclasses(cls):
        for sub in cls.__subclasses__():
            yield sub
            yield from subclasses(sub)
    # every model / preprocessing / search class; metric helpers such as
    # LabelEncoder (used inside sklearn.metrics) are not training and stay enabled
    training_modules = ("sklearn.linear_model", "sklearn.tree", "sklearn.neighbors",
                        "sklearn.naive_bayes", "sklearn.svm", "sklearn.pipeline",
                        "sklearn.impute", "sklearn.preprocessing._data",
                        "sklearn.model_selection", "sklearn.calibration")
    targets = [c for c in set(subclasses(BaseEstimator))
               if c.__module__.startswith(training_modules)]
    assert len(targets) > 20
    for cls in targets:
        for meth in ("fit", "fit_transform", "partial_fit", "fit_predict"):
            if meth in cls.__dict__:
                monkeypatch.setattr(cls, meth, forbidden)

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    options = at.selectbox[0].options
    assert options == list(MODEL_FILES)
    saved = pd.read_csv(COMPARISON_FILE, index_col="Model")
    for name in options:
        at.selectbox[0].select(name).run()
        assert not at.exception, (name, [e.value for e in at.exception])
        assert not at.error, (name, [e.value for e in at.error])
        html = " ".join(m.value for m in at.markdown)
        for metric in METRIC_NAMES:                       # six metric cards
            assert f"<div class='lbl'>{metric}</div>" in html
        acc = f"{saved.loc[name, 'Accuracy']:.4f}"
        assert acc in html, (name, acc)                   # card shows real value
    assert len(at.tabs) == 4
    assert calls == []


def test_imputers_hold_training_medians_only(models):
    """Leakage check: every pipeline's imputer learned TRAIN medians."""
    train_median = pd.read_csv(TRAIN_DATA_FILE)[FEATURES].median().values
    for name, m in models.items():
        assert np.allclose(m.named_steps["imputer"].statistics_, train_median), name
