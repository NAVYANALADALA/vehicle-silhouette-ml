"""
Evaluation utilities shared by training.py (offline) and app.py (online).

Using the SAME function in both places guarantees the numbers shown in the
Streamlit app are computed exactly like the numbers in the README table.

Problem type: 4-class (multiclass) classification with balanced classes.
Averaging choices (documented in the README):
  * Precision / Recall / F1 -> MACRO average: every vehicle class counts
    equally, which matches the goal "recognise all four vehicle types".
  * AUC -> One-vs-Rest ROC-AUC, MACRO average, computed from predicted
    class PROBABILITIES (never from hard labels).
  * MCC -> sklearn's multiclass generalisation (Gorodkin's R_K statistic),
    computed from the full K x K confusion matrix.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def compute_metrics(y_true, y_pred, y_proba, class_labels, average="macro"):
    """Return the six assignment metrics as a dict of floats.

    y_proba must have one column per class, in the order of `class_labels`
    (this is the order of model.classes_ for sklearn estimators).
    """
    y_true = np.asarray(y_true)
    present = np.unique(y_true)

    # ROC-AUC is undefined for a class absent from y_true. If the uploaded
    # file contains fewer than all classes, compute OvR AUC only over the
    # classes present, re-normalising their probability columns.
    if len(present) < 2:
        auc = float("nan")
    elif len(present) == len(class_labels):
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr",
                            average=average, labels=class_labels)
    else:
        idx = [list(class_labels).index(c) for c in present]
        sub = y_proba[:, idx]
        sub = sub / np.clip(sub.sum(axis=1, keepdims=True), 1e-12, None)
        if len(present) == 2:
            auc = roc_auc_score(y_true == present[1], sub[:, 1])
        else:
            auc = roc_auc_score(y_true, sub, multi_class="ovr",
                                average=average, labels=list(present))

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": float(auc),
        "Precision": precision_score(y_true, y_pred, average=average,
                                     labels=class_labels, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=average,
                               labels=class_labels, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=average,
                       labels=class_labels, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def evaluate_model(model, X, y, class_labels=None):
    """Predict with an already-fitted pipeline and score it. Never fits."""
    labels = list(model.classes_) if class_labels is None else class_labels
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    metrics = compute_metrics(y, y_pred, y_proba, labels)
    return metrics, y_pred, y_proba


def confusion_frame(y_true, y_pred, class_labels):
    """Confusion matrix as a labelled DataFrame (rows = actual)."""
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    return pd.DataFrame(cm,
                        index=[f"actual: {c}" for c in class_labels],
                        columns=[f"pred: {c}" for c in class_labels])


def report_frame(y_true, y_pred, class_labels):
    """sklearn classification report as a DataFrame."""
    rep = classification_report(y_true, y_pred, labels=class_labels,
                                output_dict=True, zero_division=0)
    return pd.DataFrame(rep).T


def compare_models(models, X, y, class_labels):
    """Score every loaded model on the SAME (uploaded) data. Never fits.
    Returns (comparison DataFrame, dict of confusion-matrix DataFrames)."""
    rows, cms = [], {}
    for name, mdl in models.items():
        m, yp, _ = evaluate_model(mdl, X, y, class_labels)
        rows.append({"Model": name, **m})
        cms[name] = confusion_frame(y, yp, class_labels)
    return pd.DataFrame(rows).set_index("Model")[METRIC_NAMES], cms
