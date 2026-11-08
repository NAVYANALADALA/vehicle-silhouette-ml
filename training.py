"""Train, tune, evaluate and save the five classifiers.

Protocol: one stratified 80:20 split (seed in project_config). Each model is a
Pipeline (imputer -> [scaler] -> classifier) tuned with 5-fold GridSearchCV on
the training split only, then scored once on the held-out test set and saved
with joblib. The test rows are never used for fitting.

Run:  python training.py
"""
import json
import platform
import socket
import time
from datetime import datetime

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree

from evaluation import METRIC_NAMES, confusion_frame, evaluate_model, report_frame
from project_config import (CLASS_LABELS, CLEAN_DATA_FILE, COMPARISON_FILE,
                            CV_FOLDS, CV_SCORING, EVAL_DIR, FEATURES,
                            METADATA_FILE, MODEL_DIR, MODEL_FILES, RANDOM_STATE,
                            TARGET, TEST_DATA_FILE, TEST_SIZE, TRAIN_DATA_FILE)


# Model definitions: preprocessing + estimator + search space
def scaled(estimator):
    """Median imputation + z-score scaling, then the estimator.
    Needed by distance / gradient / margin based models (LR, KNN, SVM)."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", estimator),
    ])


def unscaled(estimator):
    """Median imputation only. Trees split on thresholds and Gaussian NB
    models each feature separately, so both are invariant to scaling."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", estimator),
    ])


def build_search_spaces():
    return {
        "Logistic Regression": (
            scaled(LogisticRegression(max_iter=5000, solver="lbfgs")),
            {"clf__C": [0.01, 0.1, 1, 10, 100, 1000]},
        ),
        "Decision Tree": (
            unscaled(DecisionTreeClassifier(random_state=RANDOM_STATE)),
            {"clf__criterion": ["gini", "entropy"],
             "clf__max_depth": [4, 6, 8, 10, None],
             "clf__min_samples_split": [2, 10, 20],
             "clf__min_samples_leaf": [1, 3, 5],
             "clf__ccp_alpha": [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]},
        ),
        "KNN": (
            scaled(KNeighborsClassifier()),
            {"clf__n_neighbors": list(range(1, 26, 2)),
             "clf__weights": ["uniform", "distance"],
             "clf__p": [1, 2]},                     # 1 = Manhattan, 2 = Euclidean
        ),
        "Naive Bayes": (
            unscaled(GaussianNB()),
            {"clf__var_smoothing": np.logspace(-11, -3, 9)},
        ),
        "Support Vector Machine": (
            scaled(SVC(random_state=RANDOM_STATE)),
            [{"clf__kernel": ["linear"], "clf__C": [0.1, 1, 10, 100]},
             {"clf__kernel": ["rbf"], "clf__C": [1, 10, 100, 1000, 3000],
              "clf__gamma": ["scale", 0.003, 0.01, 0.03, 0.1]}],
        ),
    }


def add_svm_probabilities(best_pipe, X_train, y_train, cv):
    """An SVM outputs signed distances to the hyperplane, not probabilities.
    AUC needs class scores that are comparable across classes, so the tuned
    SVC is wrapped in Platt (sigmoid) calibration. ensemble=False: the SVC is
    refitted on the full training set and the sigmoid is fitted on 5-fold
    out-of-fold decision values (the same idea as the older
    SVC(probability=True), which is deprecated from scikit-learn 1.9)."""
    pipe = clone(best_pipe)
    svc = pipe.named_steps["clf"]
    pipe.steps[-1] = ("clf", CalibratedClassifierCV(svc, method="sigmoid",
                                                    cv=cv, ensemble=False))
    return pipe.fit(X_train, y_train)               # TRAIN DATA ONLY


def banner():
    info = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "joblib": joblib.__version__,
        "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    print("=" * 72)
    print(" ML Assignment 2 - Vehicle Silhouette Classification (5 models)")
    for k, v in info.items():
        print(f"   {k:<13}: {v}")
    print("=" * 72)
    return info


def main():
    env = banner()
    for d in (MODEL_DIR, EVAL_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # 1. load + split
    df = pd.read_csv(CLEAN_DATA_FILE)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    assert set(X_train.index).isdisjoint(X_test.index)
    print(f"\nTrain: {X_train.shape}  Test: {X_test.shape}")
    print("Train class counts:", y_train.value_counts().to_dict())
    print("Test  class counts:", y_test.value_counts().to_dict())

    pd.concat([X_train, y_train], axis=1).to_csv(TRAIN_DATA_FILE, index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(TEST_DATA_FILE, index=False)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    test_rows, train_rows, cv_rows, fitted = [], [], [], {}
    reports, cms, best_params = {}, {}, {}

    # 2-4. tune, refit, evaluate
    for name, (pipe, grid) in build_search_spaces().items():
        t0 = time.time()
        search = GridSearchCV(pipe, grid, scoring=CV_SCORING, cv=cv,
                              n_jobs=-1, refit=True, return_train_score=True)
        search.fit(X_train, y_train)               # TRAIN DATA ONLY
        model = search.best_estimator_
        if name == "Support Vector Machine":
            model = add_svm_probabilities(model, X_train, y_train, cv)
        fitted[name] = model
        secs = time.time() - t0

        i = search.best_index_
        cvr = search.cv_results_
        params = {}
        for k, v in search.best_params_.items():
            v = v.item() if hasattr(v, "item") else v
            params[k.replace("clf__", "")] = float(f"{v:.4g}") if isinstance(v, float) else v
        best_params[name] = params
        cv_rows.append({"Model": name,
                        "CV macro-F1 mean": cvr["mean_test_score"][i],
                        "CV macro-F1 std": cvr["std_test_score"][i],
                        "CV train macro-F1": cvr["mean_train_score"][i],
                        "Candidates": len(cvr["params"]),
                        "Search time (s)": round(secs, 1),
                        "Best params": json.dumps(params)})

        m_test, y_pred, _ = evaluate_model(model, X_test, y_test)
        m_train, _, _ = evaluate_model(model, X_train, y_train)
        test_rows.append({"Model": name, **m_test})
        train_rows.append({"Model": name, **m_train})
        reports[name] = report_frame(y_test, y_pred, CLASS_LABELS)
        cms[name] = confusion_frame(y_test, y_pred, CLASS_LABELS)

        joblib.dump(model, MODEL_DIR / MODEL_FILES[name], compress=3)
        print(f"\n[{name}] best {params}  ({secs:.1f}s)")
        print("   test : " + "  ".join(f"{k}={v:.4f}" for k, v in m_test.items()))

    # 5. persist results
    comp = pd.DataFrame(test_rows).set_index("Model")
    comp.round(4).to_csv(COMPARISON_FILE)
    train_df = pd.DataFrame(train_rows).set_index("Model")
    train_df.round(4).to_csv(EVAL_DIR / "train_set_metrics.csv")
    gap = (train_df - comp)[["Accuracy", "F1", "MCC"]]
    gap.round(4).to_csv(EVAL_DIR / "train_minus_test_gap.csv")
    cv_df = pd.DataFrame(cv_rows).set_index("Model")
    cv_df.round(4).to_csv(EVAL_DIR / "cv_summary.csv")
    with open(EVAL_DIR / "classification_reports.txt", "w", encoding="utf-8") as fh:
        for name in MODEL_FILES:
            fh.write(f"===== {name} =====\n{reports[name].round(4).to_string()}\n\n"
                     f"Confusion matrix (rows = actual)\n{cms[name].to_string()}\n\n")

    make_figures(fitted, comp, cms, X_test, y_test)

    meta = {
        "dataset": "UCI Statlog (Vehicle Silhouettes), id 149",
        "features": FEATURES, "target": TARGET, "class_labels": CLASS_LABELS,
        "random_state": RANDOM_STATE, "test_size": TEST_SIZE,
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "cv": f"{CV_FOLDS}-fold stratified, scoring={CV_SCORING}",
        "averaging": "macro (precision/recall/F1), OvR macro (AUC)",
        "best_params": best_params, "model_files": MODEL_FILES,
        "test_metrics": comp.round(4).to_dict(orient="index"),
        "environment": env,
    }
    METADATA_FILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    pd.set_option("display.width", 140)
    print("\n" + "=" * 72)
    print(" HOLD-OUT TEST RESULTS (n = %d, macro averages)" % len(X_test))
    print("=" * 72)
    print(comp[METRIC_NAMES].round(4).to_string())
    print("\nCross-validation (train only):")
    print(cv_df[["CV macro-F1 mean", "CV macro-F1 std", "CV train macro-F1"]].round(4).to_string())
    print("\nTrain - test gap:")
    print(gap.round(4).to_string())
    print(f"\nSaved 5 models to {MODEL_DIR.name}/ and results to outputs/")
    print(f"Pin this in requirements.txt ->  scikit-learn=={sklearn.__version__}")


def make_figures(fitted, comp, cms, X_test, y_test):
    sns.set_theme(style="whitegrid")

    # confusion matrices
    fig, axes = plt.subplots(1, 5, figsize=(26, 5.2))
    for ax, (name, cm) in zip(axes, cms.items()):
        sns.heatmap(cm.values, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, ax=ax)
        ax.set(title=f"{name}\nacc = {comp.loc[name, 'Accuracy']:.3f}",
               xlabel="Predicted class", ylabel="Actual class")
    fig.suptitle("Confusion matrices on the hold-out test set", fontsize=15)
    fig.tight_layout()
    fig.savefig(EVAL_DIR / "confusion_matrices.png", dpi=120)
    plt.close(fig)

    # metric comparison bar chart
    long = comp[METRIC_NAMES].reset_index().melt(id_vars="Model",
                                                  var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=long, x="Metric", y="Score", hue="Model", ax=ax)
    ax.set(title="Test-set metrics for the five classifiers (macro averages)",
           ylim=(0, 1.05), ylabel="Score", xlabel="Metric")
    ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(EVAL_DIR / "metric_comparison.png", dpi=120)
    plt.close(fig)

    # macro-average one-vs-rest ROC curves
    y_bin = label_binarize(y_test, classes=CLASS_LABELS)
    fig, ax = plt.subplots(figsize=(7, 6))
    grid = np.linspace(0, 1, 201)
    for name, model in fitted.items():
        proba = model.predict_proba(X_test)
        tprs = []
        for k in range(len(CLASS_LABELS)):
            disp = RocCurveDisplay.from_predictions(y_bin[:, k], proba[:, k])
            plt.close(disp.figure_)
            tprs.append(np.interp(grid, disp.fpr, disp.tpr))
        ax.plot(grid, np.mean(tprs, axis=0),
                label=f"{name} (AUC = {comp.loc[name, 'AUC']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set(title="Macro-average one-vs-rest ROC curves (test set)",
           xlabel="False positive rate", ylabel="True positive rate")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(EVAL_DIR / "roc_curves.png", dpi=120)
    plt.close(fig)

    # top of the fitted decision tree
    tree = fitted["Decision Tree"].named_steps["clf"]
    fig, ax = plt.subplots(figsize=(22, 9))
    plot_tree(tree, max_depth=3, feature_names=FEATURES, class_names=CLASS_LABELS,
              filled=True, rounded=True, fontsize=8, ax=ax, impurity=True)
    ax.set_title(f"Decision tree - first 3 levels (full depth = {tree.get_depth()}, "
                 f"leaves = {tree.get_n_leaves()})")
    fig.savefig(EVAL_DIR / "decision_tree_top_levels.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
