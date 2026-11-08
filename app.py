"""Streamlit app: evaluate the five saved models on an uploaded test CSV.

Models are only loaded and used for prediction; nothing is refitted on the
uploaded data. Run with:  streamlit run app.py
"""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import streamlit as st

from evaluation import (METRIC_NAMES, compare_models, confusion_frame,
                        evaluate_model, report_frame)
from project_config import (CLASS_LABELS, FEATURES, METADATA_FILE, MODEL_DIR,
                            MODEL_FILES, TARGET, TEST_DATA_FILE)
from validation import MAX_UPLOAD_BYTES, read_uploaded_csv, validate_upload

st.set_page_config(page_title="SilhouetteLab | Vehicle classifier",
                   page_icon="🚐", layout="wide")

ACCENT = "#0f766e"
CLASS_COLORS = dict(zip(CLASS_LABELS, ["#2a6f97", "#e07a5f", "#81b29a", "#9a6fb0"]))
METRIC_HELP = {
    "Accuracy": "Share of all silhouettes classified correctly.",
    "AUC": "One-vs-rest ROC-AUC (macro) from predicted probabilities: ranking quality, 0.5 = chance.",
    "Precision": "Macro precision: of the vehicles predicted as class k, how many really are k.",
    "Recall": "Macro recall: of the vehicles that are class k, how many were found.",
    "F1": "Macro F1: harmonic mean of precision and recall, averaged over classes.",
    "MCC": "Matthews correlation: uses the whole confusion matrix; 1 perfect, 0 random.",
}

st.markdown(f"""
<style>
  .block-container {{padding-top: 1.6rem; max-width: 1250px;}}
  .hero {{background: linear-gradient(120deg, #0b3954 0%, {ACCENT} 100%);
          color: #fff; padding: 1.3rem 1.6rem; border-radius: 14px; margin-bottom: 1rem;}}
  .hero h1 {{color:#fff; font-size: 1.75rem; margin: 0 0 .25rem 0;}}
  .hero p  {{margin: 0; opacity: .9; font-size: .97rem;}}
  .mcard {{border: 1px solid rgba(128,128,128,.25); border-left: 5px solid {ACCENT};
           border-radius: 10px; padding: .7rem .9rem; background: rgba(15,118,110,.05);}}
  .mcard .lbl {{font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; opacity: .75;}}
  .mcard .val {{font-size: 1.6rem; font-weight: 700; line-height: 1.2;}}
  .mcard .sub {{font-size: .72rem; opacity: .7;}}
  .pill {{display:inline-block; padding:.12rem .55rem; border-radius:999px; font-size:.78rem;
          color:#fff; margin-right:.3rem;}}
</style>
""", unsafe_allow_html=True)


# cache_resource: load the pipelines once per server process, not on every rerun
@st.cache_resource(show_spinner="Loading saved models ...")
def load_models():
    models, problems = {}, []
    for name, fname in MODEL_FILES.items():
        path = MODEL_DIR / fname
        try:
            models[name] = joblib.load(path)
        except Exception as exc:                       # show, don't crash
            problems.append(f"{name}: could not load {path.name} ({exc})")
    return models, problems


@st.cache_data
def load_metadata():
    try:
        return json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


@st.cache_data
def parse_upload(data: bytes, name: str):
    return read_uploaded_csv(data, name)


def metric_cards(metrics):
    cols = st.columns(6)
    for col, name in zip(cols, METRIC_NAMES):
        val = metrics[name]
        txt = "n/a" if pd.isna(val) else f"{val:.4f}"
        col.markdown(f"<div class='mcard' title='{METRIC_HELP[name]}'>"
                     f"<div class='lbl'>{name}</div><div class='val'>{txt}</div>"
                     f"<div class='sub'>{'macro' if name in ('Precision','Recall','F1') else ('OvR macro' if name=='AUC' else '')}</div>"
                     f"</div>", unsafe_allow_html=True)


def render_fig(fig):
    """Render a matplotlib figure and free its memory (long-running server)."""
    st.pyplot(fig)
    plt.close(fig)


def plot_confusion(cm_df, title):
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    sns.heatmap(cm_df.values, annot=True, fmt="d", cmap="Greens", cbar=False,
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
                linewidths=.5, linecolor="white", ax=ax)
    ax.set(xlabel="Predicted class", ylabel="Actual class", title=title)
    fig.tight_layout()
    return fig


st.markdown("""
<div class="hero">
  <h1>🚐 SilhouetteLab - Vehicle silhouette classifier</h1>
  <p>Evaluate five pre-trained classifiers (UCI Statlog Vehicle Silhouettes) on a
  labelled test CSV. Models are loaded from disk; nothing is retrained on your file.</p>
</div>""", unsafe_allow_html=True)

models, load_problems = load_models()
meta = load_metadata()
for p in load_problems:
    st.error(p)
if not models:
    st.stop()

trained_ver = meta.get("environment", {}).get("scikit-learn")
if trained_ver and trained_ver != sklearn.__version__:
    st.warning(f"Models were saved with scikit-learn {trained_ver} but this server runs "
               f"{sklearn.__version__}. Pin scikit-learn=={trained_ver} in requirements.txt.")

with st.sidebar:
    st.header("1 · Test data")
    uploaded = st.file_uploader("Upload test CSV (features + true label)", type=["csv"],
                                help=f"Needs the 18 feature columns and the target "
                                     f"'{TARGET}'. Max {MAX_UPLOAD_BYTES // 2**20} MB.")
    if uploaded is None:
        use_sample = st.checkbox("No file yet? Use the bundled test_data.csv",
                                 value=True)
    else:
        use_sample = False                   # an uploaded file always wins
        st.caption(f"Using uploaded file: **{uploaded.name}**")
    if TEST_DATA_FILE.exists():
        st.download_button("⬇ Download sample test_data.csv",
                           TEST_DATA_FILE.read_bytes(), "test_data.csv", "text/csv")

    st.header("2 · Model")
    model_name = st.selectbox("Choose a classifier", list(models.keys()), index=0)
    st.caption("Best hyper-parameters (5-fold CV on training data):")
    st.json(meta.get("best_params", {}).get(model_name, {}), expanded=True)

    with st.expander("How to use"):
        st.markdown(
            f"1. Upload a CSV with columns `{', '.join(FEATURES[:3])}, ...` "
            f"(all 18) and `{TARGET}` (bus / opel / saab / van).\n"
            "2. Pick a model from the dropdown.\n"
            "3. Read the metrics, confusion matrix and report.\n"
            "4. Open **Compare all models** to see the five side by side.")

with st.expander("📄 Expected CSV format (schema)", expanded=False):
    st.markdown(f"One row per vehicle silhouette. **Required:** the 18 numeric feature "
                f"columns below **and** the true label column `{TARGET}` with values "
                f"`{'`, `'.join(CLASS_LABELS)}`. Column order and letter case do not matter; "
                "extra columns are ignored.")
    st.dataframe(pd.DataFrame({"column": FEATURES + [TARGET],
                               "type": ["integer / numeric"] * len(FEATURES) + ["text label"]}),
                 hide_index=True, width="stretch", height=250)

raw_df, source = None, None
if uploaded is not None:
    raw_df, read_error = parse_upload(uploaded.getvalue(), uploaded.name)
    if read_error:
        st.error(read_error)
        st.stop()
    source = uploaded.name
elif use_sample and TEST_DATA_FILE.exists():
    raw_df, source = pd.read_csv(TEST_DATA_FILE), "test_data.csv (bundled)"

if raw_df is None:
    st.info("⬅ Upload a test CSV in the sidebar (or tick 'Use the bundled test_data.csv').")
    st.stop()

result = validate_upload(raw_df, FEATURES, TARGET, CLASS_LABELS)
for w in result.warnings:
    st.warning(w)
if not result.ok:
    for e in result.errors:
        st.error(e)
    st.stop()
X, y = result.X, result.y
st.markdown(f"**Data:** `{source}` · **{len(X)} rows** · **{X.shape[1]} features** · "
            f"target column detected: `{TARGET}` ({y.nunique()} classes)")

tab_eval, tab_cmp, tab_data, tab_about = st.tabs(
    ["📊 Model evaluation", "🏁 Compare all models", "🗂 Uploaded data", "ℹ️ About"])

with tab_eval:
    model = models[model_name]
    metrics, y_pred, y_proba = evaluate_model(model, X, y, CLASS_LABELS)
    st.subheader(f"{model_name}  ·  {len(y)} rows from {source}")
    metric_cards(metrics)
    if pd.isna(metrics["AUC"]):
        st.info("AUC is not defined because the uploaded data contains only one class "
                "(ROC needs both positives and negatives).")
    st.write("")

    c1, c2 = st.columns([1, 1.15])
    with c1:
        cm = confusion_frame(y, y_pred, CLASS_LABELS)
        render_fig(plot_confusion(cm, f"Confusion matrix - {model_name}"))
        st.caption("Rows are the true class, columns the predicted class. The diagonal "
                   "holds correct predictions; off-diagonal cells show which classes are "
                   "confused (on this dataset usually Opel ↔ Saab).")
    with c2:
        st.markdown("**Classification report**")
        rep = report_frame(y, y_pred, CLASS_LABELS)
        rep.loc["accuracy", "support"] = len(y)
        rep["support"] = rep["support"].astype(int)
        st.dataframe(rep.style.format({"precision": "{:.3f}", "recall": "{:.3f}",
                                       "f1-score": "{:.3f}"})
                     .background_gradient(subset=["f1-score"], cmap="Greens", vmin=0, vmax=1),
                     width="stretch")
        correct = (y_pred == y.values).sum()
        st.markdown(f"Correct: **{correct} / {len(y)}** · Misclassified: **{len(y) - correct}**")

    pred_df = X.copy()
    pred_df[TARGET] = y
    pred_df["predicted"] = y_pred
    pred_df["correct"] = pred_df["predicted"] == pred_df[TARGET]
    for i, c in enumerate(CLASS_LABELS):
        pred_df[f"p_{c}"] = np.round(y_proba[:, i], 4)
    with st.expander("Prediction details"):
        only_wrong = st.toggle("Show misclassified rows only")
        show = pred_df[~pred_df["correct"]] if only_wrong else pred_df
        st.dataframe(show, width="stretch", height=300)
    st.download_button("⬇ Download predictions (CSV)", pred_df.to_csv(index=False).encode(),
                       f"predictions_{model_name.lower().replace(' ', '_')}.csv", "text/csv")

with tab_cmp:
    comp, cms = compare_models(models, X, y, CLASS_LABELS)   # computed on THIS upload

    st.subheader("All five models on the same uploaded data")
    st.dataframe(comp.style.format("{:.4f}").highlight_max(axis=0, color="#b7e4c7"),
                 width="stretch")
    best = comp["MCC"].idxmax()
    st.success(f"Highest MCC on this file: **{best}** (MCC = {comp.loc[best, 'MCC']:.4f}, "
               f"F1 = {comp.loc[best, 'F1']:.4f}, AUC = {comp.loc[best, 'AUC']:.4f}).")

    long = comp.reset_index().melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(11, 4.3))
    sns.barplot(data=long, x="Metric", y="Score", hue="Model", ax=ax, palette="viridis")
    ax.set(ylim=(0, 1.05), title="Metric comparison (macro averages)", xlabel="", ylabel="Score")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, title="Model")
    fig.tight_layout()
    render_fig(fig)

    st.markdown("**Confusion matrices**")
    cols = st.columns(5)
    for col, (name, cmf) in zip(cols, cms.items()):
        with col:
            render_fig(plot_confusion(cmf, name))
    st.download_button("⬇ Download comparison (CSV)", comp.round(4).to_csv().encode(),
                       "model_comparison_uploaded.csv", "text/csv")

with tab_data:
    a, b, c = st.columns(3)
    a.metric("Rows used", len(X))
    b.metric("Features", X.shape[1])
    c.metric("Classes present", y.nunique())
    counts = y.value_counts().reindex(CLASS_LABELS, fill_value=0)
    st.markdown(" ".join(f"<span class='pill' style='background:{CLASS_COLORS[k]}'>"
                         f"{k}: {v}</span>" for k, v in counts.items()),
                unsafe_allow_html=True)
    st.dataframe(pd.concat([X, y], axis=1).head(50), width="stretch", height=320)
    st.caption("Summary statistics of the uploaded features")
    st.dataframe(X.describe().T.round(2), width="stretch")

with tab_about:
    st.markdown(f"""
**Data:** UCI Statlog (Vehicle Silhouettes), 846 rows, 18 shape features, classes
bus / opel / saab / van. 80:20 stratified split (seed {meta.get('random_state', '-')});
`test_data.csv` is the 20 % hold-out.

**Models:** each file in `model/` is a scikit-learn Pipeline fitted on the training
split only (median imputer, StandardScaler for LR/KNN/SVM, then the classifier).

**Metrics:** precision, recall and F1 are macro averages; AUC is one-vs-rest macro
computed from predicted probabilities; MCC is the multiclass Matthews coefficient.
""")
    st.caption(f"Server: scikit-learn {sklearn.__version__} · models trained with "
               f"{trained_ver or 'unknown'}")
