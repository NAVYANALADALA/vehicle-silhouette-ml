"""Shared settings. Training and the app import the feature order, labels,
seed and paths from here so a saved model always sees columns in the order
it was trained on."""
from pathlib import Path

# Paths (relative to this file, so they work locally, on BITS Virtual Lab
# and on Streamlit Community Cloud alike)
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "model"
OUTPUT_DIR = ROOT_DIR / "outputs"
EDA_DIR = OUTPUT_DIR / "eda"
EVAL_DIR = OUTPUT_DIR / "evaluation"

CLEAN_DATA_FILE = DATA_DIR / "vehicle_silhouettes.csv"
TRAIN_DATA_FILE = DATA_DIR / "train_data.csv"
TEST_DATA_FILE = ROOT_DIR / "test_data.csv"          # required at repo root
METADATA_FILE = MODEL_DIR / "model_metadata.json"
COMPARISON_FILE = OUTPUT_DIR / "model_comparison.csv"

# Dataset definition: UCI Statlog (Vehicle Silhouettes), dataset id 149
# The 18 features are listed in the SAME order as the UCI documentation.
FEATURES = [
    "compactness",
    "circularity",
    "distance_circularity",
    "radius_ratio",
    "pr_axis_aspect_ratio",
    "max_length_aspect_ratio",
    "scatter_ratio",
    "elongatedness",
    "pr_axis_rectangularity",
    "max_length_rectangularity",
    "scaled_variance_major",
    "scaled_variance_minor",
    "scaled_radius_of_gyration",
    "skewness_major",
    "skewness_minor",
    "kurtosis_minor",
    "kurtosis_major",
    "hollows_ratio",
]
TARGET = "vehicle_class"
CLASS_LABELS = ["bus", "opel", "saab", "van"]   # alphabetical = sklearn order

# Class counts published by UCI for the 846-instance file set.
# Used as an integrity check when the raw data is converted.
EXPECTED_CLASS_COUNTS = {"opel": 212, "saab": 217, "bus": 218, "van": 199}

# Experiment settings
RANDOM_STATE = 2026     # one seed for split, CV folds and stochastic models
TEST_SIZE = 0.20        # 80:20 stratified hold-out
CV_FOLDS = 5            # folds for hyper-parameter search on TRAIN data only
CV_SCORING = "f1_macro" # classes are balanced and equally important

# Display order + file names of the five classifiers
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}
