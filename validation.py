"""
Validation of a user-uploaded test CSV before it reaches a saved model.

The checks are deliberately strict about anything that would make the
metrics meaningless (no target, missing feature columns, unknown labels)
and lenient about cosmetic issues (column order, letter case, extra columns),
which are fixed automatically and reported as warnings.

No statistics are learned here - missing feature values are left as NaN and
are filled later by the imputer that was fitted on the TRAINING data inside
each saved pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import io

import pandas as pd

MAX_UPLOAD_BYTES = 10 * 1024 * 1024     # matches .streamlit/config.toml
MAX_ROWS = 100_000                     # free-tier memory guard


@dataclass
class ValidationResult:
    X: pd.DataFrame | None = None
    y: pd.Series | None = None
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def ok(self):
        return not self.errors and self.X is not None


def _norm(name):
    return str(name).strip().lower().replace(" ", "_").replace("-", "_").replace(".", "_")


def validate_upload(df, features, target, class_labels):
    res = ValidationResult()

    if df is None or df.shape[0] == 0:
        res.errors.append("The file has no data rows.")
        return res

    # column names: tolerate case / spaces / order
    df = df.copy()
    df.columns = [_norm(c) for c in df.columns]
    if df.columns.duplicated().any():
        dup = sorted(set(df.columns[df.columns.duplicated()]))
        res.errors.append(f"Duplicate column names: {dup}")
        return res

    if target not in df.columns:
        res.errors.append(
            f"Target column '{target}' is missing. The test CSV must contain the "
            "true labels so that metrics can be computed.")
    missing = [c for c in features if c not in df.columns]
    if missing:
        res.errors.append(f"{len(missing)} required feature column(s) missing: {missing}")
    if res.errors:
        return res

    extra = [c for c in df.columns if c not in features + [target]]
    if extra:
        res.warnings.append(f"Ignored {len(extra)} unexpected column(s): {extra}")

    # target
    y = df[target].astype("string").str.strip().str.lower()
    no_label = y.isna() | (y == "")
    if no_label.any():
        res.warnings.append(f"Dropped {int(no_label.sum())} row(s) with an empty target.")
    unknown = ~y.isin(class_labels) & ~no_label
    if unknown.any():
        bad = sorted(y[unknown].unique().tolist())[:10]
        res.warnings.append(
            f"Dropped {int(unknown.sum())} row(s) with unknown class label(s) {bad}; "
            f"allowed labels are {class_labels}.")
    keep = ~(no_label | unknown)

    # features: numeric check
    X = df.loc[:, features]                      # enforce training column order
    X_num = X.apply(pd.to_numeric, errors="coerce")
    non_numeric = (X_num.isna() & X.notna()).sum()
    non_numeric = non_numeric[non_numeric > 0]
    if not non_numeric.empty:
        res.errors.append(
            "Non-numeric values found in: " +
            ", ".join(f"{c} ({n} cells)" for c, n in non_numeric.items()))
        return res

    n_nan = int(X_num[keep].isna().sum().sum())
    if n_nan:
        res.warnings.append(
            f"{n_nan} missing feature value(s) will be filled with the TRAINING-set "
            "median stored in each saved pipeline (no refitting).")
    all_nan = X_num.isna().all(axis=1) & keep
    if all_nan.any():
        res.warnings.append(f"Dropped {int(all_nan.sum())} row(s) with no feature values.")
        keep &= ~all_nan

    negative = (X_num < 0).sum()
    negative = negative[negative > 0]
    if not negative.empty:
        res.warnings.append(
            "Negative values are impossible for these shape measurements and were "
            f"never seen in training: {dict(negative)}. Predictions may be unreliable.")

    if keep.sum() == 0:
        res.errors.append("No valid rows remain after validation.")
        return res

    res.X = X_num[keep].reset_index(drop=True)
    res.y = y[keep].astype(str).reset_index(drop=True)
    if res.y.nunique() < len(class_labels):
        res.warnings.append(
            f"Only {res.y.nunique()} of {len(class_labels)} classes are present; "
            "AUC is computed over the classes present.")
    return res


def read_uploaded_csv(data, name="upload.csv", max_bytes=MAX_UPLOAD_BYTES):
    """Parse uploaded bytes into a DataFrame.

    Returns (DataFrame, None) on success or (None, user-facing message).
    Handles wrong extension, empty, oversized, non-text and malformed files.
    """
    if not str(name).lower().endswith(".csv"):
        return None, "Please upload a file with a .csv extension."
    if data is None or len(data) == 0:
        return None, "The uploaded file is empty (0 bytes)."
    if len(data) > max_bytes:
        return None, (f"The file is {len(data) / 1e6:.1f} MB; the limit is "
                      f"{max_bytes / 1e6:.0f} MB. Upload only the test data.")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = data.decode("latin-1")
        except UnicodeDecodeError:          # pragma: no cover - latin-1 decodes any byte
            return None, "The file is not a readable text/CSV file."
    if "\x00" in text:
        return None, "The file looks binary (e.g. Excel or zip), not a plain-text CSV."
    try:
        df = pd.read_csv(io.StringIO(text))
    except pd.errors.EmptyDataError:
        return None, "The uploaded file contains no columns or rows."
    except pd.errors.ParserError as exc:
        return None, f"The CSV is malformed and could not be parsed: {exc}"
    if df.shape[1] < 2:
        return None, ("Only one column was found. Make sure the file is comma-separated "
                      "and has a header row.")
    if len(df) > MAX_ROWS:
        return None, f"The file has {len(df):,} rows; the limit is {MAX_ROWS:,}."
    return df, None
