"""Exploratory data analysis -> figures and eda_report.md in outputs/eda/.

Descriptive only: nothing computed here is used by the models.
Run:  python eda.py
"""
import matplotlib
matplotlib.use("Agg")                       # headless (BITS lab / servers)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import f_classif

from project_config import CLASS_LABELS, CLEAN_DATA_FILE, EDA_DIR, FEATURES, TARGET

PALETTE = dict(zip(CLASS_LABELS, ["#2a6f97", "#e07a5f", "#81b29a", "#9a6fb0"]))
sns.set_theme(style="whitegrid", context="notebook")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(EDA_DIR / name, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CLEAN_DATA_FILE)
    X, y = df[FEATURES], df[TARGET]
    lines = ["# EDA report - UCI Statlog (Vehicle Silhouettes)\n"]

    def section(title, body):
        lines.append(f"\n## {title}\n\n{body}\n")
        print(f"\n=== {title} ===\n{body}")

    # structure
    section("Shape", f"{df.shape[0]} rows x {df.shape[1]} columns "
                     f"({len(FEATURES)} features + 1 target)")
    section("First five rows", df.head().to_markdown(index=False))
    section("Last five rows", df.tail().to_markdown(index=False))
    section("Data types", df.dtypes.astype(str).value_counts().to_markdown())

    # data quality
    missing = int(df.isna().sum().sum())
    dups = int(df.duplicated().sum())
    feat_dups = int(X.duplicated().sum())
    section("Missing values and duplicates",
            f"Missing cells: {missing}\n\nDuplicate rows: {dups}\n\n"
            f"Rows with identical features: {feat_dups}")

    # descriptive statistics
    desc = X.describe().T
    desc["skew"] = X.skew()
    section("Descriptive statistics", desc.round(2).to_markdown())

    # target distribution
    counts = y.value_counts().reindex(CLASS_LABELS)
    section("Target distribution",
            (pd.DataFrame({"count": counts,
                           "share_%": (100 * counts / len(y)).round(1)})
             .to_markdown()))
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index, counts.values,
                  color=[PALETTE[c] for c in counts.index])
    ax.bar_label(bars, labels=[f"{v} ({100*v/len(y):.1f}%)" for v in counts.values])
    ax.set(title="Class distribution of vehicle_class (n = 846)",
           xlabel="Vehicle class", ylabel="Number of silhouettes")
    ax.set_ylim(0, counts.max() * 1.15)
    save(fig, "01_target_distribution.png")

    # correlation
    corr = X.corr()
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(corr, cmap="RdBu_r", vmin=-1, vmax=1, annot=True, fmt=".2f",
                annot_kws={"size": 7}, square=True, ax=ax,
                cbar_kws={"label": "Pearson correlation"})
    ax.set_title("Pearson correlation between the 18 shape features")
    save(fig, "03_correlation_heatmap.png")
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
    strong = upper[upper.abs() >= 0.90].sort_values(key=abs, ascending=False)
    section("Strongly correlated pairs (|r| >= 0.90)",
            f"{len(strong)} pairs\n\n" +
            strong.round(3).rename("r").to_frame().to_markdown())

    # outliers (IQR rule)
    q1, q3 = X.quantile(.25), X.quantile(.75)
    iqr = q3 - q1
    out = ((X < q1 - 1.5 * iqr) | (X > q3 + 1.5 * iqr)).sum()
    section("Outliers per feature (1.5 x IQR rule)",
            out[out > 0].sort_values(ascending=False).rename("n_outliers")
            .to_frame().to_markdown())
    z = (X - X.mean()) / X.std()
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=z, orient="v", ax=ax, color="#9ecae1", fliersize=3)
    ax.set(title="Box plots of standardised features (z-scores) - outlier view",
           xlabel="Feature", ylabel="z-score")
    ax.tick_params(axis="x", rotation=60)
    save(fig, "04_boxplots_standardised.png")

    # feature-target relationship
    f_stat, p_val = f_classif(X, y)
    anova = (pd.DataFrame({"F_statistic": f_stat, "p_value": p_val}, index=FEATURES)
             .sort_values("F_statistic", ascending=False))
    section("One-way ANOVA F-test of each feature vs class",
            anova.to_markdown(floatfmt=".3g"))
    top = list(anova.index[:4]) + list(anova.index[-2:])
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, col in zip(axes.ravel(), top):
        sns.boxplot(data=df, x=TARGET, y=col, order=CLASS_LABELS, hue=TARGET,
                    palette=PALETTE, legend=False, ax=ax)
        ax.set(title=f"{col} (F = {anova.loc[col, 'F_statistic']:.0f})",
               xlabel="Vehicle class", ylabel=col)
    fig.suptitle("Class-wise comparison: 4 most and 2 least discriminative features",
                 fontsize=15)
    save(fig, "05_classwise_boxplots.png")

    # opel vs saab: the hard pair
    two = df[df[TARGET].isin(["opel", "saab"])]
    f2, _ = f_classif(two[FEATURES], two[TARGET])
    section("Opel vs Saab separability (ANOVA F on the two car classes only)",
            pd.Series(f2, index=FEATURES).sort_values(ascending=False)
            .round(1).rename("F_statistic").to_frame().to_markdown())

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.scatterplot(data=df, x="scatter_ratio", y="skewness_major",
                    hue=TARGET, hue_order=CLASS_LABELS, palette=PALETTE,
                    alpha=.7, s=25, ax=ax)
    ax.set(title="Two features: vans/buses separate, Opel and Saab overlap",
           xlabel="scatter_ratio", ylabel="skewness_major")
    ax.legend(title="Vehicle class")
    save(fig, "06_scatter_opel_saab_overlap.png")

    (EDA_DIR / "eda_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved figures and eda_report.md to {EDA_DIR}")


if __name__ == "__main__":
    main()
