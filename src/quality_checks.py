# src/quality_checks.py
'''
- Stage分布（群別）
- 年齢（box）
- 性別（bar）
- 追跡期間（box） 
- 欠測率（table）
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_numeric_dtype


# --------------------------------------------------------
# Helper: safe column getter
# --------------------------------------------------------
def _col_exists(df, col):
    return col in df.columns


# --------------------------------------------------------
# 1) Stage distribution (stacked % bar)
# --------------------------------------------------------
def plot_stage_distribution(df, group_col="group"):
    candidates = [
        "stage_num",
        "STAGE",
        "PATHOLOGIC_STAGE",
        "AJCC_PATHOLOGIC_TUMOR_STAGE",
        "CLINICAL_STAGE",
    ]

    stage_col = next((c for c in candidates if c in df.columns), None)
    if stage_col is None:
        return None

    stage = df[[group_col, stage_col]].copy()

    if is_numeric_dtype(stage[stage_col]):
        s = pd.to_numeric(stage[stage_col], errors="coerce")
        s = s.round().astype("Int64")
        stage[stage_col] = s.map({
            1: "Stage I",
            2: "Stage II",
            3: "Stage III",
            4: "Stage IV",
        })

    stage[stage_col] = stage[stage_col].fillna("Unknown")

    stage_counts = (
        stage.groupby(group_col)[stage_col]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )

    order = ["Stage I", "Stage II", "Stage III", "Stage IV", "Unknown"]
    cols = [c for c in order if c in stage_counts.columns]
    stage_counts = stage_counts[cols]

    fig, ax = plt.subplots(figsize=(6, 4))
    stage_counts.plot(kind="bar", stacked=True, ax=ax)

    ax.set_ylabel("Proportion")
    group_sizes = get_group_sizes(df, group_col)
    title_suffix = " / ".join(
        [f"{g} (n={group_sizes[g]})" for g in group_sizes]
    )
    ax.set_title(f"Stage Distribution by Group\n{title_suffix}")
    ax.legend(title="Stage", bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    return fig



# --------------------------------------------------------
# 2) Age distribution (boxplot)
# --------------------------------------------------------
def plot_age_distribution(df, group_col="group", age_col="AGE"):
    if not _col_exists(df, age_col):
        return None

    fig, ax = plt.subplots(figsize=(7, 5))

    df.boxplot(column=age_col, by=group_col, ax=ax)
    group_sizes = get_group_sizes(df, group_col)
    title_suffix = " / ".join(
        [f"{g} (n={group_sizes[g]})" for g in group_sizes]
    )
    ax.set_title(f"Age Distribution by Group\n{title_suffix}")
    ax.set_ylabel("Age")
    plt.suptitle("")
    plt.tight_layout()

    return fig


# --------------------------------------------------------
# 3) Gender distribution (stacked bar)
# --------------------------------------------------------
def plot_gender_distribution(df, group_col="group", sex_col="SEX"):
    if not _col_exists(df, sex_col):
        return None

    sex = df[[group_col, sex_col]].copy()
    sex[sex_col] = sex[sex_col].fillna("Unknown")

    counts = (
        sex.groupby(group_col)[sex_col]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    counts.plot(kind="bar", stacked=True, ax=ax)

    ax.set_ylabel("Proportion")
    group_sizes = get_group_sizes(df, group_col)
    title_suffix = " / ".join(
        [f"{g} (n={group_sizes[g]})" for g in group_sizes]
    )
    ax.set_title(f"Gender Distribution by Group\n{title_suffix}")
    ax.legend(title="Gender", bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    return fig


# --------------------------------------------------------
# 4) Follow-up time (OS_MONTHS) boxplot
# --------------------------------------------------------
def plot_followup_distribution(df, group_col="group", time_col="OS_MONTHS"):
    if not _col_exists(df, time_col):
        return None

    fig, ax = plt.subplots(figsize=(7, 5))

    df.boxplot(column=time_col, by=group_col, ax=ax)
    group_sizes = get_group_sizes(df, group_col)
    title_suffix = " / ".join(
        [f"{g} (n={group_sizes[g]})" for g in group_sizes]
    )
    ax.set_title(f"Follow-up Time by Group\n{title_suffix}")
    ax.set_ylabel("Months")
    plt.suptitle("")
    plt.tight_layout()
    return fig


# --------------------------------------------------------
# 5) Missingness table
# --------------------------------------------------------
def compute_missing_table(
    df,
    group_col="group",
    cols_to_check=("STAGE", "AGE", "SEX", "OS_MONTHS", "is_dead"),
):
    rows = []

    for col in cols_to_check:
        if col not in df.columns:
            continue

        tmp = df[[group_col, col]].copy()
        miss = tmp[col].isna()

        miss_rate = (
            tmp.groupby(group_col)[col]
            .apply(lambda x: x.isna().mean())
            .to_dict()
        )

        row = {"Variable": col}
        row.update({f"{g}": f"{miss_rate.get(g, 0)*100:.1f}%" for g in df[group_col].unique()})
        rows.append(row)

    return pd.DataFrame(rows)

# --------------------------------------------------------
# X) Helper
# --------------------------------------------------------
def get_group_sizes(df, group_col="group"):
    counts = df[group_col].value_counts()
    order = ["HiHi", "HiLo", "LoHi", "LoLo", "High", "Low"]
    counts = counts.reindex([g for g in order if g in counts.index])
    return counts.to_dict()

# --------------------------------------------------------
# X) 各群ごとの主要カラムの欠測率をテーブルで返す
# --------------------------------------------------------
def missingness_by_group(df, group_col="group"):
    if group_col not in df.columns:
        return None

    cols = [
        "OS_MONTHS",
        "is_dead",
        "AGE",
        "SEX",
        "stage_num",
    ]

    # 存在する列だけ使う
    cols = [c for c in cols if c in df.columns]

    result = {}

    for g, sub in df.groupby(group_col):
        miss = sub[cols].isna().mean() * 100
        result[g] = miss

    if not result:
        return None

    out = pd.DataFrame(result).T
    out.index.name = "Group"
    return out.round(1)


# --------------------------------------------------------
# 6) Expression by Clinical Category (Boxplot + Swarmplot)
# --------------------------------------------------------

def plot_expression_by_category(df, expr_col, category_col, title="Expression by Category", max_points=300):
    if not _col_exists(df, expr_col) or not _col_exists(df, category_col):
        return None

    # Drop NaNs for plotting
    plot_df = df[[category_col, expr_col]].dropna()
    if plot_df.empty:
        return None
        
    # Standardize stage names if dealing with stage
    if "stage" in category_col.lower() or "stage_num" in category_col.lower():
        if is_numeric_dtype(plot_df[category_col]):
            s = pd.to_numeric(plot_df[category_col], errors="coerce").round().astype("Int64")
            plot_df[category_col] = s.map({1: "Stage I", 2: "Stage II", 3: "Stage III", 4: "Stage IV"}).fillna("Unknown")
            
    # Sort categories alphabetically or logically
    categories = sorted(plot_df[category_col].unique().tolist())
    # Move "Unknown", "NA" to the end if present
    for na_term in ["unknown", "na", "nan", "none"]:
        for c in list(categories):
            if str(c).lower() == na_term:
                categories.remove(c)
                categories.append(c)

    # Set up figure size depending on number of categories
    width = max(6, int(len(categories) * 1.5))
    fig, ax = plt.subplots(figsize=(width, 5))
    
    # Aesthetic palette
    palette = sns.color_palette("Set2", n_colors=len(categories))

    # Boxplot
    sns.boxplot(
        x=category_col, 
        y=expr_col, 
        data=plot_df, 
        order=categories,
        ax=ax,
        palette=palette,
        hue=category_col,
        legend=False,
        fliersize=0, # hide outliers to let swarmplot show them
        boxprops=dict(alpha=0.6)
    )
    
    # Swarmplot (subsample if too many points to avoid taking forever / cluttering)
    if len(plot_df) > max_points:
        swarm_df = plot_df.sample(n=max_points, random_state=42)
        sns.stripplot(
            x=category_col, 
            y=expr_col, 
            data=swarm_df, 
            order=categories,
            ax=ax,
            color=".25",
            alpha=0.6,
            jitter=True,
            size=3
        )
    else:
        sns.swarmplot(
            x=category_col, 
            y=expr_col, 
            data=plot_df, 
            order=categories,
            ax=ax,
            color=".25",
            alpha=0.8,
            size=4
        )

    # Calculate p-value (Kruskal-Wallis H-test for non-parametric comparison)
    try:
        from scipy.stats import kruskal
        groups = [plot_df[plot_df[category_col] == c][expr_col].values for c in categories if len(plot_df[plot_df[category_col] == c]) > 0]
        if len(groups) > 1:
            stat, p = kruskal(*groups)
            title += f" (Kruskal-Wallis p={p:.3e})"
    except Exception:
        pass

    ax.set_title(title)
    ax.set_ylabel(expr_col)
    ax.set_xlabel(category_col.replace("_", " ").title())
    
    # Rotate x labels if many categories or long names
    if len(categories) > 5 or any(len(str(c)) > 10 for c in categories):
        plt.xticks(rotation=45, ha='right')
        
    plt.tight_layout()
    return fig
