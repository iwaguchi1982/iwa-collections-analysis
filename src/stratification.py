import pandas as pd
import numpy as np
from typing import List, Dict, Any
from .survival import run_cox

def run_subgroup_analysis(
    df: pd.DataFrame,
    gene_col: str,
    duration_col: str = "OS_MONTHS",
    event_col: str = "is_dead",
    target_var: str = "is_high",
    covariates_to_test: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Runs Cox proportional hazards models for the target_var (e.g., 'is_high') 
    across various clinical subgroups defined by covariates_to_test.
    
    Args:
        df: DataFrame containing survival data, the target variable, and clinical covariates.
        gene_col: Name of the gene being analyzed (for display purposes).
        duration_col: Time to event column.
        event_col: Event indicator column (1=dead, 0=censored).
        target_var: The primary binary variable to test (e.g., 'is_high' expression).
        covariates_to_test: List of clinical columns to stratify by (e.g., ['STAGE', 'SEX']).
        
    Returns:
        List of dictionaries containing subgroup results (HR, p-value, N, etc.).
    """
    results = []
    
    # 1. First, calculate the "Overall" (All Patients) baseline effect
    cox_all = run_cox(df, duration_col=duration_col, event_col=event_col, main_var=target_var, covariates=[])
    if cox_all["ok"]:
        results.append({
            "Covariate": "Overall",
            "Subgroup": "All Patients",
            "N": len(df),
            "Events": int(df[event_col].sum()),
            "HR": cox_all["hr"],
            "HR_lower": cox_all["hr_l"],
            "HR_upper": cox_all["hr_u"],
            "p_value": cox_all["p"],
            "is_significant": cox_all["p"] < 0.05
        })

    if not covariates_to_test:
        return results

    # 2. Iterate through requested clinical covariates
    for cov in covariates_to_test:
        if cov not in df.columns:
            continue
            
        # Drop missing values for this specific covariate
        df_cov = df.dropna(subset=[cov]).copy()
        if len(df_cov) == 0:
            continue
            
        # Determine if covariate is continuous or categorical
        # Heuristic: If numeric and >10 unique values -> Continuous, else Categorical
        if pd.api.types.is_numeric_dtype(df_cov[cov]) and df_cov[cov].nunique() > 10:
            # Split continuous variable by median
            median_val = df_cov[cov].median()
            df_cov[f"{cov}_group"] = np.where(df_cov[cov] >= median_val, f"High (>= {median_val:.1f})", f"Low (< {median_val:.1f})")
            group_col = f"{cov}_group"
        else:
            group_col = cov
            
        # Get unique values, filtering out common "missing/unknown" string representations
        unique_groups = [g for g in df_cov[group_col].unique() if str(g).lower() not in ["n/a", "nan", "unknown", "not reported", ""]]
        unique_groups = sorted(unique_groups)

        for group_val in unique_groups:
            # Subset the data to just this specific subgroup
            df_sub = df_cov[df_cov[group_col] == group_val].copy()
            
            # Require at least 20 patients and 5 events to run a meaningful Cox model
            if len(df_sub) < 20 or df_sub[event_col].sum() < 5:
                continue
                
            # Also require that the target variable (is_high) actually varies in this subgroup!
            if df_sub[target_var].nunique() < 2:
                continue

            # Run isolated Cox model
            cox_sub = run_cox(df_sub, duration_col=duration_col, event_col=event_col, main_var=target_var, covariates=[])
            
            if cox_sub["ok"]:
                results.append({
                    "Covariate": cov,
                    "Subgroup": str(group_val),
                    "N": len(df_sub),
                    "Events": int(df_sub[event_col].sum()),
                    "HR": cox_sub["hr"],
                    "HR_lower": cox_sub["hr_l"],
                    "HR_upper": cox_sub["hr_u"],
                    "p_value": cox_sub["p"],
                    "is_significant": cox_sub["p"] < 0.05
                })

    return results

def get_top_recommended_subgroup(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Finds the specific subgroup where the target gene has the lowest p-value
    and strongest hazard ratio (excluding the 'Overall' group).
    """
    subgroups = [r for r in results if r["Covariate"] != "Overall" and r["is_significant"]]
    
    if not subgroups:
        return None
        
    # Sort by p-value ascending, then by Hazard Ratio magnitude descending
    # For HR magnitude, we want to maximize the distance from 1.0
    # i.e., HR=0.2 (protective) or HR=5.0 (risk) are both "strong"
    subgroups.sort(key=lambda x: (x["p_value"], -abs(np.log(x["HR"]))))
    
    return subgroups[0]

import matplotlib.pyplot as plt

def plot_subgroup_forest(results: List[Dict[str, Any]], title: str = "Subgroup Analysis Forest Plot"):
    """
    Plots a Forest Plot from the output of run_subgroup_analysis.
    """
    if not results:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.axis("off")
        ax.text(0.5, 0.5, "No subgroup results to plot.", ha="center", va="center")
        return fig
    
    # Create DataFrame for plotting
    df = pd.DataFrame(results)
    
    # Sort logically: Overall first, then grouped by Covariate
    # Create a custom sorter
    covs = df["Covariate"].unique()
    cov_order = {"Overall": 0}
    for i, c in enumerate([c for c in covs if c != "Overall"]):
        cov_order[c] = i + 1
        
    df["_order1"] = df["Covariate"].map(cov_order)
    df["_order2"] = df["Subgroup"]
    df = df.sort_values(["_order1", "_order2"]).reset_index(drop=True)
    
    # Reverse for plotting top-to-bottom
    df = df.iloc[::-1].reset_index(drop=True)
    
    # Determine figure height
    fig_h = max(3.0, 0.5 * len(df) + 1.0)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    
    y_pos = np.arange(len(df))
    
    # Plot reference line for HR=1
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1, zorder=1)
    
    labels = []
    
    for i, row in df.iterrows():
        y = y_pos[i]
        hr, lo, hi = row["HR"], row["HR_lower"], row["HR_upper"]
        p_val = row["p_value"]
        is_overall = (row["Covariate"] == "Overall")
        
        # Style
        color = "darkred" if is_overall else "darkblue"
        lw = 2.5 if is_overall else 1.5
        ms = 9 if is_overall else 6
        
        # Plot range line
        ax.plot([lo, hi], [y, y], color=color, linewidth=lw, alpha=0.8, zorder=2)
        # Plot point estimate
        marker = "D" if is_overall else "o"
        ax.plot(hr, y, marker=marker, markersize=ms, color=color, zorder=3)
        
        # Label text construction
        if is_overall:
            lbl = f"Overall (N={row['N']})"
        else:
            lbl = f"{row['Covariate']}: {row['Subgroup']} (N={row['N']})"
        labels.append(lbl)
        
        # Add p-value text to the right
        sig_star = "*" if p_val < 0.05 else ""
        sig_star += "*" if p_val < 0.01 else ""
        sig_star += "*" if p_val < 0.001 else ""
        
        ax.text(
            1.02, y, f"HR={hr:.2f}  p={p_val:.3e}{sig_star}",
            transform=ax.get_yaxis_transform(),
            va="center", ha="left", fontsize=9,
            fontweight="bold" if is_overall or p_val < 0.05 else "normal"
        )
        
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xscale("log")
    ax.set_xlabel("Hazard Ratio (log scale)")
    ax.set_title(title, pad=15)
    
    # Dynamically scale x-axis bounds
    xmin = df["HR_lower"].min()
    xmax = df["HR_upper"].max()
    if np.isfinite(xmin) and np.isfinite(xmax) and xmin > 0:
        ax.set_xlim(xmin * 0.7, xmax * 1.5)
        
    # Remove top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig
