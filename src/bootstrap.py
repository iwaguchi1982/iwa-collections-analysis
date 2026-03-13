import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from src.survival import find_optimal_1gene_cutoff
import joblib

def _run_single_1gene_bootstrap(
    df: pd.DataFrame,
    gene: str,
    time_col: str,
    event_col: str,
    base_cutoff_pct: float,
    use_auto_cutoff: bool,
    seed: int
) -> Optional[Dict[str, float]]:
    # Resample with replacement
    df_boot = df.sample(n=len(df), replace=True, random_state=seed)
    
    cutoff_pct = base_cutoff_pct
    
    # If using auto-cutoff, re-find the optimal cutoff on this specific bootstrap sample
    if use_auto_cutoff:
        opt_res = find_optimal_1gene_cutoff(
            df_boot, 
            gene, 
            time_col=time_col, 
            event_col=event_col, 
            min_pct=20, 
            max_pct=80
        )
        if opt_res["ok"]:
            cutoff_pct = opt_res["best_pct"]
            
    # Apply cutoff
    val = df_boot["expression"].quantile(cutoff_pct / 100.0)
    df_boot["is_high"] = (df_boot["expression"] >= val).astype(int)
    
    # Needs at least some events in both groups
    if df_boot["is_dead"].sum() < 5 or df_boot["is_high"].nunique() < 2:
        return None
        
    try:
        # Run Cox
        cph = CoxPHFitter(penalizer=0.01)
        cph.fit(df_boot[[time_col, event_col, "is_high"]], duration_col=time_col, event_col=event_col)
        hr = cph.hazard_ratios_["is_high"]
        
        # Run Log-rank
        high_mask = df_boot["is_high"] == 1
        lr = logrank_test(
            df_boot.loc[high_mask, time_col],
            df_boot.loc[~high_mask, time_col],
            event_observed_A=df_boot.loc[high_mask, event_col],
            event_observed_B=df_boot.loc[~high_mask, event_col]
        )
        p_val = lr.p_value
        
        return {
            "hr": hr,
            "p_val": p_val,
            "cutoff_pct": cutoff_pct
        }
    except Exception:
        return None


def run_1gene_bootstrap(
    df: pd.DataFrame,
    gene: str,
    time_col: str = "OS_MONTHS",
    event_col: str = "is_dead",
    base_cutoff_pct: float = 50.0,
    use_auto_cutoff: bool = False,
    n_iterations: int = 100,
    n_jobs: int = -1
) -> Dict[str, Any]:
    """
    Run bootstrap resampling to assess the stability of 1-Gene survival metrics.
    """
    if df is None or df.empty or "expression" not in df.columns:
        return {"ok": False, "reason": "Invalid or missing dataframe."}
        
    # Generate random seeds based on a fixed initialization for reproducibility within this run
    rng = np.random.default_rng(42)
    seeds = rng.integers(0, 1000000, size=n_iterations)
    
    # Run parallel
    results = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_run_single_1gene_bootstrap)(
            df, gene, time_col, event_col, base_cutoff_pct, use_auto_cutoff, int(seed)
        ) for seed in seeds
    )
    
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        return {"ok": False, "reason": "All bootstrap iterations failed (likely insufficient events in samples)."}
        
    df_res = pd.DataFrame(valid_results)
    
    # Calculate stability metrics
    p_vals = df_res["p_val"]
    sig_frac = (p_vals < 0.05).mean()
    
    hr_median = df_res["hr"].median()
    hr_ci_lower = df_res["hr"].quantile(0.025)
    hr_ci_upper = df_res["hr"].quantile(0.975)
    
    if sig_frac >= 0.90:
        stability = "🟢 Stable"
    elif sig_frac >= 0.50:
        stability = "🟡 Borderline"
    else:
        stability = "🔴 Unstable"
        
    return {
        "ok": True,
        "results_df": df_res,
        "n_successful": len(valid_results),
        "sig_fraction": sig_frac,
        "hr_median": hr_median,
        "hr_ci_lower": hr_ci_lower,
        "hr_ci_upper": hr_ci_upper,
        "stability_label": stability
    }


def _run_single_2gene_bootstrap(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    seed: int
) -> Optional[Dict[str, float]]:
    df_boot = df.sample(n=len(df), replace=True, random_state=seed)
    
    if df_boot[event_col].sum() < 5 or df_boot["Interaction"].nunique() < 2:
        return None
        
    try:
        cph = CoxPHFitter(penalizer=0.01)
        cph.fit(
            df_boot[[time_col, event_col, "Gene1_High", "Gene2_High", "Interaction"]], 
            duration_col=time_col, 
            event_col=event_col
        )
        
        hr_int = cph.hazard_ratios_["Interaction"]
        p_int = cph.summary.loc["Interaction", "p"]
        
        return {
            "hr_int": hr_int,
            "p_int": p_int
        }
    except Exception:
        return None


def run_2gene_bootstrap(
    df: pd.DataFrame,
    time_col: str = "OS_MONTHS",
    event_col: str = "is_dead",
    n_iterations: int = 100,
    n_jobs: int = -1
) -> Dict[str, Any]:
    """
    Run bootstrap resampling to assess the stability of 2-Gene Interaction metrics.
    Assumes df already contains 'Gene1_High', 'Gene2_High', and 'Interaction' columns.
    """
    req_cols = [time_col, event_col, "Gene1_High", "Gene2_High", "Interaction"]
    if df is None or df.empty or not all(c in df.columns for c in req_cols):
        return {"ok": False, "reason": "Missing required interaction columns in dataframe."}
        
    rng = np.random.default_rng(42)
    seeds = rng.integers(0, 1000000, size=n_iterations)
    
    results = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_run_single_2gene_bootstrap)(
            df, time_col, event_col, int(seed)
        ) for seed in seeds
    )
    
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        return {"ok": False, "reason": "All bootstrap iterations failed."}
        
    df_res = pd.DataFrame(valid_results)
    
    p_vals = df_res["p_int"]
    sig_frac = (p_vals < 0.05).mean()
    
    hr_median = df_res["hr_int"].median()
    hr_ci_lower = df_res["hr_int"].quantile(0.025)
    hr_ci_upper = df_res["hr_int"].quantile(0.975)
    
    if sig_frac >= 0.90:
        stability = "🟢 Stable"
    elif sig_frac >= 0.50:
        stability = "🟡 Borderline"
    else:
        stability = "🔴 Unstable"
        
    return {
        "ok": True,
        "results_df": df_res,
        "n_successful": len(valid_results),
        "sig_fraction": sig_frac,
        "hr_median": hr_median,
        "hr_ci_lower": hr_ci_lower,
        "hr_ci_upper": hr_ci_upper,
        "stability_label": stability
    }

import matplotlib.pyplot as plt
import seaborn as sns

def plot_bootstrap_results(res: Dict[str, Any], title: str = "Bootstrap Robustness") -> plt.Figure:
    df_res = res["results_df"]
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"{title} (n={res['n_successful']}) - {res['stability_label']}", fontsize=12)
    
    # Plot HR
    hr_col = "hr" if "hr" in df_res.columns else "hr_int"
    sns.histplot(df_res[hr_col], ax=axes[0], kde=True, color="skyblue")
    axes[0].axvline(res['hr_median'], color="red", linestyle="--", label=f"Median: {res['hr_median']:.2f}")
    axes[0].axvline(1.0, color="gray", linestyle="-")
    axes[0].set_title("Hazard Ratio Distribution")
    axes[0].set_xlabel("Hazard Ratio")
    axes[0].legend()
    
    # Plot P-value
    p_col = "p_val" if "p_val" in df_res.columns else "p_int"
    sns.histplot(df_res[p_col], ax=axes[1], bins=20, color="salmon")
    axes[1].axvline(0.05, color="red", linestyle="--", label="p = 0.05")
    axes[1].set_title("P-value Distribution")
    axes[1].set_xlabel("P-value")
    axes[1].legend()
    
    plt.tight_layout()
    return fig
