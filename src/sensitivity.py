# src/sensitivity.py
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from src.survival import run_cox, calculate_rmst, apply_landmark

def run_cutoff_scan(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    gene_col: str,
    covariates: List[str],
    min_pct: int = 20,
    max_pct: int = 80,
    step: int = 5
) -> Dict[str, Any]:
    """
    Scans different percentiles for cutoff and returns HR, p-value, and sample size N.
    Returns: {"ok": True, "results": [{"pct": int, "hr": float, "p": float, "n": int, "events": int}, ...]}
    """
    if gene_col not in df.columns:
        return {"ok": False, "reason": f"Gene column {gene_col} not found"}

    results = []
    pcts = list(range(min_pct, max_pct + 1, step))
    
    # Ensure baseline N/events are sufficient before scanning
    for pct in pcts:
        q = pct / 100.0
        cutoff_val = df[gene_col].quantile(q)
        
        # Binary grouping
        df_tmp = df.copy()
        df_tmp["group_binary"] = (df_tmp[gene_col] >= cutoff_val).astype(int)
        
        res = run_cox(
            df_tmp,
            duration_col=duration_col,
            event_col=event_col,
            main_var="group_binary",
            covariates=covariates
        )
        
        if res.get("ok"):
            results.append({
                "pct": pct,
                "cutoff_val": float(cutoff_val),
                "hr": float(res["hr"]),
                "p": float(res["p"]),
                "n": int(res["n"]),
                "events": int(res["events"])
            })
        else:
            results.append({
                "pct": pct,
                "cutoff_val": float(cutoff_val),
                "hr": np.nan,
                "p": np.nan,
                "n": int(res.get("n", 0)),
                "events": int(res.get("events", 0))
            })
            
    return {"ok": True, "results": results}

def run_covariate_sensitivity(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    main_var: str,
    predefined_sets: Dict[str, List[str]]
) -> Dict[str, Any]:
    """
    Runs Cox models with different covariate adjustment sets.
    """
    results = {}
    for label, covs in predefined_sets.items():
        res = run_cox(df, duration_col, event_col, main_var, covs)
        results[label] = res
        
    return {"ok": True, "models": results}

def run_method_comparison(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    group_col: str,
    tau: Optional[float] = None,
    landmark_time: Optional[float] = None
) -> Dict[str, Any]:
    """
    Compares Cox, Landmark (Late phase), and RMST.
    """
    # 1. Cox
    cox_res = run_cox(df, duration_col, event_col, group_col, [])
    
    # 2. RMST
    rmst_res = calculate_rmst(df, duration_col, event_col, group_col, tau=tau)
    
    # 3. Landmark Analysis (Late phase)
    landmark_res = {"ok": False}
    if landmark_time and landmark_time > 0:
        _, late_df = apply_landmark(df, duration_col, event_col, landmark_time)
        if late_df is not None and not late_df.empty:
            landmark_res = run_cox(late_df, duration_col, event_col, group_col, [])
    
    return {
        "ok": True,
        "cox": cox_res,
        "rmst": rmst_res,
        "landmark": landmark_res,
        "consistency": _evaluate_consistency(cox_res, rmst_res, landmark_res)
    }

def _evaluate_consistency(cox_res: Dict, rmst_res: Dict, landmark_res: Dict) -> str:
    """ Evaluates whether Cox and RMST (and optionally Landmark) agree on direction. """
    if not cox_res.get("ok") or not rmst_res.get("ok"):
        return "Unknown"
    
    hr_cox = cox_res["hr"]
    vals = rmst_res["rmst_values"]
    
    # Check Cox vs RMST
    rmst_consistent = False
    if "1" in vals and "0" in vals:
        rmst_diff = vals["1"] - vals["0"]
        if (hr_cox > 1 and rmst_diff < 0) or (hr_cox < 1 and rmst_diff > 0):
            rmst_consistent = True
            
    # Check Cox vs Landmark (if available)
    landmark_consistent = True # Default true if not ran
    if landmark_res.get("ok"):
        hr_land = landmark_res["hr"]
        if (hr_cox > 1 and hr_land < 1) or (hr_cox < 1 and hr_land > 1):
            landmark_consistent = False
            
    if rmst_consistent and landmark_consistent:
        return "Consistent"
    elif rmst_consistent or landmark_consistent:
        return "Partially Consistent"
    else:
        return "Conflicting"

def classify_robustness(
    cutoff_res: List[Dict],
    covariate_res: Dict[str, Dict],
    method_res: Dict
) -> Dict[str, Any]:
    """
    Classifies the target and provides structured reasons.
    """
    reasons = []
    
    # 1. Cutoff Stability
    valid_p = [r["p"] for r in cutoff_res if pd.notna(r["p"])]
    sig_pct = sum(1 for p in valid_p if p < 0.05) / len(cutoff_res) if cutoff_res else 0
    cutoff_stable = sig_pct > 0.5 
    if cutoff_stable:
        reasons.append("Stable across broad cutoff range")
    else:
        reasons.append("Significant results limited to narrow cutoff window")
    
    # 2. Covariate Stability
    cov_variants = [m.get("ok") and m.get("p", 1.0) < 0.05 for m in covariate_res.values()]
    cov_stable = all(cov_variants) if cov_variants else False
    
    # Check drift
    unadj = covariate_res.get("Unadjusted", {})
    full = covariate_res.get("Full (Age, Sex, Stage)", {})
    drift_small = True
    if unadj.get("ok") and full.get("ok"):
        drift = abs(unadj["hr"] - full["hr"]) / unadj["hr"]
        if drift < 0.2:
            reasons.append(f"Low HR drift ({drift:.1%}) after adjustment")
        else:
            reasons.append(f"Significant HR drift ({drift:.1%}) detected")
            drift_small = False
    
    if cov_stable:
        reasons.append("Consistently significant across adjustment models")
    else:
        reasons.append("Significance lost in some adjusted models")
    
    # 3. Method Consistency
    consistency = method_res.get("consistency")
    method_stable = (consistency == "Consistent")
    if method_stable:
        reasons.append("Direction consistent across Cox/RMST/Landmark")
    else:
        reasons.append(f"Method consistency: {consistency}")
    
    # 4. Data Sufficiency
    min_n = min([r.get("n", 0) for r in cutoff_res]) if cutoff_res else 0
    data_sufficient = min_n > 50
    if not data_sufficient:
        reasons.append(f"Relatively small sample size (N_min={min_n})")

    score = 0
    if cutoff_stable: score += 1
    if cov_stable and drift_small: score += 1
    if method_stable: score += 1
    
    classification = "Fragile"
    if score == 3: classification = "Stable"
    elif score >= 1: classification = "Moderately sensitive"
    
    return {
        "classification": classification,
        "details": {
            "cutoff_stability": "High" if cutoff_stable else "Low",
            "covariate_stability": "High" if cov_stable else "Low",
            "method_consistency": "High" if method_stable else "Low",
            "data_sufficiency": "High" if data_sufficient else "Low"
        },
        "reasons": reasons
    }

def summarize_sensitivity_results(
    cutoff_res: List[Dict],
    covariate_res: Dict[str, Dict],
    method_res: Dict,
    classification: Dict
) -> Dict[str, str]:
    """
    Generates text summaries for the dashboard and reports.
    """
    cls = classification["classification"]
    
    # Cutoff Check
    sig_range = [r["pct"] for r in cutoff_res if pd.notna(r["p"]) and r["p"] < 0.05]
    if sig_range:
        cutoff_sum = f"Significant in {min(sig_range)}%-{max(sig_range)}% percentile range."
    else:
        cutoff_sum = "No significant percentile found."
        
    # Covariate Check
    cov_dir = []
    unadj = covariate_res.get("Unadjusted", {})
    full = covariate_res.get("Full (Age, Sex, Stage)", {})
    if unadj.get("ok") and full.get("ok"):
        hr_drift = abs(unadj["hr"] - full["hr"]) / unadj["hr"]
        cov_sum = f"HR drift after full adjustment: {hr_drift:.1%}."
    else:
        cov_sum = "Incomplete covariate data."
        
    # Method Check
    method_sum = f"Directional consistency: {method_res.get('consistency')}."
    
    overall = f"Target is classified as **{cls}**. "
    if cls == "Stable":
        overall += "The association is highly consistent across multiple analytic settings."
    elif cls == "Moderately sensitive":
        overall += "The result holds generally, but exhibits some dependence on specific parameters."
    else:
        overall += "The significance appears heavily dependent on specific analytic choices or lacks robustness."
        
    return {
        "cutoff_summary": cutoff_sum,
        "covariate_summary": cov_sum,
        "method_summary": method_sum,
        "overall_interpretation": overall
    }
