# src/cohort_shift.py
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from src.survival import run_cox
from src.missingness import get_missingness_stats
from src.analysis_issues import Issue

def calculate_smd(group1: pd.Series, group2: pd.Series, is_categorical: bool = False) -> float:
    """
    Calculates the Standardized Mean Difference (SMD) between two groups.
    """
    if is_categorical:
        # For categorical, we use the formula for proportion-based SMD or general categorical
        # Here we use a simplified version for the most frequent category or just return 0 for now
        # Actually, for multiple categories, SMD is often calculated via covariance matrix
        # Let's implement a simplified version for binary or just mean of dummies
        g1_counts = group1.value_counts(normalize=True)
        g2_counts = group2.value_counts(normalize=True)
        all_cats = set(g1_counts.index) | set(g2_counts.index)
        
        diffs = []
        for cat in all_cats:
            p1 = g1_counts.get(cat, 0)
            p2 = g2_counts.get(cat, 0)
            # p_pooled = (p1 + p2) / 2
            # denom = np.sqrt(p_pooled * (1 - p_pooled))
            # diffs.append(abs(p1 - p2) / denom if denom > 0 else 0)
            diffs.append(abs(p1 - p2))
        return float(np.mean(diffs)) # Simplified "average proportion difference" as a proxy for SMD in categorical
    else:
        m1, m2 = group1.mean(), group2.mean()
        v1, v2 = group1.var(), group2.var()
        if np.isnan(v1): v1 = 0
        if np.isnan(v2): v2 = 0
        denom = np.sqrt((v1 + v2) / 2)
        return float(abs(m1 - m2) / denom) if denom > 0 else 0.0

def compare_clinical_composition(
    df_disc: pd.DataFrame, 
    df_val: pd.DataFrame, 
    covariates: List[str],
    categorical_covariates: List[str]
) -> List[Dict[str, Any]]:
    """
    Compares clinical distributions between cohorts.
    """
    results = []
    for col in covariates:
        if col not in df_disc.columns or col not in df_val.columns:
            continue
        
        is_cat = col in categorical_covariates
        smd = calculate_smd(df_disc[col].dropna(), df_val[col].dropna(), is_categorical=is_cat)
        
        results.append({
            "column": col,
            "is_categorical": is_cat,
            "smd": smd,
            "disc_summary": df_disc[col].describe().to_dict() if not is_cat else df_disc[col].value_counts(normalize=True).to_dict(),
            "val_summary": df_val[col].describe().to_dict() if not is_cat else df_val[col].value_counts(normalize=True).to_dict(),
            "shift_severity": "Substantial" if smd > 0.2 else "Moderate" if smd > 0.1 else "Minimal"
        })
    return results

def compare_outcome_structure(
    df_disc: pd.DataFrame, 
    df_val: pd.DataFrame,
    duration_col: str = "OS_MONTHS",
    event_col: str = "is_dead"
) -> Dict[str, Any]:
    """
    Compares survival outcome structure.
    """
    def get_metrics(df, name):
        raw_n = len(df)
        df_clean = df.dropna(subset=[duration_col, event_col])
        eligible_n = len(df_clean)
        events = int(df_clean[event_col].sum())
        event_rate = events / eligible_n if eligible_n > 0 else 0
        median_fu = df_clean[duration_col].median()
        censoring_rate = 1.0 - event_rate
        
        return {
            "cohort": name,
            "raw_n": raw_n,
            "eligible_n": eligible_n,
            "events": events,
            "event_rate": event_rate,
            "median_followup": float(median_fu),
            "censoring_rate": censoring_rate
        }
    
    disc_m = get_metrics(df_disc, "Discovery")
    val_m = get_metrics(df_val, "Validation")
    
    # Calculate differences
    er_diff = abs(disc_m["event_rate"] - val_m["event_rate"])
    fu_ratio = val_m["median_followup"] / disc_m["median_followup"] if disc_m["median_followup"] > 0 else 1.0
    
    severity = "Minimal"
    if er_diff > 0.15 or fu_ratio < 0.5 or fu_ratio > 2.0:
        severity = "Substantial"
    elif er_diff > 0.05 or fu_ratio < 0.8 or fu_ratio > 1.25:
        severity = "Moderate"
        
    return {
        "discovery": disc_m,
        "validation": val_m,
        "event_rate_diff": er_diff,
        "followup_ratio": fu_ratio,
        "shift_severity": severity
    }

def compare_molecular_shift(
    df_disc: pd.DataFrame, 
    df_val: pd.DataFrame, 
    gene_label: str
) -> Dict[str, Any]:
    """
    Focuses on distribution summary (Density/Violin, Median/IQR, Quantiles).
    """
    target_col = "expression" # Internal name used after processing
    if target_col not in df_disc.columns or target_col not in df_val.columns:
        return {"ok": False, "error": "Target expression column not found"}
    
    d_vals = df_disc[target_col].dropna()
    v_vals = df_val[target_col].dropna()
    
    smd = calculate_smd(d_vals, v_vals, is_categorical=False)
    
    res = {
        "ok": True,
        "gene": gene_label,
        "smd": smd,
        "discovery": {
            "median": float(d_vals.median()),
            "iqr": float(d_vals.quantile(0.75) - d_vals.quantile(0.25)),
            "q25": float(d_vals.quantile(0.25)),
            "q75": float(d_vals.quantile(0.75)),
            "mean": float(d_vals.mean()),
            "std": float(d_vals.std())
        },
        "validation": {
            "median": float(v_vals.median()),
            "iqr": float(v_vals.quantile(0.75) - v_vals.quantile(0.25)),
            "q25": float(v_vals.quantile(0.25)),
            "q75": float(v_vals.quantile(0.75)),
            "mean": float(v_vals.mean()),
            "std": float(v_vals.std())
        },
        "shift_severity": "Substantial" if smd > 0.2 else "Moderate" if smd > 0.1 else "Minimal"
    }
    return res

def compare_cutoff_behavior(
    df_disc: pd.DataFrame, 
    df_val: pd.DataFrame, 
    gene_label: str,
    disc_cutoff_val: float,
    disc_cutoff_type: str,
    direction: str,
    disc_cutoff_percentile: Optional[float] = None
) -> Dict[str, Any]:
    """
    Analyzes how the discovery cutoff partitions the validation cohort.
    """
    target_col = "expression"
    if target_col not in df_val.columns:
        return {"ok": False}
    
    v_vals = df_val[target_col].dropna()
    n_val = len(v_vals)
    
    # 1. Apply Discovery Cutoff to Validation
    if direction == "High vs Low":
        v_high_mask = v_vals >= disc_cutoff_val
    else:
        v_high_mask = v_vals <= disc_cutoff_val # Potentially inverted
        
    v_high_n = int(v_high_mask.sum())
    v_high_frac = v_high_n / n_val if n_val > 0 else 0.5
    
    # 2. What percentile does the discovery cutoff map to in validation?
    current_percentile = (v_vals < disc_cutoff_val).mean() * 100
    
    # 3. If we used the SAME percentile in validation, what would the cutoff be?
    target_cutoff_val = disc_cutoff_val
    if disc_cutoff_percentile is not None:
        target_cutoff_val = v_vals.quantile(disc_cutoff_percentile / 100)
    
    # Balance shift
    # Assume discovery was roughly balanced if it's median, or use the provided percentile
    target_frac = (100 - disc_cutoff_percentile) / 100 if disc_cutoff_percentile is not None else 0.5
    frac_diff = abs(v_high_frac - target_frac)
    
    severity = "Minimal"
    if frac_diff > 0.2: severity = "Substantial"
    elif frac_diff > 0.1: severity = "Moderate"

    return {
        "ok": True,
        "disc_cutoff_val": disc_cutoff_val,
        "disc_cutoff_type": disc_cutoff_type,
        "disc_cutoff_percentile": disc_cutoff_percentile,
        "val_actual_high_frac": v_high_frac,
        "val_actual_high_n": v_high_n,
        "val_mapped_percentile": current_percentile,
        "val_equivalent_cutoff": float(target_cutoff_val),
        "fraction_drift": frac_diff,
        "shift_severity": severity
    }

def compare_data_quality(
    df_disc: pd.DataFrame, 
    df_val: pd.DataFrame, 
    covariates: List[str]
) -> Dict[str, Any]:
    """
    Compares missingness rates between cohorts by reusing 0012 logic.
    """
    disc_stats = get_missingness_stats(df_disc, covariates)
    val_stats = get_missingness_stats(df_val, covariates)
    
    m_diff = abs(disc_stats["any_missing_rate"] - val_stats["any_missing_rate"])
    
    severity = "Minimal"
    if m_diff > 0.2: severity = "Substantial"
    elif m_diff > 0.1: severity = "Moderate"
    
    return {
        "discovery_stats": disc_stats,
        "validation_stats": val_stats,
        "overall_missing_diff": m_diff,
        "shift_severity": severity
    }

def classify_cohort_shift(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Internal helper to return shift labels and drivers.
    """
    severities = []
    drivers = []
    
    # Check Clinical
    clin = results.get("clinical", [])
    max_smd = 0
    worst_clin = None
    for c in clin:
        if c["smd"] > max_smd:
            max_smd = c["smd"]
            worst_clin = c["column"]
    
    if max_smd > 0.2: 
        severities.append(3)
        drivers.append(f"Clinical Composition ({worst_clin})")
    elif max_smd > 0.1:
        severities.append(2)
        drivers.append(f"Clinical Composition ({worst_clin})")
    else:
        severities.append(1)
        
    # Check Outcome
    out = results.get("outcome", {})
    s_out = out.get("shift_severity", "Minimal")
    if s_out == "Substantial": 
        severities.append(3)
        drivers.append("Outcome Structure (Event rate/Follow-up)")
    elif s_out == "Moderate":
        severities.append(2)
        drivers.append("Outcome Structure")
    else:
        severities.append(1)
        
    # Check Molecular
    mol = results.get("molecular", {})
    s_mol = mol.get("shift_severity", "Minimal")
    if s_mol == "Substantial": 
        severities.append(3)
        drivers.append("Target Molecular Distribution")
    elif s_mol == "Moderate":
        severities.append(2)
        drivers.append("Target Molecular Distribution")
    else:
        severities.append(1)
        
    # Check Cutoff
    cut = results.get("cutoff", {})
    s_cut = cut.get("shift_severity", "Minimal")
    if s_cut == "Substantial": 
        severities.append(3)
        drivers.append("Cutoff Group Balance")
    elif s_cut == "Moderate":
        severities.append(2)
        drivers.append("Cutoff Behavior")
    else:
        severities.append(1)
        
    # Check Data Quality
    dq = results.get("data_quality", {})
    s_dq = dq.get("shift_severity", "Minimal")
    if s_dq == "Substantial": 
        severities.append(3)
        drivers.append("Data Completeness / Missingness")
    elif s_dq == "Moderate":
        severities.append(2)
        drivers.append("Data Completeness")
    else:
        severities.append(1)
        
    avg_sev = np.mean(severities)
    label = "Minimal"
    if avg_sev > 2.2 or any(s == 3 for s in severities): label = "Substantial"
    elif avg_sev > 1.4: label = "Moderate"
    
    # Interpretation Text
    interp = ""
    if label == "Substantial":
        interp = "Significant differences detected between cohorts. Validation outcome should be interpreted with extreme caution as the underlying populations or data structures differ fundamentally."
    elif label == "Moderate":
        interp = "Moderate cohort shift observed. Differences in composition or molecular distribution may partially explain discrepancies in validation results."
    else:
        interp = "Cohorts are broadly comparable. Validation results are likely reflecting true biological (re)producibility."
        
    return {
        "shift_label": label,
        "interpretation_text": interp,
        "top_shift_drivers": drivers[:3],
        "block_labels": {
            "clinical": results.get("clinical_severity", "Minimal"), # We'll set this in summarize
            "outcome": s_out,
            "molecular": s_mol,
            "cutoff": s_cut,
            "data_quality": s_dq
        }
    }

def summarize_cohort_shift_results(
    df_disc: pd.DataFrame,
    df_val: pd.DataFrame,
    gene_label: str,
    covariates: List[str],
    categorical_covariates: List[str],
    disc_cutoff_val: float,
    disc_cutoff_type: str,
    direction: str,
    disc_cutoff_percentile: Optional[float] = None
) -> Dict[str, Any]:
    """
    Orchestrates the full cohort shift analysis.
    """
    res = {}
    res["clinical"] = compare_clinical_composition(df_disc, df_val, covariates, categorical_covariates)
    res["clinical_severity"] = "Minimal"
    if res["clinical"]:
        max_smd = max([c["smd"] for c in res["clinical"]])
        res["clinical_severity"] = "Substantial" if max_smd > 0.2 else "Moderate" if max_smd > 0.1 else "Minimal"
        
    res["outcome"] = compare_outcome_structure(df_disc, df_val)
    res["molecular"] = compare_molecular_shift(df_disc, df_val, gene_label)
    res["cutoff"] = compare_cutoff_behavior(df_disc, df_val, gene_label, disc_cutoff_val, disc_cutoff_type, direction, disc_cutoff_percentile)
    res["data_quality"] = compare_data_quality(df_disc, df_val, covariates)
    
    summary = classify_cohort_shift(res)
    res["summary"] = summary
    
    issues = []
    if summary["shift_label"] == "Substantial":
        issues.append(Issue(
            category="cohort shift",
            severity="warning",
            kind="caution",
            title="Substantial Cohort Shift Detected",
            detail=f"Cohorts differ significantly in: {', '.join(summary['top_shift_drivers'])}.",
            evidence=f"shift_label={summary['shift_label']}",
            recommendation="Review the Cohort Shift Insight dashboard. Validation failure may be due to population differences.",
            source_module="cohort_shift.summarize_cohort_shift_results"
        ))
    elif summary["shift_label"] == "Moderate":
        issues.append(Issue(
            category="cohort shift",
            severity="info",
            kind="caution",
            title="Moderate Cohort Shift observed",
            detail=f"Some differences in {', '.join(summary['top_shift_drivers'])} may affect validation consistency.",
            evidence=f"shift_label={summary['shift_label']}",
            recommendation="Consider if observed differences explain discrepancies in effect size.",
            source_module="cohort_shift.summarize_cohort_shift_results"
        ))
        
    res["issues"] = issues
    return res
