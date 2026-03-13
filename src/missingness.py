# src/missingness.py
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from src.analysis_issues import Issue
from src.survival import run_cox

def get_missingness_stats(df: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
    """
    Calculates missing rates and patterns for the specified columns.
    """
    total_n = len(df)
    stats = []
    for col in columns:
        if col not in df.columns:
            continue
        m_count = df[col].isna().sum()
        m_rate = m_count / total_n
        severity = "Low"
        if m_rate > 0.2: severity = "Severe"
        elif m_rate > 0.05: severity = "Moderate"
        
        stats.append({
            "column": col,
            "missing_count": int(m_count),
            "missing_rate": float(m_rate),
            "severity": severity
        })
    
    # Pattern: Rows with any missing
    any_missing_mask = df[columns].isna().any(axis=1)
    any_missing_n = int(any_missing_mask.sum())
    
    return {
        "total_n": total_n,
        "any_missing_n": any_missing_n,
        "any_missing_rate": any_missing_n / total_n if total_n > 0 else 0,
        "variable_stats": stats
    }

def apply_handling_strategy(
    df: pd.DataFrame, 
    strategy: str, 
    columns: List[str], 
    categorical_columns: List[str]
) -> Dict[str, Any]:
    """
    Applies a missing data handling strategy and returns processed data with rich metadata.
    Strategies: 'complete-case', 'simple-imputation', 'missing-category'
    """
    res = {
        "df_processed": None,
        "strategy_name": strategy,
        "columns": columns,
        "retained_n": 0,
        "dropped_n": 0,
        "imputation_counts": {},
        "warnings": [],
        "notes": ""
    }
    
    work_df = df.copy()
    
    if strategy == "complete-case":
        # Drop any row with NaN in target columns
        before_n = len(work_df)
        work_df = work_df.dropna(subset=columns)
        res["df_processed"] = work_df
        res["retained_n"] = len(work_df)
        res["dropped_n"] = before_n - len(work_df)
        res["notes"] = "Standard complete-case analysis (listwise deletion)."
        
    elif strategy == "simple-imputation":
        impute_log = {}
        for col in columns:
            if col not in work_df.columns: continue
            missing_mask = work_df[col].isna()
            m_count = int(missing_mask.sum())
            if m_count == 0: continue
            
            if col in categorical_columns:
                # Mode
                mode_val = work_df[col].mode()
                fill_val = mode_val[0] if not mode_val.empty else "Unknown"
            else:
                # Median
                fill_val = work_df[col].median()
            
            work_df[col] = work_df[col].fillna(fill_val)
            impute_log[col] = m_count
            
        res["df_processed"] = work_df
        res["retained_n"] = len(work_df)
        res["imputation_counts"] = impute_log
        res["notes"] = "Mean/Mode imputation based on variable type."

    elif strategy == "missing-category":
        impute_log = {}
        for col in columns:
            if col not in work_df.columns: continue
            if col not in categorical_columns:
                res["warnings"].append(f"Ignoring continuous variable '{col}' for missing-category strategy.")
                continue
            
            missing_mask = work_df[col].isna()
            m_count = int(missing_mask.sum())
            if m_count == 0: continue
            
            # For categorical, we cast to string and fill with "Missing"
            work_df[col] = work_df[col].astype(str).replace(['nan', 'None', 'NaN'], 'Missing')
            impute_log[col] = m_count
            
        res["df_processed"] = work_df
        res["retained_n"] = len(work_df)
        res["imputation_counts"] = impute_log
        res["notes"] = "Encoded missing values as a distinct 'Missing' category."
        
    else:
        res["warnings"].append(f"Unknown strategy '{strategy}'. Returning raw data.")
        res["df_processed"] = work_df
        res["retained_n"] = len(work_df)
        
    return res

def run_missing_sensitivity_analysis(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    main_var: str,
    covariates: List[str],
    categorical_covariates: List[str]
) -> Dict[str, Any]:
    """
    Runs Cox models across different handling strategies and captures drift.
    """
    strategies = ["complete-case", "simple-imputation", "missing-category"]
    analysis_results = []
    
    # All columns involved in the model
    all_cols = [main_var] + covariates
    
    for s_name in strategies:
        h_res = apply_handling_strategy(df, s_name, all_cols, categorical_covariates)
        
        # Run Cox on the processed DF
        if h_res["df_processed"] is not None and len(h_res["df_processed"]) > 0:
            cox_res = run_cox(
                h_res["df_processed"],
                duration_col,
                event_col,
                main_var,
                covariates
            )
            
            entry = {
                "strategy": s_name,
                "ok": cox_res.get("ok", False),
                "hr": cox_res.get("hr", np.nan),
                "hr_l": cox_res.get("hr_l", np.nan),
                "hr_u": cox_res.get("hr_u", np.nan),
                "p": cox_res.get("p", np.nan),
                "n": int(cox_res.get("n", 0)),
                "events": int(cox_res.get("events", 0)),
                "handling_meta": h_res
            }
            analysis_results.append(entry)
            
    # Calculate drift metrics relative to complete-case (benchmark)
    cc_entry = next((r for r in analysis_results if r["strategy"] == "complete-case"), None)
    drift_metrics = []
    if cc_entry and cc_entry["ok"]:
        cc_hr = cc_entry["hr"]
        for r in analysis_results:
            if r["strategy"] == "complete-case": continue
            if r["ok"] and not np.isnan(r["hr"]):
                drift = abs(r["hr"] - cc_hr) / cc_hr
                drift_metrics.append({"strategy": r["strategy"], "hr_drift": drift})

    return {
        "strategy_results": analysis_results,
        "drift_metrics": drift_metrics,
        "benchmark_hr": cc_entry["hr"] if cc_entry and cc_entry["ok"] else np.nan
    }

def classify_missingness_robustness(analysis_res: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classifies the susceptibility of findings to missing data handling.
    """
    results = analysis_res["strategy_results"]
    drifts = analysis_res["drift_metrics"]
    
    # 1. Directional Consistency
    hrs = [r["hr"] for r in results if r["ok"] and not np.isnan(r["hr"])]
    direction_consistent = False
    if hrs:
        all_above = all(h > 1 for h in hrs)
        all_below = all(h < 1 for h in hrs)
        direction_consistent = all_above or all_below
        
    # 2. HR Drift
    max_drift = max([d["hr_drift"] for d in drifts]) if drifts else 0
    drift_low = max_drift < 0.2
    
    # 3. Significance Retention
    p_vals = [r["p"] for r in results if r["ok"] and not np.isnan(r["p"])]
    sig_retained = all(p < 0.05 for p in p_vals) if p_vals else False
    
    # 4. Sample Retention (Complete-case benchmark)
    cc_entry = next((r for r in results if r["strategy"] == "complete-case"), None)
    total_n = cc_entry["handling_meta"]["retained_n"] + cc_entry["handling_meta"]["dropped_n"] if cc_entry else 0
    retained_n = cc_entry["n"] if cc_entry else 0
    retention_rate = retained_n / total_n if total_n > 0 else 1.0
    retention_high = retention_rate > 0.8
    
    score = 0
    if direction_consistent: score += 1
    if drift_low: score += 1
    if sig_retained: score += 1
    if retention_high: score += 1
    
    classification = "Fragile"
    if score >= 4: classification = "Robust"
    elif score >= 2: classification = "Moderately sensitive"
    
    return {
        "classification": classification,
        "score": score,
        "details": {
            "direction_consistent": direction_consistent,
            "max_drift": float(max_drift),
            "sig_retained": sig_retained,
            "retention_rate": float(retention_rate)
        }
    }

def summarize_missingness_results(
    stats: Dict[str, Any],
    analysis_res: Dict[str, Any],
    robustness: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generates a high-level summary and interpretation of missing data impacts.
    """
    heavy_vars = [s["column"] for s in stats["variable_stats"] if s["severity"] == "Severe"]
    
    cc_entry = next((r for r in analysis_res["strategy_results"] if r["strategy"] == "complete-case"), None)
    max_drop_n = cc_entry["handling_meta"]["dropped_n"] if cc_entry else 0
    
    classification = robustness["classification"]
    details = robustness["details"]
    
    interpretation = ""
    if classification == "Robust":
        interpretation = "The association remains highly stable across multiple missing data handling strategies, with high sample retention and minimal effect size drift."
    elif classification == "Moderately sensitive":
        interpretation = f"The association is directionally consistent, but shows some sensitivity to handling strategies (Max HR Drift: {details['max_drift']:.1%}). "
        if details["retention_rate"] < 0.8:
            interpretation += f"Significant sample loss ({1.0 - details['retention_rate']:.1%}) observed under complete-case analysis."
    else:
        interpretation = "The finding is fragile regarding missing data treatment. Conclusions may reverse or lose significance depending on the handling strategy or sample exclusion."
    
    return {
        "classification": classification,
        "heavy_variables": heavy_vars,
        "max_sample_drop": max_drop_n,
        "max_hr_drift": details["max_drift"],
        "interpretation": interpretation
    }
