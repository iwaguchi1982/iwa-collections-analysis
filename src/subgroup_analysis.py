# src/subgroup_analysis.py
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from src.survival import run_cox
from lifelines import CoxPHFitter

# Constraints
MIN_SUBGROUP_N = 20
MIN_SUBGROUP_EVENTS = 5

def check_subgroup_eligibility(
    df: pd.DataFrame, 
    subgroup_col: str, 
    main_var: str,
    min_n: int = MIN_SUBGROUP_N, 
    min_events: int = MIN_SUBGROUP_EVENTS
) -> Dict[str, Any]:
    """
    Groups the dataframe by subgroup_col and checks if each group has enough data.
    Returns: {lvl: {"eligible": bool, "n": int, "events": int, "reason": str}}
    """
    if subgroup_col not in df.columns:
        return {"ok": False, "reason": f"Column {subgroup_col} not found."}
    
    # Check for missing labels
    total_len = len(df)
    df_clean = df.dropna(subset=[subgroup_col])
    missing_count = total_len - len(df_clean)
    
    if df_clean.empty:
        return {"ok": False, "reason": "no data after dropping NAs for subgroup"}
    
    levels = df_clean[subgroup_col].unique()
    if len(levels) < 2:
        # We can still check eligibility for the one level, but note it
        pass

    eligibility = {}
    for lvl in levels:
        mask = df_clean[subgroup_col] == lvl
        df_lvl = df_clean[mask]
        n = int(len(df_lvl))
        events = int(df_lvl["is_dead"].sum())
        
        # Check imbalance if main_var exists
        imbalance_reason = ""
        if main_var in df_lvl.columns:
            counts = df_lvl[main_var].value_counts()
            if len(counts) < 2:
                imbalance_reason = "no variation in target group"
            else:
                ratio = counts.min() / counts.sum()
                if ratio < 0.05:
                    imbalance_reason = "one arm too imbalanced (<5%)"
        
        is_eligible = True
        reason = ""
        
        if n < min_n:
            is_eligible = False
            reason = "too few patients"
        elif events < min_events:
            is_eligible = False
            reason = "too few events"
        elif imbalance_reason:
            is_eligible = False
            reason = imbalance_reason
        
        eligibility[str(lvl)] = {
            "eligible": is_eligible,
            "n": n,
            "events": events,
            "reason": reason
        }
        
    return {"ok": True, "eligibility": eligibility, "missing_labels": missing_count}

def run_subgroup_batch_analysis(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    main_var: str,
    subgroup_col: str,
    covariates: List[str] = [],
) -> Dict[str, Any]:
    """
    Runs Cox analysis for each eligible level of a subgroup.
    Returns standardized list of results.
    """
    elig_res = check_subgroup_eligibility(df, subgroup_col, main_var)
    if not elig_res["ok"]:
        return elig_res
    
    standard_results = []
    
    for lvl, info in elig_res["eligibility"].items():
        base_res = {
            "subgroup_variable": subgroup_col,
            "subgroup_level": lvl,
            "N": info["n"],
            "events": info["events"],
            "HR": None,
            "CI_low": None,
            "CI_high": None,
            "p_value": None,
            "status": "Eligible" if info["eligible"] else "Excluded",
            "note": info["reason"]
        }
        
        if not info["eligible"]:
            standard_results.append(base_res)
            continue
            
        # Filter df for this level
        df_lvl = df[df[subgroup_col].astype(str) == lvl].copy()
        
        # Run Cox
        cox_res = run_cox(
            df_lvl,
            duration_col=duration_col,
            event_col=event_col,
            main_var=main_var,
            covariates=covariates
        )
        
        if cox_res["ok"]:
            base_res.update({
                "HR": cox_res.get("hr"),
                "CI_low": cox_res.get("hr_l"),
                "CI_high": cox_res.get("hr_u"),
                "p_value": cox_res.get("p"),
                "status": "Success",
                "note": ""
            })
        else:
            base_res.update({
                "status": "Failed",
                "note": cox_res.get("reason", "Analysis failed")
            })
            
        standard_results.append(base_res)
        
    return {"ok": True, "subgroup_results": standard_results}

def run_interaction_test(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    main_var: str,
    subgroup_col: str,
    covariates: List[str] = []
) -> Dict[str, Any]:
    """
    Calculates interaction p-value between main_var and subgroup_col.
    Returns: {interaction_p: float|None, status: str, warning: str}
    """
    if subgroup_col not in df.columns:
        return {"interaction_p": None, "status": "Failed", "warning": f"Column {subgroup_col} not found."}
        
    # Prepare data
    cols = [duration_col, event_col, main_var, subgroup_col] + covariates
    df_mini = df[cols].dropna().copy()
    
    # Check if subgroup has more than 1 level
    levels = df_mini[subgroup_col].nunique()
    if levels < 2:
        return {"interaction_p": None, "status": "Skipped", "warning": "Subgroup has only 1 level."}
    
    if levels > 6:
        return {"interaction_p": None, "status": "Skipped", "warning": "Too many levels (>6) for stable interaction test."}
        
    # Check if each arm has events
    for lvl in df_mini[subgroup_col].unique():
        ev = df_mini.loc[df_mini[subgroup_col] == lvl, event_col].sum()
        if ev == 0:
            return {"interaction_p": None, "status": "Failed", "warning": f"Level '{lvl}' has zero events."}

    # lifelines requires numerical inputs
    try:
        df_model = df_mini.copy()
        dummies = pd.get_dummies(df_model[subgroup_col], prefix="sub", drop_first=True)
        dummy_cols = dummies.columns.tolist()
        df_model = pd.concat([df_model.drop(columns=[subgroup_col]), dummies], axis=1)
        
        interaction_terms = []
        for dcol in dummy_cols:
            term_name = f"inter_{dcol}"
            df_model[term_name] = df_model[main_var] * df_model[dcol]
            interaction_terms.append(term_name)
            
        cph = CoxPHFitter()
        cph.fit(df_model, duration_col=duration_col, event_col=event_col)
        
        summary = cph.summary
        # Check if any interaction term failed (NaN p-values)
        if summary.loc[interaction_terms, "p"].isna().any():
             return {"interaction_p": None, "status": "Failed", "warning": "Unstable model (HR overflow or single-member group)."}

        inter_p = float(summary.loc[interaction_terms, "p"].min()) if interaction_terms else 1.0
        
        return {
            "interaction_p": inter_p,
            "status": "Success",
            "warning": ""
        }
    except Exception as e:
        return {"interaction_p": None, "status": "Failed", "warning": str(e)}

def classify_subgroup_consistency(
    overall_hr: float,
    subgroup_results: Dict[str, Any],
    interaction_p: Optional[float]
) -> str:
    """
    Classifies the consistency of effects across subgroups.
    """
    valid_hrs = [res["HR"] for res in subgroup_results if res["status"] == "Success" and res["HR"] is not None]
    
    if not valid_hrs:
        return "Insufficient Data"
    
    # Check if all directions match the overall direction
    overall_dir = 1 if overall_hr > 1 else -1
    directions = [1 if hr > 1 else -1 for hr in valid_hrs]
    all_same_dir = all(d == overall_dir for d in directions)
    
    if interaction_p is not None and interaction_p < 0.05:
        return "Heterogeneous (Potential Subgroup-Specific Effect)"
    
    if all_same_dir:
        return "Broadly Consistent"
    else:
        return "Directionally Inconsistent"

def generate_subgroup_interpretation(
    consistency: str,
    interaction_p: Optional[float],
    top_diff_lvl: Optional[str] = None
) -> str:
    """
    Generates a natural language summary.
    """
    if consistency == "Broadly Consistent":
        return "The prognostic effect is maintained across major subgroups. No strong evidence of effect modification."
    elif consistency == "Heterogeneous (Potential Subgroup-Specific Effect)":
        msg = f"Potential effect modification detected (interaction p={interaction_p:.3f}). "
        if top_diff_lvl:
            msg += f"Effect varies significantly in the '{top_diff_lvl}' stratum."
        return msg
    elif consistency == "Directionally Inconsistent":
        return "The effect direction varies between subgroups. Use caution when generalizing outcomes."
    else:
        return "Subgroup analysis was limited by sample size or event counts."
