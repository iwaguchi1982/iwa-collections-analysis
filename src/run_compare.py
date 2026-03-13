# src/run_compare.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

@dataclass
class RunSnapshot:
    run_id: str
    label: str
    created_at: str
    analysis_type: str  # "1-gene", "validation", etc.
    
    # Metadata
    dataset: str
    gene: str
    omics: str
    endpoint: str
    cutoff_percentile: Optional[float] = None
    cutoff_value: Optional[float] = None
    covariates: List[str] = field(default_factory=list)
    subgroup: Optional[str] = None
    
    # Results
    hr: Optional[float] = None
    hr_lower: Optional[float] = None
    hr_upper: Optional[float] = None
    p_value: Optional[float] = None
    logrank_p: Optional[float] = None
    n_total: int = 0
    n_events: int = 0
    
    # Absolute survival (optional)
    surv_3y_high: Optional[float] = None
    surv_3y_low: Optional[float] = None
    surv_5y_high: Optional[float] = None
    surv_5y_low: Optional[float] = None
    
    # Issues & Interpretation
    issues: List[Dict[str, Any]] = field(default_factory=list)
    interpretation: str = ""
    
    # Plot references (optional)
    plot_refs: Dict[str, str] = field(default_factory=dict)

def capture_run_snapshot(
    label: str,
    analysis_type: str,
    metadata: Dict[str, Any],
    results: Dict[str, Any],
    issues: List[Any],
    interpretation: str = ""
) -> RunSnapshot:
    """Creates a RunSnapshot from raw analysis components."""
    # Convert issues to dicts safely
    serialized_issues = []
    for iss in issues:
        if hasattr(iss, 'to_dict'):
            serialized_issues.append(iss.to_dict())
        else:
            # Fallback if it's already a dict or something else
            try:
                serialized_issues.append(dict(iss))
            except:
                serialized_issues.append({"title": str(iss), "severity": "info", "kind": "caution"})
            
    return RunSnapshot(
        run_id=datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
        label=label,
        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        analysis_type=analysis_type,
        dataset=metadata.get("dataset", "Unknown"),
        gene=metadata.get("gene", "Unknown"),
        omics=metadata.get("omics", "Expression"),
        endpoint=metadata.get("endpoint", "OS"),
        cutoff_percentile=metadata.get("cutoff_percentile"),
        cutoff_value=metadata.get("cutoff_value"),
        covariates=metadata.get("covariates", []),
        subgroup=metadata.get("subgroup"),
        hr=results.get("hr"),
        hr_lower=results.get("hr_l"),
        hr_upper=results.get("hr_u"),
        p_value=results.get("p"),
        logrank_p=results.get("logrank_p"),
        n_total=results.get("n", 0),
        n_events=results.get("events", 0),
        surv_3y_high=results.get("surv_prob_3y_high"),
        surv_3y_low=results.get("surv_prob_3y_low"),
        surv_5y_high=results.get("surv_prob_5y_high"),
        surv_5y_low=results.get("surv_prob_5y_low"),
        issues=serialized_issues,
        interpretation=interpretation
    )

def compare_snapshots(s1: RunSnapshot, s2: RunSnapshot) -> Dict[str, Any]:
    """Compares two snapshots and computes deltas."""
    diff = {
        "metadata_changed": [],
        "metrics_delta": {},
        "issue_diff": {
            "new": [],
            "resolved": [],
            "persistent": []
        }
    }
    
    # Metadata comparison
    fields_to_check = ["dataset", "gene", "omics", "endpoint", "cutoff_percentile", "covariates", "subgroup"]
    for f in fields_to_check:
        v1 = getattr(s1, f)
        v2 = getattr(s2, f)
        if v1 != v2:
            diff["metadata_changed"].append(f)
            
    # Metrics delta
    for m in ["hr", "p_value", "logrank_p", "n_total", "n_events"]:
        v1 = getattr(s1, m)
        v2 = getattr(s2, m)
        if v1 is not None and v2 is not None:
            diff["metrics_delta"][m] = v2 - v1
                
    # Issue comparison (by title for simplicity)
    s1_titles = {iss.get("title") for iss in s1.issues if iss.get("title")}
    s2_titles = {iss.get("title") for iss in s2.issues if iss.get("title")}
    
    diff["issue_diff"]["new"] = [iss for iss in s2.issues if iss.get("title") not in s1_titles]
    diff["issue_diff"]["resolved"] = [iss for iss in s1.issues if iss.get("title") not in s2_titles]
    diff["issue_diff"]["persistent"] = [iss for iss in s2.issues if iss.get("title") in s1_titles]
    
    return diff

def generate_comparison_summary(s1: RunSnapshot, s2: RunSnapshot, diff: Dict[str, Any]) -> str:
    """Generates a detailed, factual summary of what changed with specific values."""
    parts = []
    
    # Directionality and Effect Strength
    if s1.hr is not None and s2.hr is not None:
        direction_msg = "Effect direction is preserved." if (s1.hr > 1 and s2.hr > 1) or (s1.hr < 1 and s2.hr < 1) else "Effect direction is reversed."
        
        strength_msg = ""
        hr1_strength = abs(s1.hr - 1)
        hr2_strength = abs(s2.hr - 1)
        if hr2_strength < hr1_strength:
            strength_msg = f"HR effect is attenuated (from {s1.hr:.2f} to {s2.hr:.2f})."
        elif hr2_strength > hr1_strength:
            strength_msg = f"HR effect is amplified (from {s1.hr:.2f} to {s2.hr:.2f})."
        else:
            strength_msg = f"HR remains stable ({s1.hr:.2f})."
            
        parts.append(f"{direction_msg} {strength_msg}")
            
    # Metadata context (Cutoff focus)
    if s1.cutoff_percentile is not None and s2.cutoff_percentile is not None:
        if s1.cutoff_percentile != s2.cutoff_percentile:
            parts.append(f"Cutoff shifted from {s1.cutoff_percentile}% to {s2.cutoff_percentile}%.")
    
    # Significance
    if s1.p_value is not None and s2.p_value is not None:
        if s1.p_value < 0.05 and s2.p_value >= 0.05:
            parts.append(f"Statistical support is weakened (lost significance: p={s1.p_value:.3e} → {s2.p_value:.3e}).")
        elif s1.p_value >= 0.05 and s2.p_value < 0.05:
            parts.append(f"Statistical support is strengthened (gained significance: p={s1.p_value:.3e} → {s2.p_value:.3e}).")
        elif s1.p_value < 0.05 and s2.p_value < 0.05:
            parts.append(f"Significance remains stable (both runs p < 0.05).")
        else:
            parts.append(f"Statistical support remains non-significant (p={s1.p_value:.3e} → {s2.p_value:.3e}).")
            
    # Issues
    issue_diff = diff.get("issue_diff", {})
    new_iss = issue_diff.get("new", [])
    if new_iss:
        parts.append(f"{len(new_iss)} new concern(s) introduced.")
        
    res_iss = issue_diff.get("resolved", [])
    if res_iss:
        parts.append(f"{len(res_iss)} previous concern(s) resolved.")
        
    if not parts:
        return "No major changes in effect size, significance, or analytical issues detected."
        
    return " ".join(parts)
