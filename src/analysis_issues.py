# src/analysis_issues.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import pandas as pd

@dataclass
class Issue:
    category: str  # data sufficiency, model assumption, missingness, sensitivity, cohort shift, implementation limitation
    severity: str  # info, warning, critical
    kind: str      # error, limitation, caution, recommendation-trigger
    title: str
    detail: str
    evidence: str = ""
    recommendation: str = ""
    source_module: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "severity": self.severity,
            "kind": self.kind,
            "title": self.title,
            "detail": self.detail,
            "evidence": self.evidence,
            "recommendation": self.recommendation,
            "source_module": self.source_module
        }

def summarize_analysis_issues(issues: List[Issue]) -> Dict[str, Any]:
    """
    Summarizes a list of issues to determine overall status and top drivers.
    """
    if not issues:
        return {
            "overall_status": "OK",
            "max_severity": "info",
            "issue_count": 0,
            "top_drivers": [],
            "summary_text": "No notable analytical issues detected."
        }
    
    # Sort: Critical > Warning > Info
    severity_map = {"critical": 3, "warning": 2, "info": 1}
    sorted_issues = sorted(issues, key=lambda x: severity_map.get(x.severity, 0), reverse=True)
    
    max_sev = sorted_issues[0].severity
    status = "Critical" if max_sev == "critical" else "Warning" if max_sev == "warning" else "OK"
    
    top_drivers = list(set([iss.category for iss in sorted_issues[:3]]))
    
    return {
        "overall_status": status,
        "max_severity": max_sev,
        "issue_count": len(issues),
        "top_drivers": top_drivers,
        "summary_text": f"Found {len(issues)} issues. Primary concerns: {', '.join(top_drivers)}."
    }

def generate_interpretation_text(issues: List[Issue]) -> str:
    """
    Generates a natural language summary of the issues for reports.
    """
    if not issues:
        return "The analysis results are robust with no major issues detected."
    
    criticals = [i for i in issues if i.severity == "critical"]
    warnings = [i for i in issues if i.severity == "warning"]
    
    parts = []
    if criticals:
        parts.append(f"CRITICAL: {len(criticals)} major issues identified that may invalidate the results.")
    if warnings:
        parts.append(f"CAUTION: {len(warnings)} potential instabilities or biases detected.")
        
    return " ".join(parts)
