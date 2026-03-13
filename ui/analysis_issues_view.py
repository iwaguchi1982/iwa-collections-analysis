# ui/analysis_issues_view.py
import streamlit as st
from typing import List, Dict, Any
from src.analysis_issues import Issue, summarize_analysis_issues

def render_issue_panel(issues: List[Issue]):
    """
    Renders a structured panel for analytical issues and limitations.
    """
    if not issues:
        return

    summary = summarize_analysis_issues(issues)
    
    # 1. Header with overall status
    st.markdown("---")
    status_icon = "⚪"
    if summary["overall_status"] == "Critical":
        status_icon = "🔴"
        st.error(f"### {status_icon} Analytical Alert: {summary['overall_status']}")
    elif summary["overall_status"] == "Warning":
        status_icon = "🟡"
        st.warning(f"### {status_icon} Analytical Alert: {summary['overall_status']}")
    else:
        status_icon = "🔵"
        st.info(f"### {status_icon} Analytical Notes")

    st.markdown(f"*{summary['summary_text']}*")
    
    # 2. Categorized List
    # We group by Kind (Error/Limitation/Caution/Recommendation)
    kinds = {
        "error": "❌ Errors (Fatal Issues)",
        "limitation": "⛓️ System Limitations",
        "caution": "⚠️ Statistical Cautions",
        "recommendation-trigger": "💡 Recommended Actions"
    }
    
    for kind_key, kind_label in kinds.items():
        kind_issues = [i for i in issues if i.kind == kind_key]
        if not kind_issues:
            continue
            
        with st.expander(f"{kind_label} ({len(kind_issues)})", expanded=(kind_key == "error" or kind_key == "caution")):
            for iss in kind_issues:
                sev_color = "red" if iss.severity == "critical" else "orange" if iss.severity == "warning" else "blue"
                
                st.markdown(f"""
                **{iss.title}**  
                <span style='color:{sev_color}; font-weight:bold;'>[{iss.severity.upper()}]</span> {iss.detail}
                """, unsafe_allow_html=True)
                
                if iss.evidence:
                    st.caption(f"Evidence: `{iss.evidence}`")
                
                if iss.recommendation:
                    st.success(f"**Action**: {iss.recommendation}")
                
                st.markdown("<div style='margin-bottom:10px;'></div>", unsafe_allow_html=True)

def render_issue_sidebar_badges(issues: List[Issue]):
    """
    Renders a small badge in the sidebar or top if issues exist.
    """
    if not issues:
        return
        
    summary = summarize_analysis_issues(issues)
    if summary["overall_status"] == "Critical":
        st.sidebar.error(f"🚨 {len(issues)} Issues detected")
    elif summary["overall_status"] == "Warning":
        st.sidebar.warning(f"⚠️ {len(issues)} Issues detected")
