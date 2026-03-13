# ui/subgroup_view.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Optional
from src.subgroup_analysis import (
    run_subgroup_batch_analysis, 
    run_interaction_test, 
    classify_subgroup_consistency, 
    generate_subgroup_interpretation
)

def render_subgroup_dashboard():
    st.header("🧬 Subgroup Batch Analysis")
    st.markdown("""
    Evaluate if the prognostic effect of your candidate gene is consistent across 
    different clinical subgroups (e.g., Sex, Stage, Age) or if there are 
    significant interactions.
    """)

    if not st.session_state.get("analysis_1gene_ready"):
        st.warning("Please run a 1-Gene Survival Analysis first to define the target gene and cutoff.")
        return

    # Extract context from session state
    merged = st.session_state["analysis_1gene_merged"]
    gene = st.session_state["analysis_1gene_gene"]
    cutoff = st.session_state["analysis_1gene_cutoff"]
    omics_used = st.session_state.get("analysis_1gene_omics", "Expression")
    covariates_actual = st.session_state.get("analysis_1gene_covariates", [])
    overall_hr = st.session_state.get("analysis_1gene_res", {}).get("hr", 1.0)
    
    st.info(f"Target: **{gene}** ({omics_used}) | Overall HR: {overall_hr:.2f}")

    # Subgroup Selection
    # Standard candidates (Priority)
    priority_cols = ["SEX", "AJCC_PATHOLOGIC_TUMOR_STAGE", "AGE_GROUP", "SUBTYPE", "stage_num"]
    
    # Add age group if AGE exists
    if "AGE" in merged.columns and "AGE_GROUP" not in merged.columns:
        median_age = merged["AGE"].median()
        merged["AGE_GROUP"] = merged["AGE"].apply(lambda x: f">{median_age}" if x > median_age else f"<={median_age}")
    
    # Curated selection
    available_cols = [c for c in priority_cols if c in merged.columns]
    
    # Optional: allow low-cardinality columns not in priority
    other_cols = []
    for c in merged.columns:
        if c not in available_cols and c not in ["PATIENT_ID", "OS_MONTHS", "is_dead", "expression", "analysis_group", "group", "is_target"]:
            if merged[c].nunique() > 1 and merged[c].nunique() <= 6:
                other_cols.append(c)
    
    # Display priority first, then others
    all_options = available_cols + sorted(other_cols)
    selected_axis = st.selectbox("Select Subgroup Axis (Recommended Candidates First):", all_options, index=0 if all_options else None)

    if not selected_axis:
        st.error("No valid clinical columns found for subgrouping.")
        return

    if st.button("🚀 Run Subgroup Analysis", use_container_width=True):
        with st.spinner(f"Analyzing {selected_axis}..."):
            # Define target group mapping (must match analysis_1gene.py)
            if omics_used == "Mutation":
                base_group, ref_group = "WT", "Mutant"
            elif omics_used == "CNA":
                base_group, ref_group = "Normal", "Alteration"
            else: # Expression
                base_group, ref_group = "Low", "High"
            
            # Ensure group and is_target exist
            merged["group"] = merged["analysis_group"]
            merged["is_target"] = (merged["group"] == ref_group).astype(int)

            # 1. Batch Model
            batch_res = run_subgroup_batch_analysis(
                merged,
                duration_col="OS_MONTHS",
                event_col="is_dead",
                main_var="is_target",
                subgroup_col=selected_axis,
                covariates=covariates_actual
            )
            
            # 2. Interaction
            inter_res = run_interaction_test(
                merged,
                duration_col="OS_MONTHS",
                event_col="is_dead",
                main_var="is_target",
                subgroup_col=selected_axis,
                covariates=covariates_actual
            )
            
            if not batch_res["ok"]:
                st.error(batch_res.get("reason", "Unknown error in batch analysis."))
                return

            sub_results = batch_res["subgroup_results"]
            inter_p = inter_res.get("interaction_p")
            inter_status = inter_res.get("status", "Failed")
            inter_warning = inter_res.get("warning", "")
            
            # 3. Consistency Classification
            consistency = classify_subgroup_consistency(overall_hr, sub_results, inter_p)
            summary_text = generate_subgroup_interpretation(consistency, inter_p)
            
            # --- Render Dashboard ---
            st.markdown("---")
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.subheader("Interpretation")
                st.metric("Subgroup Consistency", consistency)
                if inter_status == "Success" and inter_p is not None:
                    st.metric("Interaction P-value", f"{inter_p:.4f}", delta="Significant" if inter_p < 0.05 else None, delta_color="inverse")
                else:
                    st.warning(f"Interaction Test: {inter_status}")
                    if inter_warning: st.caption(f"Note: {inter_warning}")
                
                st.write(summary_text)

            with col_b:
                st.subheader("Subgroup Forest Plot")
                fig = plot_subgroup_forest(sub_results, selected_axis, overall_hr)
                st.pyplot(fig)

            # Detailed Table
            st.subheader("Detailed Results per Stratum")
            df_display = pd.DataFrame(sub_results)
            # Pretty formatting for display
            df_display = df_display.rename(columns={
                "subgroup_level": "Level",
                "N": "N",
                "events": "Events",
                "HR": "HR",
                "CI_low": "CI Low",
                "CI_high": "CI High",
                "p_value": "P-value",
                "status": "Status",
                "note": "Exclusion/Warning"
            })
            
            # Display nicely
            st.dataframe(df_display[[
                "Level", "N", "Events", "HR", "P-value", "Status", "Exclusion/Warning"
            ]], use_container_width=True)

def plot_subgroup_forest(sub_results: List[Dict[str, Any]], axis_name: str, overall_hr: float):
    # Prepare data
    labels = []
    hrs = []
    hr_ls = []
    hr_us = []
    ps = []
    ns = []
    
    # Filter only successful ones
    for res in sub_results:
        if res["status"] == "Success" and res["HR"] is not None:
            labels.append(res["subgroup_level"])
            hrs.append(res["HR"])
            hr_ls.append(res["CI_low"])
            hr_us.append(res["CI_high"])
            ps.append(res["p_value"])
            ns.append(res["N"])

    if not labels:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Insufficient data for Forest Plot", ha='center', va='center')
        return fig

    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.6 * len(labels) + 1)))
    
    for i in range(len(labels)):
        ax.plot([hr_ls[i], hr_us[i]], [y[i], y[i]], color='tab:blue', lw=1.5)
        ax.plot(hrs[i], y[i], marker='s', color='tab:blue', markersize=8)
        # Add N and events at the end? 
        ax.text(ax.get_xlim()[1] * 1.05 if ax.get_xlim()[1] > 0 else 2, y[i], f"N={ns[i]} p={ps[i]:.2g}", va='center')

    ax.axvline(1.0, color='red', linestyle='--', alpha=0.6)
    ax.axvline(overall_hr, color='gray', linestyle=':', alpha=0.4, label="Overall HR")
    
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Hazard Ratio (HR) with 95% CI")
    ax.set_title(f"Subgroup Analysis: {axis_name}")
    
    plt.tight_layout()
    return fig
