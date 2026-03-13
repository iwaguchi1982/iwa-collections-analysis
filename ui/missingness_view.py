# ui/missingness_view.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from src.missingness import (
    get_missingness_stats,
    run_missing_sensitivity_analysis,
    classify_missingness_robustness,
    summarize_missingness_results
)
from typing import Dict, Any, List

def render_missingness_dashboard(
    *,
    df_clin: pd.DataFrame,
    df_exp: pd.DataFrame,
    gene: str,
    get_gene_exp: Any,
    add_covariate_columns: Any,
    deps: Dict[str, Any]
) -> None:
    st.title(f"🧩 Missing Data Treatment Framework: {gene}")
    st.markdown("---")
    
    # 2-Gene Disclaimer if applicable (this logic can be refined if gene contains ' + ')
    if " + " in gene:
        st.warning("⚠️ **Preview Mode**: 2-gene analysis missingness support is limited in this version.")

    if df_clin.empty or df_exp.empty:
        st.warning("Please upload Clinical and Expression data first.")
        return

    exp_data = get_gene_exp(df_exp, gene)
    if exp_data is None:
        st.error(f"Gene '{gene}' not found.")
        return

    # Data Preparation
    merged = pd.merge(df_clin, exp_data, on="PATIENT_ID")
    merged = add_covariate_columns(merged)
    
    # Identify clinical columns for analysis
    # We focus on AGE, is_male, and stage_num as primary clinical covariates
    target_covs = [c for c in ["AGE", "is_male", "stage_num"] if c in merged.columns]
    categorical_covs = [c for c in ["is_male", "stage_num"] if c in target_covs]
    
    # 1. Missingness Overview
    st.subheader("📊 Missingness Overview")
    stats = get_missingness_stats(merged, target_covs)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Patients (N)", stats["total_n"])
    col2.metric("Patients with Any Missing", stats["any_missing_n"])
    col3.metric("Overall Missing Rate", f"{stats['any_missing_rate']:.1%}")
    
    # Heatmap of missing patterns
    if stats["any_missing_n"] > 0:
        missing_matrix = merged[target_covs].isna().astype(int)
        fig_heat = px.imshow(
            missing_matrix.T,
            labels=dict(x="Patients", y="Variables", color="Is Missing"),
            color_continuous_scale=[[0, 'lightgrey'], [1, 'firebrick']],
            title="Missing Data Pattern (Red = Missing)"
        )
        fig_heat.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_heat, use_container_width=True)
        
        # Variable Summary Table
        st.dataframe(pd.DataFrame(stats["variable_stats"]).style.format({"missing_rate": "{:.1%}"}))
    else:
        st.success("No missing data detected in primary clinical covariates.")

    st.markdown("---")

    # Action Button
    if st.button("Run Missing Data Analysis", key="btn_run_missingness"):
        with st.spinner("Comparing handling strategies..."):
            
            # 2. Handling Comparison
            analysis_res = run_missing_sensitivity_analysis(
                merged, "OS_MONTHS", "is_dead", "expression", 
                target_covs, categorical_covs
            )
            
            # 3. Robustness Summary
            robustness = classify_missingness_robustness(analysis_res)
            summary = summarize_missingness_results(stats, analysis_res, robustness)
            
            cls = robustness["classification"]
            color = {"Robust": "green", "Moderately sensitive": "orange", "Fragile": "red"}.get(cls, "gray")
            st.markdown(f"### Robustness Summary: :{color}[{cls}]")
            st.info(summary["interpretation"])
            
            # Evidence
            details = robustness["details"]
            st.markdown("**Evidence Checklist:**")
            col_ev1, col_ev2 = st.columns(2)
            col_ev1.markdown(f"- {'✅' if details['direction_consistent'] else '❌'} Directional Consistency")
            col_ev1.markdown(f"- {'✅' if details['sig_retained'] else '❌'} Significance Retention")
            col_ev2.markdown(f"- {'✅' if details['max_drift'] < 0.2 else '⚠️'} Low HR Drift ({details['max_drift']:.1%})")
            col_ev2.markdown(f"- {'✅' if details['retention_rate'] > 0.8 else '⚠️'} High Sample Retention ({details['retention_rate']:.1%})")

            st.markdown("---")
            
            # Handling Comparison Panel
            st.subheader("⚖️ Handling Strategy Comparison")
            st.caption("Side-by-side evaluation of Cox model results.")
            
            results_df = pd.DataFrame(analysis_res["strategy_results"])
            if not results_df.empty:
                # Forest Plot
                fig_forest = go.Figure()
                fig_forest.add_trace(go.Scatter(
                    x=results_df["hr"], y=results_df["strategy"], mode='markers',
                    error_x=dict(type='data', symmetric=False, array=results_df["hr_u"] - results_df["hr"], arrayminus=results_df["hr"] - results_df["hr_l"]),
                    marker=dict(size=10, color='royalblue')
                ))
                fig_forest.add_vline(x=1, line_dash="dash", line_color="gray")
                fig_forest.update_layout(title="Effect Size Comparison", xaxis_title="HR (95% CI)", yaxis_autorange="reversed")
                st.plotly_chart(fig_forest, use_container_width=True)
                
                # Detailed Table
                disp_cols = ["strategy", "hr", "p", "n", "events"]
                st.dataframe(results_df[disp_cols].style.format({"hr": "{:.2f}", "p": "{:.2e}"}))
            
            st.markdown("---")
            
            # Sample Retention Panel
            st.subheader("📉 Sample Retention Analysis")
            st.caption("Visualizing patient loss across strategies.")
            
            retention_data = []
            for r in analysis_res["strategy_results"]:
                meta = r["handling_meta"]
                retention_data.append({
                    "Strategy": r["strategy"],
                    "Status": "Retained",
                    "Count": meta["retained_n"]
                })
                if meta["dropped_n"] > 0:
                    retention_data.append({
                        "Strategy": r["strategy"],
                        "Status": "Dropped",
                        "Count": meta["dropped_n"]
                    })
            
            df_ret = pd.DataFrame(retention_data)
            fig_ret = px.bar(
                df_ret, x="Strategy", y="Count", color="Status",
                color_discrete_map={"Retained": "green", "Dropped": "red"},
                title="Patients Retained vs Dropped"
            )
            st.plotly_chart(fig_ret, use_container_width=True)
            
            # Imputation Metrics
            with st.expander("View Imputation/Handling Details"):
                for r in analysis_res["strategy_results"]:
                    st.markdown(f"**{r['strategy']}**")
                    st.write(f"Notes: {r['handling_meta']['notes']}")
                    if r['handling_meta']['imputation_counts']:
                        st.write("Imputation counts:", r['handling_meta']['imputation_counts'])
                    if r['handling_meta']['warnings']:
                        for w in r['handling_meta']['warnings']:
                            st.warning(w)

    else:
        st.info("Click **Run Missing Data Analysis** to evaluate the impact of missing values.")
