# ui/sensitivity_view.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.sensitivity import (
    run_cutoff_scan,
    run_covariate_sensitivity,
    run_method_comparison,
    classify_robustness,
    summarize_sensitivity_results
)
from typing import Dict, Any, List

def render_sensitivity_dashboard(
    *,
    df_clin: pd.DataFrame,
    df_exp: pd.DataFrame,
    gene: str,
    get_gene_exp: Any,
    add_covariate_columns: Any,
    deps: Dict[str, Any]
) -> None:
    st.title(f"🔍 Sensitivity & Robustness: {gene}")
    st.markdown("---")
    
    # Initial Guards
    if df_clin.empty or df_exp.empty:
        st.warning("Please upload Clinical and Expression data first.")
        return

    exp_data = get_gene_exp(df_exp, gene)
    if exp_data is None:
        st.error(f"Gene '{gene}' not found in expression data.")
        return

    # Data Preparation
    merged = pd.merge(df_clin, exp_data, on="PATIENT_ID")
    merged = add_covariate_columns(merged)
    
    # Check OS columns
    if "OS_MONTHS" not in merged.columns or "is_dead" not in merged.columns:
        st.error("Survival columns (OS_MONTHS, is_dead) missing after normalization.")
        return

    # Sidebar Controls for Sensitivity
    st.sidebar.subheader("Sensitivity Settings")
    scan_step = st.sidebar.slider("Cutoff Scan Step (%)", 1, 10, 5)
    
    if st.sidebar.button("Run Global Sensitivity Analysis", key="btn_run_sensitivity"):
        with st.spinner("Analyzing robustness across analytic choices..."):
            
            # 1. Cutoff Scan
            cutoff_res = run_cutoff_scan(
                merged, "OS_MONTHS", "is_dead", "expression", 
                covariates=[], # unadjusted scan for simplicity
                min_pct=20, max_pct=80, step=scan_step
            )
            
            # 2. Covariate Sensitivity (Nested sets)
            sets = {
                "Unadjusted": [],
                "Basic (Age, Sex)": [c for c in ["AGE", "is_male"] if c in merged.columns],
                "Full (Age, Sex, Stage)": [c for c in ["AGE", "is_male", "stage_num"] if c in merged.columns]
            }
            # We use 50% as the reference cutoff for covariate sensitivity
            mid_val = merged["expression"].quantile(0.5)
            merged["ref_group"] = (merged["expression"] >= mid_val).astype(int)
            cov_res = run_covariate_sensitivity(merged, "OS_MONTHS", "is_dead", "ref_group", sets)
                      # 3. Method Comparison
            # RMST tau at median or max
            tau = merged["OS_MONTHS"].median()
            # If deps doesn't have landmark_time, use a reasonable default (e.g. 1/3 of max or median)
            lt = deps.get("landmark_time", merged["OS_MONTHS"].median())
            method_res = run_method_comparison(merged, "OS_MONTHS", "is_dead", "ref_group", tau=tau, landmark_time=lt)
            
            # 4. Classification
            classification = classify_robustness(cutoff_res.get("results", []), cov_res.get("models", {}), method_res)
            summaries = summarize_sensitivity_results(cutoff_res.get("results", []), cov_res.get("models", {}), method_res, classification)
            
            # --- Rendering ---
            
            # Overview Panel
            cls = classification["classification"]
            color = {"Stable": "green", "Moderately sensitive": "orange", "Fragile": "red"}.get(cls, "gray")
            st.markdown(f"### Robustness Summary: :{color}[{cls}]")
            st.info(summaries["overall_interpretation"])
            
            # Reasons List
            with st.expander("🔍 Classification Evidence", expanded=True):
                for reason in classification["reasons"]:
                    if any(x in reason.lower() for x in ["stable", "low", "consistent"]):
                        st.markdown(f"- ✅ {reason}")
                    else:
                        st.markdown(f"- ⚠️ {reason}")

            col_sum1, col_sum2, col_sum3 = st.columns(3)
            
            def get_color_tag(val: str) -> str:
                if val == "High": return ":green[High]"
                if val == "Moderate": return ":orange[Moderate]"
                if val == "Low": return ":red[Low]"
                return val

            with col_sum1:
                st.markdown(f"**Cutoff Stability**\n\n{get_color_tag(classification['details']['cutoff_stability'])}")
            with col_sum2:
                # Rename Model -> Adjustment
                st.markdown(f"**Adjustment Robustness**\n\n{get_color_tag(classification['details']['covariate_stability'])}")
            with col_sum3:
                st.markdown(f"**Method Consistency**\n\n{get_color_tag(classification['details']['method_consistency'])}")

            st.markdown("---")
            
            # Cutoff Plot
            st.subheader("📊 Cutoff Percentile Scan")
            st.caption("How results change by shifting the high/low threshold.")
            
            df_scan = pd.DataFrame(cutoff_res["results"])
            if not df_scan.empty:
                fig_scan = go.Figure()
                # HR line
                fig_scan.add_trace(go.Scatter(
                    x=df_scan["pct"], y=df_scan["hr"], name="Hazard Ratio",
                    line=dict(color='blue', width=2)
                ))
                # P-value line (secondary axis)
                fig_scan.add_trace(go.Scatter(
                    x=df_scan["pct"], y=df_scan["p"], name="P-value",
                    line=dict(color='red', dash='dash'), yaxis="y2"
                ))
                
                fig_scan.update_layout(
                    xaxis_title="Cutoff Percentile (%)",
                    yaxis_title="Hazard Ratio (HR)",
                    yaxis2=dict(title="P-value", overlaying="y", side="right", range=[0, 1]),
                    legend=dict(x=0.01, y=0.99),
                    hovermode="x unified"
                )
                # Significant region
                sig_pts = df_scan[df_scan["p"] < 0.05]
                if not sig_pts.empty:
                    fig_scan.add_vrect(
                        x0=sig_pts["pct"].min(), x1=sig_pts["pct"].max(),
                        fillcolor="green", opacity=0.1, layer="below", line_width=0,
                        annotation_text="Significant Region"
                    )
                st.plotly_chart(fig_scan, use_container_width=True)
                
                # Sample Size N Table
                with st.expander("View Scan Data (N and Events)"):
                    st.dataframe(df_scan[["pct", "cutoff_val", "hr", "p", "n", "events"]].style.format({"hr": "{:.2f}", "p": "{:.2e}"}))
            
            st.markdown("---")
            
            # Covariate Forest Plot
            st.subheader("🌲 Adjustment Robustness")
            st.caption("Do results hold after adjusting for clinical variables?")
            
            cov_models = cov_res["models"]
            forest_df = []
            for label, m in cov_models.items():
                if m["ok"]:
                    forest_df.append({
                        "Model": label,
                        "HR": m["hr"],
                        "L": m["hr_l"],
                        "U": m["hr_u"],
                        "P": m["p"],
                        "N": m["n"]
                    })
            
            if forest_df:
                df_f = pd.DataFrame(forest_df)
                fig_f = go.Figure()
                fig_f.add_trace(go.Scatter(
                    x=df_f["HR"], y=df_f["Model"], mode='markers',
                    error_x=dict(type='data', symmetric=False, array=df_f["U"] - df_f["HR"], arrayminus=df_f["HR"] - df_f["L"]),
                    marker=dict(size=10, color='royalblue')
                ))
                fig_f.add_vline(x=1, line_dash="dash", line_color="gray")
                fig_f.update_layout(title="Effect Size Comparison", xaxis_title="HR (95% CI)", yaxis_autorange="reversed")
                st.plotly_chart(fig_f, use_container_width=True)
                st.write(summaries["covariate_summary"])
            
            st.markdown("---")
            
            # Method Comparison
            st.subheader("⚖️ Analytic Method Consistency")
            st.caption("Cross-check results using three distinct analytic frameworks.")
            
            c_res = method_res["cox"]
            r_res = method_res["rmst"]
            l_res = method_res["landmark"]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**1. Cox (Overall)**")
                if c_res["ok"]:
                    st.write(f"HR: {c_res['hr']:.2f} (p={c_res['p']:.2e})")
                else:
                    st.write("Failed")
            with col2:
                st.markdown("**2. RMST (Model-free)**")
                if r_res["ok"]:
                    vals = r_res["rmst_values"]
                    st.write(f"Tau: {r_res['tau']:.1f} months")
                    st.write(vals)
                else:
                    st.write("Failed")
            with col3:
                st.markdown("**3. Landmark (Late)**")
                if l_res["ok"]:
                    st.write(f"HR: {l_res['hr']:.2f} (p={l_res['p']:.2e})")
                    st.caption(f"Time > {lt:.1f}m")
                else:
                    st.write("Not run or failed")
            
            st.success(summaries["method_summary"])
            
    else:
        st.info("Click **Run Global Sensitivity Analysis** to evaluate the robustness of this candidate.")
