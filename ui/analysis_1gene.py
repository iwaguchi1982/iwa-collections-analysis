from __future__ import annotations
from typing import Any, Dict, List
import pandas as pd
import numpy as np
import streamlit as st
from src.analysis_issues import Issue, generate_interpretation_text
from ui.analysis_issues_view import render_issue_panel
from src.run_compare import capture_run_snapshot


def render_1gene_analysis(
    *,
    dataset_name: str,
    df_clin: pd.DataFrame,
    df_exp: pd.DataFrame,
    df_mut: pd.DataFrame = None,
    df_cna: pd.DataFrame = None,
    gene: str,
    threshold_pct: int,
    use_auto_cutoff: bool = False,
    covariates: List[str],
    get_gene_exp,
    add_covariate_columns,
    deps: Dict[str, Any],
    covariates_actual: List[str] = [],
    rmst_time: float = 0.0,
    calculate_rmst: Any = None,
) -> None:
    plot_km = deps["plot_km"]
    run_cox = deps["run_cox"]
    plot_forest_hr = deps["plot_forest_hr"]
    plot_compare_main_effect = deps["plot_compare_main_effect"]
    metric_p_display = deps["metric_p_display"]
    logrank_test = deps["logrank_test"]
    find_optimal_1gene_cutoff = deps.get("find_optimal_1gene_cutoff")
    run_ph_test = deps.get("run_ph_test")
    plot_schoenfeld_residuals = deps.get("plot_schoenfeld_residuals")

    # （あなたの既存変数 forest_mode を使うなら depsで渡すか、ここで固定）
    forest_mode = deps.get("forest_mode", "Simple (Adjusted only)")

    threshold = float(threshold_pct) / 100.0

    st.sidebar.markdown("---")
    st.sidebar.subheader("🌟 Omics Selection")
    omics_type = st.sidebar.radio(
        "Survival Analysis Feature",
        ["Expression", "Mutation", "CNA"],
        index=0,
        help="Choose which omics data to use for KM and Cox."
    )

    run_now = st.sidebar.button("Run Analysis", key="btn_run_1gene") or st.session_state.pop("auto_run_analysis", False)
    if run_now:
        exp_data = get_gene_exp(df_exp, gene)
        if exp_data is None:
            st.error(f"Gene not found: {gene}")
            st.stop()

        merged0 = pd.merge(df_clin, exp_data, on="PATIENT_ID")
        merged0 = add_covariate_columns(merged0)

        best_p_val_display = None
        if use_auto_cutoff and find_optimal_1gene_cutoff and omics_type == "Expression":
            with st.spinner("Finding optimal cutoff..."):
                opt_res = find_optimal_1gene_cutoff(
                    merged0,
                    duration_col="OS_MONTHS",
                    event_col="is_dead",
                    expr_col="expression"
                )
                if opt_res["ok"]:
                    threshold = float(opt_res["optimal_pct"]) / 100.0
                    cutoff = opt_res["optimal_cutoff"]
                    best_p_val_display = opt_res["min_p"]
                else:
                    st.warning("Auto-Cutoff failed. Using default.")
                    cutoff = merged0["expression"].quantile(threshold)
        else:
            cutoff = merged0["expression"].quantile(threshold)
        
        if df_mut is not None:
            gene_mut = df_mut[df_mut["Hugo_Symbol"] == gene] if "Hugo_Symbol" in df_mut.columns else pd.DataFrame()
            if not gene_mut.empty:
                mut_samples = gene_mut["Tumor_Sample_Barcode"].astype(str).str[:12].unique() if "Tumor_Sample_Barcode" in gene_mut.columns else pd.Series(dtype=str)
                merged0["mut_status"] = merged0["PATIENT_ID"].isin(mut_samples).map({True: "Mutant", False: "WT"})
            else:
                merged0["mut_status"] = "WT"
        else:
            merged0["mut_status"] = "N/A"

        if df_cna is not None:
             cna_exp = get_gene_exp(df_cna, gene)
             if cna_exp is not None:
                 cna_exp = cna_exp.rename(columns={"expression": "cna_value"})
                 merged0 = pd.merge(merged0, cna_exp[["PATIENT_ID", "cna_value"]], on="PATIENT_ID", how="left")
                 def map_cna(v):
                     if pd.isna(v): return "Unknown"
                     if v >= 1: return "Amplification"
                     if v <= -1: return "Deletion"
                     return "Diploid"
                 merged0["cna_status"] = merged0["cna_value"].apply(map_cna)
             else:
                 merged0["cna_status"] = "N/A"
        else:
            merged0["cna_status"] = "N/A"

        if omics_type == "Expression":
            merged0["analysis_group"] = (merged0["expression"] >= cutoff).map({True: "High", False: "Low"})
        elif omics_type == "Mutation":
            merged0["analysis_group"] = merged0["mut_status"]
            if "N/A" in merged0["mut_status"].values:
                st.warning("Mutation data not available. Try another option.")
                st.stop()
        elif omics_type == "CNA":
             merged0["analysis_group"] = merged0["cna_status"]
             if "N/A" in merged0["cna_status"].values:
                 st.warning("CNA data not available for this gene.")
                 st.stop()

        st.session_state["analysis_1gene_ready"] = True
        st.session_state["analysis_1gene_gene"] = gene
        st.session_state["analysis_1gene_cutoff"] = float(cutoff)
        st.session_state["analysis_1gene_threshold"] = float(threshold)
        st.session_state["analysis_1gene_omics"] = omics_type
        if best_p_val_display is not None:
             st.session_state["analysis_1gene_auto_p"] = best_p_val_display
        else:
             st.session_state.pop("analysis_1gene_auto_p", None)
        st.session_state["analysis_1gene_merged"] = merged0.copy()
        st.session_state["analysis_1gene_covariates"] = covariates_actual
        st.session_state["analysis_1gene_res"] = None # Reset results until run

    # --- render cached 1-gene analysis ---
    if st.session_state.get("analysis_1gene_ready") and isinstance(st.session_state.get("analysis_1gene_merged"), pd.DataFrame):
        merged = st.session_state["analysis_1gene_merged"].copy()
        gene = st.session_state.get("analysis_1gene_gene", gene)
        cutoff = float(st.session_state.get("analysis_1gene_cutoff", merged["expression"].median()))
        threshold = float(st.session_state.get("analysis_1gene_threshold", threshold))
        merged["group"] = merged["analysis_group"]

        omics_used = st.session_state.get("analysis_1gene_omics", "Expression")

        st.header(f"🧪 1-Gene Survival ({omics_used}): {gene}")
        st.caption(f"Dataset: {dataset_name}")
        if omics_used == "Expression":
            cut_msg = f"Cutoff: {threshold*100:.0f}th percentile (expression >= {cutoff:.4g} => High)"
            auto_p = st.session_state.get("analysis_1gene_auto_p")
            if auto_p is not None:
                st.success(f"✨ Auto-Cutoff Applied: Optimal cut at {threshold*100:.0f}th percentile (p-value={auto_p:.4e})")
            st.caption(cut_msg)

        fig_km = plot_km(merged, group_col="group", title=f"Kaplan–Meier ({omics_used})")
        st.pyplot(fig_km)

        with st.expander("Number At Risk", expanded=False):
            from src.survival import get_at_risk_df
            df_risk = get_at_risk_df(merged, "group")
            st.dataframe(df_risk, use_container_width=True, hide_index=True)

        if omics_used == "Expression":
            ref_group = "High"
            base_group = "Low"
        elif omics_used == "Mutation":
            ref_group = "Mutant"
            base_group = "WT"
        else: # CNA
            ref_group = "Amplification"
            base_group = "Diploid"

        # Safe logrank for main 2 groups if possible
        has_ref = (merged["group"] == ref_group).any()
        has_base = (merged["group"] == base_group).any()
        p_lr = 1.0
        if has_ref and has_base:
            lr = logrank_test(
                merged.loc[merged["group"] == ref_group, "OS_MONTHS"],
                merged.loc[merged["group"] == base_group, "OS_MONTHS"],
                merged.loc[merged["group"] == ref_group, "is_dead"],
                merged.loc[merged["group"] == base_group, "is_dead"],
            )
            p_lr = float(lr.p_value)

        merged["is_target"] = (merged["group"] == ref_group).astype(int)
        cox_res = run_cox(
            merged,
            duration_col="OS_MONTHS",
            event_col="is_dead",
            main_var="is_target",
            covariates=covariates_actual,
        )
        st.session_state["analysis_1gene_res"] = cox_res

        st.markdown("---")
        st.subheader("Statistical Significance & Risk Analysis")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Log-rank p-value", metric_p_display(p_lr))
            st.caption(f"2-group comparison ({ref_group} vs {base_group})")
            st.success("✅ Significant" if p_lr < 0.05 else "⚠️ Not Significant")

        with col2:
            if cox_res["ok"]:
                st.metric(f"Adjusted HR ({ref_group} vs {base_group})", f"{cox_res['hr']:.2f}")
                st.caption(f"95% CI: [{cox_res['hr_l']:.2f} – {cox_res['hr_u']:.2f}]")
                st.caption(f"Cox N={cox_res['n']}, events={cox_res['events']}")
            else:
                st.metric(f"Adjusted HR ({ref_group} vs {base_group})", "N/A")
                st.caption(cox_res["reason"])

        with col3:
            st.metric("Adjusted p-value", f"{cox_res['p']:.4e}" if cox_res["ok"] else "N/A")
            st.caption("Cox model (optional covariate adjustment)")
            if cox_res["ok"] and cox_res["p"] < 0.05:
                st.success("✅ Independent Prognostic Factor (adjusted)")
            elif cox_res["ok"]:
                st.warning("⚠️ Not independent (adjusted)")


        # --- [SYS-26-0020] Analysis Issues & Limitations ---
        all_issues = []
        if cox_res.get("issues"):
            all_issues.extend(cox_res["issues"])
            
        # Add other logic-based issues if needed (e.g. log-rank significance or subgroup size)
        if p_lr < 0.05 and cox_res["ok"] and cox_res["p"] >= 0.05:
            all_issues.append(Issue(
                category="sensitivity",
                severity="warning",
                kind="caution",
                title="Log-rank vs Adjusted Cox discrepancy",
                detail="Association lost significance after covariate adjustment.",
                evidence=f"logrank_p={p_lr:.3f}, adjusted_p={cox_res['p']:.3f}",
                recommendation="Review the impact of individual covariates in the forest plot.",
                source_module="analysis_1gene"
            ))

        render_issue_panel(all_issues)

        # --- [SYS-26-0014] Run Comparison Snapshot ---
        if "saved_runs" not in st.session_state:
            st.session_state.saved_runs = {}

        save_col1, save_col2 = st.columns([2, 1])
        with save_col1:
            run_label = st.text_input("Run Label (e.g. 'Discovery / 50th / Full')", value=f"{dataset_name} / {omics_used} / {gene}", key="run_label_input")
        with save_col2:
            st.write("##") # padding
            if st.button("💾 Save Run for Comparison", use_container_width=True):
                meta = {
                    "dataset": dataset_name,
                    "gene": gene,
                    "omics": omics_used,
                    "endpoint": "OS", # Hardcoded for now
                    "cutoff_percentile": (threshold * 100) if omics_used == "Expression" else None,
                    "covariates": covariates_actual,
                    "subgroup": None # Future expansion
                }
                # Add absolute survival probs if available
                res_to_save = cox_res.copy()
                res_to_save["logrank_p"] = p_lr
                
                snapshot = capture_run_snapshot(
                    label=run_label,
                    analysis_type="1-gene",
                    metadata=meta,
                    results=res_to_save,
                    issues=all_issues,
                    interpretation=generate_interpretation_text(all_issues)
                )
                st.session_state.saved_runs[snapshot.run_id] = snapshot
                st.success(f"Run saved! Total saved runs: {len(st.session_state.saved_runs)}")

        if cox_res["ok"]:
            if forest_mode == "Simple (Adjusted only)":
                fig_forest = plot_forest_hr(cox_res["model"], highlight="is_target", title="Adjusted Cox Forest (HR scale)")
                st.pyplot(fig_forest)
            else:
                unadj = run_cox(
                    merged,
                    duration_col="OS_MONTHS",
                    event_col="is_dead",
                    main_var="is_target",
                    covariates=[],
                )
                fig_cmp = plot_compare_main_effect(unadj, cox_res, main_label=f"{ref_group} vs {base_group}")
                st.pyplot(fig_cmp)

            st.dataframe(merged["group"].value_counts().to_frame().T, hide_index=True)

            # PH Diagnostics
            if run_ph_test and plot_schoenfeld_residuals and cox_res["model"]:
                # Needs the dataframe that was actually fitted: merged with NAs dropped from covariates
                df_fit = cox_res["data"]

                ph_res = run_ph_test(cox_res["model"], df_fit)
                if ph_res["ok"]:
                    target_p = ph_res["p_values"].get("is_target", 1.0)

                    with st.expander("🩺 Diagnostics: Proportional Hazards (PH) Assumption"):
                        if target_p < 0.05:
                            st.error(f"⚠️ **PH Assumption Violated (p={target_p:.3e})**: The effect of the target gene changes significantly over time. Standard Cox Average HR is mathematically invalid here. Please consult the 'Time-Dependent Analysis' module below.")
                        else:
                            st.success(f"✅ **PH Assumption Met (p={target_p:.3f})**: The hazard ratio of the target gene is relatively constant over time.")

                        fig_res = plot_schoenfeld_residuals(cox_res["model"], df_fit, "is_target", title="Schoenfeld Residuals: is_target")
                        if fig_res:
                            st.pyplot(fig_res)

                        st.caption("A violation of the PH assumption requires advanced time-dependent modeling (e.g. Time-Varying Cox) to properly map when the effect vanishes or reverses.")
                else:
                    st.error(f"PH test failed: {ph_res.get('reason')}")

        st.markdown("---")
        st.subheader("⏱️ Time-Dependent Analysis & Report Bundle")
        st.info("Robust multi-angle evaluation of non-proportional hazards, effect decay, and crossing survival curves. Bundled for automated reporting.")
        
        get_survival_probability = deps.get("get_survival_probability")
        if get_survival_probability:
            prob_res = get_survival_probability(merged, duration_col="OS_MONTHS", event_col="is_dead", group_col="group", time_points=[36.0, 60.0])
            if prob_res["ok"]:
                pvals = prob_res["prob_values"]
                v36_high = pvals.get(ref_group, {}).get(36.0)
                v36_low = pvals.get(base_group, {}).get(36.0)
                v60_high = pvals.get(ref_group, {}).get(60.0)
                v60_low = pvals.get(base_group, {}).get(60.0)

                if v36_high is not None and v36_low is not None:
                    with st.expander("🛡️ Absolute Risk (3/5-Year Survival Rate)", expanded=False):
                        st.markdown("**Kaplan-Meier estimates of survival probability at specific time points.**")
                        c1, c2, c3 = st.columns(3)
                        c1.metric(f"3-Year Survival: {ref_group}", f"{v36_high*100:.1f}%")
                        c2.metric(f"3-Year Survival: {base_group}", f"{v36_low*100:.1f}%")
                        diff36 = (v36_high - v36_low) * 100
                        c3.metric("Absolute Risk Diff (3-Yr)", f"{diff36:+.1f}%", delta=f"{diff36:+.1f}%", delta_color="normal" if diff36>0 else "inverse")

                        if v60_high is not None and v60_low is not None:
                            d1, d2, d3 = st.columns(3)
                            d1.metric(f"5-Year Survival: {ref_group}", f"{v60_high*100:.1f}%")
                            d2.metric(f"5-Year Survival: {base_group}", f"{v60_low*100:.1f}%")
                            diff60 = (v60_high - v60_low) * 100
                            d3.metric("Absolute Risk Diff (5-Yr)", f"{diff60:+.1f}%", delta=f"{diff60:+.1f}%", delta_color="normal" if diff60>0 else "inverse")

        # rmst_time and calculate_rmst are passed as kwargs
        if calculate_rmst:
            rmst_res = calculate_rmst(merged, duration_col="OS_MONTHS", event_col="is_dead", group_col="group", tau=rmst_time if rmst_time > 0 else None)
            if rmst_res["ok"]:
                tau = rmst_res["tau"]
                vals = rmst_res["rmst_values"]
                v_high = vals.get(ref_group)
                v_low = vals.get(base_group)

                if v_high is not None and v_low is not None:
                    diff = v_high - v_low
                    with st.expander(f"⏲️ Restricted Mean Survival Time (RMST) @ {tau:.1f} Months", expanded=False):
                        st.markdown(
                            "**Restricted Mean Survival Time (RMST)** measures the area under the Kaplan-Meier curve. "
                            "It represents the expected survival time within that window. "
                        )
                        c1, c2, c3 = st.columns(3)
                        c1.metric(f"RMST: {ref_group}", f"{v_high:.1f} mo")
                        c2.metric(f"RMST: {base_group}", f"{v_low:.1f} mo")

                        # Show difference with color coding
                        diff_color = "normal" if diff > 0 else "inverse"
                        c3.metric("Difference (Effect Size)", f"{diff:+.1f} mo", delta=f"{diff:+.1f} Months", delta_color=diff_color)

        run_cox_time_varying = deps.get("run_cox_time_varying")
        plot_hr_over_time = deps.get("plot_hr_over_time")
        get_time_varying_snapshots = deps.get("get_time_varying_snapshots")

        if run_cox_time_varying and plot_hr_over_time and get_time_varying_snapshots:
            if locals().get("ph_violated_1g", False):
                st.info("🎯 **Recommended:** A Proportional Hazards violation was detected above (Curves likely cross). "
                        "This Time-Varying model is mathematically necessary to correctly interpret the dynamic effect.")

            with st.expander("⏳ Time-Varying Coefficient Cox Model (HR over Time)", expanded=True):
                st.markdown(
                    "Explicitly models how the Hazard Ratio changes over time. "
                    "When Proportional Hazards are violated, the standard HR is a misleading average. "
                    "This model mathematically captures whether the true effect vanishes, grows, or reverses later on."
                )
                
                time_func = st.radio("Time Function Interaction", ["log(time)", "linear(time)"], index=0, horizontal=True)
                t_trans = 'log' if 'log' in time_func else 'linear'
                if t_trans == 'linear':
                    st.caption("Warning: linear(time) gives extreme weight to distant events. log(time) is often more stable.")
                
                st.caption("⏳ **Note:** This analysis converts data into an 'episodic' format and may take significantly longer than standard Cox models.")
                if st.button("Calculate Time-Varying HR Curve", key="btn_tv"):
                    with st.spinner("Fitting non-proportional Time-Varying Cox Model..."):
                        tv_res = run_cox_time_varying(
                            merged, 
                            duration_col="OS_MONTHS", 
                            event_col="is_dead", 
                            main_var="is_target", 
                            covariates=covariates_actual,
                            time_transform=t_trans
                        )
                        
                        if tv_res["ok"]:
                            st.success("✅ Time-Varying Model Fitted Successfully")
                            
                            c_t1, c_t2 = st.columns([2, 1])
                            with c_t1:
                                fig_tv = plot_hr_over_time(tv_res, title=f"Time-Varying HR(t) ({ref_group} vs {base_group})")
                                st.pyplot(fig_tv)
                                st.caption("If the curve crosses '1.0' (dotted line), the effect direction **may reverse** from protective to detrimental (or vice versa) over time.")
                            with c_t2:
                                st.markdown("##### Representative HR Snapshots")
                                df_snaps = get_time_varying_snapshots(tv_res, timepoints=[12, 24, 36, 60])
                                st.dataframe(df_snaps.style.format({"HR(t)": "{:.2f}", "95% CI Lower": "{:.2f}", "95% CI Upper": "{:.2f}", "Month": "{:.0f}"}), hide_index=True, use_container_width=True)
                                
                                p_t = tv_res["p_time"]
                                st.markdown(f"**Interaction p-value:** `{p_t:.3e}`")
                                if p_t < 0.05:
                                    st.info("The effect changes significantly over time.")
                                else:
                                    st.write("The effect decay is not statistically significant.")
                                
                                # Early/Late Summary
                                if not df_snaps.empty:
                                    st.markdown("##### Phase Summary")
                                    hr_12 = df_snaps.loc[df_snaps["Month"] == 12, "HR(t)"]
                                    hr_late = df_snaps.iloc[-1]["HR(t)"]
                                    m_late = df_snaps.iloc[-1]["Month"]
                                    if len(hr_12) > 0:
                                        st.markdown(f"- **Early Phase (12m):** HR = `{hr_12.values[0]:.2f}`")
                                    st.markdown(f"- **Late Phase ({m_late:.0f}m):** HR = `{hr_late:.2f}`")
                                    
                                    # Very rough crossing detection
                                    if len(hr_12) > 0 and (hr_12.values[0] - 1.0) * (hr_late - 1.0) < 0:
                                        st.markdown("- **Crossing Detected:** The effect reverses direction during follow-up.")
                        else:
                            st.error(f"Time-Varying fitting failed: {tv_res.get('reason')}")

        landmark_time = deps.get("landmark_time", 0)
        apply_landmark = deps.get("apply_landmark")
        if landmark_time > 0 and apply_landmark:
            with st.expander(f"⏱️ Piecewise Landmark Analysis (Cutoff: {landmark_time} Months)", expanded=False):
                st.info(
                    f"**Time-dependent Analysis**: Data is split into 'Early' (0–{landmark_time} months) "
                    f"and 'Late' (> {landmark_time} months) cohorts to handle non-proportional hazards (crossing curves)."
                )

                df_early, df_late = apply_landmark(merged, duration_col="OS_MONTHS", event_col="is_dead", landmark_time=landmark_time)

                col_l1, col_l2 = st.columns(2)

                with col_l1:
                    st.markdown(f"**Early Cohort (0 to {landmark_time} Months)**")
                    if not df_early.empty and df_early["is_dead"].sum() >= 5:
                        fig_early = plot_km(df_early, group_col="group", title=f"KM: Early (<= {landmark_time}m)")
                        st.pyplot(fig_early)
                        cox_early = run_cox(df_early, duration_col="OS_MONTHS", event_col="is_dead", main_var="is_target", covariates=covariates_actual)
                        if cox_early["ok"]:
                            sig_color = "normal" if cox_early['p'] < 0.05 else "off"
                            st.metric(f"Early HR ({ref_group} vs {base_group})", f"{cox_early['hr']:.2f}", delta=f"p={cox_early['p']:.4f}", delta_color=sig_color)
                            st.caption(f"95% CI: [{cox_early['hr_l']:.2f} – {cox_early['hr_u']:.2f}] | Events={cox_early['events']}")
                        else:
                            st.warning("Cox model failed for Early cohort.")
                    else:
                        st.warning("Not enough events for Early Cohort.")

                with col_l2:
                    st.markdown(f"**Late Cohort (> {landmark_time} Months)**")
                    if not df_late.empty and df_late["is_dead"].sum() >= 5:
                        fig_late = plot_km(df_late, group_col="group", title=f"KM: Late (> {landmark_time}m)")
                        st.pyplot(fig_late)
                        cox_late = run_cox(df_late, duration_col="OS_MONTHS", event_col="is_dead", main_var="is_target", covariates=covariates_actual)
                        if cox_late["ok"]:
                            sig_color = "normal" if cox_late['p'] < 0.05 else "off"
                            st.metric(f"Late HR ({ref_group} vs {base_group})", f"{cox_late['hr']:.2f}", delta=f"p={cox_late['p']:.4f}", delta_color=sig_color)
                            st.caption(f"95% CI: [{cox_late['hr_l']:.2f} – {cox_late['hr_u']:.2f}] | Events={cox_late['events']}")
                        else:
                            st.warning("Cox model failed for Late cohort.")
                    else:
                        st.warning("Not enough events for Late Cohort.")

        # ----------------------------------------------------
        # 1.54 Mechanism Exploration (Pathway Analysis)
        # ----------------------------------------------------
        run_gsea_prerank = deps.get("run_gsea_prerank")
        plot_top_pathways = deps.get("plot_top_pathways")
        plot_gsea_enrichment = deps.get("plot_gsea_enrichment")
        LIBRARY_MAPPING = deps.get("LIBRARY_MAPPING", {})
        
        if run_gsea_prerank and plot_top_pathways and plot_gsea_enrichment and get_gene_exp:
            st.markdown("---")
            st.subheader("🧬 Mechanism Exploration (Hypothesis-Generating Pathway Analysis)")
            st.markdown(
                f"Identify which biological pathways co-vary with **{gene}** expression across the cohort. "
                "This provides a hypothesis for the underlying functional mechanism driving the survival differences."
            )
            
            with st.expander("Mechanism & Pathways (GSEA)", expanded=False):
                col_g1, col_g2 = st.columns([1, 2])
                with col_g1:
                    lib_keys = list(LIBRARY_MAPPING.keys()) if LIBRARY_MAPPING else ["Hallmark", "KEGG"]
                    lib_choice = st.selectbox("Gene Set Library", lib_keys, index=0)
                    lib_name = LIBRARY_MAPPING.get(lib_choice, "MSigDB_Hallmark_2020")
                    st.caption(f"Engine: `gseapy.prerank`\nLibrary: `{lib_name}`")
                    
                    calc_gsea = st.button("Run Rank-Based GSEA", type="primary", use_container_width=True)
                
                with col_g2:
                    st.info(
                        "**Methodology:** All genes are ranked by their Pearson correlation with the target gene. "
                        "Threshold-free Gene Set Enrichment Analysis (GSEA) is then performed to find globally enriched/depleted pathways."
                    )
                
                if calc_gsea:
                    with st.spinner(f"Running GSEA against {lib_name}..."):
                        df_exp_full = df_exp # Use the provided full expression matrix
                        if df_exp_full is not None and not df_exp_full.empty:
                            gsea_res = run_gsea_prerank(df_exp_full, target_gene=gene, gene_set=lib_name, n_jobs=-1)
                            
                            if gsea_res.get("ok"):
                                st.success(f"✅ GSEA Completed: {gsea_res['n_genes_ranked']:,} genes ranked.")
                                
                                df_res = gsea_res["results_df"]
                                
                                # Render Top Pathways Bar Chart
                                st.markdown("#### Top Enriched Pathways")
                                fig_gsea = plot_top_pathways(df_res, top_n=10, mode="GSEA")
                                st.pyplot(fig_gsea)
                                
                                # Render Rich Table with Leading Edge
                                st.markdown("#### Detailed Pathway Results")
                                st.dataframe(
                                    df_res.style.format({
                                        "ES": "{:.2f}", 
                                        "NES": "{:.2f}", 
                                        "NOM p-val": "{:.2e}", 
                                        "FDR q-val": "{:.2e}"
                                    }),
                                    use_container_width=True,
                                    hide_index=True
                                )
                                
                                # Store for optional individual plot rendering
                                st.session_state["last_gsea_res"] = gsea_res
                            else:
                                st.error(f"GSEA failed: {gsea_res.get('reason')}")
                                
                # If cached GSEA results exist, offer plotting
                if "last_gsea_res" in st.session_state and st.session_state["last_gsea_res"].get("ok"):
                    res_cache = st.session_state["last_gsea_res"]
                    if res_cache.get("library") == lib_name: # Only show if library matches
                        st.markdown("---")
                        st.markdown("#### Specific Pathway Enrichment Plot")
                        terms = res_cache["results_df"]["Term"].tolist()
                        sel_term = st.selectbox("Select Pathway to Plot", terms)
                        if st.button("Generate Enrichment Plot"):
                            fig_ep = plot_gsea_enrichment(res_cache, sel_term)
                            st.pyplot(fig_ep)

        # ----------------------------------------------------
        # 1.5 Target Druggability & Safety Profile
        # ----------------------------------------------------
        st.markdown("---")

        use_mock = deps.get("use_mock", True)
        title_suffix = "(Mock)" if use_mock else "(Live API)"
        st.subheader(f"💊 Target Profiling & Druggability {title_suffix}")

        if use_mock:
            st.markdown("Assess whether this gene possesses characteristics suitable for drug development. *Note: Data is mock/simulated for demonstration.*")
        else:
            st.markdown("Assess whether this gene possesses characteristics suitable for drug development. *Data fetched from OpenTargets API.*")

        try:
            from src.druggability import get_target_profile
            prof = get_target_profile(gene, use_mock=use_mock)

            col_p1, col_p2, col_p3, col_p4 = st.columns(4)
            with col_p1:
                st.markdown("**Modality**")
                st.write(prof.get("modality", "Unknown"))
            with col_p2:
                st.markdown("**Subcellular Location**")
                st.write(prof.get("location", "Unknown"))
            with col_p3:
                ess_score = prof.get("essentiality_prob", 0)
                st.metric("CRISPR Essentiality", f"{ess_score:.0%}")
            with col_p4:
                tox_score = prof.get("toxicity_risk", 0)
                st.metric("Norm. Tissue Toxicity Risk", f"{tox_score}/100")

            # Progress bar for overall score
            ov_score = prof.get("overall_score", 0)
            score_color = "green" if ov_score >= 70 else ("orange" if ov_score >= 40 else "red")
            st.write(f"**Overall Druggability Score:** {ov_score}/100")
            st.progress(ov_score / 100.0)

            # Warnings and Notes
            if prof.get("notes"):
                st.info(f"**Annotations:** {prof['notes']}")

            if ess_score > 0.8:
                st.warning("⚠️ **High Essentiality Risk:** This gene appears pan-essential. Systemic inhibition may cause severe severe toxicity.")
            if tox_score > 75:
                st.warning("⚠️ **High Off-Target Toxicity Risk:** High expression detected in vital normal tissues (e.g., Heart, Brain, Liver).")
            if "Intracellular" in prof.get("location", "") and "Antibody" in prof.get("modality", ""):
                st.warning("⚠️ **Modality Mismatch:** Target is intracellular but annotated for antibodies. Consider small molecules or degraders.")

        except Exception as e:
            st.error(f"Failed to load target profile: {e}")

        # ----------------------------------------------------
        # 1.55 Known Drugs & Repurposing Opportunities
        # ----------------------------------------------------
        st.markdown("---")
        st.subheader(f"⚕️ Known Drugs & Repurposing Opportunities {title_suffix}")

        if use_mock:
            st.markdown(f"Identify existing approved or investigational therapies targeting **{gene}** that could be repurposed for this cohort.")
        else:
            st.markdown(f"Identify existing approved or investigational therapies targeting **{gene}** that could be repurposed for this cohort. *Data fetched from OpenTargets API.*")

        try:
            from src.known_drugs import get_known_drugs
            d_list = get_known_drugs(gene, use_mock=use_mock)
            if not d_list:
                st.info(f"No known targeted therapies found for {gene} in the current database.")
            else:
                df_drugs = pd.DataFrame(d_list)

                # Highlight approved drugs
                def style_approved(val):
                    color = '#2e7d32' if val == 'Approved' else ''
                    font = 'bold' if val == 'Approved' else 'normal'
                    return f'color: {color}; font-weight: {font}'

                # Render elegantly
                st.dataframe(
                    df_drugs.style.map(style_approved, subset=['phase']),
                    use_container_width=True,
                    hide_index=True
                )
        except Exception as e:
            st.error(f"Failed to load known drugs: {e}")

        # ----------------------------------------------------
        # 1.57 Bootstrap Robustness Evaluation
        # ----------------------------------------------------
        st.markdown("---")
        st.subheader("🧪 Bootstrap Robustness Evaluation")
        st.markdown(
            "Repeatedly resample patients (with replacement) to re-evaluate the Survival Hazard Ratio and Log-rank p-value "
            "across hundreds of simulated cohorts. This quantifies whether the discovered effect is statistically stable or an overfitted artifact."
        )
        
        run_1gene_bootstrap = deps.get("run_1gene_bootstrap")
        plot_bootstrap_results = deps.get("plot_bootstrap_results")
        
        if run_1gene_bootstrap and plot_bootstrap_results:
            with st.expander("Evaluate Stability via Bootstrap", expanded=False):
                boot_n = st.radio("Number of Resampling Iterations", [100, 500], horizontal=True, 
                                  help="100 is fast for preview. 500 is recommended for publication stability.")
                
                if st.button("Run Bootstrap Robustness Test"):
                    with st.spinner(f"Running {boot_n} bootstrap iterations..."):
                        boot_res = run_1gene_bootstrap(
                            merged, 
                            duration_col="OS_MONTHS", 
                            event_col="is_dead", 
                            covariates=covariates_actual,
                            gene_col=target_col,
                            threshold=threshold,
                            n_iterations=boot_n,
                            auto_p=auto_p is not None # re-optimize cutoff if auto was used
                        )
                        st.session_state["1gene_boot_res"] = boot_res
                
                # Display cached bootstrap
                if "1gene_boot_res" in st.session_state:
                    res = st.session_state["1gene_boot_res"]
                    st.success(f"**Stability Rating:** {res['stability_rating']}")
                    
                    st.markdown(f"- **HR 95% Empirical CI**: [{res['hr_ci'][0]:.2f} - {res['hr_ci'][1]:.2f}]")
                    st.markdown(f"- **Fraction of significant runs (Log-rank)**: {res['sig_fraction']*100:.1f}%")
                    
                    fig = plot_bootstrap_results(res)
                    st.pyplot(fig)


        # ----------------------------------------------------
        # 1.60 Target Triage & Prioritization (Translational Score)
        # ----------------------------------------------------
        st.markdown("---")
        st.subheader("🎯 Target Triage & Prioritization")
        st.markdown(
            "Synthesizes biological evidence, druggability, safety, and competitive landscape into a unified translation priority score. "
            "Select your evaluation strategy to weigh the translation axes accordingly."
        )
        triage_mode = st.radio(
            "Evaluation Strategy Mode", 
            ["Aligned with Novel Target Discovery (Novelty-First)", "Aligned with Drug Repurposing (Repurposing-First)", "Balanced"],
            index=2,
            horizontal=False
        )
        
        # Map to engine modes
        if "Novel" in triage_mode:
            engine_mode = "Novel-target-first"
        elif "Repurposing" in triage_mode:
            engine_mode = "Repurposing-first"
        else:
            engine_mode = "Balanced"

        if st.button("Generate Triage Dashboard", type="primary"):
            from src.triage import TargetTriageEngine
            import plotly.graph_objects as go
            
            with st.spinner("Synthesizing multi-omics evidence..."):
                engine = TargetTriageEngine(mode=engine_mode)
                
                # Extract previously calculated data from session or current variables
                gsea_hit = False
                if "last_gsea_res" in st.session_state and st.session_state["last_gsea_res"].get("ok"):
                    # Check if any FDR < 0.25 (standard GSEA threshold)
                    g_res = st.session_state["last_gsea_res"]["gsea_obj"].res2d
                    if (g_res["FDR q-val"] < 0.25).any():
                        gsea_hit = True

                # Extract Drug info
                has_appr = False
                has_clin = False
                try:
                    from src.known_drugs import get_known_drugs
                    d_list = get_known_drugs(gene, use_mock=use_mock)
                    for d in d_list:
                        if d["phase"] == "Approved":
                            has_appr = True
                        elif "Phase" in d["phase"]:
                            has_clin = True
                except:
                    pass

                # Build Triage Dictionary
                triage_input = {
                    "target_symbol": gene,
                    "hr": cox_res.get("hr") if cox_res.get("ok") else None,
                    "p_value": cox_res.get("p") if cox_res.get("ok") else None,
                    "psm_maintained": psm_res.get("p") < 0.05 if ("psm_res" in locals() and psm_res and psm_res.get("ok")) else None,
                    "time_varying_stable": target_p >= 0.05 if ("target_p" in locals()) else None,
                    "gsea_hallmark_hit": gsea_hit,
                    "accessibility": prof.get("location") if ("prof" in locals() and prof) else None,
                    "target_class": prof.get("modality") if ("prof" in locals() and prof) else None,
                    "is_essential": ess_score > 0.8 if ("ess_score" in locals() and pd.notna(ess_score)) else None,
                    "normal_tissue_expression": "High" if ("tox_score" in locals() and pd.notna(tox_score) and tox_score > 75) else ("Low" if ("tox_score" in locals() and pd.notna(tox_score) and tox_score < 25) else "Medium"),
                    "known_drugs": {
                        "has_approved": has_appr,
                        "has_clinical": has_clin
                    }
                }
                
                res = engine.evaluate(triage_input)
                
                st.markdown(f"### Score: **{res['total_score']:.0f} / 100**")
                
                # Status Badge
                if res['priority'] == "High":
                    st.success(f"**Translational Priority:** {res['priority']} ({res['confidence']})")
                elif res['priority'] == "Medium":
                    st.warning(f"**Translational Priority:** {res['priority']} ({res['confidence']})")
                else:
                    st.error(f"**Translational Priority:** {res['priority']} ({res['confidence']})")
                
                colA, colB = st.columns([1.2, 1])
                
                with colA:
                    # Pros & Cons
                    st.markdown("#### Strengths & Weaknesses")
                    for p in res['pros']:
                        st.markdown(f"✅ {p}")
                    for c in res['cons']:
                        st.markdown(f"⚠️ {c}")
                    
                    if res.get('missing_data', 0) > 0:
                        st.info(f"ℹ️ {res['missing_data']} evidence domains were missing and marked as 'uncertain'. Run GSEA and PSM to complete the profile.")

                with colB:
                    # Radar Chart
                    categories = ['Evidence', 'Druggability', 'Safety', f'Strategy<br>({engine_mode})']
                    r_values = [
                        res["breakdown"]["Evidence"]["score"] / res["breakdown"]["Evidence"]["max"] * 100,
                        res["breakdown"]["Druggability"]["score"] / res["breakdown"]["Druggability"]["max"] * 100,
                        res["breakdown"]["Safety"]["score"] / res["breakdown"]["Safety"]["max"] * 100,
                        res["breakdown"]["Translation Strategy"]["score"] / res["breakdown"]["Translation Strategy"]["max"] * 100
                    ]
                    # close the polygon
                    categories.append(categories[0])
                    r_values.append(r_values[0])
                    
                    fig_radar = go.Figure(data=go.Scatterpolar(
                        r=r_values,
                        theta=categories,
                        fill='toself',
                        marker=dict(color='#ff4b4b' if res['priority'] == "High" else '#4b8bba')
                    ))
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(visible=True, range=[0, 100])
                        ),
                        showlegend=False,
                        margin=dict(l=40, r=40, t=20, b=20),
                        height=300
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

                # Breakdown Table
                with st.expander("Detailed Evidence Breakdown"):
                    all_details = []
                    for cat in ["Evidence", "Druggability", "Safety", "Translation Strategy"]:
                        for row in res["breakdown"][cat]["details"]:
                            all_details.append({
                                "Category": cat,
                                "Subdomain": row["Subdomain"],
                                "Score": row["Score"],
                                "Evidence": row["Evidence"]
                            })
                    df_details = pd.DataFrame(all_details)
                    st.dataframe(df_details, use_container_width=True, hide_index=True)
                col_b1, col_b2 = st.columns([1, 2])
                with col_b1:
                    n_iter_str = st.selectbox("Bootstrap Iterations", ["100 (Fast Preview)", "500 (Publication Check)"], index=0)
                    n_iter = 100 if "100" in n_iter_str else 500
                    trigger_boot = st.button(f"Run {n_iter} Resamples", type="secondary", use_container_width=True)
                
                if trigger_boot:
                    with st.spinner(f"Running {n_iter} bootstrap parallel iterations..."):
                        # Re-run from the merged dataset
                        boot_res = run_1gene_bootstrap(
                            merged,
                            gene="expression", # The expression column is literally named 'expression' in `merged`
                            time_col="OS_MONTHS",
                            event_col="is_dead",
                            base_cutoff_pct=threshold_pct,
                            use_auto_cutoff=use_auto_cutoff,
                            n_iterations=n_iter,
                            n_jobs=-1
                        )
                    
                    if not boot_res["ok"]:
                        st.error(f"Bootstrap failed: {boot_res.get('reason')}")
                    else:
                        st.success(f"**Robustness Score: {boot_res['stability_label']}** (Significant in {boot_res['sig_fraction']:.1%} of resamples)")
                        c_b1, c_b2 = st.columns(2)
                        c_b1.metric("Median HR", f"{boot_res['hr_median']:.2f}", f"95% Empirical CI: [{boot_res['hr_ci_lower']:.2f} - {boot_res['hr_ci_upper']:.2f}]", delta_color="off")
                        c_b2.metric("Significant Runs (p < 0.05)", f"{boot_res['sig_fraction']:.1%}")
                        
                        fig_boot = plot_bootstrap_results(boot_res, title=f"1-Gene Cutoff Robustness")
                        st.pyplot(fig_boot)
                        st.caption("A wide HR distribution spanning across 1.0 (the dotted grey line) indicates high instability.")

        # ----------------------------------------------------
        # 1.58 Companion Diagnostics & Patient Stratification
        # ----------------------------------------------------
        st.markdown("---")
        st.subheader("🎯 Companion Diagnostics & Patient Stratification")
        st.markdown(
            f"Identify which specific patient subgroups receive the greatest prognostic benefit from **{gene}** expression. "
            "This highlights potential target populations for clinical trials."
        )

        try:
            from src.stratification import run_subgroup_analysis, plot_subgroup_forest, get_top_recommended_subgroup

            # Select relevant clinical variables for stratification
            potential_covariates = ["STAGE", "PATHOLOGIC_STAGE", "SEX", "AGE", "CANCER_TYPE", "CANCER_TYPE_DETAILED", "MSI", "TMB", "mut_status"]
            available_covs = [c for c in potential_covariates if c in merged0.columns and merged0[c].nunique() > 1]

            if not available_covs:
                st.info("Not enough clinical covariates available for subgroup stratification.")
            else:
                with st.spinner("Running subgroup Cox regressions..."):
                    # target_var is the dichotomized expression 'is_high' (1/0)
                    strat_results = run_subgroup_analysis(
                        merged0,
                        gene_col=gene,
                        duration_col="OS_MONTHS",
                        event_col="is_dead",
                        target_var="is_high",
                        covariates_to_test=available_covs
                    )

                if not strat_results:
                    st.info("Subgroup analysis did not yield valid models (likely too few events).")
                else:
                    col_s1, col_s2 = st.columns([1, 2])

                    top_sub = get_top_recommended_subgroup(strat_results)

                    with col_s1:
                        st.markdown("#### Primary Indication Hypothesis")
                        if top_sub:
                            hr = top_sub['HR']
                            direction = "poor" if hr > 1.0 else "favorable"
                            st.success(
                                f"**Recommended Target Population:**\n\n"
                                f"**{top_sub['Subgroup']}** (within {top_sub['Covariate']}).\n\n"
                                f"In this subgroup, {gene} expression represents a highly significant **{direction}** prognostic factor "
                                f"(HR={hr:.2f}, p={top_sub['p_value']:.2e})."
                            )
                        else:
                            st.info("No single subgroup stands out significantly beyond the overall population effect.")

                        # Toggle raw data
                        with st.expander("View Raw Subgroup Data"):
                            df_s = pd.DataFrame(strat_results)
                            st.dataframe(df_s, use_container_width=True)

                    with col_s2:
                        # Plot Forest Plot
                        fig_strat = plot_subgroup_forest(strat_results, title=f"{gene} Prognostic Impact by Subgroup")
                        st.pyplot(fig_strat, use_container_width=True)

        except Exception as e:
            st.error(f"Failed to run subgroup stratification: {e}")

        # ----------------------------------------------------
        # 1.6 Expression vs Clinical Features
        # ----------------------------------------------------
        st.markdown("---")
        st.subheader("📊 Expression vs Clinical Features")
        st.markdown(
            "Explore whether the target gene is differentially expressed across different clinical subgroups "
            "(e.g., Stage, Gender, Cancer Type / PanCancer)."
        )

        plot_expression_by_category = deps.get("plot_expression_by_category")
        if plot_expression_by_category:
            with st.expander("Show Expression Distribution Plots", expanded=False):
                col_cat1, col_cat2 = st.columns(2)
                with col_cat1:
                    # Select category
                    cat_cols = [c for c in df_clin.columns if c not in ["PATIENT_ID", "OS_MONTHS", "is_dead"]]

                    # Try to put common categories at the top
                    common = [c for c in ["STAGE", "PATHOLOGIC_STAGE", "SEX", "CANCER_TYPE", "CANCER_TYPE_DETAILED"] if c in cat_cols]
                    others = [c for c in cat_cols if c not in common]
                    cat_options = common + others

                    if not cat_options:
                        st.warning("No categorical variables available in the dataset.")
                    else:
                        selected_cat = st.selectbox("Select Clinical Variable", options=cat_options)

                        btn_plot_cat = st.button("Generate Boxplot vs " + selected_cat)
                        if btn_plot_cat:
                            with st.spinner(f"Plotting Expression by {selected_cat}..."):
                                fig_cat = plot_expression_by_category(
                                    df=merged, 
                                    expr_col="expression", 
                                    category_col=selected_cat,
                                    title=f"{gene} Expression by {selected_cat}"
                                )
                                if fig_cat:
                                    st.pyplot(fig_cat)
                                else:
                                    st.warning(f"Could not generate plot for {selected_cat}. Check data availability.")

        # ----------------------------------------------------
        # 2. Integration: Expression vs Mutation / CNA Boxplots
        # ----------------------------------------------------
        st.markdown("---")
        st.subheader("🧬 Multi-Omics Integration")
        if "mut_status" in merged.columns and "cna_status" in merged.columns:
            st.caption("Investigate relationships between mRNA Expression and Mutation/CNA statuses.")

            with st.expander("Show Omics Integration Plots", expanded=False):
                col1, col2 = st.columns(2)

                with col1:
                    has_mut = (merged["mut_status"] != "N/A").any()
                    if has_mut and plot_expression_by_category:
                        fig_mut_box = plot_expression_by_category(
                            df=merged, 
                            expr_col="expression", 
                            category_col="mut_status",
                            title=f"{gene} Expression vs Mut Status"
                        )
                        if fig_mut_box:
                            st.pyplot(fig_mut_box)
                        else:
                            st.info("Insufficient data for Mutation plot.")
                    else:
                        st.info("Mutation data not available.")

                with col2:
                    has_cna = (merged["cna_status"] != "N/A").any()
                    if has_cna and plot_expression_by_category:
                        valid_cna_df = merged[merged["cna_status"] != "Unknown"]
                        fig_cna_box = plot_expression_by_category(
                            df=valid_cna_df, 
                            expr_col="expression", 
                            category_col="cna_status",
                            title=f"{gene} Expression vs CNA Status"
                        )
                        if fig_cna_box:
                            st.pyplot(fig_cna_box)
                        else:
                            st.info("Insufficient data for CNA plot.")
                    else:
                        st.info("CNA data not available.")

        # ----------------------------------------------------
        # 3. Nomogram & Risk Score Simulator
        # ----------------------------------------------------
        if cox_res["ok"] and len(covariates) > 0:
            st.markdown("---")
            st.subheader("📊 Nomogram & Risk Score Simulator (Beta)")
            st.markdown(
                "Build a personalized prognostic score based on the **Adjusted Cox Model** coefficients. "
                "Adjust the patient profile below to calculate their estimated relative risk."
            )

            model = cox_res["model"]
            coefs = model.params_

            with st.expander("Interactive Risk Calculator", expanded=True):
                col_inputs, col_results = st.columns([1, 1])

                patient_profile = {}
                base_score = 0.0

                with col_inputs:
                    st.markdown("**Patient Profile**")
                    # Main variable (is_target) means the Gene expression group
                    is_high = st.checkbox(f"{gene} Expression = {ref_group} (vs {base_group})", value=True)
                    patient_profile["is_target"] = 1.0 if is_high else 0.0

                    if "AGE" in covariates_actual:
                        age = st.slider("Age", 20, 90, 60, step=1)
                        patient_profile["AGE"] = float(age)

                    if "is_male" in covariates_actual:
                        is_male = st.radio("Gender", ["Female", "Male"], index=1)
                        patient_profile["is_male"] = 1.0 if is_male == "Male" else 0.0

                    if "stage_num" in covariates_actual:
                        stage = st.slider("Stage (I-IV)", 1, 4, 2, step=1)
                        patient_profile["stage_num"] = float(stage)

                    if "MSI" in covariates_actual:
                        _max_msi = float(merged["MSI"].max()) if "MSI" in merged.columns and not merged["MSI"].isna().all() else 2.0
                        msi_val = st.slider("MSI Score", 0.0, max(2.0, _max_msi), min(0.5, _max_msi), step=0.1)
                        patient_profile["MSI"] = float(msi_val)

                    if "TMB" in covariates_actual:
                        _max_tmb = float(merged["TMB"].max()) if "TMB" in merged.columns and not merged["TMB"].isna().all() else 100.0
                        tmb_val = st.slider("TMB (Nonsynonymous)", 0.0, max(10.0, _max_tmb), min(10.0, _max_tmb), step=1.0)
                        patient_profile["TMB"] = float(tmb_val)

                with col_results:
                    # Calculate Risk Score (Linear Predictor: sum of beta * x)
                    linear_predictor = sum(coefs.get(var, 0) * val for var, val in patient_profile.items())

                    # Convert to Hazard Ratio relative to baseline profile (all 0 or mean)
                    # For simplicity, we just show the HR relative to the "reference patient" where all covariates = 0
                    hazard_ratio = np.exp(linear_predictor)

                    st.markdown("**Predicted Risk**")

                    # Calculate the baseline survival probability at median followup
                    median_time = merged["OS_MONTHS"].median()
                    # A very rough approximation of baseline survival at median time
                    base_surv = 0.5 

                    # S(t) = S0(t)^exp(beta*x)
                    indiv_surv = base_surv ** hazard_ratio

                    st.metric("Relative Risk (HR vs reference)", f"{hazard_ratio:.2f}x")

                    color = "red" if hazard_ratio > 1 else "green"
                    st.markdown(
                        f"""
                        <div style='padding:15px; background-color:#f8f9fa; border-radius:5px; border-left:5px solid {color}'>
                            <h4 style='margin:0'>Points Breakdown:</h4>
                            <ul style='font-size:0.9em; margin-top:10px'>
                        """
                        + "".join([f"<li><b>{v}</b>: {coefs.get(v, 0):.3f} × {val} = {coefs.get(v, 0)*val:.3f}</li>" for v, val in patient_profile.items() if v in coefs])
                        + f"""
                            </ul>
                            <hr>
                            <strong>Total Score (Linear Predictor, βx):</strong> {linear_predictor:.3f}<br>
                            <small><em>Estimated survival probability changes exponentially with this score.</em></small>
                        </div>
                        """, unsafe_allow_html=True
                    )