from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import streamlit as st
from src.analysis_issues import Issue
from ui.analysis_issues_view import render_issue_panel
import matplotlib.pyplot as plt
from typing import Tuple
from src.quality_checks import compute_missing_table



def ensure_cancer_type(merged: pd.DataFrame, df_clin: pd.DataFrame) -> pd.DataFrame:
    """Ensure CANCER_TYPE exists in merged when available in df_clin."""
    if "CANCER_TYPE" not in merged.columns and "CANCER_TYPE" in df_clin.columns:
        if "PATIENT_ID" in merged.columns and "PATIENT_ID" in df_clin.columns:
            merged = merged.merge(
                df_clin[["PATIENT_ID", "CANCER_TYPE"]].drop_duplicates(),
                on="PATIENT_ID",
                how="left",
            )
    return merged


def render_2gene_analysis(
    *,
    dataset_name: str,
    df_clin: pd.DataFrame,
    df_exp: pd.DataFrame,
    gene1: str,
    gene2: str,
    cutoff_pct: int,
    covariates: List[str],
    get_gene_exp,
    add_covariate_columns,
    deps: Dict[str, Any],
    covariates_actual: List[str] = [],
    rmst_time: float = 0.0,
    calculate_rmst: Any = None,
) -> None:
    plot_km = deps["plot_km"]
    run_cox_4group = deps["run_cox_4group"]
    run_cox_interaction = deps["run_cox_interaction"]
    format_interaction_table_for_forest = deps["format_interaction_table_for_forest"]
    plot_interaction_forest = deps["plot_interaction_forest"]
    plot_forest_contrasts = deps["plot_forest_contrasts"]
    summarize_by_cancer_type = deps["summarize_by_cancer_type"]
    run_cox_4group_stratified = deps["run_cox_4group_stratified"]
    run_cut_smoothness_scan = deps["run_cut_smoothness_scan"]
    plot_cut_smoothness_curve = deps["plot_cut_smoothness_curve"]
    run_ph_test = deps.get("run_ph_test")
    plot_schoenfeld_residuals = deps.get("plot_schoenfeld_residuals")
    multivariate_logrank_test = deps["multivariate_logrank_test"]
    get_survival_probability = deps.get("get_survival_probability")

    _assess_survival_guardrails = deps["_assess_survival_guardrails"]
    render_guardrail_box = deps["render_guardrail_box"]
    metric_p_display = deps["metric_p_display"]

    plot_stage_distribution = deps["plot_stage_distribution"]
    plot_age_distribution = deps["plot_age_distribution"]
    plot_gender_distribution = deps["plot_gender_distribution"]
    plot_followup_distribution = deps["plot_followup_distribution"]
    compute_missing_table = deps["compute_missing_table"]
    """
    UI wrapper for 2-gene analysis. Keeps session_state caching behavior in iwa-collections-analysis.py compatible.
    """
    st.header(f"🧪 2-Gene Survival: {gene1} × {gene2}")
    run_now = st.sidebar.button("Run Detailed Analysis") or st.session_state.pop("auto_run_detailed", False)
    if run_now:
        exp1 = get_gene_exp(df_exp, gene1)
        exp2 = get_gene_exp(df_exp, gene2)
        if exp1 is None or exp2 is None:
            st.error(f"Gene not found: {gene1 if exp1 is None else gene2}")
            st.stop()

        m_exp = pd.merge(exp1, exp2, on="PATIENT_ID", suffixes=("_1", "_2"))
        merged0 = pd.merge(df_clin, m_exp, on="PATIENT_ID")
        merged0 = add_covariate_columns(merged0)
        # ensure CANCER_TYPE persists
        if "CANCER_TYPE" in df_clin.columns and "CANCER_TYPE" not in merged0.columns:
            merged0 = merged0.merge(
                df_clin[["PATIENT_ID", "CANCER_TYPE"]].drop_duplicates(),
                on="PATIENT_ID",
                how="left",
            )

        st.session_state["analysis_merged"] = merged0.copy()
        st.session_state["analysis_ready"] = True
        st.session_state["analysis_meta"] = {"gene1": gene1, "gene2": gene2}

    # --- render cached 2-gene analysis (so it doesn't disappear on reruns) ---
    if st.session_state.get("analysis_ready") and isinstance(st.session_state.get("analysis_merged"), pd.DataFrame):
        merged = st.session_state["analysis_merged"].copy()
        if "CANCER_TYPE" not in merged.columns and "CANCER_TYPE" in df_clin.columns:
            merged = merged.merge(
                df_clin[["PATIENT_ID", "CANCER_TYPE"]].drop_duplicates(),
                on="PATIENT_ID",
                how="left",
            )
        meta = st.session_state.get("analysis_meta", {})
        gene1 = str(meta.get("gene1", gene1)).upper()
        gene2 = str(meta.get("gene2", gene2)).upper()

        # percentile thresholds (always from sidebar state)
        th1_pct = int(st.session_state.get("analysis_th1_pct", cutoff_pct))
        th2_pct = int(st.session_state.get("analysis_th2_pct", cutoff_pct))
        th1 = th1_pct / 100.0
        th2 = th2_pct / 100.0

        # recompute cutoffs + groups on each rerun (cheap + avoids storing figs)
        c1 = merged["expression_1"].quantile(th1)
        c2 = merged["expression_2"].quantile(th2)
        g1_hi = merged["expression_1"] >= c1
        g2_hi = merged["expression_2"] >= c2
        merged["group"] = np.select(
            [g1_hi & g2_hi, g1_hi & (~g2_hi), (~g1_hi) & g2_hi],
            ["HiHi", "HiLo", "LoHi"],
            default="LoLo",
        )

        # ここから下は、今のiwa-collections-analysis.pyの「Detailed analysis表示」ブロックをそのまま移植する領域
        pass
        
        # Plots: KM + scatter
        col_plot1, col_plot2 = st.columns([1.2, 1])
        color_map = {"HiHi": "red", "HiLo": "orange", "LoHi": "green", "LoLo": "blue"}

        with col_plot1:
            st.subheader("Survival Analysis")
            fig_km = plot_km(
                merged,
                group_col="group",
                title="Kaplan–Meier (4-way)",
                color_map=color_map,
                group_order=["HiHi", "HiLo", "LoHi", "LoLo"],
            )
            st.pyplot(fig_km)
            
            with st.expander("Number At Risk", expanded=False):
                from src.survival import get_at_risk_df
                df_risk = get_at_risk_df(merged, "group", group_order=["HiHi", "HiLo", "LoHi", "LoLo"])
                st.dataframe(df_risk, use_container_width=True, hide_index=True)
            
            import io
            buf_km = io.BytesIO()
            fig_km.savefig(buf_km, format="pdf", bbox_inches="tight")
            st.download_button(
                label="Download KM Plot (PDF)",
                data=buf_km.getvalue(),
                file_name=f"KM_plot_{gene1}_{gene2}.pdf",
                mime="application/pdf",
                key=f"dl_km_plot_{dataset_name}"
            )

        with col_plot2:
            st.subheader("Correlation Plot")
            fig_sc, ax_sc = plt.subplots(figsize=(7.5, 7.0))
            for g in ["HiHi", "HiLo", "LoHi", "LoLo"]:
                mask = merged["group"] == g
                if mask.sum() > 0:
                    ax_sc.scatter(
                        merged.loc[mask, "expression_1"],
                        merged.loc[mask, "expression_2"],
                        label=g,
                        alpha=0.6,
                        c=color_map.get(g),
                        edgecolors="w",
                    )
            ax_sc.axvline(c1, color="gray", linestyle="--")
            ax_sc.axhline(c2, color="gray", linestyle="--")
            corr = merged[["expression_1", "expression_2"]].corr().iloc[0, 1]
            ax_sc.set_title(f"r = {corr:.3f}")
            ax_sc.set_xlabel(gene1)
            ax_sc.set_ylabel(gene2)
            ax_sc.legend()
            st.pyplot(fig_sc)
            
            buf_sc = io.BytesIO()
            fig_sc.savefig(buf_sc, format="pdf", bbox_inches="tight")
            st.download_button(
                label="Download Scatter Plot (PDF)",
                data=buf_sc.getvalue(),
                file_name=f"Scatter_plot_{gene1}_{gene2}.pdf",
                mime="application/pdf",
                key=f"dl_sc_plot_{dataset_name}"
            )

        st.markdown("---")
        st.subheader("Group Background Check")

        col1, col2 = st.columns(2)

        with col1:
            fig_stage = plot_stage_distribution(merged)
            if fig_stage:
                st.pyplot(fig_stage, use_container_width=True)

            fig_age = plot_age_distribution(merged)
            if fig_age:
                st.pyplot(fig_age, use_container_width=True)

        with col2:
            fig_sex = plot_gender_distribution(merged)
            if fig_sex:
                st.pyplot(fig_sex, use_container_width=True)

            fig_fu = plot_followup_distribution(merged)
            if fig_fu:
                st.pyplot(fig_fu, use_container_width=True)

        st.markdown("### Missingness by Group")
        df_miss = compute_missing_table(merged)
        st.dataframe(df_miss, use_container_width=True)

        # ----------------------------------------------------
        # 1.4 Target Druggability & Safety Profile (Dual)
        # ----------------------------------------------------
        st.markdown("---")
        use_mock = deps.get("use_mock", True)
        title_suffix = "(Mock)" if use_mock else "(Live API)"
        st.subheader(f"💊 Target Profiling & Druggability {title_suffix}")

        if use_mock:
            st.markdown(f"Assess whether **{gene1}** and **{gene2}** possess characteristics suitable for drug development or combination therapy. *Note: Data is mock/simulated for demonstration.*")
        else:
            st.markdown(f"Assess whether **{gene1}** and **{gene2}** possess characteristics suitable for drug development or combination therapy. *Data fetched from OpenTargets API.*")

        try:
            from src.druggability import get_target_profile
            p1 = get_target_profile(gene1, use_mock=use_mock)
            p2 = get_target_profile(gene2, use_mock=use_mock)

            col_g1, col_g2 = st.columns(2)

            def render_gene_profile(gene_sym, prof):
                st.markdown(f"#### {gene_sym} Profile")
                st.markdown("**Modality**")
                st.write(prof.get("modality", "Unknown"))
                st.markdown("**Subcellular Location**")
                st.write(prof.get("location", "Unknown"))

                c1, c2 = st.columns(2)
                ess_score = prof.get("essentiality_prob", 0)
                tox_score = prof.get("toxicity_risk", 0)
                c1.metric("CRISPR Ess.", f"{ess_score:.0%}")
                c2.metric("Tissue Tox.", f"{tox_score}/100")

                ov_score = prof.get("overall_score", 0)
                st.write(f"**Overall Score:** {ov_score}/100")
                st.progress(ov_score / 100.0)

                if prof.get("notes"):
                    st.caption(prof["notes"])

                if ess_score > 0.8:
                    st.warning("⚠️ High Essentiality (Toxicity Risk)")
                elif tox_score > 75:
                    st.warning("⚠️ High Off-Target Toxicity Risk")

            with col_g1:
                render_gene_profile(gene1, p1)
            with col_g2:
                render_gene_profile(gene2, p2)

        except Exception as e:
            st.error(f"Failed to load target profiles: {e}")

        # ----------------------------------------------------
        # 1.45 Known Drugs & Repurposing Opportunities (Dual)
        # ----------------------------------------------------
        st.markdown("---")
        st.subheader(f"⚕️ Known Drugs & Repurposing Opportunities {title_suffix}")

        if use_mock:
            st.markdown(f"Identify existing approved or investigational therapies targeting **{gene1}** or **{gene2}** that could be repurposed for this cohort.")
        else:
            st.markdown(f"Identify existing approved or investigational therapies targeting **{gene1}** or **{gene2}** that could be repurposed for this cohort. *Data fetched from OpenTargets API.*")

        try:
            from src.known_drugs import get_known_drugs
            d1_list = get_known_drugs(gene1, use_mock=use_mock)
            d2_list = get_known_drugs(gene2, use_mock=use_mock)

            def style_approved(val):
                color = '#2e7d32' if val == 'Approved' else ''
                font = 'bold' if val == 'Approved' else 'normal'
                return f'color: {color}; font-weight: {font}'

            col_d1, col_d2 = st.columns(2)

            with col_d1:
                st.markdown(f"#### {gene1} Drugs")
                if not d1_list:
                    st.info(f"No known therapies for {gene1}.")
                else:
                    df1 = pd.DataFrame(d1_list)
                    st.dataframe(df1.style.applymap(style_approved, subset=['phase']), use_container_width=True, hide_index=True)

            with col_d2:
                st.markdown(f"#### {gene2} Drugs")
                if not d2_list:
                    st.info(f"No known therapies for {gene2}.")
                else:
                    df2 = pd.DataFrame(d2_list)
                    st.dataframe(df2.style.applymap(style_approved, subset=['phase']), use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Failed to load known drugs: {e}")

        # Stats section

        # Stats section
        st.markdown("---")
        st.subheader("Statistical Significance & Risk Analysis")

        # --- Overall log-rank (4-way) はそのまま ---
        res_all = multivariate_logrank_test(merged["OS_MONTHS"], merged["group"], merged["is_dead"])
        p_all = float(res_all.p_value)

        # --- 4群Cox（同一モデル） ---
        guard_cox4 = _assess_survival_guardrails(
            merged,
            duration_col="OS_MONTHS",
            event_col="is_dead",
            group_col="group",
            n_groups=4,
            n_covariates=len(covariates) if covariates else 0,
            model_name="Cox 4-group",
        )
        render_guardrail_box("4-group Cox (same model)", guard_cox4)

        if guard_cox4.get("level") == "DANGER":
            cox4 = {"ok": False, "skipped": True, "reason": "Guardrails: danger (see Help → Guardrails)."}
        else:
            # --- pre-clean for Cox (avoid NaNs) ---
            merged_cox = merged.copy()
            merged_cox["OS_MONTHS"] = pd.to_numeric(merged_cox.get("OS_MONTHS"), errors="coerce")
            merged_cox["is_dead"] = (pd.to_numeric(merged_cox.get("is_dead"), errors="coerce") > 0).astype(int)

            # keep only columns needed for modeling (group + covariates)
            keep_cols = ["OS_MONTHS", "is_dead", "group"] + (covariates or [])
            keep_cols = [c for c in keep_cols if c in merged_cox.columns]
            merged_cox = merged_cox[keep_cols].dropna()
            cov_used = [c for c in (covariates or []) if c in merged_cox.columns]

            cox4 = run_cox_4group(
                merged_cox,
                duration_col="OS_MONTHS",
                event_col="is_dead",
                group_col="group",
                ref_group="LoLo",
                covariates=cov_used,
            )

        # 画面表示（例：HiHi vs LoLoをメインに見せる）
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Overall log-rank p-value (4-way)", metric_p_display(p_all))
            st.caption("4-group comparison (HiHi/HiLo/LoHi/LoLo)")
            st.success("✅ Significant" if p_all < 0.05 else "⚠️ Not Significant")

        with col2:
            if cox4["ok"]:
                if cox4.get("warnings"):
                    for w in cox4["warnings"]:
                        st.warning(w)
                    st.info("Tip: When cutoffs result in empty groups, Continuous + Interaction model below is more reliable.")

                tbl = cox4["table"].copy()
                actual_ref = cox4.get("ref_group", "LoLo")
                # HiHi vs XX を探してメイン表示（なければ先頭）
                pick = tbl[tbl["contrast"].str.startswith("HiHi vs")]
                row = pick.iloc[0] if len(pick) else tbl.iloc[0]

                st.metric(f"Adjusted HR ({row.get('contrast', 'N/A')})", f"{row.get('hr', 0):.2f}")
                st.caption(f"95% CI: [{row.get('hr_l', 0):.2f} – {row.get('hr_u', 0):.2f}]")
                st.caption(f"Cox (4-group, same model) N={cox4['n']}, events={cox4['events']}")
                st.caption(f"'{actual_ref}' is the reference group")

            else:
                st.metric("Adjusted HR (4-group)", "N/A")
                st.caption(cox4["reason"])

        with col3:
            st.write("Sample Counts:")
            st.dataframe(merged["group"].value_counts().to_frame().T, hide_index=True)

        # --- [SYS-26-0020] Analysis Issues & Limitations ---
        all_issues = []
        if cox4.get("issues"):
            all_issues.extend(cox4["issues"])
        # We'll also collect from 'inter' results later after it's defined

        # rmst_time and calculate_rmst are passed as kwargs
        if calculate_rmst:
            rmst_res = calculate_rmst(merged, duration_col="OS_MONTHS", event_col="is_dead", group_col="group", tau=rmst_time if rmst_time > 0 else None)
            if rmst_res["ok"]:
                tau = rmst_res["tau"]
                vals = rmst_res["rmst_values"]
                v_hihi = vals.get("HiHi")
                v_lolo = vals.get("LoLo")

                if v_hihi is not None and v_lolo is not None:
                    diff = v_hihi - v_lolo
                    with st.expander(f"⏲️ Restricted Mean Survival Time (RMST) @ {tau:.1f} Months", expanded=True):
                        st.markdown(
                            "**RMST Highlights**: Useful when KM curves cross. Compares expected survival months within the timeframe between extreme groups (HiHi vs LoLo)."
                        )
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric(f"RMST: HiHi", f"{v_hihi:.1f} mo")
                        if "HiLo" in vals: c2.metric(f"RMST: HiLo", f"{vals.get('HiLo'):.1f} mo")
                        if "LoHi" in vals: c3.metric(f"RMST: LoHi", f"{vals.get('LoHi'):.1f} mo")
                        c4.metric(f"RMST: LoLo", f"{v_lolo:.1f} mo")

                        st.markdown(f"**Difference (HiHi vs LoLo):** {diff:+.1f} Months")
                        st.caption(f"Note: Absolute months gained/lost over {tau:.1f} months. Exact asymptotic p-values for RMST are not natively computed here; use this as an effect size indicator.")

        if get_survival_probability:
            prob_res = get_survival_probability(merged, duration_col="OS_MONTHS", event_col="is_dead", group_col="group", time_points=[36.0, 60.0])
            if prob_res["ok"]:
                pvals = prob_res["prob_values"]
                v36_hihi = pvals.get("HiHi", {}).get(36.0)
                v36_lolo = pvals.get("LoLo", {}).get(36.0)
                v60_hihi = pvals.get("HiHi", {}).get(60.0)
                v60_lolo = pvals.get("LoLo", {}).get(60.0)

                if v36_hihi is not None and v36_lolo is not None:
                    with st.expander("🛡️ Absolute Risk (3/5-Year Survival Rate)", expanded=True):
                        st.markdown("**Kaplan-Meier estimates of survival probability at specific time points.**")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("3-Year Survival: HiHi", f"{v36_hihi*100:.1f}%")
                        c2.metric("3-Year Survival: LoLo", f"{v36_lolo*100:.1f}%")
                        diff36 = (v36_hihi - v36_lolo) * 100
                        c3.metric("Absolute Risk Diff (3-Yr)", f"{diff36:+.1f}%", delta=f"{diff36:+.1f}%", delta_color="normal" if diff36>0 else "inverse")

                        if v60_hihi is not None and v60_lolo is not None:
                            d1, d2, d3 = st.columns(3)
                            d1.metric("5-Year Survival: HiHi", f"{v60_hihi*100:.1f}%")
                            d2.metric("5-Year Survival: LoLo", f"{v60_lolo*100:.1f}%")
                            diff60 = (v60_hihi - v60_lolo) * 100
                            d3.metric("Absolute Risk Diff (5-Yr)", f"{diff60:+.1f}%", delta=f"{diff60:+.1f}%", delta_color="normal" if diff60>0 else "inverse")
        st.markdown("---")
        st.subheader("Continuous + Interaction (Cox) - Synergy Assessment")
        st.caption("Cox: gene1(z) + gene2(z) + gene1×gene2 + covariates. Evaluates additive vs synergistic effects.")

        guard_inter = _assess_survival_guardrails(
            merged,
            duration_col="OS_MONTHS",
            event_col="is_dead",
            group_col="group",
            n_groups=4,
            n_covariates=len(covariates) if covariates else 0,
            model_name="Cox interaction",
        )
        render_guardrail_box("Continuous + Interaction (Cox)", guard_inter)

        # --- pre-clean for interaction Cox models (avoid NaNs) ---
        merged_inter = merged.copy()
        merged_inter["OS_MONTHS"] = pd.to_numeric(merged_inter.get("OS_MONTHS"), errors="coerce")
        merged_inter["is_dead"] = (pd.to_numeric(merged_inter.get("is_dead"), errors="coerce") > 0).astype(int)
        _drop_cols = ["OS_MONTHS", "is_dead", "expression_1", "expression_2"] + [c for c in covariates if c in merged_inter.columns]
        merged_inter = merged_inter.dropna(subset=_drop_cols)

        if guard_inter.get("level") == "DANGER":
            inter = {"ok": False, "skipped": True, "reason": "Guardrails: danger (see Help → Guardrails)."}
        else:
            inter = run_cox_interaction(
                merged_inter,
                duration_col="OS_MONTHS",
                event_col="is_dead",
                x1_col="expression_1",
                x2_col="expression_2",
                covariates=covariates_actual,
                zscore=True,
            )

        if not inter["ok"]:
            st.warning(inter["reason"])
        else:
            forest_df = format_interaction_table_for_forest(
                table=inter["table"],
                gene1_label=gene1,   # UIで選んだ表示名（例: TP53）
                gene2_label=gene2,   # UIで選んだ表示名（例: EGFR）
            )

            fig = plot_interaction_forest(
                forest_df,
                title=f"Interaction Forest: {gene1} & {gene2} (z + interaction)"
            )
            st.pyplot(fig, clear_figure=True)

            p_int = float("nan")
            if forest_df is not None and not forest_df.empty:
                s = forest_df.loc[forest_df["is_interaction"], "p"]
                if len(s) > 0:
                    p_int = float(s.iloc[0])

            if pd.notna(p_int) and p_int >= 0.05:
                st.caption(
                    "Tip: A non-significant interaction term means the effect of the two genes is likely additive (independent), rather than synergistic. "
                    "See Help → Interaction Forest for interpretation and caveats."
                )
            else:
                st.caption("✅ Significant Interaction: This suggests a synergistic (or antagonistic) effect beyond simple addition. See Help → Interaction Forest.")

            if inter.get("issues"):
                all_issues.extend(inter["issues"])
            
            render_issue_panel(all_issues)

            st.caption(f"N={inter['n']}, events={inter['events']}")
            tbl = inter["table"].copy()

            # 表示名を gene symbol に置換（UIの分かりやすさ向上）
            if "label" in tbl.columns:
                tbl["label"] = tbl["label"].astype(str)
                tbl["label"] = (
                    tbl["label"]
                    .str.replace("expression_1", gene1, regex=False)
                    .str.replace("expression_2", gene2, regex=False)
                )

            # interaction行を判定
            def highlight_interaction(row):
                is_inter = str(row.get("term", "")).lower() in ("x1_x2",) or "interaction" in str(row.get("label", "")).lower()
                if is_inter:
                    return ["font-weight: bold; background-color: #fca5a5"] * len(row) if p_int < 0.05 else ["font-weight: bold; background-color: #fff3cd"] * len(row)
                return [""] * len(row)

            styled = (
                tbl.style
                .apply(highlight_interaction, axis=1)
                .format({"hr": "{:.2f}", "hr_l": "{:.2f}", "hr_u": "{:.2f}", "p": "{:.3e}"})
            )

            st.dataframe(styled, use_container_width=True)

            st.write("DEBUG INFO:", "run_ph_test=", bool(run_ph_test), "plot_schoenfeld_residuals=", bool(plot_schoenfeld_residuals), "model_in_inter=", "model" in inter, "model_truthy=", bool(inter.get("model")))

            # PH Diagnostics
            if run_ph_test and plot_schoenfeld_residuals and "model" in inter and inter["model"]:
                # We need the exact training dataframe from the fitted lifelines model:
                df_fit = inter["data"]

                ph_violated_2g = False
                ph_res = run_ph_test(inter["model"], df_fit)
                if ph_res["ok"]:
                    target_p = ph_res["p_values"].get("x1_x2", 1.0)
                    if target_p < 0.05:
                        ph_violated_2g = True

                    with st.expander("🩺 Diagnostics: Proportional Hazards (PH) Assumption"):
                        if target_p < 0.05:
                            st.error(f"⚠️ **PH Assumption Violated (p={target_p:.3e})**: The effect of the interaction term changes significantly over time. Standard Cox Average HR is mathematically invalid here. Please consult the 'Time-Dependent Analysis' module below.")
                        else:
                            st.success(f"✅ **PH Assumption Met (p={target_p:.3f})**: The hazard ratio of the interaction term is relatively constant over time.")

                        fig_res = plot_schoenfeld_residuals(inter["model"], df_fit, "x1_x2", title=f"Schoenfeld Residuals: Interaction ({gene1} x {gene2})")
                        if fig_res:
                            st.pyplot(fig_res)

                        st.caption("A violation of the PH assumption requires advanced time-dependent modeling (e.g. Time-Varying Cox) to properly map when the effect vanishes or reverses.")
                else:
                    st.error(f"PH test failed for interaction model: {ph_res.get('reason')}")

            # ----------------------------------------------------
            # Time-Dependent Analysis (Interaction Effect)
            # ----------------------------------------------------
            run_cox_time_varying = deps.get("run_cox_time_varying")
            plot_hr_over_time = deps.get("plot_hr_over_time")
            get_time_varying_snapshots = deps.get("get_time_varying_snapshots")

            if run_cox_time_varying and plot_hr_over_time and get_time_varying_snapshots and "model" in inter and inter["model"]:
                st.markdown("---")
                st.subheader("⏱️ Time-Dependent Analysis (Interaction Effect)")
                st.info("Robust multi-angle evaluation of non-proportional hazards and effect decay over time for the synergistic interaction.")
                
                if locals().get("ph_violated_2g", False):
                    st.info("🎯 **Recommended:** A Proportional Hazards violation was detected above (Curves likely cross). "
                            "This Time-Varying model is mathematically necessary to correctly interpret the dynamic effect.")

                with st.expander("⏳ Time-Varying Coefficient Cox Model (HR over Time)", expanded=True):
                    st.markdown(
                        f"Explicitly models how the **Interaction Hazard Ratio ({gene1} × {gene2})** changes over time. "
                        "When Proportional Hazards are violated, the standard interaction HR is a misleading average. "
                        "This model mathematically captures whether the true synergy vanishes, grows, or reverses later on."
                    )
                    
                    time_func = st.radio("Time Function Interaction", ["log(time)", "linear(time)"], index=0, horizontal=True, key="tv_2g_func")
                    t_trans = 'log' if 'log' in time_func else 'linear'
                    if t_trans == 'linear':
                        st.caption("Warning: linear(time) gives extreme weight to distant events. log(time) is often more stable.")
                    
                    st.caption("⏳ **Note:** This analysis converts data into an 'episodic' format and may take significantly longer than standard Cox models.")
                    if st.button("Calculate Time-Varying Interaction Curve", key="btn_tv_2g"):
                        with st.spinner("Fitting non-proportional Time-Varying Cox Model..."):
                            df_fit = inter["data"]
                            # Fixed covariates should be everything else in the model EXCEPT the interaction term itself, duration, and event.
                            fixed_covs = [col for col in list(df_fit.columns) if col not in ["OS_MONTHS", "is_dead", "x1_x2"]]
                            
                            tv_res = run_cox_time_varying(
                                df_fit, 
                                duration_col="OS_MONTHS", 
                                event_col="is_dead", 
                                main_var="x1_x2", 
                                covariates=fixed_covs,
                                time_transform=t_trans
                            )
                            
                            if tv_res["ok"]:
                                st.success("✅ Time-Varying Model Fitted Successfully")
                                
                                c_t1, c_t2 = st.columns([2, 1])
                                with c_t1:
                                    fig_tv = plot_hr_over_time(tv_res, title=f"Time-Varying Interaction HR(t) ({gene1} × {gene2})")
                                    st.pyplot(fig_tv)
                                    st.caption("If the curve crosses '1.0' (dotted line), the synergistic interaction **may reverse** into antagonism over time.")
                                with c_t2:
                                    st.markdown("##### Representative HR Snapshots")
                                    df_snaps = get_time_varying_snapshots(tv_res, timepoints=[12, 24, 36, 60])
                                    st.dataframe(df_snaps.style.format({"HR(t)": "{:.2f}", "95% CI Lower": "{:.2f}", "95% CI Upper": "{:.2f}", "Month": "{:.0f}"}), hide_index=True, use_container_width=True)
                                    
                                    p_t = tv_res["p_time"]
                                    st.markdown(f"**Interaction*Time p-value:** `{p_t:.3e}`")
                                    if p_t < 0.05:
                                        st.info("The synergy/antagonism changes significantly over time.")
                                    else:
                                        st.write("The interaction decay is not statistically significant.")
                                        
                                    # Early/Late Summary
                                    if not df_snaps.empty:
                                        st.markdown("##### Phase Summary")
                                        hr_12 = df_snaps.loc[df_snaps["Month"] == 12, "HR(t)"]
                                        hr_late = df_snaps.iloc[-1]["HR(t)"]
                                        m_late = df_snaps.iloc[-1]["Month"]
                                        if len(hr_12) > 0:
                                            st.markdown(f"- **Early Phase (12m):** HR = `{hr_12.values[0]:.2f}`")
                                        st.markdown(f"- **Late Phase ({m_late:.0f}m):** HR = `{hr_late:.2f}`")
                                        
                                        if len(hr_12) > 0 and (hr_12.values[0] - 1.0) * (hr_late - 1.0) < 0:
                                            st.markdown("- **Crossing Detected:** The interaction reverses direction during follow-up.")
                            else:
                                st.error(f"Time-Varying fitting failed: {tv_res.get('reason')}")

        # ----------------------------------------------------
        # Bootstrap Robustness Evaluation (2-Gene Interaction)
        # ----------------------------------------------------
        run_2gene_bootstrap = deps.get("run_2gene_bootstrap")
        plot_bootstrap_results = deps.get("plot_bootstrap_results")
        
        if run_2gene_bootstrap and plot_bootstrap_results:
            st.markdown("---")
            st.subheader("🧪 Bootstrap Robustness Evaluation (Interaction)")
            st.markdown(
                "Repeatedly resample patients (with replacement) to re-evaluate the **Interaction Hazard Ratio** and Log-rank p-value "
                "across hundreds of simulated cohorts. This quantifies whether the discovered synergy is statistically stable or an overfitted artifact."
            )
            
            with st.expander("Evaluate Stability via Bootstrap", expanded=False):
                col_b1, col_b2 = st.columns([1, 2])
                with col_b1:
                    n_iter_str = st.selectbox("Bootstrap Iterations", ["100 (Fast Preview)", "500 (Publication Check)"], index=0, key="n_iter_2g")
                    n_iter = 100 if "100" in n_iter_str else 500
                    trigger_boot = st.button(f"Run {n_iter} Resamples", type="secondary", use_container_width=True, key="btn_boot_2g")
                
                if trigger_boot:
                    with st.spinner(f"Running {n_iter} bootstrap parallel iterations..."):
                        
                        # We need to construct the exact dataframe expected by the Cox model
                        # The interaction model internally creates "Gene1_High", "Gene2_High", and "Interaction"
                        boot_df = merged_inter.copy()
                        # Standardize numeric columns just like run_cox_interaction does
                        if True: # zscore logic is standardized in ui
                            m1 = boot_df["expression_1"].mean()
                            s1 = boot_df["expression_1"].std() or 1e-9
                            boot_df["Gene1_High"] = (boot_df["expression_1"] - m1) / s1
                            
                            m2 = boot_df["expression_2"].mean()
                            s2 = boot_df["expression_2"].std() or 1e-9
                            boot_df["Gene2_High"] = (boot_df["expression_2"] - m2) / s2
                            
                            boot_df["Interaction"] = boot_df["Gene1_High"] * boot_df["Gene2_High"]
                        
                        boot_res = run_2gene_bootstrap(
                            boot_df,
                            time_col="OS_MONTHS",
                            event_col="is_dead",
                            n_iterations=n_iter,
                            n_jobs=-1
                        )
                    
                    if not boot_res["ok"]:
                        st.error(f"Bootstrap failed: {boot_res.get('reason')}")
                    else:
                        st.success(f"**Robustness Score: {boot_res['stability_label']}** (Significant in {boot_res['sig_fraction']:.1%} of resamples)")
                        c_b1, c_b2 = st.columns(2)
                        c_b1.metric("Median Interaction HR", f"{boot_res['hr_median']:.2f}", f"95% Empirical CI: [{boot_res['hr_ci_lower']:.2f} - {boot_res['hr_ci_upper']:.2f}]", delta_color="off")
                        c_b2.metric("Significant Runs (p < 0.05)", f"{boot_res['sig_fraction']:.1%}")
                        
                        fig_boot = plot_bootstrap_results(boot_res, title=f"2-Gene Interaction Robustness")
                        st.pyplot(fig_boot)
                        st.caption("A wide HR distribution spanning across 1.0 (the dotted grey line) indicates high instability.")

        # 4群のHR表を表示（不安定さの可視化）
        if cox4.get("ok", False):
            tbl = cox4.get("table", None)

            if isinstance(tbl, pd.DataFrame) and not tbl.empty:
                st.markdown("#### 4-group Cox (LoLo reference) — all contrasts")
                st.dataframe(
                    tbl.style.format(
                        {"hr": "{:.2f}", "hr_l": "{:.2f}", "hr_u": "{:.2f}", "p": "{:.3e}"}
                    ),
                    use_container_width=True,
                )

                st.subheader("4-group Cox Forest (LoLo reference)")
                fig = plot_forest_contrasts(
                    tbl,  # ✅ これだけでForest描ける
                    title="4-group Cox Forest — LoLo reference",
                    xlabel="Hazard Ratio (HR) with 95% CI",
                    p_digits=3,
                    use_logx=False,
                )
                st.pyplot(fig)

            else:
                st.warning("Cox4 result is OK, but 'table' is missing/empty.")
        else:
            st.warning(f"4-group Cox failed: {cox4.get('reason', 'unknown reason')}")

        # ---- PanCancer bias check (by cancer type) ----
        if "CANCER_TYPE" in merged.columns:
            with st.expander("PanCancer bias check: by cancer type", expanded=False):
                st.caption(
                    "Cancer-type stratification to check whether the signal is driven by specific cancers. "
                    "AutoDiscovery is screening; validate here."
                )

                # 1) per-cancer summary
                df_ct_summary = summarize_by_cancer_type(
                    merged,
                    duration_col="OS_MONTHS",
                    event_col="is_dead",
                    group_col="group",
                    cancer_col="CANCER_TYPE",
                    covariates=covariates_actual,
                    min_n=80,
                    min_events=20,
                )
                if "guardrails_status" in merged.attrs and merged.attrs["guardrails_status"] == "danger":
                    st.warning("Driver heuristic disabled because guardrails=danger.")
                else:
                    # --- "Possibly driven by ..." (conservative heuristic) ---
                    driver_msg = None
                    driver_reason = None

                    df_driver = df_ct_summary.copy()

                    # 候補指標：まずは km_logrank_p（見やすい）、なければ cox_*_p の最小を使う
                    if "km_logrank_p" in df_driver.columns:
                        df_driver["km_logrank_p"] = pd.to_numeric(df_driver["km_logrank_p"], errors="coerce")

                    # 3コントラストの最小p（Cox側の保険）
                    cox_p_cols = [c for c in ["cox_HiHi_p", "cox_HiLo_p", "cox_LoHi_p"] if c in df_driver.columns]
                    if cox_p_cols:
                        for c in cox_p_cols:
                            df_driver[c] = pd.to_numeric(df_driver[c], errors="coerce")
                        df_driver["cox_min_p"] = df_driver[cox_p_cols].min(axis=1, skipna=True)
                    else:
                        df_driver["cox_min_p"] = np.nan

                    # どちらで判定するか：kmが取れていればkm優先、なければcox_min_p
                    score_col = "km_logrank_p" if df_driver["km_logrank_p"].notna().any() else "cox_min_p"

                    # N/events 条件（信頼性担保）
                    df_driver["N"] = pd.to_numeric(df_driver.get("N", np.nan), errors="coerce")
                    df_driver["events"] = pd.to_numeric(df_driver.get("events", np.nan), errors="coerce")
                    df_ok = df_driver.dropna(subset=[score_col, "N", "events"]).copy()
                    df_ok = df_ok[(df_ok["N"] >= 80) & (df_ok["events"] >= 20)]

                    if len(df_ok) >= 2:
                        df_ok = df_ok.sort_values(score_col, ascending=True)
                        best = df_ok.iloc[0]
                        second = df_ok.iloc[1]

                        best_p = float(best[score_col])
                        second_p = float(second[score_col])
                        best_ct = str(best["CANCER_TYPE"])

                        # --- Conservative trigger ---
                        # 1) bestがそこそこ強い  2) 2位との差が十分ある（10倍以上）
                        # 3) 2位も弱すぎるとは限らないが、差が大きい場合だけ“偏りの可能性”として出す
                        if (best_p < 1e-2) and (second_p / max(best_p, 1e-300) >= 10.0):
                            driver_msg = f"Possibly driven by **{best_ct}** (heuristic)"
                            driver_reason = f"{score_col}: best={best_p:.2e}, second={second_p:.2e} (ratio={second_p/max(best_p,1e-300):.1f}×)"
                        else:
                            driver_msg = "No clear single cancer-type driver detected (heuristic)."
                            driver_reason = f"{score_col}: best={best_p:.2e}, second={second_p:.2e} (ratio={second_p/max(best_p,1e-300):.1f}×)"
                    elif len(df_ok) == 1:
                        driver_msg = "No clear driver (only one cancer type passed N/events thresholds)."
                    else:
                        driver_msg = "No clear driver (insufficient per-cancer data after thresholds)."

                    # 表示（過信しない文言）
                    st.info(driver_msg)
                    if driver_reason:
                        st.caption(
                            f"Reason: {driver_reason}. "
                            "This is a screening heuristic, not a causal conclusion."
                        )

                # debug ---
                # st.write("df_ct columns:", df_ct_summary.columns.tolist())
                # st.write("df_ct dtypes:", df_ct_summary.dtypes)
                # st.dataframe(df_ct_summary.head(5), use_container_width=True)
                # debyg end ---

                df_ct_summary = df_ct_summary.copy()
                df_ct_summary = df_ct_summary.replace({None: np.nan})
                df_ct_summary["km_logrank_p"] = pd.to_numeric(df_ct_summary["km_logrank_p"], errors="coerce")

                if df_ct_summary.empty:
                    st.info("CANCER_TYPE not available or no data after filtering.")
                else:
                    st.markdown("### Per-cancer summary (within each cancer type)")
                    st.dataframe(
                        df_ct_summary.style.format({
                            "km_logrank_p": "{:.3e}",
                            "cox_HiHi_vs_LoLo_HR": "{:.2f}",
                            "cox_HiHi_p": "{:.3e}",
                            "cox_HiLo_vs_LoLo_HR": "{:.2f}",
                            "cox_HiLo_p": "{:.3e}",
                            "cox_LoHi_vs_LoLo_HR": "{:.2f}",
                            "cox_LoHi_p": "{:.3e}",
                        }),
                        use_container_width=True,
                    )

                    st.caption(
                        "Interpretation tip: If only one or two cancers show strong signal and others are null, "
                        "the PanCancer result may be driven by those cancers."
                    )

                # 2) stratified Cox across cancers (one model)
                st.markdown("---")
                st.markdown("### Stratified Cox (strata=CANCER_TYPE) — pooled across cancers")
                merged_s = merged.copy()

                # keep needed cols only (avoid unexpected NaNs)
                keep = ["OS_MONTHS", "is_dead", "group", "CANCER_TYPE"] + [c for c in (covariates or []) if c in merged_s.columns]
                merged_s = merged_s[keep].copy()

                strat = run_cox_4group_stratified(
                    merged_s,
                    duration_col="OS_MONTHS",
                    event_col="is_dead",
                    group_col="group",
                    ref_group="LoLo",
                    covariates=[c for c in (covariates or []) if c in merged_s.columns],
                    strata_col="CANCER_TYPE",
                    min_n=200,
                    min_events=50,
                )

                if not strat.get("ok"):
                    st.warning(strat.get("reason", "stratified cox failed"))
                else:
                    st.caption(f"N={strat['n']}, events={strat['events']} (baseline hazards stratified by cancer type)")
                    tbl = strat["table"]
                    st.dataframe(
                        tbl.style.format({"hr": "{:.2f}", "hr_l": "{:.2f}", "hr_u": "{:.2f}", "p": "{:.3e}"}),
                        use_container_width=True,
                    )

        # ---- Cut smoothness scan (4-group log-rank stability) ----
        with st.expander("Cut smoothness scan", expanded=False):
            st.caption("Scan percentile cutoffs (min→max) and plot the stability of 4-group log-rank p-values.")

            fragment_decorator = getattr(st, "fragment", getattr(st, "experimental_fragment", lambda f: f))

            @fragment_decorator
            def _render_cutscan():
                if (not st.session_state.get("analysis_ready")) or (st.session_state.get("analysis_merged") is None):
                    st.info("Run Detailed Analysis first (needed to prepare merged data).")
                else:
                    merged = st.session_state["analysis_merged"]
                    meta = st.session_state.get("analysis_meta") or {}
                    g1 = meta.get("gene1", gene1)
                    g2 = meta.get("gene2", gene2)

                    # If the analysis setting changed, invalidate previous scan result
                    min_pct = st.slider("min percentile", 5, 30, 10, key=f"cutscan_min_{dataset_name}")
                    max_pct = st.slider("max percentile", 40, 80, 60, key=f"cutscan_max_{dataset_name}")
                    step    = st.slider("step", 1, 10, 5, key=f"cutscan_step_{dataset_name}")

                    sig = (
                        dataset_name,
                        str(g1),
                        str(g2),
                        int(st.session_state.get("analysis_th1_pct", 0)),
                        int(st.session_state.get("analysis_th2_pct", 0)),
                        int(min_pct),
                        int(max_pct),
                        int(step),
                    )
                    if st.session_state.get("cutscan_sig") != sig:
                        st.session_state["cutscan_sig"] = sig
                        st.session_state.pop("cutscan_result", None)

                    # Guardrails (based on current 4-way grouping)
                    guard_cutscan = _assess_survival_guardrails(
                        merged,
                        duration_col="OS_MONTHS",
                        event_col="is_dead",
                        group_col="group",
                        n_groups=4,
                        n_covariates=0,
                        model_name="KM (4-way)",
                    )
                    render_guardrail_box("Cut smoothness scan", guard_cutscan)

                    danger_cutscan = guard_cutscan.get("level") == "DANGER"

                    if st.button("Run cutoff scan", key=f"cutscan_run_btn_{dataset_name}", disabled=danger_cutscan):
                        merged_cs = merged[["OS_MONTHS", "is_dead", "expression_1", "expression_2"]].copy()
                        merged_cs["OS_MONTHS"] = pd.to_numeric(merged_cs["OS_MONTHS"], errors="coerce")
                        merged_cs["is_dead"] = (pd.to_numeric(merged_cs["is_dead"], errors="coerce") > 0).astype(int)

                        scan = run_cut_smoothness_scan(
                            merged_cs,
                            duration_col="OS_MONTHS",
                            event_col="is_dead",
                            x1_col="expression_1",
                            x2_col="expression_2",
                            min_pct=min_pct,
                            max_pct=max_pct,
                            step=step,
                        )
                        st.session_state["cutscan_result"] = scan

                    scan = st.session_state.get("cutscan_result")
                    if isinstance(scan, dict) and scan:
                        fig = plot_cut_smoothness_curve(scan, title=f"Cut smoothness: {g1} × {g2}")
                        st.pyplot(fig, clear_figure=True)

                        import io
                        buf = io.BytesIO()
                        fig.savefig(buf, format="pdf", bbox_inches="tight")
                        st.download_button(
                            label="Download Plot as PDF",
                            data=buf.getvalue(),
                            file_name=f"cut_smoothness_scan_{g1}_{g2}.pdf",
                            mime="application/pdf",
                            key=f"download_cutscan_btn_{dataset_name}"
                        )
                    else:
                        st.info("Run cutoff scan to generate the smoothness curve.")
            _render_cutscan()

        # ---- Detailed Exploration (Clinical Features) ----
        plot_expression_by_category = deps.get("plot_expression_by_category")
        if plot_expression_by_category:
            st.markdown("---")
            st.subheader("📊 Expression vs Clinical Features (Detailed Exploration)")
            st.caption(
                "Explore whether the target genes are differentially expressed across different clinical subgroups "
                "(e.g., Stage, Gender, Cancer Type / PanCancer)."
            )
            with st.expander("Show Expression Distribution Plots", expanded=False):
                cat_cols = [c for c in df_clin.columns if c not in ["PATIENT_ID", "OS_MONTHS", "is_dead"]]
                common = [c for c in ["STAGE", "PATHOLOGIC_STAGE", "SEX", "CANCER_TYPE", "CANCER_TYPE_DETAILED"] if c in cat_cols]
                others = [c for c in cat_cols if c not in common]
                cat_options = common + others

                if not cat_options:
                    st.warning("No categorical variables available in the dataset.")
                else:
                    selected_cat = st.selectbox("Select Clinical Variable", options=cat_options, key=f"2gene_cat_select_{dataset_name}")
                    btn_plot_cat = st.button("Generate Boxplots vs " + selected_cat, key=f"2gene_cat_btn_{dataset_name}")

                    if btn_plot_cat:
                        with st.spinner(f"Plotting Expression by {selected_cat}..."):
                            col_g1, col_g2 = st.columns(2)
                            with col_g1:
                                fig_cat1 = plot_expression_by_category(
                                    df=merged, 
                                    expr_col="expression_1", 
                                    category_col=selected_cat,
                                    title=f"{gene1} Expression by {selected_cat}"
                                )
                                if fig_cat1:
                                    st.pyplot(fig_cat1, clear_figure=True)
                                else:
                                    st.warning(f"Could not generate plot for {gene1}.")

                            with col_g2:
                                fig_cat2 = plot_expression_by_category(
                                    df=merged, 
                                    expr_col="expression_2", 
                                    category_col=selected_cat,
                                    title=f"{gene2} Expression by {selected_cat}"
                                )
                                if fig_cat2:
                                    st.pyplot(fig_cat2, clear_figure=True)
                                else:
                                    st.warning(f"Could not generate plot for {gene2}.")

        # ----------------------------------------------------
        # 1.60 Target Triage & Prioritization (Translational Score)
        # ----------------------------------------------------
        st.markdown("---")
        st.subheader("🎯 2-Gene Target Triage & Prioritization")
        st.markdown(
            "Synthesizes biological evidence, druggability, safety, and competitive landscape for the interacting gene pair into a unified translation priority score."
        )
        triage_mode = st.radio(
            "Evaluation Strategy Mode", 
            ["Aligned with Novel Target Discovery (Novelty-First)", "Aligned with Drug Repurposing (Repurposing-First)", "Balanced"],
            index=2,
            horizontal=False,
            key="2gene_triage_mode"
        )
        
        # Map to engine modes
        if "Novel" in triage_mode:
            engine_mode = "Novel-target-first"
        elif "Repurposing" in triage_mode:
            engine_mode = "Repurposing-first"
        else:
            engine_mode = "Balanced"

        if st.button("Generate Joint Triage Dashboard", type="primary"):
            from src.triage import TargetTriageEngine
            import plotly.graph_objects as go
            
            with st.spinner("Synthesizing multi-omics evidence for both genes..."):
                engine = TargetTriageEngine(mode=engine_mode)
                
                # Extract previously calculated data for Interaction
                hr_val = None
                p_val = None
                # Use 'inter' instead of 'interaction_res'
                if 'inter' in locals() and inter.get("ok"):
                    hr_val = inter.get("hr_interaction")
                    p_val = inter.get("p_interaction")
                
                # Basic Drug Info combined for both genes
                has_appr = False
                has_clin = False
                try:
                    from src.known_drugs import get_known_drugs
                    for g in [gene1, gene2]:
                        d_list = get_known_drugs(g, use_mock=use_mock)
                        for d in d_list:
                            if d["phase"] == "Approved":
                                has_appr = True
                            elif "Phase" in d["phase"]:
                                has_clin = True
                except:
                    pass

                # Build Triage Dictionary combining features of Gene1 and Gene2
                # Note: Currently it applies worst-case safety and best-case druggability as an approximation
                triage_input = {
                    "target_symbol": f"{gene1} & {gene2}",
                    "hr": hr_val,
                    "p_value": p_val,
                    "psm_maintained": None, # PSM is not universally defined for continuous interactions yet
                    "time_varying_stable": True if ('target_p' in locals() and target_p >= 0.05) else None,
                    "gsea_hallmark_hit": None, # GSEA not natively supported for 2-Gene yet
                    "accessibility": locals().get("prof_1", {}).get("location") if "prof_1" in locals() else None,
                    "target_class": locals().get("prof_1", {}).get("modality") if "prof_1" in locals() else None,
                    "is_essential": (locals().get("ess1", 0) > 0.8 or locals().get("ess2", 0) > 0.8) if ("ess1" in locals() and "ess2" in locals()) else None,
                    "normal_tissue_expression": "High" if locals().get("tox1", 0) > 75 else "Medium",
                    "known_drugs": {
                        "has_approved": has_appr,
                        "has_clinical": has_clin
                    }
                }
                
                res = engine.evaluate(triage_input)
                
                st.markdown(f"### Joint Interaction Score: **{res['total_score']:.0f} / 100**")
                
                # Status Badge
                if res['priority'] == "High":
                    st.success(f"**Translational Priority:** {res['priority']} ({res['confidence']})")
                elif res['priority'] == "Medium":
                    st.warning(f"**Translational Priority:** {res['priority']} ({res['confidence']})")
                else:
                    st.error(f"**Translational Priority:** {res['priority']} ({res['confidence']})")
                
                colA, colB = st.columns([1.2, 1])
                
                with colA:
                    st.markdown("#### Strengths & Weaknesses")
                    for p in res['pros']:
                        st.markdown(f"✅ {p}")
                    for c in res['cons']:
                        st.markdown(f"⚠️ {c}")
                    
                    if res.get('missing_data', 0) > 0:
                        st.info(f"ℹ️ {res['missing_data']} evidence domains were missing (expected for 2-Gene Interaction models).")

                with colB:
                    # Radar Chart
                    categories = ['Evidence', 'Druggability', 'Safety', f'Strategy<br>({engine_mode})']
                    r_values = [
                        res["breakdown"]["Evidence"]["score"] / res["breakdown"]["Evidence"]["max"] * 100,
                        res["breakdown"]["Druggability"]["score"] / res["breakdown"]["Druggability"]["max"] * 100,
                        res["breakdown"]["Safety"]["score"] / res["breakdown"]["Safety"]["max"] * 100,
                        res["breakdown"]["Translation Strategy"]["score"] / res["breakdown"]["Translation Strategy"]["max"] * 100
                    ]
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