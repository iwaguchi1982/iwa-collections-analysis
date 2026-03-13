# ui/validation_view.py
import streamlit as st
import pandas as pd
from typing import Dict, Any, List
from src.cohort_shift import summarize_cohort_shift_results
from ui.cohort_shift_view import render_cohort_shift_dashboard
from src.analysis_issues import Issue
from ui.analysis_issues_view import render_issue_panel

def render_validation_view(
    *,
    data_root: str,
    dataset_options: List[str],
    current_dataset: str,
    deps: Dict[str, Any]
):
    st.header("🧪 Validation (Cross-Cohort)")
    st.info(
        "**仮説の独立検証**: Discovery（探索）データセットで見つけたしきい値(Percentile)のルールを、"
        "別の独立した Validation（検証）データセットに強制適用し、結果が再現するか（過学習していないか）をテストします。"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        disc_ds = st.selectbox(
            "Discovery Dataset (探索用)",
            options=dataset_options,
            index=dataset_options.index(current_dataset) if current_dataset in dataset_options else 0,
            key="val_disc_ds"
        )
    with col2:
        val_options = ["[Upload Custom CSV/TSV]"] + dataset_options
        val_ds = st.selectbox(
            "Validation Dataset (検証用)",
            options=val_options,
            index=1 if len(dataset_options) > 1 else 0,
            key="val_val_ds"
        )
    with col3:
        gene = st.text_input("Gene Symbol", value="EGFR", key="val_gene").strip().upper()

    upload_clin_file = None
    upload_exp_file = None
    if val_ds == "[Upload Custom CSV/TSV]":
        with st.expander("📂 Custom Dataset Upload Settings", expanded=True):
            st.markdown("""
            **外部データセット (GEO等) のCSV/TSVファイルをアップロードしてください:**
            - **Clinical Data**: `PATIENT_ID`, `OS_MONTHS`, `is_dead` (1=dead, 0=alive) の3列が必須です。
            - **Expression Data**: `PATIENT_ID` と目的の遺伝子名（例:`EGFR`）の列を持つWide形式、または `Hugo_Symbol`と`expression`を持つLong形式。
            """)
            col_u1, col_u2 = st.columns(2)
            with col_u1:
                upload_clin_file = st.file_uploader("Clinical File", type=["csv", "tsv", "txt"], key="val_up_clin")
            with col_u2:
                upload_exp_file = st.file_uploader("Expression File", type=["csv", "tsv", "txt"], key="val_up_exp")

    col4, col5 = st.columns(2)
    with col4:
        transfer_method = st.radio(
            "Transfer Method (バッチエフェクト吸収)",
            ["Percentile 転送 (順位ベース)", "Z-score 転送 (標準化ベース)"],
            index=0,
            key="val_transfer"
        )
    with col5:
        if "Percentile" in transfer_method:
            threshold_val = st.slider(
                "Threshold (Percentile) defined in Discovery",
                10, 90, 50,
                key="val_th_pct"
            )
        else:
            threshold_val = st.slider(
                "Threshold (Z-score) defined in Discovery",
                -3.0, 3.0, 1.0, step=0.1,
                key="val_th_z"
            )

    run_btn = st.button("Run Validation", type="primary", use_container_width=True)
    
    if not run_btn and not st.session_state.get("val_has_run", False):
        return
        
    st.session_state["val_has_run"] = True

    if not gene:
        st.warning("Please enter a gene symbol.")
        return
        
    st.divider()

    # Load data functions from dependencies
    load_dataset_cached = deps["load_dataset_cached"]
    get_gene_exp = deps["get_gene_exp"]
    plot_km = deps["plot_km"]
    run_cox = deps["run_cox"]
    add_covariate_columns = deps["add_covariate_columns"]

    # 1. Load Data
    with st.spinner("Loading datasets..."):
        try:
            df_clin_disc, df_exp_disc, _, _ = load_dataset_cached(disc_ds, data_root)
            
            if val_ds == "[Upload Custom CSV/TSV]":
                if upload_clin_file is None or upload_exp_file is None:
                    st.warning("Please upload both Clinical and Expression files for Custom Validation.")
                    return
                # Parse Custom Clinical
                sep_clin = "," if upload_clin_file.name.endswith(".csv") else "\t"
                df_clin_val = pd.read_csv(upload_clin_file, sep=sep_clin)
                # Base normalization (ensure PATIENT_ID strings)
                if "PATIENT_ID" in df_clin_val.columns:
                    df_clin_val["PATIENT_ID"] = df_clin_val["PATIENT_ID"].astype(str)
                else:
                    first_col = df_clin_val.columns[0]
                    df_clin_val = df_clin_val.rename(columns={first_col: "PATIENT_ID"})
                    df_clin_val["PATIENT_ID"] = df_clin_val["PATIENT_ID"].astype(str)
                
                # Parse Custom Expression
                sep_exp = "," if upload_exp_file.name.endswith(".csv") else "\t"
                df_exp_val = pd.read_csv(upload_exp_file, sep=sep_exp)
                if "PATIENT_ID" not in df_exp_val.columns:
                    df_exp_val = df_exp_val.rename(columns={df_exp_val.columns[0]: "PATIENT_ID"})
                df_exp_val["PATIENT_ID"] = df_exp_val["PATIENT_ID"].astype(str)
                
                # Auto-detect Wide vs Long formats and normalize to Long for compatibility
                if "Hugo_Symbol" not in df_exp_val.columns or "expression" not in df_exp_val.columns:
                    # Look for gene column in Wide format
                    if gene in df_exp_val.columns:
                        long_df = df_exp_val[["PATIENT_ID", gene]].copy()
                        long_df = long_df.rename(columns={gene: "expression"})
                        long_df["Hugo_Symbol"] = gene
                        df_exp_val = long_df
            else:
                df_clin_val, df_exp_val, _, _ = load_dataset_cached(val_ds, data_root)
        except Exception as e:
            st.error(f"Failed to load datasets: {e}")
            return

    # Helper function to process a single cohort
    def _process_cohort(df_c, df_e, ds_name, is_discovery=True):
        exp_data = get_gene_exp(df_e, gene)
        if exp_data is None or exp_data.empty:
            return None, f"Gene '{gene}' not found in {ds_name}."
            
        merged = pd.merge(df_c, exp_data, on="PATIENT_ID")
        merged = add_covariate_columns(merged) # Standardize clinical cols
        
        merged["OS_MONTHS"] = pd.to_numeric(merged["OS_MONTHS"], errors="coerce")
        merged["is_dead"] = (pd.to_numeric(merged["is_dead"], errors="coerce") > 0).astype(int)
        merged["expression"] = pd.to_numeric(merged["expression"], errors="coerce")
        merged = merged.dropna(subset=["OS_MONTHS", "is_dead", "expression"])
        
        if len(merged) < 20 or merged["is_dead"].sum() < 5:
            return None, f"Insufficient survival data in {ds_name}."
            
        mean_exp = merged["expression"].mean()
        std_exp = merged["expression"].std()
        if std_exp == 0 or pd.isna(std_exp):
            std_exp = 1e-9
        merged["z_score"] = (merged["expression"] - mean_exp) / std_exp
        
        if "Percentile" in transfer_method:
            cut_val = merged["expression"].quantile(threshold_val / 100.0)
            merged["is_high"] = (merged["expression"] >= cut_val).astype(int)
            cutoff_label = f"Expression >= {cut_val:.4f}"
        else:
            cut_val = threshold_val
            merged["is_high"] = (merged["z_score"] >= cut_val).astype(int)
            cutoff_label = f"Z-score >= {cut_val:.2f}"
        
        merged["group"] = merged["is_high"].map({1: "High", 0: "Low"})
        
        res = {
            "df": merged,
            "cutoff_label": cutoff_label,
            "cut_val": cut_val,
            "n": len(merged),
            "events": int(merged["is_dead"].sum()),
            "ds_name": ds_name
        }
        return res, None

    res_disc, err_disc = _process_cohort(df_clin_disc, df_exp_disc, disc_ds, is_discovery=True)
    res_val, err_val = _process_cohort(df_clin_val, df_exp_val, val_ds, is_discovery=False)

    if err_disc:
        st.error(f"Discovery Cohort Error: {err_disc}")
        return
    if err_val:
        st.error(f"Validation Cohort Error: {err_val}")
        return

    # Render results side-by-side
    col_d, col_v = st.columns(2)
    
    def _render_results(res, col, title_icon):
        with col:
            st.subheader(f"{title_icon} {res['ds_name']}")
            st.caption(f"Valid N={res['n']}, Events={res['events']} | {res['cutoff_label']}")
            
            fig_km = plot_km(res['df'], group_col="group", title=f"KM: {res['ds_name']}")
            st.pyplot(fig_km)
            
            cox_res = run_cox(
                res['df'], 
                duration_col="OS_MONTHS", 
                event_col="is_dead", 
                main_var="is_high",
                covariates=[] 
            )
            
            if cox_res["ok"]:
                hr = cox_res["hr"]
                p = cox_res["p"]
                
                color = "#2e7d32" if p < 0.05 else "#616161"
                bg_color = "rgba(46, 125, 50, 0.1)" if p < 0.05 else "rgba(97, 97, 97, 0.1)"
                st.markdown(
                    f"<div style='padding: 10px; border-radius: 5px; background-color: {bg_color}; border: 1px solid {color};'>"
                    f"<b>Hazard Ratio (High vs Low):</b> {hr:.2f} (95% CI: {cox_res['hr_l']:.2f}-{cox_res['hr_u']:.2f})<br>"
                    f"<b>Cox p-value:</b> <span style='color:{color}; font-weight:bold;'>{p:.4e}</span>"
                    f"</div>", 
                    unsafe_allow_html=True
                )
            else:
                st.warning(f"Cox model failed: {cox_res.get('reason')}")
            
            return cox_res.get("p", 1.0)

    p_disc = _render_results(res_disc, col_d, "🔍 Discovery:")
    p_val = _render_results(res_val, col_v, "✅ Validation:")

    with st.expander("🔎 Discovery Context & Reliability Check (Multiple Testing & Cut Smoothness)", expanded=False):
        st.markdown(
            "**Multiple Testing Warning**: In the Discovery phase, Auto Discovery tests thousands of genes, and Auto Cutoff scans up to ~80 different percentiles. "
            "Because we perform so many tests on the same Discovery dataset, the *nominal p-value* often grossly overestimates the true significance (False Positives due to Multiple Comparisons). "
            "Validation on a completely untouched dataset is the most robust way to prove the finding isn't merely an artifact of random noise or 'Cutoff Optimization'."
        )
        try:
            from src.survival import run_1gene_cut_smoothness_scan, plot_cut_smoothness_curve
            if res_disc and "df" in res_disc:
                sc = run_1gene_cut_smoothness_scan(res_disc["df"], x_col="expression")
                if sc.get("ok"):
                    fig_sc = plot_cut_smoothness_curve(sc, title="Discovery Cohort: Cutoff Smoothness Scan", xlabel="Percentile Cutoff (%)")
                    st.pyplot(fig_sc, clear_figure=True)
                    st.caption(
                        "**Cut Smoothness**: A reliable biomarker should show a stable 'valley' of significance (p < 0.05) across multiple adjacent cutoffs, rather than a single sharp isolated dip. "
                        "If the effect is only significant at one specific percentile (e.g. exactly 34%), it's highly likely to be overfitted noise."
                    )
        except ImportError:
            pass

    st.divider()
    if p_disc < 0.05 and p_val < 0.05:
        st.success("**Interpretation (再現成功 💮)**: The threshold rule discovered in the Discovery cohort successfully validated (p < 0.05) in the independent Validation cohort. The effect is robust across datasets!")
    elif p_disc < 0.05 and p_val >= 0.05:
        st.warning("**Interpretation (再現失敗 ⚠️)**: The effect was significant in Discovery, but *failed to replicate* in Validation. This suggests the Discovery result might be overfitted (過学習) to its specific cohort characteristics, or that the cohorts differ significantly in baseline biology.")
    elif p_disc >= 0.05:
        st.info("**Interpretation**: The effect wasn't significant in the Discovery cohort to begin with.")

    # --- [SYS-26-0013] Cohort Shift Insight ---
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🔍 Cohort Shift Insight (Recommended to interpret replication failure)", expanded=False):
        st.markdown("*Analyze clinical, molecular, and outcome differences between Discovery and Validation cohorts.*")
        
        # Standard covariates to compare
        std_covariates = ["AGE", "is_male", "stage_num", "MSI", "TMB"]
        cat_covariates = ["is_male", "stage_num"]
        
        with st.spinner("Calculating cohort shift metrics..."):
            try:
                shift_res = summarize_cohort_shift_results(
                    df_disc=res_disc["df"],
                    df_val=res_val["df"],
                    gene_label=gene, # Pass original symbol for display
                    covariates=std_covariates,
                    categorical_covariates=cat_covariates,
                    disc_cutoff_val=res_disc["cut_val"],
                    disc_cutoff_type="Percentile" if "Percentile" in transfer_method else "Z-score",
                    direction="High vs Low", # Default for validation
                    disc_cutoff_percentile=threshold_val if "Percentile" in transfer_method else None
                )
                render_cohort_shift_dashboard(shift_res, df_disc=res_disc["df"], df_val=res_val["df"])
            except Exception as e:
                st.error(f"Failed to generate cohort shift insight: {e}")

    # --- [SYS-26-0020] Analysis Issues & Limitations ---
    all_issues = []
    # 1. Collect from Cohort Shift
    if "shift_res" in locals() and shift_res.get("issues"):
        all_issues.extend(shift_res["issues"])
        
    # 2. Replication Logic-based issues
    if p_disc < 0.05 and p_val >= 0.05:
        all_issues.append(Issue(
            category="sensitivity",
            severity="critical" if p_val > 0.2 else "warning",
            kind="error" if p_val > 0.5 else "caution",
            title="Replication Failure",
            detail=f"Association found in {disc_ds} vanished in {val_ds}.",
            evidence=f"p_disc={p_disc:.3e}, p_val={p_val:.3f}",
            recommendation="Review the Cohort Shift Insight panel above to see if population differences explain the failure.",
            source_module="validation_view"
        ))
        
    render_issue_panel(all_issues)
