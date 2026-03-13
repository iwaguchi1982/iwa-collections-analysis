# ui/psm_view.py
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns

def match_pairs(treated_df, control_df, score_col, caliper=0.2):
    """
    Perform greedy 1:1 matching without replacement based on propensity scores.
    """
    treated = treated_df.copy().reset_index()
    control = control_df.copy().reset_index()
    
    # Randomize order to avoid sorting bias in greedy matching
    treated = treated.sample(frac=1, random_state=42).reset_index(drop=True)
    control = control.sample(frac=1, random_state=42).reset_index(drop=True)
    
    matched_control_indices = set()
    matches = []
    
    caliper_val = caliper * pd.concat([treated[score_col], control[score_col]]).std()
    
    for idx, t_row in treated.iterrows():
        t_score = t_row[score_col]
        
        # Find available controls
        available_controls = control[~control.index.isin(matched_control_indices)]
        
        if available_controls.empty:
            break
            
        # Calculate distances
        distances = np.abs(available_controls[score_col] - t_score)
        
        # Find closest
        min_idx = distances.idxmin()
        min_dist = distances.loc[min_idx]
        
        if min_dist <= caliper_val:
            matches.append({
                "treated_index": t_row["index"],
                "control_index": control.loc[min_idx, "index"],
                "distance": min_dist
            })
            matched_control_indices.add(min_idx)
            
    return pd.DataFrame(matches)

def plot_ps_distribution(df, score_col, group_col, title="Propensity Score Distribution"):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.kdeplot(data=df, x=score_col, hue=group_col, common_norm=False, fill=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Propensity Score")
    plt.tight_layout()
    return fig

def render_psm_view(
    *,
    dataset_name: str,
    df_clin: pd.DataFrame,
    df_exp: pd.DataFrame,
    deps: Dict[str, Any]
):
    st.header(f"⚖️ Propensity Score Matching (Beta) - {dataset_name}")
    st.info(
        "**究極の交絡調整**: 患者背景（年齢、性別、ステージなど）をロジスティック回帰で加味して「傾向スコア（Propensity Score）」を算出し、"
        "そのスコアが近い High 群 と Low 群 の患者を 1対1 で抽出（マッチング）します。"
        "これにより、RCT（ランダム化比較試験）に近い擬似的な集団を作成して生存曲線を比較します。"
    )

    get_gene_exp = deps["get_gene_exp"]
    add_covariate_columns = deps["add_covariate_columns"]
    plot_km = deps["plot_km"]
    run_cox = deps["run_cox"]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        gene = st.text_input("Target Gene", value="EGFR", key="psm_gene").strip().upper()
    with col2:
        threshold_pct = st.slider("High Expression Percentile", 10, 90, 50, key="psm_th")
    with col3:
        caliper = st.slider("Caliper (Standard Deviations)", 0.05, 1.0, 0.2, step=0.05, key="psm_caliper",
                             help="最大許容距離。小さいほど厳密なマッチングになりますが、ペア数（N数）は減ります。")

    st.markdown("### Select Covariates for Matching (交絡因子)")
    
    merged_pre = add_covariate_columns(df_clin.copy())
    
    # Available standard variables
    available_covs = []
    if "AGE" in merged_pre.columns and merged_pre["AGE"].notna().any():
        available_covs.append("AGE")
    if "is_male" in merged_pre.columns and merged_pre["is_male"].notna().any():
        available_covs.append("is_male")
    if "stage_num" in merged_pre.columns and merged_pre["stage_num"].notna().any():
        available_covs.append("stage_num")
    if "MSI" in merged_pre.columns and merged_pre["MSI"].notna().any():
        available_covs.append("MSI")
    if "TMB" in merged_pre.columns and merged_pre["TMB"].notna().any():
        available_covs.append("TMB")
        
    selected_covs = st.multiselect(
        "Match on variables",
        options=available_covs,
        default=available_covs,
        key="psm_covs"
    )
    
    run_btn = st.button("Run Propensity Score Matching", type="primary", use_container_width=True)
    if not run_btn and not st.session_state.get("psm_has_run", False):
        return
        
    st.session_state["psm_has_run"] = True
    
    if not gene or not selected_covs:
        st.warning("Please enter a gene and select at least one covariate.")
        return

    st.divider()
    
    exp_data = get_gene_exp(df_exp, gene)
    if exp_data is None or exp_data.empty:
        st.error(f"Gene '{gene}' not found in dataset.")
        return
        
    merged = pd.merge(merged_pre, exp_data, on="PATIENT_ID")
    
    # Clean data for PSM
    cols_to_keep = ["PATIENT_ID", "OS_MONTHS", "is_dead", "expression"] + selected_covs
    merged = merged[cols_to_keep].copy()
    
    # Drop rows with NAs in essential columns
    merged["OS_MONTHS"] = pd.to_numeric(merged["OS_MONTHS"], errors="coerce")
    merged["is_dead"] = (pd.to_numeric(merged["is_dead"], errors="coerce") > 0).astype(int)
    merged["expression"] = pd.to_numeric(merged["expression"], errors="coerce")
    for cov in selected_covs:
        merged[cov] = pd.to_numeric(merged[cov], errors="coerce")
    
    merged = merged.dropna().copy()
    
    if len(merged) < 20:
        st.error("Insufficient complete data points for PSM.")
        return
        
    # Define Target Group
    cut_val = merged["expression"].quantile(threshold_pct / 100.0)
    merged["is_target"] = (merged["expression"] >= cut_val).astype(int)
    merged["group"] = merged["is_target"].map({1: "High", 0: "Low"})
    
    with st.spinner("Calculating Propensity Scores and Matching..."):
        try:
            # Propensity Score Model
            X = merged[selected_covs].astype(float)
            X = sm.add_constant(X)
            y = merged["is_target"].astype(float)
            
            logit_model = sm.Logit(y, X).fit(disp=False)
            merged["propensity_score"] = logit_model.predict(X)
            
            # Matching
            treated = merged[merged["is_target"] == 1]
            control = merged[merged["is_target"] == 0]
            
            matches = match_pairs(treated, control, "propensity_score", caliper=caliper)
            
        except Exception as e:
            st.error(f"Error during PSM modeling/matching: {e}")
            return
            
    if matches.empty:
         st.warning("No valid matches found with the specified caliper constraint. Try increasing the caliper.")
         return
         
    matched_indices = pd.concat([matches["treated_index"], matches["control_index"]])
    matched_df = merged.loc[matched_indices].copy()
    
    col_u, col_m = st.columns(2)
    
    with col_u:
        st.subheader("Before Matching (Unadjusted)")
        st.caption(f"N = {len(merged)} | High = {len(treated)}, Low = {len(control)}")
        
        fig_ps_u = plot_ps_distribution(merged, "propensity_score", "group", "PS Dist: Before Matching")
        st.pyplot(fig_ps_u)
        
        fig_km_u = plot_km(merged, group_col="group", title="KM: Unmatched")
        st.pyplot(fig_km_u)
        
        cox_u = run_cox(merged, "OS_MONTHS", "is_dead", "is_target", covariates=[])
        if cox_u["ok"]:
            st.markdown(f"**Unadjusted HR:** {cox_u['hr']:.2f} (p={cox_u['p']:.4f})")
            
    with col_m:
        st.subheader("After Matching (PSM Adjusted)")
        st.caption(f"Matched N = {len(matched_df)} (1:1 Pair = {len(matches)})")
        
        fig_ps_m = plot_ps_distribution(matched_df, "propensity_score", "group", "PS Dist: After Matching")
        st.pyplot(fig_ps_m)
        
        fig_km_m = plot_km(matched_df, group_col="group", title="KM: PSM Matched")
        st.pyplot(fig_km_m)
        
        cox_m = run_cox(matched_df, "OS_MONTHS", "is_dead", "is_target", covariates=[])
        if cox_m["ok"]:
             color = "#2e7d32" if cox_m['p'] < 0.05 else "#616161"
             st.markdown(
                f"<div style='padding:10px; border-left:4px solid {color}; background-color:rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.1);'>"
                f"<b>PSM HR: <span style='color:{color};'>{cox_m['hr']:.2f}</span></b> (95% CI: {cox_m['hr_l']:.2f}-{cox_m['hr_u']:.2f})<br>"
                f"<b>PSM p-value: <span style='color:{color};'>{cox_m['p']:.4e}</span></b>"
                f"</div>", unsafe_allow_html=True
             )
    
    # Covariate Balance summary
    st.markdown("---")
    st.subheader("Covariate Balance Check")
    st.caption("Average values of covariates before and after matching. Standardized Mean Difference (SMD) should ideally be < 0.1 after matching.")
    
    # Compare Means
    def calc_smd(df_t, df_c, covs):
         res = []
         for cov in covs:
             m_t = df_t[cov].mean()
             m_c = df_c[cov].mean()
             v_t = df_t[cov].var()
             v_c = df_c[cov].var()
             pool_sd = np.sqrt((v_t + v_c)/2) if (v_t+v_c) > 0 else 1e-9
             smd = abs(m_t - m_c) / pool_sd
             res.append({"Covariate": cov, "Treated Mean": m_t, "Control Mean": m_c, "SMD": smd})
         return pd.DataFrame(res)
         
    smd_pre = calc_smd(treated, control, selected_covs)
    smd_post = calc_smd(matched_df[matched_df["is_target"]==1], matched_df[matched_df["is_target"]==0], selected_covs)
    
    smd_df = pd.DataFrame({"Covariate": selected_covs})
    smd_df["SMD (Unmatched)"] = smd_pre["SMD"].values
    smd_df["SMD (Matched)"] = smd_post["SMD"].values
    
    st.dataframe(smd_df.style.background_gradient(cmap="coolwarm", subset=["SMD (Unmatched)", "SMD (Matched)"], vmin=0, vmax=0.3), use_container_width=True)
