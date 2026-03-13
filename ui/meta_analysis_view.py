# ui/meta_analysis_view.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List

def plot_meta_forest(results_df: pd.DataFrame, title="Meta-Analysis Forest Plot"):
    if results_df.empty:
        return None
        
    df = results_df.dropna(subset=['hr', 'hr_l', 'hr_u']).copy()
    if df.empty:
        return None
        
    # Sort for visual presentation
    df = df[::-1].reset_index(drop=True)
    
    # Calculate figure height dynamically
    fig_height = max(4, len(df) * 0.8)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    
    y_pos = np.arange(len(df))
    
    # Clip extreme HR CIs for plotting purposes to avoid squishing the main plot
    df['plot_hr_l'] = df['hr_l'].clip(lower=0.01)
    df['plot_hr_u'] = df['hr_u'].clip(upper=100)
    
    lower_err = df['hr'] - df['plot_hr_l']
    upper_err = df['plot_hr_u'] - df['hr']
    
    # In matplotlib errorbar, xerr must be positive
    # Some extreme cases might cause floating issues
    lower_err = np.maximum(0, lower_err)
    upper_err = np.maximum(0, upper_err)
    
    # Plot points
    ax.scatter(df['hr'], y_pos, marker='s', color='#1f77b4', s=100, zorder=3)
    
    # Plot error bars
    ax.errorbar(df['hr'], y_pos, xerr=[lower_err, upper_err], fmt='none', color='black', capsize=5, zorder=2)
    
    # Vertical line at HR = 1
    ax.axvline(x=1, color='red', linestyle='--', linewidth=1.5, zorder=1)
    
    ax.set_yticks(y_pos)
    
    yticklabels = []
    for i, row in df.iterrows():
        p_val_str = f"p={row['p']:.2e}" if row['p'] < 0.001 else f"p={row['p']:.4f}"
        sig_star = "*" if row['p'] < 0.05 else ""
        label = f"{row['Dataset']}\nN={row['n']}, E={row['events']} | {p_val_str} {sig_star}"
        yticklabels.append(label)
        
    ax.set_yticklabels(yticklabels)
    
    ax.set_xlabel("Hazard Ratio (Log Scale) with 95% CI")
    ax.set_title(title)
    
    # Log scale is standard for HR forest plots
    ax.set_xscale('log')
    ticks = [0.1, 0.2, 0.5, 1, 2, 5, 10]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    
    # Grid lines to guide the eye
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Set x limits to avoid extreme outliers squishing the plot
    ax.set_xlim(0.1, 10)
    
    plt.tight_layout()
    return fig


def render_meta_analysis_view(
    *,
    data_root: str,
    dataset_options: List[str],
    deps: Dict[str, Any]
):
    st.header("🌐 Meta-Analysis (Forest Plot over Cohorts)")
    st.info(
        "**複数データセットの一括検証**: 指定した条件（遺伝子としきい値）を複数のデータセットに一斉に適用し、"
        "その効果（ハザード比）を一望できる強力な俯瞰ツールです。結果の再現性と広範な臨床的意義（汎がん性）を評価します。"
    )

    # UI Settings
    col1, col2 = st.columns(2)
    with col1:
        gene = st.text_input("Gene Symbol", value="EGFR", key="meta_gene").strip().upper()
        
        selected_datasets = st.multiselect(
            "Select Datasets for Meta-Analysis",
            options=dataset_options,
            default=dataset_options[:min(5, len(dataset_options))],
            key="meta_ds"
        )
        
    with col2:
        transfer_method = st.radio(
            "Threshold Method",
            ["Percentile (各コホート内の順位)", "Z-score (各コホート内の標準化)"],
            index=0,
            key="meta_transfer"
        )
        
        if "Percentile" in transfer_method:
            threshold_val = st.slider("Threshold (Percentile)", 10, 90, 50, key="meta_th_pct")
        else:
            threshold_val = st.slider("Threshold (Z-score)", -3.0, 3.0, 1.0, step=0.1, key="meta_th_z")

    run_btn = st.button("Run Meta-Analysis", type="primary", use_container_width=True)
    
    if not run_btn and not st.session_state.get("meta_has_run", False):
        return
        
    st.session_state["meta_has_run"] = True

    if not gene:
        st.warning("Please enter a gene symbol.")
        return
        
    if not selected_datasets:
        st.warning("Please select at least one dataset.")
        return
        
    st.divider()
    
    load_dataset_cached = deps["load_dataset_cached"]
    get_gene_exp = deps["get_gene_exp"]
    run_cox = deps["run_cox"]
    add_covariate_columns = deps["add_covariate_columns"]

    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, ds_name in enumerate(selected_datasets):
        status_text.text(f"Processing {ds_name}...")
        try:
            df_c, df_e, _, _ = load_dataset_cached(ds_name, data_root)
            
            exp_data = get_gene_exp(df_e, gene)
            if exp_data is None or exp_data.empty:
                results.append({"Dataset": ds_name, "Error": "Gene not found"})
                continue
                
            merged = pd.merge(df_c, exp_data, on="PATIENT_ID")
            merged = add_covariate_columns(merged)
            
            merged["OS_MONTHS"] = pd.to_numeric(merged["OS_MONTHS"], errors="coerce")
            merged["is_dead"] = (pd.to_numeric(merged["is_dead"], errors="coerce") > 0).astype(int)
            merged["expression"] = pd.to_numeric(merged["expression"], errors="coerce")
            merged = merged.dropna(subset=["OS_MONTHS", "is_dead", "expression"])
            
            events = int(merged["is_dead"].sum())
            n = len(merged)
            
            if n < 20 or events < 5:
                results.append({"Dataset": ds_name, "Error": f"Not enough data (N={n}, Events={events})"})
                continue
                
            mean_exp = merged["expression"].mean()
            std_exp = merged["expression"].std()
            if std_exp == 0 or pd.isna(std_exp):
                std_exp = 1e-9
            merged["z_score"] = (merged["expression"] - mean_exp) / std_exp
            
            if "Percentile" in transfer_method:
                cut_val = merged["expression"].quantile(threshold_val / 100.0)
                merged["is_high"] = (merged["expression"] >= cut_val).astype(int)
            else:
                cut_val = threshold_val
                merged["is_high"] = (merged["z_score"] >= cut_val).astype(int)
                
            cox_res = run_cox(
                merged, 
                duration_col="OS_MONTHS", 
                event_col="is_dead", 
                main_var="is_high",
                covariates=[] 
            )
            
            if cox_res["ok"]:
                results.append({
                    "Dataset": ds_name,
                    "n": n,
                    "events": events,
                    "hr": cox_res["hr"],
                    "hr_l": cox_res["hr_l"],
                    "hr_u": cox_res["hr_u"],
                    "p": cox_res["p"],
                    "Error": None
                })
            else:
                results.append({"Dataset": ds_name, "Error": f"Cox model failed: {cox_res.get('reason')}"})
                
        except Exception as e:
             results.append({"Dataset": ds_name, "Error": f"Error: {e}"})
             
        progress_bar.progress((i + 1) / len(selected_datasets))
        
    status_text.empty()
    progress_bar.empty()
    
    # Process Results
    df_res = pd.DataFrame(results)
    
    # Split successful and failed
    if df_res.empty:
        st.warning("No results to display.")
        return
        
    df_success = df_res[df_res["Error"].isnull()].copy()
    df_failed = df_res[df_res["Error"].notnull()].copy()
    
    if not df_success.empty:
        st.subheader(f"Meta-Analysis Forest Plot: {gene}")
        fig = plot_meta_forest(df_success, title=f"Hazard Ratios across {len(df_success)} cohorts (High vs Low)")
        if fig:
            st.pyplot(fig)
        else:
            st.warning("Failed to generate Forest Plot.")
            
        st.divider()
        st.subheader("Results Summary")
        
        # Formatting table
        disp_df = df_success[["Dataset", "n", "events", "hr", "hr_l", "hr_u", "p"]].copy()
        disp_df["HR (95% CI)"] = disp_df.apply(lambda r: f"{r['hr']:.2f} ({r['hr_l']:.2f}-{r['hr_u']:.2f})", axis=1)
        disp_df["P-value"] = disp_df["p"].apply(lambda p: f"{p:.2e}" if p < 0.001 else f"{p:.4f}")
        disp_df["Sig"] = disp_df["p"].apply(lambda p: "✨" if p < 0.05 else "")
        
        disp_df = disp_df[["Dataset", "n", "events", "HR (95% CI)", "P-value", "Sig"]]
        disp_df.columns = ["Cohort", "N", "Events", "HR (95% CI)", "P-value", "Sig"]
        
        st.dataframe(disp_df, use_container_width=True, hide_index=True)
            
        # Simple meta-synthesis logic (counting significant findings)
        sig_count = (df_success["p"] < 0.05).sum()
        cons_hr1 = (df_success["hr"] > 1).sum()
        cons_hr0 = (df_success["hr"] < 1).sum()
        
        st.markdown("### 💡 Synthesis")
        st.write(f"- Analyzed **{len(df_success)}** valid cohorts.")
        st.write(f"- Significant associations (p < 0.05) found in **{sig_count}** cohorts.")
        st.write(f"- Trend: HR > 1 in {cons_hr1} cohorts, HR < 1 in {cons_hr0} cohorts.")
        if sig_count == len(df_success) and len(df_success) > 1:
             st.success("Extremely robust finding: Significant across all selected cohorts!")
    else:
        st.error("No valid survival analysis results could be generated for the selected datasets.")
        
    if not df_failed.empty:
        with st.expander("Show skipped/failed datasets"):
            st.dataframe(df_failed[["Dataset", "Error"]], use_container_width=True, hide_index=True)
