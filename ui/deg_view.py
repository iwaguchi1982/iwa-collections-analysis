# ui/deg_view.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, Any

def plot_volcano(deg_df: pd.DataFrame, title="Volcano Plot", fc_cutoff=1.0, p_cutoff=0.05):
    """
    Plots a Volcano Plot from a DEG results DataFrame containing 'logFC' and 'p_value'.
    """
    if deg_df.empty or "logFC" not in deg_df.columns or "p_value" not in deg_df.columns:
        return None
        
    df = deg_df.copy()
    
    # Handling exact zero p-values to avoid log10(0) = -inf
    p_min = df[df["p_value"] > 0]["p_value"].min()
    if pd.isna(p_min):
        p_min = 1e-300
    df["p_value"] = df["p_value"].replace(0, p_min * 0.1)
    df["-log10(p)"] = -np.log10(df["p_value"])

    # Determine colors
    conditions = [
        (df["logFC"] >= fc_cutoff) & (df["p_value"] < p_cutoff),
        (df["logFC"] <= -fc_cutoff) & (df["p_value"] < p_cutoff),
    ]
    choices = ["Up", "Down"]
    df["Significance"] = np.select(conditions, choices, default="Not Sig")
    
    color_map = {"Up": "#d62728", "Down": "#1f77b4", "Not Sig": "#bcbd22"}
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for sig_status in ["Up", "Down", "Not Sig"]:
        subset = df[df["Significance"] == sig_status]
        ax.scatter(subset["logFC"], subset["-log10(p)"], 
                   c=color_map[sig_status], label=sig_status, 
                   alpha=0.7, edgecolors="white", linewidth=0.5)
                   
    # Lines
    ax.axhline(-np.log10(p_cutoff), color="gray", linestyle="--", linewidth=1)
    ax.axvline(fc_cutoff, color="gray", linestyle="--", linewidth=1)
    ax.axvline(-fc_cutoff, color="gray", linestyle="--", linewidth=1)
    
    # Annotate top genes
    top_genes = pd.concat([
        df[df["Significance"] == "Up"].nlargest(5, "-log10(p)"),
        df[df["Significance"] == "Down"].nlargest(5, "-log10(p)")
    ])
    
    for _, row in top_genes.iterrows():
        ax.annotate(row["Gene"], (row["logFC"], row["-log10(p)"]),
                    xytext=(3, 3), textcoords="offset points",
                    fontsize=8, alpha=0.8)

    ax.set_title(title)
    ax.set_xlabel("Log2 Fold Change")
    ax.set_ylabel("-Log10 P-value")
    ax.legend()
    plt.tight_layout()
    return fig

def plot_pca(df_exp: pd.DataFrame, meta_df: pd.DataFrame, group_col: str, title="PCA"):
    """
    Plots a highly simplified 2D PCA using SVD on the expression matrix.
    df_exp: wide format (rows=Patient, cols=Gene)
    meta_df: must have PATIENT_ID and the group_col
    """
    # Merge and align
    merged = pd.merge(meta_df[["PATIENT_ID", group_col]], df_exp, on="PATIENT_ID").dropna(subset=[group_col])
    if merged.empty:
        return None
        
    group_labels = merged[group_col].values
    gene_cols = [c for c in merged.columns if c not in ["PATIENT_ID", group_col]]
    
    if len(gene_cols) < 2:
        return None
        
    # Explicitly cast to float to avoid object array issues
    X = merged[gene_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
    
    # Standardize
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1 # Prevent division by zero
    X_scaled = (X - X_mean) / X_std
    
    # PCA via SVD
    U, s, Vt = np.linalg.svd(X_scaled, full_matrices=False)
    PCs = U * s
    
    # Explained variance roughly
    variance_explained = (s ** 2) / np.sum(s ** 2)
    
    pca_df = pd.DataFrame({
        "PC1": PCs[:, 0],
        "PC2": PCs[:, 1],
        "Group": group_labels
    })
    
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Group", alpha=0.8, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({variance_explained[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({variance_explained[1]*100:.1f}%)")
    plt.tight_layout()
    
    return fig

def render_deg_view(
    *,
    dataset_name: str,
    df_clin: pd.DataFrame,
    df_exp: pd.DataFrame,
    deps: Dict[str, Any]
):
    st.header(f"🧬 Differential Expression (Outcome-based DEG) - {dataset_name}")
    st.info(
        "**結果からの逆探索**: 生存期間（例: 長期生存 vs 短期死亡）に基づいて患者を意図的に2群に分類し、"
        "その群間で発現が有意に異なる「未知の予後関連遺伝子群（ドライバー候補）」を探索します。"
    )
    
    st.subheader("1. Define Outcome Groups (生存結果による患者分類)")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Good Outcome (長期生存群)**")
        good_months = st.number_input("Minimum Survival Months", min_value=12, max_value=240, value=60, help="この期間以上生存した患者。打ち切り（Censored）を含むかどうかも指定可能。")
        good_include_censored = st.checkbox("Include Censored patients in Good Outcome", value=True)
        
    with col2:
        st.markdown("**Poor Outcome (短期死亡群)**")
        poor_months = st.number_input("Maximum Survival Months", min_value=0, max_value=60, value=24, help="この期間未満に死亡した（Event=1）患者。")
        
    # Apply logic
    df = df_clin.copy()
    if "OS_MONTHS" not in df.columns or "is_dead" not in df.columns:
        st.error("Survival data required.")
        return
        
    df["OS_MONTHS"] = pd.to_numeric(df["OS_MONTHS"], errors="coerce")
    df["is_dead"] = pd.to_numeric(df["is_dead"], errors="coerce")
    df = df.dropna(subset=["OS_MONTHS", "is_dead"])
    
    conditions = [
        (df["OS_MONTHS"] >= good_months) & ((df["is_dead"] == 0) | good_include_censored),
        (df["OS_MONTHS"] <= poor_months) & (df["is_dead"] == 1)
    ]
    choices = ["Good Outcome", "Poor Outcome"]
    df["Outcome_Group"] = np.select(conditions, choices, default=pd.NA)
    
    meta_df = df.dropna(subset=["Outcome_Group"])[["PATIENT_ID", "Outcome_Group", "OS_MONTHS", "is_dead"]]
    
    st.write(f"**Group Sizes**: Good = {len(meta_df[meta_df['Outcome_Group'] == 'Good Outcome'])}, Poor = {len(meta_df[meta_df['Outcome_Group'] == 'Poor Outcome'])}")
    
    if len(meta_df[meta_df['Outcome_Group'] == 'Good Outcome']) < 5 or len(meta_df[meta_df['Outcome_Group'] == 'Poor Outcome']) < 5:
        st.warning("各グループに最低5人の患者が必要です。基準（Months）を緩和してください。")
        return
        
    st.divider()
    
    st.subheader("2. Run Differential Expression (DEG)")
    st.write("注：プロトタイプ版では処理速度を優先し、データセットに含まれる遺伝子からランダムにサンプリングした部分集合（Max 2000遺伝子）、または指定したパスウェイの遺伝子リストに対してT検定を行います。")
    
    deg_mode = st.radio("Target Genes", ["Random Sample (Fast Discovery)", "Custom Gene List"], horizontal=True)
    
    custom_genes_text = ""
    if deg_mode == "Custom Gene List":
        custom_genes_text = st.text_area("Gene Symbols (comma or newline separated)", value="EGFR, KRAS, TP53, PTEN, MYC, ERBB2")
        
    run_deg = st.button("Run DEG & PCA", type="primary")
    
    if run_deg:
        with st.spinner("Analyzing Expression..."):
            # Check if df_exp is long format (common in this app)
            is_long = "Hugo_Symbol" in df_exp.columns and "expression" in df_exp.columns
            
            if is_long:
                all_genes = df_exp["Hugo_Symbol"].dropna().unique().tolist()
            else:
                all_genes = [c for c in df_exp.columns if c != "PATIENT_ID"]
            
            if deg_mode == "Random Sample (Fast Discovery)":
                import random
                target_genes = random.sample(all_genes, min(2000, len(all_genes)))
            else:
                import re
                input_genes = [g.strip().upper() for g in re.split(r'[,\n]+', custom_genes_text) if g.strip()]
                target_genes = [g for g in input_genes if g in all_genes]
                
            if not target_genes:
                st.error("No valid genes found in the dataset.")
                return
                
            # Filter expression matrix and convert to wide format if necessary
            if is_long:
                exp_subset_long = df_exp[df_exp["Hugo_Symbol"].isin(target_genes)]
                # Drop duplicates just in case
                exp_subset_long = exp_subset_long.drop_duplicates(subset=["PATIENT_ID", "Hugo_Symbol"])
                exp_wide = exp_subset_long.pivot(index="PATIENT_ID", columns="Hugo_Symbol", values="expression").reset_index()
            else:
                exp_wide = df_exp[["PATIENT_ID"] + target_genes].copy()
                
            # --- Auto Log Transform for Linear Scale Data ---
            numeric_cols = [c for c in exp_wide.columns if c != "PATIENT_ID"]
            
            # 1. Safely cast to numeric, filling NaNs (missing expression) with 0
            for col in numeric_cols:
                exp_wide[col] = pd.to_numeric(exp_wide[col], errors='coerce').fillna(0)
            
            # 2. Fast check if it's likely linear scale (max value > 50)
            sample_max = exp_wide[numeric_cols].max().max()
            if not pd.isna(sample_max) and sample_max > 50:
                # Add offset to avoid log(0) or log(-negative)
                min_val = exp_wide[numeric_cols].min().min()
                offset = abs(min_val) + 1 if min_val < 0 else 1
                # 3. Apply log2 transform
                for col in numeric_cols:
                    exp_wide[col] = np.log2(exp_wide[col] + offset)
            
            merged = pd.merge(meta_df, exp_wide, on="PATIENT_ID")
            
            # Split Data
            good_df = merged[merged["Outcome_Group"] == "Good Outcome"]
            poor_df = merged[merged["Outcome_Group"] == "Poor Outcome"]
            
            results = []
            
            progress_bar = st.progress(0)
            for i, gene in enumerate(target_genes):
                if i % 100 == 0:
                     progress_bar.progress(i / len(target_genes))
                
                # Drop NaNs and ensure numeric
                g_good = pd.to_numeric(good_df[gene], errors='coerce').dropna().values
                g_poor = pd.to_numeric(poor_df[gene], errors='coerce').dropna().values
                
                if len(g_good) < 3 or len(g_poor) < 3:
                    continue
                    
                # Skip if variance is exactly 0 in both groups
                if np.var(g_good) == 0 and np.var(g_poor) == 0:
                     continue
                    
                # We calculate Fold Change as log2(Mean Poor / Mean Good)
                # Since RNA-Seq might be z-score, linear mean could be tricky, 
                # but we'll use a simple mean difference as proxy for logFC if data is already log-transformed
                
                mean_good = np.mean(g_good)
                mean_poor = np.mean(g_poor)
                
                log_fc = mean_poor - mean_good
                
                # T-test
                try:
                    t_stat, p_val = stats.ttest_ind(g_poor, g_good, equal_var=False)
                    if np.isnan(p_val):
                        continue
                except Exception:
                    continue
                
                results.append({
                    "Gene": gene,
                    "Mean_Good": mean_good,
                    "Mean_Poor": mean_poor,
                    "logFC": log_fc,
                    "p_value": p_val
                })
                
            progress_bar.empty()
            
            res_df = pd.DataFrame(results)
            if res_df.empty:
                 st.error("Failed to perform DEG calculation.")
                 return
                 
            # Sorting
            res_df = res_df.sort_values("p_value")
            
            st.success("Analysis Complete!")
            
            tab_voclano, tab_pca, tab_table = st.tabs(["🌋 Volcano Plot", "🧩 PCA Plot", "📋 Top Genes Table"])
            
            with tab_voclano:
                st.markdown("**Volcano Plot (Poor vs Good)**")
                st.caption("Positive logFC (Right) = Highly expressed in **Poor** Outcome group (Bad Genes). Negative logFC (Left) = Highly expressed in **Good** Outcome group (Protective Genes).")
                fig_vol = plot_volcano(res_df, title="DEG Volcano Plot", fc_cutoff=0.5, p_cutoff=0.05)
                if fig_vol:
                    st.pyplot(fig_vol)
                else:
                    st.warning("Failed to generate Volcano Plot.")
                    
            with tab_pca:
                st.markdown("**Principal Component Analysis (PCA)**")
                st.caption("Shows how well the selected genes can separate the Good and Poor outcome groups linearly.")
                fig_pca = plot_pca(merged[["PATIENT_ID"] + target_genes], meta_df, "Outcome_Group")
                if fig_pca:
                    st.pyplot(fig_pca)
                else:
                    st.warning("Failed to generate PCA Plot.")
                    
            with tab_table:
                st.markdown("**Significant Genes List**")
                disp_df = res_df.copy()
                disp_df["p_value"] = disp_df["p_value"].apply(lambda p: f"{p:.4e}")
                disp_df["logFC"] = disp_df["logFC"].round(4)
                st.dataframe(disp_df, use_container_width=True)

            # ----------------------------------------------------
            # 2.5 Over-Representation Analysis (ORA)
            # ----------------------------------------------------
            st.divider()
            st.subheader("🧪 Over-Representation Analysis (ORA)")
            st.markdown(
                "Perform Over-Representation Analysis (ORA) on the significant differentially expressed genes to discover over-represented biological pathways."
            )
            
            run_ora = deps.get("run_ora")
            LIBRARY_MAPPING = deps.get("LIBRARY_MAPPING", {})
            plot_top_pathways = deps.get("plot_top_pathways")
            
            if run_ora and plot_top_pathways:
                with st.expander("Configure & Run ORA (Enrichr)", expanded=True):
                    c_ora1, c_ora2, c_ora3, c_ora4 = st.columns(4)
                    with c_ora1:
                        logfc_th = st.number_input("Absolute logFC Threshold", min_value=0.0, max_value=5.0, value=0.5, step=0.1)
                    with c_ora2:
                        pval_th = st.number_input("P-value Threshold", min_value=0.001, max_value=0.5, value=0.05, step=0.01)
                    with c_ora3:
                        direction = st.selectbox("Direction", ["Both (Up & Down)", "Up Only (Poor Outcome)", "Down Only (Good Outcome)"])
                    with c_ora4:
                        lib_keys = list(LIBRARY_MAPPING.keys()) if LIBRARY_MAPPING else ["Hallmark", "KEGG"]
                        lib_choice = st.selectbox("Pathway Library", lib_keys, index=0)
                        
                    if st.button("Run ORA Pathway Enrichment", type="primary", use_container_width=True):
                        # Filter DEG list
                        if "Up " in direction:
                            sig_df = res_df[(res_df["p_value"] < pval_th) & (res_df["logFC"] > logfc_th)]
                        elif "Down " in direction:
                            sig_df = res_df[(res_df["p_value"] < pval_th) & (res_df["logFC"] < -logfc_th)]
                        else:
                            sig_df = res_df[(res_df["p_value"] < pval_th) & (res_df["logFC"].abs() > logfc_th)]
                            
                        deg_list = sig_df["Gene"].tolist()
                        background_list = res_df["Gene"].tolist()
                        
                        st.write(f"**Input Genes:** {len(deg_list)} (Background universe: {len(background_list)})")
                        
                        if len(deg_list) < 5:
                            st.warning("Too few significant genes to run reliable ORA. Please relax the thresholds.")
                        else:
                            with st.spinner("Running Over-Representation Analysis..."):
                                lib_name = LIBRARY_MAPPING.get(lib_choice, "MSigDB_Hallmark_2020")
                                ora_res = run_ora(deg_list=deg_list, background_list=background_list, gene_set=lib_name)
                                
                                if ora_res.get("ok"):
                                    st.success(f"✅ ORA Completed against `{lib_name}`")
                                    df_ora = ora_res["results_df"]
                                    
                                    # Plot
                                    st.markdown("#### Top Enriched Pathways")
                                    fig_ora = plot_top_pathways(df_ora, top_n=10, mode="ORA")
                                    st.pyplot(fig_ora)
                                    
                                    # Table
                                    st.markdown("#### Enrichment Results Table")
                                    # Enrichr columns: Term, Overlap, P-value, Adjusted P-value, Old P-value, Old Adjusted P-value, Odds Ratio, Combined Score, Genes
                                    # Keep necessary ones
                                    keep_cols = ["Term", "Overlap", "P-value", "Adjusted P-value", "Combined Score", "Genes"]
                                    avail_cols = [c for c in keep_cols if c in df_ora.columns]
                                    disp_ora = df_ora[avail_cols].copy()
                                    
                                    st.dataframe(
                                        disp_ora.style.format({
                                            "P-value": "{:.2e}", 
                                            "Adjusted P-value": "{:.2e}", 
                                            "Combined Score": "{:.1f}"
                                        }),
                                        use_container_width=True,
                                        hide_index=True
                                    )
                                else:
                                    st.error(f"ORA failed: {ora_res.get('reason')}")
