import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gseapy as gp
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor

# Supported libraries
LIBRARY_MAPPING = {
    "Hallmark": "MSigDB_Hallmark_2020",
    "KEGG": "KEGG_2021_Human",
    "Reactome": "Reactome_2022"
}

def run_gsea_prerank(
    df_exp: pd.DataFrame, 
    target_gene: str, 
    gene_set: str = "MSigDB_Hallmark_2020",
    n_jobs: int = 4
) -> Dict[str, Any]:
    """
    Runs threshold-free GSEA (Preranked) by correlating target_gene with all other genes.
    """
    if df_exp is None or df_exp.empty:
        return {"ok": False, "reason": "Expression data is missing or empty."}
        
    try:
        # Check if df is in long format:
        is_long = "Hugo_Symbol" in df_exp.columns and "PATIENT_ID" in df_exp.columns and "expression" in df_exp.columns
        if is_long:
            # Pivot to patient x gene (wide)
            df_wide = df_exp.pivot_table(index='PATIENT_ID', columns='Hugo_Symbol', values='expression', aggfunc='mean')
            if target_gene not in df_wide.columns:
                 return {"ok": False, "reason": f"Target gene '{target_gene}' not found after pivoting long expression data."}
            df_exp_numeric = df_wide
            target_expr = df_exp_numeric[target_gene]
        else:
            # Detect the gene column for wide format
            gene_cols = ["Hugo_Symbol", "HUGO_SYMBOL", "gene", "GENE", "Gene Symbol", "GENE_SYMBOL"]
            gene_col = next((c for c in gene_cols if c in df_exp.columns), None)
            
            if not gene_col:
                # Fallback: assume the index contains the gene symbols if no explicit column
                df_exp_numeric = df_exp.select_dtypes(include=[np.number])
                if target_gene not in df_exp_numeric.index:
                    return {"ok": False, "reason": f"Target gene '{target_gene}' not found in expression data index."}
                target_expr = df_exp_numeric.loc[target_gene]
                df_exp_numeric = df_exp_numeric.T # Make it patient x gene
            else:
                # Gene symbols are in a column, values are in other columns
                if target_gene not in df_exp[gene_col].values:
                    return {"ok": False, "reason": f"Target gene '{target_gene}' not found in expression data."}
                
                # Set the gene column as index and extract numeric data
                df_exp_numeric = df_exp.set_index(gene_col).select_dtypes(include=[np.number])
                target_expr = df_exp_numeric.loc[target_gene]
                df_exp_numeric = df_exp_numeric.T # Make it patient x gene
            
        # Ensure target_expr is a Series (in case of duplicate gene rows)
        if isinstance(target_expr, pd.DataFrame):
            target_expr = target_expr.iloc[0]
            
        target_expr = target_expr.astype(float)
            
        std_val = target_expr.std()
        # Check if the target gene expression has zero variance (all values are identical)
        if std_val == 0 or np.isnan(std_val):
            debug_info = f"shape={target_expr.shape}, std={std_val}, head={target_expr.head(3).to_dict()}"
            return {"ok": False, "reason": f"Target gene '{target_gene}' has zero variance across samples (constant expression). Debug: {debug_info}"}
            
        # 2. Calculate Pearson correlation between target and all other genes
        # Since df_exp_numeric is now patient x gene, we can correlate directly
        # .corrwith correlates each column of the caller with the passed Series.
        if len(df_exp_numeric) < 3:
             return {"ok": False, "reason": "Too few samples (<3) to calculate meaningful correlation."}
             
        corr_series = df_exp_numeric.corrwith(target_expr, method='pearson', drop=True)
        
        # Drop self-correlation and NaNs
        if target_gene in corr_series.index:
            corr_series = corr_series.drop(target_gene)
        corr_series = corr_series.dropna()
        
        if corr_series.empty:
            return {
                "ok": False, 
                "reason": "Could not calculate valid correlations. Expression matrix might be incorrectly formatted or lack variance."
            }
            
        # 3. Create Ranked List for GSEA
        rnk = corr_series.sort_values(ascending=False).reset_index()
        rnk.columns = [0, 1] # gseapy expects exact column positions (Gene, Score)
        
        # 4. Run gseapy.prerank
        pre_res = gp.prerank(
            rnk=rnk,
            gene_sets=gene_set,
            threads=n_jobs,
            min_size=5,
            max_size=1000,
            permutation_num=100, # 100 for speed in UI, normally 1000
            outdir=None,
            seed=42,
            verbose=False
        )
        
        if pre_res.res2d.empty:
            return {"ok": False, "reason": f"No significantly enriched pathways found in {gene_set}."}
            
        # 5. Format Output
        df_res = pre_res.res2d.copy()
        
        # Ensure we just return necessary columns and rename them cleanly
        keep_cols = ['Term', 'ES', 'NES', 'NOM p-val', 'FDR q-val', 'Gene %', 'Lead_genes']
        if not all(c in df_res.columns for c in keep_cols):
             # Fallback if gseapy version changes column names
             available = list(df_res.columns)
             keep_cols = [c for c in keep_cols if c in available]
             
        df_res = df_res[keep_cols]
        df_res = df_res.sort_values(by="NES", ascending=False, key=abs)
        
        return {
            "ok": True,
            "results_df": df_res.reset_index(drop=True),
            "gsea_obj": pre_res,
            "n_genes_ranked": len(rnk),
            "library": gene_set
        }
        
    except Exception as e:
        return {"ok": False, "reason": f"GSEA Prerank failed: {str(e)}"}

def run_ora(
    deg_list: List[str], 
    background_list: Optional[List[str]] = None,
    gene_set: str = "MSigDB_Hallmark_2020"
) -> Dict[str, Any]:
    """
    Runs Over-Representation Analysis (Enrichr API) on a thresholded list of genes.
    """
    if not deg_list:
        return {"ok": False, "reason": "No input genes provided for ORA."}
        
    try:
        # Note: Enrichr expects background as an integer (approx ~20000 for human usually, or specific lists)
        # gseapy.enrichr allows passing background as int or list of genes.
        background = background_list if background_list and len(background_list) > 0 else 20000
        
        enr = gp.enrichr(
            gene_list=deg_list,
            gene_sets=gene_set,
            background=background,
            outdir=None,
            verbose=False
        )
        
        if enr.results.empty:
            return {"ok": False, "reason": f"No significantly enriched pathways found in {gene_set}."}
            
        df_res = enr.results.copy()
        df_res = df_res.sort_values(by="Adjusted P-value", ascending=True)
        
        return {
            "ok": True,
            "results_df": df_res.reset_index(drop=True),
            "library": gene_set,
            "n_input_genes": len(deg_list)
        }
        
    except Exception as e:
        return {"ok": False, "reason": f"ORA (Enrichr) failed: {str(e)}"}

def plot_gsea_enrichment(gsea_res: Dict[str, Any], term: str) -> plt.Figure:
    """
    Plots the classic GSEA Enrichment curve for a specific Term.
    """
    if not gsea_res.get("ok"):
        fig, ax = plt.subplots()
        return fig
        
    pre_res = gsea_res["gsea_obj"]
    
    # Check if term exists
    if term not in pre_res.res2d['Term'].values:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"Term '{term}' not found", ha='center', va='center')
        return fig
        
    from gseapy.plot import gseaplot
    
    # We must extract the figure directly from the returned axes
    # gseaplot officially returns an Axes tuple/list and creates its own figure
    axes = gseaplot(
        rank_metric=pre_res.ranking,
        term=term,
        **pre_res.results[term]
    )
    
    # Extract the figure from the first axis
    if isinstance(axes, (list, tuple)) and len(axes) > 0:
        fig = axes[0].figure
    else:
        fig = axes.figure
        
    return fig

def plot_top_pathways(df_res: pd.DataFrame, top_n: int = 10, mode: str = "GSEA") -> plt.Figure:
    """
    Plots a horizontal bar chart of the top NES scores (GSEA) or -log10(FDR) (ORA).
    """
    fig, ax = plt.subplots(figsize=(7, max(4, top_n * 0.3)))
    
    if mode == "GSEA":
        if "NES" not in df_res.columns:
            return fig
            
        # Top positive and Top negative NES
        df_sorted = df_res.sort_values(by="NES", ascending=False)
        top_pos = df_sorted.head(top_n//2)
        top_neg = df_sorted.tail(top_n//2)
        plot_df = pd.concat([top_pos, top_neg]).drop_duplicates()
        plot_df = plot_df.sort_values("NES", ascending=True)
        
        colors = ['tomato' if val < 0 else 'skyblue' for val in plot_df["NES"]]
        ax.barh(plot_df["Term"], plot_df["NES"], color=colors)
        ax.set_xlabel("Normalized Enrichment Score (NES)")
        ax.set_title(f"Top {len(plot_df)} Pathways by NES")
        
        # Add FDR text labels
        for i, (_, row) in enumerate(plot_df.iterrows()):
            fdr = row.get("FDR q-val", 1.0)
            ax.text(
                row["NES"], 
                i, 
                f" q={fdr:.2e}", 
                va='center', 
                ha='left' if row["NES"] > 0 else 'right',
                fontsize=8,
                alpha=0.7
            )
            
    else: # ORA mode
        if "Adjusted P-value" not in df_res.columns:
            return fig
            
        plot_df = df_res.sort_values("Adjusted P-value").head(top_n).copy()
        plot_df["-log10(FDR)"] = -np.log10(plot_df["Adjusted P-value"] + 1e-10)
        plot_df = plot_df.sort_values("-log10(FDR)", ascending=True)
        
        ax.barh(plot_df["Term"], plot_df["-log10(FDR)"], color="mediumseagreen")
        ax.set_xlabel("-log10(FDR)")
        ax.set_title(f"Top {len(plot_df)} Over-Represented Pathways")
        
    ax.grid(axis='x', linestyle=':', alpha=0.6)
    plt.tight_layout()
    return fig
