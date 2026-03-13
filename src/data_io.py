# src/data_io.py
import os
import pandas as pd
import streamlit as st

from .preprocess import normalize_clinical
try:
    from .auto_discovery import get_gene_exp_long_or_wide as _get_gene_exp_any
except Exception:
    _get_gene_exp_any = None


CLINICAL_CANDIDATES = [
    "clinical.parquet",            # ✅ 優先: Parquet化されたデータ
    "data_clinical_patient.parquet",
    "data_clinical.parquet",
    "data_clinical_patient.txt",
    "data_clinical_patient.tsv",
    "data_clinical_patient.csv",
    "data_clinical.txt",  # 一部データセット
]

EXPR_CANDIDATES = [
    "expression.parquet",          # ✅ 優先: Parquet化されたデータ
    "data_mrna_seq_v2_rsem.parquet",
    "data_mrna_seq_v2_rsem_zscores_ref_diploid_samples.parquet",
    "data_rna_seq_v2_mrna_median_Zscores.parquet",
    "data_mrna_seq_v2_zscores_ref_all_samples.parquet",
    "data_expression.parquet",
    # よくある順に優先
    "data_mrna_seq_v2_rsem.txt",
    "data_mrna_seq_v2_rsem_zscores_ref_diploid_samples.txt",
    "data_rna_seq_v2_mrna_median_Zscores.txt",
    "data_mrna_seq_v2_zscores_ref_all_samples.txt",
    "data_expression.txt",
]

MUT_CANDIDATES = [
    "mutation.parquet",
    "data_mutations.parquet",
    "data_mutations.txt",
]

CNA_CANDIDATES = [
    "cna.parquet",
    "data_cna.parquet",
    "data_linear_CNA.parquet",
    "data_cna.txt",
    "data_linear_CNA.txt",
]


def list_datasets(data_root: str) -> list[str]:
    if not os.path.isdir(data_root):
        return []
    dirs = []
    for name in sorted(os.listdir(data_root)):
        p = os.path.join(data_root, name)
        if not os.path.isdir(p):
            continue
        # clinical/exprのどちらかが見つかるディレクトリを候補に
        if _find_first_existing(p, CLINICAL_CANDIDATES) and _find_first_existing(p, EXPR_CANDIDATES):
            dirs.append(name)
    return dirs


def _find_first_existing(dirpath: str, filenames: list[str]) -> str | None:
    for fn in filenames:
        p = os.path.join(dirpath, fn)
        if os.path.exists(p):
            return p
    return None


def _read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    df = None
    if ext == ".parquet":
        return pd.read_parquet(path, engine="pyarrow")
    
    # Text fallback with PyArrow engine for speed
    elif ext in [".csv"]:
        try:
            df = pd.read_csv(path, engine="pyarrow", comment="#")
        except Exception:
            df = pd.read_csv(path, comment="#", low_memory=False)
    else: # default: tsv
        try:
            df = pd.read_csv(path, sep="\t", engine="pyarrow", comment="#")
        except Exception:
            df = pd.read_csv(path, sep="\t", comment="#", low_memory=False)

    # Auto-caching to parquet for extremely fast subsequent loads
    pq_path = os.path.splitext(path)[0] + ".parquet"
    if not os.path.exists(pq_path):
        try:
            df.to_parquet(pq_path, engine="pyarrow")
        except Exception:
            pass

    return df


@st.cache_data
def load_merged_data(data_dir: str):
    import concurrent.futures
    """
    Returns (df_clin, df_exp, df_mut, df_cna) where:
      df_clin is normalized to common columns: PATIENT_ID/OS_MONTHS/OS_STATUS/is_dead/AGE/SEX/stage...
      df_exp is raw expression matrix dataframe.
      df_mut is raw mutation dataframe, or None if not available.
      df_cna is raw CNA dataframe, or None if not available.
    """
    clin_path = _find_first_existing(data_dir, CLINICAL_CANDIDATES)
    exp_path = _find_first_existing(data_dir, EXPR_CANDIDATES)
    mut_path = _find_first_existing(data_dir, MUT_CANDIDATES)
    cna_path = _find_first_existing(data_dir, CNA_CANDIDATES)

    if clin_path is None:
        raise FileNotFoundError(f"Clinical file not found in {data_dir}. Expected one of: {CLINICAL_CANDIDATES}")
    if exp_path is None:
        raise FileNotFoundError(f"Expression file not found in {data_dir}. Expected one of: {EXPR_CANDIDATES}")

    def _load_clin():
        df_clin_raw = _read_table(clin_path)
        sample_clin_path = _find_first_existing(data_dir, ["data_clinical_sample.parquet", "data_clinical_sample.txt"])
        if sample_clin_path:
            df_sample_raw = _read_table(sample_clin_path)
            if "PATIENT_ID" in df_clin_raw.columns and "PATIENT_ID" in df_sample_raw.columns:
                cols_to_use = df_sample_raw.columns.difference(df_clin_raw.columns).tolist() + ["PATIENT_ID"]
                df_clin_raw = pd.merge(df_clin_raw, df_sample_raw[cols_to_use], on="PATIENT_ID", how="left")
        return normalize_clinical(df_clin_raw)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        f_clin = executor.submit(_load_clin)
        f_exp = executor.submit(_read_table, exp_path)
        f_mut = executor.submit(_read_table, mut_path) if mut_path else None
        f_cna = executor.submit(_read_table, cna_path) if cna_path else None

        df_clin = f_clin.result()
        df_exp = f_exp.result()
        df_mut = f_mut.result() if f_mut else None
        df_cna = f_cna.result() if f_cna else None

    return df_clin, df_exp, df_mut, df_cna


@st.cache_data
def get_gene_exp(df_exp: pd.DataFrame, gene_symbol: str):
    # ✅ PanCancer(long) / wide 対応の共通実装が使えるならそれを使う
    if _get_gene_exp_any is not None:
        return _get_gene_exp_any(df_exp, gene_symbol)
    """
    Robust gene expression extractor for different matrix formats.
    - Detect gene symbol column (Hugo_Symbol / gene / Gene Symbol ...)
    - Ignore common metadata columns (Entrez_Gene_Id etc.)
    - Return: columns [expression, PATIENT_ID], index=SAMPLE_ID
    """
    gene_cols = ["Hugo_Symbol", "HUGO_SYMBOL", "gene", "GENE", "Gene Symbol", "GENE_SYMBOL"]
    gene_col = None
    for c in gene_cols:
        if c in df_exp.columns:
            gene_col = c
            break
    if gene_col is None:
        # 最低限 cBioPortal の標準である Hugo_Symbol が無いと厳しい
        return None

    # 対象遺伝子行
    row = df_exp[df_exp[gene_col] == gene_symbol]
    if row.empty:
        return None

    row0 = row.iloc[0]

    # サンプル列 = gene_colやEntrez列などの“メタ列”以外
    meta_cols = set([gene_col, "Entrez_Gene_Id", "ENTREZ_GENE_ID", "entrez_gene_id", "Entrez Gene Id"])
    sample_cols = [c for c in df_exp.columns if c not in meta_cols]

    # ただし、先頭に余計な列が混ざるケースがあるので「数値化できる列」に絞る（保険）
    vals = pd.to_numeric(row0[sample_cols], errors="coerce")
    vals = vals.dropna()

    if exp_vals.empty:
        return None

    exp_vals = vals.to_frame(name="expression")
    exp_vals.index.name = "SAMPLE_ID"
    # cBioPortal (TCGA) IDs generally start with 'TCGA-' or contain it. If not, don't truncate.
    idx_str = exp_vals.index.astype(str)
    # Just assign the whole ID. dataset_registry handles TCGA 12-char truncation upstream now.
    exp_vals["PATIENT_ID"] = idx_str
    return exp_vals

@st.cache_data
def get_gene_signature_exp(df_exp: pd.DataFrame, genes: list[str]):
    """
    Calculate a Gene Signature score (mean Z-score) for a list of genes.
    Returns DataFrame: [PATIENT_ID, expression (the score), signature_meta]
    """
    if not genes or df_exp is None or df_exp.empty:
        return None

    valid_exps = []
    found_genes = []
    missing_genes = []

    for g in genes:
        g = g.strip().upper()
        if not g:
            continue
        exp_df = get_gene_exp(df_exp, g)
        if exp_df is not None and not exp_df.empty:
            # Calculate Z-score for this gene across all valid patients
            vals = exp_df["expression"]
            mean_val = vals.mean()
            std_val = vals.std(ddof=0)
            
            if std_val > 0:
                z_scores = (vals - mean_val) / std_val
            else:
                z_scores = pd.Series(0, index=vals.index) # If all values are the same
                
            exp_df["z_score"] = z_scores
            exp_df = exp_df.rename(columns={"z_score": f"z_{g}"})
            valid_exps.append(exp_df[["PATIENT_ID", f"z_{g}"]])
            found_genes.append(g)
        else:
            missing_genes.append(g)

    if not valid_exps:
        return None, {"found": [], "missing": genes}

    # Merge all Z-scores on PATIENT_ID
    merged_z = valid_exps[0]
    for df_z in valid_exps[1:]:
        merged_z = pd.merge(merged_z, df_z, on="PATIENT_ID", how="outer")

    # Calculate mean Z-score per patient
    z_cols = [f"z_{g}" for g in found_genes]
    merged_z["expression"] = merged_z[z_cols].mean(axis=1) # The mean Z-score is the proxy expression

    meta = {
        "found": found_genes,
        "missing": missing_genes
    }
    
    return merged_z[["PATIENT_ID", "expression"]].dropna(), meta
