"""
Auto Discovery: robust to expression formats (long/wide) and flexible argument names.
Adds smoother progress updates across phases so the UI doesn't look "stuck at 100%".

- Accepts cutoff_pct or rank_threshold_pct (alias)
- Ignores unknown kwargs
- Handles df_exp in long or wide schema
"""

from __future__ import annotations

from typing import Callable, Dict, Any, List, Optional, Tuple
import itertools
import numpy as np
import pandas as pd
from lifelines.statistics import multivariate_logrank_test


_LONG_GENE_COL_CANDIDATES = ["Hugo_Symbol", "HUGO_SYMBOL", "gene", "GENE", "Gene", "symbol", "SYMBOL"]
_LONG_EXPR_COL_CANDIDATES = ["expression", "EXPR", "expr", "value", "VALUE"]
_ID_COL_CANDIDATES = ["PATIENT_ID", "patient_id", "Patient_ID", "sample", "SAMPLE_ID", "Sample", "ID"]


def infer_long_schema(df_exp: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if df_exp is None or df_exp.empty:
        return (None, None, None)

    id_col = next((c for c in _ID_COL_CANDIDATES if c in df_exp.columns), None)
    gene_col = next((c for c in _LONG_GENE_COL_CANDIDATES if c in df_exp.columns), None)
    expr_col = next((c for c in _LONG_EXPR_COL_CANDIDATES if c in df_exp.columns), None)

    if id_col and gene_col and expr_col:
        return (id_col, gene_col, expr_col)
    return (None, None, None)


def list_candidate_genes(df_exp: pd.DataFrame, target_n: int) -> List[str]:
    id_col, gene_col, expr_col = infer_long_schema(df_exp)
    if gene_col is not None:
        genes = df_exp[gene_col].astype(str).dropna().unique().tolist()
        return genes[: int(target_n)]

    exclude = set(_ID_COL_CANDIDATES + ["CANCER_TYPE"])
    gene_cols = [c for c in df_exp.columns if c not in exclude]
    return gene_cols[: int(target_n)]


def get_gene_exp_long_or_wide(df_exp: pd.DataFrame, gene: str) -> Optional[pd.DataFrame]:
    if df_exp is None or df_exp.empty:
        return None

    id_col, gene_col, expr_col = infer_long_schema(df_exp)
    gene = str(gene)

    if gene_col is not None:
        sub = df_exp.loc[df_exp[gene_col].astype(str) == gene, [id_col, expr_col]].copy()
        if sub.empty:
            return None
        sub = sub.rename(columns={id_col: "PATIENT_ID", expr_col: "expression"})
        return sub

    # wide
    idc = next((c for c in _ID_COL_CANDIDATES if c in df_exp.columns), None)
    if idc is None:
        if not isinstance(df_exp.index, pd.RangeIndex):
            tmp = df_exp.copy()
            tmp["PATIENT_ID"] = tmp.index.astype(str)
            df_exp = tmp
            idc = "PATIENT_ID"
        else:
            return None

    if gene not in df_exp.columns:
        return None

    sub = df_exp[[idc, gene]].copy()
    sub = sub.rename(columns={idc: "PATIENT_ID", gene: "expression"})
    return sub


def run_topn_pairing_search(
    df_clin: pd.DataFrame,
    df_exp: pd.DataFrame,
    *,
    target_n: int = 100,
    cutoff_pct: Optional[int] = None,
    rank_threshold_pct: Optional[int] = None,
    show_only_q: bool = True,
    q_cutoff: float = 0.1,
    progress_callback: Optional[Callable[[float], Any]] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Search top-N gene pairs by 4-way log-rank p-value (HiHi/HiLo/LoHi/LoLo).

    Progress:
      - 0.00–0.20: preparing per-gene expression vectors (precompute)
      - 0.20–1.00: scanning gene pairs
    """
    if cutoff_pct is None and rank_threshold_pct is not None:
        cutoff_pct = int(rank_threshold_pct)
    if cutoff_pct is None:
        cutoff_pct = 25

    if progress_callback:
        progress_callback(0.0)

    candidates = list_candidate_genes(df_exp, target_n=int(target_n))
    if len(candidates) < 2:
        if progress_callback:
            progress_callback(1.0)
        return pd.DataFrame()

    df_clin0 = df_clin.copy()
    need_cols = ["PATIENT_ID", "OS_MONTHS", "is_dead"]
    for c in need_cols:
        if c not in df_clin0.columns:
            raise ValueError(f"Clinical missing required column: {c}")

    df_clin0 = df_clin0[need_cols].copy()
    df_clin0["OS_MONTHS"] = pd.to_numeric(df_clin0["OS_MONTHS"], errors="coerce")
    df_clin0["is_dead"] = (pd.to_numeric(df_clin0["is_dead"], errors="coerce") > 0).astype(int)
    df_clin0 = df_clin0.dropna(subset=["OS_MONTHS", "is_dead", "PATIENT_ID"])

    # Precompute gene expression vectors
    gene_to_series: Dict[str, pd.Series] = {}
    total_genes = len(candidates)
    for idx, g in enumerate(candidates, start=1):
        gexp = get_gene_exp_long_or_wide(df_exp, g)
        if gexp is None:
            if progress_callback:
                progress_callback(0.20 * (idx / total_genes))
            continue

        m = pd.merge(df_clin0[["PATIENT_ID"]], gexp, on="PATIENT_ID", how="left")
        s = pd.to_numeric(m["expression"], errors="coerce")
        if s.notna().sum() < 10:
            if progress_callback:
                progress_callback(0.20 * (idx / total_genes))
            continue

        gene_to_series[str(g)] = s

        if progress_callback:
            progress_callback(0.20 * (idx / total_genes))

    genes = list(gene_to_series.keys())
    if len(genes) < 2:
        if progress_callback:
            progress_callback(1.0)
        return pd.DataFrame()

    out_rows: List[Dict[str, Any]] = []
    n_pairs = len(genes) * (len(genes) - 1) // 2
    done = 0
    q = float(cutoff_pct) / 100.0

    for (g1, g2) in itertools.combinations(genes, 2):
        done += 1
        if progress_callback and (done % 25 == 0 or done == n_pairs):
            progress_callback(0.20 + 0.80 * (done / n_pairs))

        x1 = gene_to_series[g1]
        x2 = gene_to_series[g2]

        tmp = df_clin0.copy()
        tmp["x1"] = x1.values
        tmp["x2"] = x2.values
        tmp = tmp.dropna(subset=["x1", "x2", "OS_MONTHS", "is_dead"])
        if tmp.shape[0] < 30:
            continue

        c1 = float(tmp["x1"].quantile(q))
        c2 = float(tmp["x2"].quantile(q))

        g1_hi = tmp["x1"] >= c1
        g2_hi = tmp["x2"] >= c2
        grp = np.where(g1_hi & g2_hi, "HiHi",
              np.where(g1_hi & ~g2_hi, "HiLo",
              np.where(~g1_hi & g2_hi, "LoHi", "LoLo")))
        tmp["group"] = grp

        try:
            lr = multivariate_logrank_test(tmp["OS_MONTHS"], tmp["group"], tmp["is_dead"])
            p = float(lr.p_value)
        except Exception:
            continue

        vc = tmp["group"].value_counts()
        out_rows.append({
            "Gene 1": g1,
            "Gene 2": g2,
            "p-value": p,
            "N_HiHi": int(vc.get("HiHi", 0)),
            "N_HiLo": int(vc.get("HiLo", 0)),
            "N_LoHi": int(vc.get("LoHi", 0)),
            "N_LoLo": int(vc.get("LoLo", 0)),
            "cutoff_pct": int(cutoff_pct),
        })

    df = pd.DataFrame(out_rows)
    if df.empty:
        if progress_callback:
            progress_callback(1.0)
        return df

    # FDR (BH)
    df = df.sort_values("p-value").reset_index(drop=True)
    m = len(df)
    ranks = np.arange(1, m + 1)
    qvals = df["p-value"].values * m / ranks
    qvals = np.minimum.accumulate(qvals[::-1])[::-1]
    df["q-value"] = qvals
    df["FDR_significant(q<0.1)"] = df["q-value"] < float(q_cutoff)

    if show_only_q:
        df = df[df["q-value"] < float(q_cutoff)].reset_index(drop=True)

    if progress_callback:
        progress_callback(1.0)

    return df
