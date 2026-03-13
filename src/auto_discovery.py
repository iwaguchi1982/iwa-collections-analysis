"""
Auto Discovery: robust to expression formats (long/wide) and flexible argument names.
Adds smoother progress updates across phases so the UI doesn't look "stuck at 100%".

- Accepts cutoff_pct or rank_threshold_pct (alias)
- Ignores unknown kwargs
- Handles df_exp in long or wide schema

Meta:
  The returned DataFrame includes diagnostics in df.attrs["meta"].
"""

from __future__ import annotations

import itertools
import numpy as np
import pandas as pd
from lifelines.statistics import multivariate_logrank_test
from joblib import Parallel, delayed

def _evaluate_pair(g1, g2, df_base, q, cutoff_pct):
    x1 = pd.to_numeric(df_base[g1], errors="coerce")
    x2 = pd.to_numeric(df_base[g2], errors="coerce")
    
    tmp = pd.DataFrame({
        "OS_MONTHS": df_base["OS_MONTHS"],
        "is_dead": df_base["is_dead"],
        "x1": x1,
        "x2": x2,
    }).dropna()
    
    n_tmp = int(tmp.shape[0])
    if n_tmp < 30:
        return None
        
    c1 = float(tmp["x1"].quantile(q))
    c2 = float(tmp["x2"].quantile(q))
    
    g1_hi = tmp["x1"] >= c1
    g2_hi = tmp["x2"] >= c2
    grp = np.where(
        g1_hi & g2_hi, "HiHi",
        np.where(g1_hi & ~g2_hi, "HiLo", np.where(~g1_hi & g2_hi, "LoHi", "LoLo"))
    )
    tmp["group"] = grp
    
    try:
        lr = multivariate_logrank_test(tmp["OS_MONTHS"], tmp["group"], tmp["is_dead"])
        p = float(lr.p_value)
    except Exception:
        return None
        
    vc = tmp["group"].value_counts()
    return {
        "Gene 1": g1,
        "Gene 2": g2,
        "p-value": p,
        "N_HiHi": int(vc.get("HiHi", 0)),
        "N_HiLo": int(vc.get("HiLo", 0)),
        "N_LoHi": int(vc.get("LoHi", 0)),
        "N_LoLo": int(vc.get("LoLo", 0)),
        "cutoff_pct": int(cutoff_pct),
        "n_tmp": n_tmp
    }


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
    _, gene_col, _ = infer_long_schema(df_exp)
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

    # Check if transposed Wide (Genes as Rows, Patients as Columns)
    # This happens when e.g. 'Hugo_Symbol' is a column, but 'expression' is not (samples are the other columns)
    tc_gene_col = next((c for c in _LONG_GENE_COL_CANDIDATES if c in df_exp.columns), None)
    if tc_gene_col is not None and expr_col is None:
        row = df_exp.loc[df_exp[tc_gene_col].astype(str) == gene]
        if row.empty:
            return None
        
        # sample columns are everything except the gene col and other metadata
        meta_cols = set([tc_gene_col, "Entrez_Gene_Id", "ENTREZ_GENE_ID", "entrez_gene_id", "Entrez Gene Id"])
        sample_cols = [c for c in df_exp.columns if c not in meta_cols]
        
        row0 = row.iloc[0]
        vals = pd.to_numeric(row0[sample_cols], errors="coerce").dropna()
        if vals.empty:
            return None
            
        sub = vals.to_frame(name="expression")
        sub.index.name = "PATIENT_ID"
        sub = sub.reset_index()
        sub["PATIENT_ID"] = sub["PATIENT_ID"].astype(str)
        return sub

    # standard wide (Patients as Rows, Genes as Columns)
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


def _attach_meta(df: pd.DataFrame, meta: Dict[str, Any]) -> pd.DataFrame:
    try:
        df.attrs["meta"] = meta
    except Exception:
        pass
    return df


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
    n_jobs: int = 1,
    **kwargs: Any,
) -> pd.DataFrame:
    """Search top-N gene pairs by 4-way log-rank p-value.

    Optimized:
      - Build candidate-gene wide matrix once (PATIENT_ID × genes)
      - Merge clinical only once
    """

    meta: Dict[str, Any] = {
        "target_n_requested": int(target_n),
        "cutoff_pct": None,
        "show_only_q": bool(show_only_q),
        "q_cutoff": float(q_cutoff),
        "schema": None,
        "candidate_genes_found": 0,
        "valid_genes_used": 0,
        "pairs_total": 0,
        "pairs_tested": 0,
        "pairs_with_p": 0,
        "rows_median_per_pair": None,
        "rows_min_per_pair": None,
        "rows_max_per_pair": None,
        "rows_median_per_gene_merge": None,
        "returned_rows_before_filter": 0,
        "returned_rows_after_filter": 0,
        "skipped_reason": None,
    }

    if cutoff_pct is None and rank_threshold_pct is not None:
        cutoff_pct = int(rank_threshold_pct)
    if cutoff_pct is None:
        cutoff_pct = 25
    meta["cutoff_pct"] = int(cutoff_pct)

    if progress_callback:
        progress_callback(0.0)

    id_col, gene_col, expr_col = infer_long_schema(df_exp)
    meta["schema"] = "long" if gene_col is not None else "wide"

    candidates = list_candidate_genes(df_exp, target_n=int(target_n))
    meta["candidate_genes_found"] = int(len(candidates))

    if len(candidates) < 2:
        meta["skipped_reason"] = "too_few_candidate_genes"
        df0 = pd.DataFrame()
        if progress_callback:
            progress_callback(1.0)
        return _attach_meta(df0, meta)

    # --- clinical preparation ---
    df_clin0 = df_clin.copy()
    need_cols = ["PATIENT_ID", "OS_MONTHS", "is_dead"]
    for c in need_cols:
        if c not in df_clin0.columns:
            raise ValueError(f"Clinical missing required column: {c}")

    df_clin0 = df_clin0[need_cols].copy()
    df_clin0["OS_MONTHS"] = pd.to_numeric(df_clin0["OS_MONTHS"], errors="coerce")
    df_clin0["is_dead"] = (pd.to_numeric(df_clin0["is_dead"], errors="coerce") > 0).astype(int)
    df_clin0 = df_clin0.dropna(subset=["OS_MONTHS", "is_dead", "PATIENT_ID"])

    # ==========================================================
    # ✅ NEW: build candidate-gene wide matrix once and merge once
    # ==========================================================
    if progress_callback:
        progress_callback(0.05)

    df_wide: Optional[pd.DataFrame] = None

    if gene_col is not None:
        # long schema: columns like PATIENT_ID / Hugo_Symbol / expression
        # Filter only candidate genes then pivot to wide.
        try:
            sub = df_exp[[id_col, gene_col, expr_col]].copy()
            sub[gene_col] = sub[gene_col].astype(str)
            sub = sub[sub[gene_col].isin([str(g) for g in candidates])]
            sub[expr_col] = pd.to_numeric(sub[expr_col], errors="coerce")

            # If multiple rows per (PATIENT_ID, gene), take mean
            df_wide = (
                sub.pivot_table(index=id_col, columns=gene_col, values=expr_col, aggfunc="mean")
                .reset_index()
                .rename(columns={id_col: "PATIENT_ID"})
            )
        except Exception:
            df_wide = None

    # wide schema fallback:
    # - If df_exp already has PATIENT_ID and candidate genes as columns, just select them
    if df_wide is None and "PATIENT_ID" in df_exp.columns:
        cand_cols = [c for c in candidates if c in df_exp.columns]
        if len(cand_cols) >= 2:
            tmpw = df_exp[["PATIENT_ID"] + cand_cols].copy()
            for c in cand_cols:
                tmpw[c] = pd.to_numeric(tmpw[c], errors="coerce")
            df_wide = tmpw

    # If still None: fall back to old per-gene extraction (rare, but keeps robustness)
    if df_wide is None:
        # fallback: build df_wide by repeated extraction without merge (still only one merge later)
        rows = []
        pid = df_clin0["PATIENT_ID"].drop_duplicates().astype(str)
        df_wide = pd.DataFrame({"PATIENT_ID": pid})
        for idx, g in enumerate(candidates, start=1):
            gexp = get_gene_exp_long_or_wide(df_exp, g)
            if gexp is None or gexp.empty:
                if progress_callback:
                    progress_callback(0.05 + 0.15 * (idx / len(candidates)))
                continue
            # gexp expected: PATIENT_ID, expression
            gg = gexp[["PATIENT_ID", "expression"]].copy()
            gg["expression"] = pd.to_numeric(gg["expression"], errors="coerce")
            gg = gg.rename(columns={"expression": str(g)})
            df_wide = df_wide.merge(gg, on="PATIENT_ID", how="left")
            if progress_callback:
                progress_callback(0.05 + 0.15 * (idx / len(candidates)))

    if progress_callback:
        progress_callback(0.20)

    # Merge once
    df_base = pd.merge(df_clin0, df_wide, on="PATIENT_ID", how="left")

    # Decide usable genes based on non-NA count
    gene_merge_ns: List[int] = []
    genes: List[str] = []
    for g in candidates:
        g = str(g)
        if g not in df_base.columns:
            continue
        n_ok = int(pd.to_numeric(df_base[g], errors="coerce").notna().sum())
        gene_merge_ns.append(n_ok)
        if n_ok >= 10:
            genes.append(g)

    meta["valid_genes_used"] = int(len(genes))
    if gene_merge_ns:
        meta["rows_median_per_gene_merge"] = float(np.median(gene_merge_ns))

    if len(genes) < 2:
        meta["skipped_reason"] = "too_few_valid_genes_after_merge"
        df0 = pd.DataFrame()
        if progress_callback:
            progress_callback(1.0)
        return _attach_meta(df0, meta)

    # ==========================================================
    # Pair scan (Parallelized via joblib)
    # ==========================================================
    n_pairs = len(genes) * (len(genes) - 1) // 2
    meta["pairs_total"] = int(n_pairs)

    q = float(cutoff_pct) / 100.0
    pairs = list(itertools.combinations(genes, 2))
    
    chunk_size = max(1, n_pairs // 10) # Update progress 10 times
    all_results = []
    
    if progress_callback:
        progress_callback(0.20)
        
    for i in range(0, n_pairs, chunk_size):
        chunk = pairs[i:i+chunk_size]
        chunk_res = Parallel(n_jobs=n_jobs, batch_size="auto")(
            delayed(_evaluate_pair)(g1, g2, df_base, q, int(cutoff_pct)) for g1, g2 in chunk
        )
        all_results.extend(chunk_res)
        
        if progress_callback:
            progress_callback(0.20 + 0.70 * min(1.0, (i + chunk_size) / n_pairs))

    out_rows: List[Dict[str, Any]] = []
    pair_ns: List[int] = []
    pairs_tested = 0

    for res in all_results:
        if res is not None:
            pairs_tested += 1
            pair_ns.append(res.pop("n_tmp"))
            out_rows.append(res)

    meta["pairs_tested"] = int(pairs_tested)
    meta["pairs_with_p"] = int(pairs_tested)

    if pair_ns:
        meta["rows_median_per_pair"] = float(np.median(pair_ns))
        meta["rows_min_per_pair"] = int(np.min(pair_ns))
        meta["rows_max_per_pair"] = int(np.max(pair_ns))

    df = pd.DataFrame(out_rows)
    if df.empty:
        meta["skipped_reason"] = "no_pairs_with_valid_p"
        if progress_callback:
            progress_callback(1.0)
        return _attach_meta(df, meta)

    # --- BH (robust float conversion) ---
    df = df.sort_values("p-value").reset_index(drop=True)
    p = pd.to_numeric(df["p-value"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    m = p.size
    ranks = np.arange(1, m + 1, dtype=float)
    qvals = p * m / ranks
    qvals = np.minimum.accumulate(qvals[::-1])[::-1]
    qvals = np.clip(qvals, 0.0, 1.0)

    df["q-value"] = qvals.astype(float)
    df["FDR_significant(q<0.1)"] = df["q-value"] < float(q_cutoff)

    meta["returned_rows_before_filter"] = int(len(df))

    if show_only_q:
        df = df[df["q-value"] < float(q_cutoff)].reset_index(drop=True)

    meta["returned_rows_after_filter"] = int(len(df))
    if show_only_q and meta["returned_rows_after_filter"] == 0:
        meta["skipped_reason"] = "filtered_all_by_q"

    if progress_callback:
        progress_callback(1.0)

    return _attach_meta(df, meta)
