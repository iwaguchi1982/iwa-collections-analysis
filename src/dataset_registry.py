from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict

import pandas as pd
import streamlit as st




CACHE_VERSION = 'transpose_v1'
import numpy as np
from src.config import is_gpu_enabled, is_db_enabled


# ---------------------------
# Expression normalization
# ---------------------------

def _normalize_tcga_expression(df_exp: pd.DataFrame, cancer: str) -> pd.DataFrame:
    """
    Return expression DF in *long* format with columns:
      PATIENT_ID, Hugo_Symbol, expression

    Handles:
      1) long format already: [PATIENT_ID, Hugo_Symbol, expression] (or similar)
      2) wide "patient-rows × gene-cols": [PATIENT_ID, GENE1, GENE2, ...]
      3) transposed "gene-rows × sample-cols": [PATIENT_ID(gene), Entrez_Gene_Id, TCGA-..-01, ...]
         This is the case you hit in PanCancer.
    """
    if df_exp is None or df_exp.empty:
        return df_exp

    df = df_exp.copy()

    # Case 3: gene-rows × sample-cols (TCGA samples appear as columns)
    tcga_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("TCGA-")]

    # ✅ gene-rows × sample-cols: gene column can be PATIENT_ID or Hugo_Symbol
    gene_col = None
    if len(tcga_cols) >= 10:
        if "PATIENT_ID" in df.columns:
            gene_col = "PATIENT_ID"
        elif "Hugo_Symbol" in df.columns:
            gene_col = "Hugo_Symbol"

    if gene_col is not None:
        gene_raw = df[gene_col].astype(str)

        # gene key may be 'BRCA:TP53' etc -> take last token as symbol
        gene = gene_raw.str.split(":", n=1).str[-1]
        gene = gene.replace({"nan": np.nan, "None": np.nan, "": np.nan})

        gene_map = pd.DataFrame({"_gene_key": gene_raw, "Hugo_Symbol": gene})

        df_m = df.drop(columns=[c for c in ["Entrez_Gene_Id"] if c in df.columns]).copy()
        df_m = df_m.rename(columns={gene_col: "_gene_key"})

        long = df_m.melt(id_vars=["_gene_key"], var_name="_sample", value_name="expression")
        long = long.merge(gene_map, on="_gene_key", how="left")

        pid = long["_sample"].astype(str).str.slice(0, 12)  # TCGA-XX-YYYY
        long["PATIENT_ID"] = (cancer + ":" + pid).astype(str)

        long = long.drop(columns=["_gene_key", "_sample"])
        long = long.dropna(subset=["Hugo_Symbol"])
        long["expression"] = pd.to_numeric(long["expression"], errors="coerce")
        return long[["PATIENT_ID", "Hugo_Symbol", "expression"]]

    # Case 1: already long-ish with gene column candidates
    for gene_col in ["Hugo_Symbol", "HUGO_SYMBOL", "gene", "GENE", "symbol", "SYMBOL"]:
        if gene_col in df.columns:
            expr_col = None
            for c in ["expression", "EXPR", "expr", "value", "VALUE"]:
                if c in df.columns:
                    expr_col = c
                    break
            if expr_col is None:
                continue
            if "PATIENT_ID" not in df.columns:
                continue
            out = df[["PATIENT_ID", gene_col, expr_col]].copy()
            out = out.rename(columns={gene_col: "Hugo_Symbol", expr_col: "expression"})
            out["PATIENT_ID"] = (cancer + ":" + out["PATIENT_ID"].astype(str).str.slice(0, 12)).astype(str)
            out["expression"] = pd.to_numeric(out["expression"], errors="coerce")
            return out[["PATIENT_ID", "Hugo_Symbol", "expression"]]

    # Case 2: patient-rows × gene-cols (wide with PATIENT_ID + gene columns)
    if "PATIENT_ID" in df.columns:
        gene_cols = [c for c in df.columns if c != "PATIENT_ID"]
        if len(gene_cols) >= 5:
            wide = df.copy()
            wide["PATIENT_ID"] = (cancer + ":" + wide["PATIENT_ID"].astype(str).str.slice(0, 12)).astype(str)
            long = wide.melt(id_vars=["PATIENT_ID"], var_name="Hugo_Symbol", value_name="expression")
            long["expression"] = pd.to_numeric(long["expression"], errors="coerce")
            return long[["PATIENT_ID", "Hugo_Symbol", "expression"]]

    return df_exp
@dataclass(frozen=True)
class DatasetSpec:
    """A selectable dataset."""
    kind: str  # "TCGA" / "TCGA_PANCAN" / "GEO"(future)
    name: str  # internal name (e.g. folder name or pancan name)
    path: str  # filesystem path to dataset root (for TCGA), "" for pancan
    cancer_type: Optional[str] = None
    members: Optional[Tuple[str, ...]] = None

    @property
    def label(self) -> str:
        if self.kind == "CUSTOM":
            return f"[Custom] {self.name}"
        if self.kind == "TCGA_PANCAN" and self.members:
            return f"PanCancer (TCGA): {self.name}"
        if self.cancer_type:
            return f"TCGA-{self.cancer_type}: {self.name}"
        return self.name


def _normalize_data_root(data_root: Union[str, Path]) -> Path:
    return data_root if isinstance(data_root, Path) else Path(str(data_root))


def _infer_cancer_type_from_folder(folder: str) -> Optional[str]:
    head = folder.split("_", 1)[0].upper()
    if 2 < len(head) <= 6 and head.isalpha() and head.isupper():
        return head
    return None


def list_dataset_specs(data_root: Union[str, Path]) -> List[DatasetSpec]:
    """
    List available datasets under data_root and add PanCancer specs.

    Current behavior:
      - Individual TCGA datasets mirror src.data_io.list_datasets(data_root)
      - PanCancer spec is created by grouping folders that share the same suffix
        (everything after the first underscore), e.g.
          LUAD_tcga_pan_can_atlas_2018
          BRCA_tcga_pan_can_atlas_2018
        -> PanCancer(TCGA): tcga_pan_can_atlas_2018
    """
    from src.data_io import list_datasets  # local import to avoid circulars

    root = _normalize_data_root(data_root)
    roots = list_datasets(str(root))

    specs: List[DatasetSpec] = []
    for r in roots:
        p = str(root / r)
        if r.startswith("Custom_"):
            specs.append(DatasetSpec(kind="CUSTOM", name=r.replace("Custom_", ""), path=p, cancer_type="Custom"))
        else:
            ctype = _infer_cancer_type_from_folder(r)
            specs.append(DatasetSpec(kind="TCGA", name=r, path=p, cancer_type=ctype))

    # group by suffix to avoid mixing different preprocess variants
    groups: Dict[str, List[str]] = {}
    for r in roots:
        parts = r.split("_", 1)
        if len(parts) == 2:
            suffix = parts[1]
            ctype = _infer_cancer_type_from_folder(r)
            if ctype:
                groups.setdefault(suffix, []).append(r)

    for suffix, members in groups.items():
        if len(members) >= 2:
            specs.append(
                DatasetSpec(
                    kind="TCGA_PANCAN",
                    name=suffix,
                    path="",
                    members=tuple(sorted(members)),
                )
            )

    return specs


def get_spec_by_label(specs: List[DatasetSpec], label: str) -> DatasetSpec:
    for s in specs:
        if s.label == label:
            return s
    return specs[0]


@st.cache_data(show_spinner=False)
def _load_tcga_dataset(path: str) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    from src.data_io import load_merged_data  # local import
    
    if is_gpu_enabled():
        pass # Placeholder: In the future, use cuDF: df_clin, df_exp = load_merged_data_cudf(path)

    df_clin, df_exp, df_mut, df_cna = load_merged_data(path)

    # cancer type from folder name (e.g. brca_tcga_pan_can_atlas_2018 -> BRCA)
    ctype = _infer_cancer_type_from_folder(Path(path).name) or "TCGA"

    # clinical: make sure PATIENT_ID exists and normalize to 12-char patient barcode
    df_clin = _ensure_patient_id_column(df_clin, context=f"clinical for {path}").copy()
    df_clin["PATIENT_ID"] = df_clin["PATIENT_ID"].astype(str).str.slice(0, 12)

    # expression: normalize to long (NOTE: for single cancer, DONT add "BRCA:" prefix)
    df_exp_long = _normalize_tcga_expression(df_exp, ctype)

    # If normalize added prefix like "BRCA:TCGA-..-..", strip it for single-cancer datasets
    if not df_exp_long.empty and "PATIENT_ID" in df_exp_long.columns:
        df_exp_long["PATIENT_ID"] = df_exp_long["PATIENT_ID"].astype(str).str.split(":", n=1).str[-1]

    return df_clin, df_exp_long, df_mut, df_cna


@st.cache_data(show_spinner=False)
def _load_custom_dataset(path: str) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    from src.data_io import load_merged_data  # local import
    
    df_clin, df_exp, df_mut, df_cna = load_merged_data(path)

    df_clin = _ensure_patient_id_column(df_clin, context=f"clinical for {path}").copy()
    df_clin["PATIENT_ID"] = df_clin["PATIENT_ID"].astype(str)

    df_exp_long = _normalize_tcga_expression(df_exp, "Custom")
    if not df_exp_long.empty and "PATIENT_ID" in df_exp_long.columns:
        df_exp_long["PATIENT_ID"] = df_exp_long["PATIENT_ID"].astype(str).str.split(":", n=1).str[-1]

    return df_clin, df_exp_long, df_mut, df_cna


def _ensure_patient_id_column(df: pd.DataFrame, *, context: str) -> pd.DataFrame:
    """
    Ensure df has a PATIENT_ID column.
    Supports common variants:
      - PATIENT_ID already present
      - index holds patient ids (non-RangeIndex / named)
      - first column holds ids (common in TSVs)
    """
    if df is None or df.empty:
        return df

    if "PATIENT_ID" in df.columns:
        return df

    # Common alternative column names
    for alt in ("patient_id", "Patient_ID", "PATIENT", "sample", "SAMPLE_ID", "Sample", "ID"):
        if alt in df.columns:
            out = df.copy()
            out = out.rename(columns={alt: "PATIENT_ID"})
            return out

    # If index looks like IDs (not default RangeIndex)
    if not isinstance(df.index, pd.RangeIndex):
        out = df.copy()
        out["PATIENT_ID"] = df.index.astype(str)
        out = out.reset_index(drop=True)
        return out

    # If first column likely is IDs
    first = df.columns[0]
    if first not in (0, "0"):
        s = df[first]
        # heuristic: many strings / contains '-' / ':'
        if s.dtype == object:
            out = df.copy()
            out = out.rename(columns={first: "PATIENT_ID"})
            return out

    # Special case for TCGA expression data with Hugo_Symbol as first column
    if first == "Hugo_Symbol":
        out = df.copy()
        out = out.rename(columns={first: "PATIENT_ID"})
        return out

    raise ValueError(f"PATIENT_ID missing in {context}. Columns={list(df.columns)[:10]}...")


@st.cache_data(show_spinner=False)
def _load_tcga_pancan_dataset(
    member_paths: Tuple[str, ...],
    member_cancers: Tuple[str, ...],
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Load multiple TCGA datasets and concatenate them into a pan-cancer dataset.

    clinical: rows=patients, needs PATIENT_ID
    expression: normalize into long format:
      PATIENT_ID (cancer:patient12), Hugo_Symbol, expression
    """
    clin_list: List[pd.DataFrame] = []
    exp_list: List[pd.DataFrame] = []

    mut_list: List[pd.DataFrame] = []
    cna_list: List[pd.DataFrame] = []

    for path, ctype in zip(member_paths, member_cancers):
        df_clin, df_exp, df_mut, df_cna = _load_tcga_dataset(path)

        # --- clinical ---
        c = _ensure_patient_id_column(df_clin, context=f"clinical for {path}").copy()
        c["CANCER_TYPE"] = ctype
        c["PATIENT_ID"] = (ctype + ":" + c["PATIENT_ID"].astype(str).str.slice(0, 12)).astype(str)
        clin_list.append(c)

        # --- expression ---
        # NOTE: df_exp may be:
        #  - gene×sample (PanCancer: PATIENT_ID column stores gene key, columns are TCGA-... samples)
        #  - patient×gene (wide)
        #  - already long
        e = _normalize_tcga_expression(df_exp, ctype)
        exp_list.append(e)

        if df_mut is not None:
            # Maybe add CANCER_TYPE or Prefix to PATIENT_ID if needed for mutation
            mut_list.append(df_mut)
        if df_cna is not None:
            cna_list.append(df_cna)

    df_clin_all = pd.concat(clin_list, axis=0, ignore_index=True)
    df_exp_all = pd.concat(exp_list, axis=0, ignore_index=True)
    
    df_mut_all = pd.concat(mut_list, axis=0, ignore_index=True) if mut_list else None
    df_cna_all = pd.concat(cna_list, axis=0, ignore_index=True) if cna_list else None

    return df_clin_all, df_exp_all, df_mut_all, df_cna_all


def load_dataset(spec: DatasetSpec) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load (clinical_df, expression_df, mut_df, cna_df) for a dataset spec."""
    if specs.kind == "TCGA":
        return _load_tcga_dataset(spec.path)

    if spec.kind == "TCGA_PANCAN":
        raise NotImplementedError(
            "TCGA_PANCAN needs a helper to resolve member paths. "
            "Use load_dataset_with_root(spec, data_root) instead."
        )

    raise NotImplementedError(f"Dataset kind '{spec.kind}' is not implemented yet.")


def load_dataset_with_root(spec: DatasetSpec, data_root: Union[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Root-aware loader (recommended). Keeps iwa-collections-analysis simple when using PanCancer specs.
    """
    root = _normalize_data_root(data_root)

    if spec.kind == "TCGA":
        return _load_tcga_dataset(spec.path)

    if spec.kind == "CUSTOM":
        return _load_custom_dataset(spec.path)

    if spec.kind == "TCGA_PANCAN":
        members = spec.members or ()
        member_paths = tuple(str(root / m) for m in members)
        member_cancers = tuple(_infer_cancer_type_from_folder(m) or "NA" for m in members)
        return _load_tcga_pancan_dataset(member_paths, member_cancers)

    raise NotImplementedError(f"Dataset kind '{spec.kind}' is not implemented yet.")
