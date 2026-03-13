# ui/sidebar.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import streamlit as st


@dataclass
class SidebarCommon:
    dataset_label: str
    mode_label: str            # "1-Gene (Hi/Lo)" or "2-Genes (4-way)"
    covariates: List[str]
    keep_autodiscovery: bool
    detected_covariates: dict[str, str] = None


@dataclass
class Sidebar2Gene:
    gene1: str
    gene2: str
    use_auto_cutoff: bool
    th1_pct: int
    th2_pct: int

@dataclass
class Sidebar1Gene:
    gene: str
    threshold_pct: int


def render_sidebar_common(
    *, 
    dataset_options: List[str], 
    default_dataset: Optional[str] = None,
    detected_covars: dict[str, str] = None
) -> SidebarCommon:
    st.sidebar.header("Dataset")
    keep_autodiscovery = st.sidebar.checkbox("Keep Auto Discovery results across datasets", value=False)

    dataset_label = st.sidebar.selectbox(
        "Choose dataset",
        options=dataset_options,
        index=(dataset_options.index(default_dataset) if default_dataset in dataset_options else 0),
        key="selected_dataset",
    )

    st.sidebar.header("Analysis Mode")
    mode_label = st.sidebar.radio(
        "",
        options=["1-Gene (Hi/Lo)", "2-Genes (4-way)", "Gene Signature (Multi-Gene)"],
        index=2,
        key="sidebar_mode_label",
    )

    st.sidebar.header("Clinical Covariates (Adjustment)")
    covariates: List[str] = []
    if st.sidebar.checkbox("Adjust for Age", value=False, key="cov_age"):
        covariates.append("age")
    if st.sidebar.checkbox("Adjust for Gender", value=False, key="cov_gender"):
        covariates.append("gender")
    if st.sidebar.checkbox("Adjust for Stage", value=False, key="cov_stage"):
        covariates.append("stage")
    if st.sidebar.checkbox("Adjust for MSI (MANTIS/SENSOR)", value=False, key="cov_msi"):
        covariates.append("msi")
    if st.sidebar.checkbox("Adjust for TMB (Nonsynonymous)", value=False, key="cov_tmb"):
        covariates.append("tmb")
        
    dynamic_covariates = []
    if detected_covars:
        st.sidebar.markdown("**Auto-Detected Covariates:**")
        for disp_name, col_name in detected_covars.items():
            encoded_col = col_name + "_encoded"
            # Encode logic adds _encoded for categorical strings, numeric pass-through
            if st.sidebar.checkbox(f"Adjust for {disp_name}", value=False, key=f"cov_dyn_{col_name}"):
                dynamic_covariates.append(encoded_col)

    st.sidebar.divider()

    return SidebarCommon(
        dataset_label=dataset_label,
        mode_label=mode_label,
        covariates=covariates + dynamic_covariates, # For old API compatibility, or keep separate
        keep_autodiscovery=keep_autodiscovery,
        detected_covariates=detected_covars or {},
    )


def render_sidebar_2gene(*, auto_cutoff: Optional[int] = None) -> Sidebar2Gene:
    # --- defaults (must be before widgets are created) ---
    st.session_state.setdefault("analysis_gene1", "ABCA6")
    st.session_state.setdefault("analysis_gene2", "ABCC6P2")
    st.session_state.setdefault("use_auto_cutoff", False)
    st.session_state.setdefault("analysis_th1_pct", 25)
    st.session_state.setdefault("analysis_th2_pct", 25)

    st.sidebar.subheader("Genes")
    gene1 = st.sidebar.text_input("Gene 1", key="analysis_gene1").strip().upper()
    gene2 = st.sidebar.text_input("Gene 2", key="analysis_gene2").strip().upper()

    st.sidebar.subheader("Thresholds")

    use_auto = st.sidebar.checkbox(
        "Use same percentile cutoff as Auto Discovery",
        key="use_auto_cutoff",
        disabled=auto_cutoff is None,
    )

    # if auto cutoff enabled, update state BEFORE sliders render
    if use_auto and auto_cutoff is not None:
        st.session_state["analysis_th1_pct"] = int(auto_cutoff)
        st.session_state["analysis_th2_pct"] = int(auto_cutoff)

    th1_pct = st.sidebar.slider(
        f"{gene1 or 'Gene1'} Threshold (%)",
        5, 95,
        key="analysis_th1_pct",
        disabled=use_auto,
    )
    th2_pct = st.sidebar.slider(
        f"{gene2 or 'Gene2'} Threshold (%)",
        5, 95,
        key="analysis_th2_pct",
        disabled=use_auto,
    )

    return Sidebar2Gene(
        gene1=gene1,
        gene2=gene2,
        use_auto_cutoff=use_auto,
        th1_pct=int(th1_pct),
        th2_pct=int(th2_pct),
    )


@dataclass
class Sidebar1Gene:
    gene: str
    threshold_pct: int
    use_auto_cutoff: bool

def render_sidebar_1gene() -> Sidebar1Gene:
    st.session_state.setdefault("analysis_gene", "EGFR")
    st.session_state.setdefault("analysis_threshold_pct", 50)
    st.session_state.setdefault("analysis_1gene_use_auto", False)

    st.sidebar.subheader("1-Gene")
    gene = st.sidebar.text_input("Gene Symbol", key="analysis_gene").strip().upper()
    
    use_auto = st.sidebar.checkbox(
        "Auto Optimal Cutoff",
        key="analysis_1gene_use_auto"
    )
    
    threshold_pct = st.sidebar.slider(
        "High Expression Threshold (Percentile)",
        10, 90,
        key="analysis_threshold_pct",
        disabled=use_auto
    )
    return Sidebar1Gene(gene=gene, threshold_pct=int(threshold_pct), use_auto_cutoff=bool(use_auto))

@dataclass
class SidebarSignature:
    genes_text: str
    threshold_pct: int
    use_auto_cutoff: bool

def render_sidebar_signature() -> SidebarSignature:
    st.session_state.setdefault("analysis_sig_genes", "EGFR\nKRAS\nTP53")
    st.session_state.setdefault("analysis_sig_threshold_pct", 50)
    st.session_state.setdefault("analysis_sig_use_auto", False)

    st.sidebar.subheader("Gene Signature")
    genes_text = st.sidebar.text_area("Gene Symbols (one per line or comma-separated)", key="analysis_sig_genes").strip()
    
    use_auto = st.sidebar.checkbox(
        "Auto Optimal Cutoff",
        key="analysis_sig_use_auto"
    )
    
    threshold_pct = st.sidebar.slider(
        "High Score Threshold (Percentile)",
        10, 90,
        key="analysis_sig_threshold_pct",
        disabled=use_auto
    )
    return SidebarSignature(genes_text=genes_text, threshold_pct=int(threshold_pct), use_auto_cutoff=bool(use_auto))

