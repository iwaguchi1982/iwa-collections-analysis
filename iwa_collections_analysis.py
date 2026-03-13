# iwa-collections-analysis.py
import streamlit as st

import pandas as pd
from src.preprocess import build_covariate_list

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from lifelines.statistics import multivariate_logrank_test, logrank_test

from src.auto_discovery import run_topn_pairing_search
from src.data_io import get_gene_exp, get_gene_signature_exp
from src.quality_checks import compute_missing_table

from ui.sidebar import render_sidebar_common, render_sidebar_2gene, render_sidebar_1gene, render_sidebar_signature
from ui.analysis_2gene import render_2gene_analysis
from ui.auto_discovery_view import render_auto_discovery
from ui.validation_view import render_validation_view
from ui.meta_analysis_view import render_meta_analysis_view
from ui.psm_view import render_psm_view
from ui.deg_view import render_deg_view
from ui.help_view import render_help
from ui.analysis_1gene import render_1gene_analysis
from ui.import_view import render_import_view
from ui.sensitivity_view import render_sensitivity_dashboard
from ui.missingness_view import render_missingness_dashboard
from ui.run_compare_view import render_run_compare_view
from ui.subgroup_view import render_subgroup_dashboard

# Dataset registry (root-aware; supports future PanCancer/GEO specs)
try:
    import src.dataset_registry as _dsreg
    list_dataset_specs = _dsreg.list_dataset_specs
    get_spec_by_label = getattr(_dsreg, "get_spec_by_label", None)
    load_dataset_with_root = getattr(_dsreg, "load_dataset_with_root", None)
    load_dataset = getattr(_dsreg, "load_dataset", None)

    HAS_REGISTRY = True
except Exception:
    HAS_REGISTRY = False

from src.preprocess import add_covariate_columns, build_covariate_list, detect_additional_covariates
from src.quality_checks import (
    plot_stage_distribution,
    plot_age_distribution,
    plot_gender_distribution,
    plot_followup_distribution,
    missingness_by_group,
    plot_expression_by_category,
)
from src.survival import (
    plot_km,
    run_cox,
    run_cox_4group,
    metric_p_display,
    plot_forest_hr,
    plot_compare_main_effect,
    plot_forest_contrasts,
    run_cox_interaction,
    format_interaction_table_for_forest,
    plot_interaction_forest,
    run_cut_smoothness_scan,
    plot_cut_smoothness_curve,
    apply_landmark,
    get_survival_probability,
    calculate_rmst,
    run_ph_test,
    plot_schoenfeld_residuals,
    run_cox_time_varying,
    plot_hr_over_time,
    get_time_varying_snapshots,
    summarize_by_cancer_type,
    run_cox_4group_stratified,
    find_optimal_1gene_cutoff
)

from src.druggability import get_target_profile
from src.pathway import (
    run_gsea_prerank, run_ora, 
    plot_gsea_enrichment, plot_top_pathways, 
    LIBRARY_MAPPING
)
from src.known_drugs import get_known_drugs

def _clear_session_keys(keys: list[str]) -> None:
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]

# =========================
# Config
# =========================
DATA_ROOT = Path.home() / "iwa-collections-analysis" / "data"

st.set_page_config(page_title="cBioPortal Target Explorer", layout="wide")
pending = st.session_state.pop("pending_analysis_settings", None)
if isinstance(pending, dict):
    # safe: happens before widgets instantiation in this run
    for k, v in pending.items():
        st.session_state[k] = v

if "auto_df" not in st.session_state:
    st.session_state["auto_df"] = None
if "auto_meta" not in st.session_state:
    st.session_state["auto_meta"] = {}

# --- Analysis defaults (for Apply-to-Analysis) ---
st.session_state.setdefault("sidebar_mode_label", "1-Gene (Hi/Lo)")
st.session_state.setdefault("analysis_gene", "EGFR")
st.session_state.setdefault("analysis_threshold_pct", 50)

# 2-gene defaults
st.session_state.setdefault("analysis_gene1", "A1BG")
st.session_state.setdefault("analysis_gene2", "ABAT")
st.session_state.setdefault("analysis_th1_pct", 50)
st.session_state.setdefault("analysis_th2_pct", 50)
st.session_state.setdefault("use_auto_cutoff", False)
st.session_state.setdefault("saved_runs", {})

# cross-tab flags
st.session_state.setdefault("auto_run_analysis", False)
st.session_state.setdefault("auto_run_detailed", False)

# =========================
# Sidebar: dataset selection
# =========================

# -----------------------------
# Reliability / Guardrails (A-B-C)
# -----------------------------
def _status_level(level: str) -> tuple[str, callable]:
    """Return (icon_label, streamlit_box_func)."""
    level = (level or "").upper()
    if level == "OK":
        return "✅ OK", st.success
    if level == "CAUTION":
        return "⚠️ 注意", st.warning
    return "⛔ 危険域", st.error

def _format_int(x) -> str:
    try:
        return f"{int(x):,}"
    except Exception:
        return str(x)

@st.cache_data(show_spinner=False)
def _load_dataset_cached(dataset_label: str, data_root: str):
    # spec は cache key に入れない（hash問題回避）
    spec0 = get_spec_by_label(specs, dataset_label) if callable(get_spec_by_label) else next(
        (s for s in specs if s.label == dataset_label), None
    )
    if spec0 is None:
        raise KeyError(f"Dataset spec not found: {dataset_label}")

    if callable(load_dataset_with_root):
        return load_dataset_with_root(spec0, data_root)
    return load_dataset(spec0)

def _assess_survival_guardrails(
    merged: pd.DataFrame,
    *,
    duration_col: str,
    event_col: str,
    group_col: str,
    n_groups: int,
    n_covariates: int,
    model_name: str,
) -> dict:
    """Heuristic guardrails for survival calculations.
    Returns dict with level in {OK, CAUTION, DANGER} and reasons + summary stats.
    """
    out = {"level": "OK", "reasons": [], "summary": {}}
    reasons: list[str] = []
    if merged is None or merged.empty:
        out["level"] = "DANGER"
        reasons.append("No data (merged is empty).")
        out["reasons"] = reasons
        return out

    # Basic column checks
    need = [duration_col, event_col, group_col]
    miss = [c for c in need if c not in merged.columns]
    if miss:
        out["level"] = "DANGER"
        reasons.append(f"Missing columns: {miss}")
        out["reasons"] = reasons
        return out

    df = merged[[duration_col, event_col, group_col]].copy()
    df[duration_col] = pd.to_numeric(df[duration_col], errors="coerce")
    df[event_col] = (pd.to_numeric(df[event_col], errors="coerce") > 0).astype(int)
    df = df.dropna()
    if df.empty:
        out["level"] = "DANGER"
        reasons.append("No valid rows after numeric coercion / dropna.")
        out["reasons"] = reasons
        return out

    g_counts = df[group_col].value_counts(dropna=False).to_dict()
    e_counts = df.groupby(group_col)[event_col].sum().to_dict()
    total_n = int(len(df))
    total_events = int(df[event_col].sum())
    min_n = int(min(g_counts.values())) if g_counts else 0
    min_events = int(min(e_counts.values())) if e_counts else 0

    out["summary"] = {
        "model": model_name,
        "total_n": total_n,
        "total_events": total_events,
        "min_group_n": min_n,
        "min_group_events": min_events,
        "n_groups": n_groups,
    }

    # Events-per-parameter (very rough) for Cox-like models
    # For 4-group: params ~= (groups-1) + covariates
    # For interaction: params ~= 3 + covariates (x1, x2, x1*x2)
    if model_name.lower().startswith("cox"):
        if "interaction" in model_name.lower():
            n_params = 3 + max(0, n_covariates)
        else:
            n_params = max(1, (n_groups - 1) + max(0, n_covariates))
        epp = (total_events / n_params) if n_params else 0.0
        out["summary"]["n_params"] = int(n_params)
        out["summary"]["events_per_param"] = float(epp)

    # ---- Heuristic thresholds ----
    if not isinstance(out.get("reasons"), list):
        out["reasons"] = []

    # Danger conditions
    if total_events < 15:
        out["level"] = "DANGER"
        reasons.append("Too few events (<15).")
    if min_n < 8:
        out["level"] = "DANGER"
        reasons.append("Some group has too few samples (<8).")
    if min_events == 0:
        out["level"] = "DANGER"
        reasons.append("Some group has 0 events (complete separation risk).")

    # Caution conditions (only if not already danger)
    if out["level"] != "DANGER":
        if total_events < 30:
            out["level"] = "CAUTION"
            reasons.append("Events are limited (<30).")
        if min_n < 20:
            out["level"] = "CAUTION"
            reasons.append("Some group has small N (<20).")
        if min_events < 5:
            out["level"] = "CAUTION"
            reasons.append("Some group has few events (<5).")

        # Cox-specific caution/danger based on events-per-parameter
        if model_name.lower().startswith("cox") and "events_per_param" in out["summary"]:
            epp = out["summary"]["events_per_param"]
            if epp < 5:
                out["level"] = "DANGER"
                reasons.append("Events per parameter is too low (<5).")
            elif epp < 10 and out["level"] == "OK":
                out["level"] = "CAUTION"
                reasons.append("Events per parameter is borderline (<10).")

    out["reasons"] = reasons
    return out

def render_guardrail_box(title: str, guard: dict, *, help_anchor: str = "Guardrails") -> None:
    lvl = (guard or {}).get("level", "DANGER")
    label, box = _status_level(lvl)
    reasons = (guard or {}).get("reasons", [])
    summary = (guard or {}).get("summary", {})
    msg_lines = [f"**{label}** — {title}"]
    if summary:
        parts = []
        if "total_n" in summary: parts.append(f"N={_format_int(summary['total_n'])}")
        if "total_events" in summary: parts.append(f"events={_format_int(summary['total_events'])}")
        if "min_group_n" in summary: parts.append(f"min group N={_format_int(summary['min_group_n'])}")
        if "min_group_events" in summary: parts.append(f"min group events={_format_int(summary['min_group_events'])}")
        if "events_per_param" in summary:
            parts.append(f"events/param={summary['events_per_param']:.1f}")
        if parts:
            msg_lines.append(" / ".join(parts))
    if reasons:
        msg_lines.append("— " + "; ".join(dict.fromkeys(reasons)))
    if lvl in ("CAUTION", "DANGER"):
        msg_lines.append(f"⚠️ **黄色/赤の場合は Help → {help_anchor} を必ず参照してください。**")
    box("\n\n".join(msg_lines))

def build_run_params_table(params: dict) -> pd.DataFrame:
    rows = []
    for k, v in params.items():
        rows.append({"Parameter": str(k), "Value": str(v)})
    return pd.DataFrame(rows)

st.sidebar.title("🧬いろいろしらべるくん🧬\nv0.1.0\nCopyright (c) 2026 J. Iwaguchi")
st.sidebar.markdown("## 📈Analysis Settings")

# -----------------------------
# Dataset selection (via dataset_registry if available; fallback to legacy data_io)
# -----------------------------
if HAS_REGISTRY:
    specs = list_dataset_specs(str(DATA_ROOT))
    labels = [s.label for s in specs]
    if not labels:
        st.error(f"No datasets found under: {DATA_ROOT}")
        st.stop()

    def _resolve_spec(specs, label: str):
        if callable(get_spec_by_label):
            return get_spec_by_label(specs, label)
        return next((s for s in specs if s.label == label), None)

    # --- Pre-load df_clin to detect dynamic covariates quickly ---
    # Retrieve the previously/currently selected dataset label from session state if available
    pre_selected_dataset = st.session_state.get("selected_dataset", labels[0])
    
    spec = _resolve_spec(specs, pre_selected_dataset)
    if spec is None:
        st.error(f"Dataset spec not found: {pre_selected_dataset}")
        st.stop()
        
    with st.spinner(f"Scanning clinical metadata for {pre_selected_dataset}..."):
        df_clin_pre, _, _, _ = _load_dataset_cached(pre_selected_dataset, str(DATA_ROOT))
        
    detected_covariates = detect_additional_covariates(df_clin_pre)

    common = render_sidebar_common(
        dataset_options=labels,
        default_dataset=pre_selected_dataset,
        detected_covars=detected_covariates
    )
    dataset_label  = common.dataset_label
    mode = common.mode_label
    covariates = common.covariates

    # --- dataset change detection & state reset (HEAVY RESULTS ONLY) ---
    prev_selected = st.session_state.get("_prev_selected_label")
    if prev_selected != dataset_label:
        st.session_state["_prev_selected_label"] = dataset_label

        keys_heavy = [
            # 2-gene heavy
            "analysis_merged",
            "analysis_ready",

            # 1-gene heavy
            "analysis_1gene_merged",
            "analysis_1gene_ready",

            # cut scan heavy
            "cutscan_result",
            "cutscan_sig",
            "cutscan_log",
        ]

        # optional: also clear stored plots if any (if you store fig objects)
        # keys_heavy += ["some_fig_key"]

        if not common.keep_autodiscovery:
            keys_heavy += ["auto_df", "auto_meta"]

        _clear_session_keys(keys_heavy)
        st.toast("Dataset changed → cleared heavy cached results", icon="🧹")
else:
    st.error("Dataset registry is required for this application.")
    st.stop()

# Render the rest of the sidebar first to avoid visual delay
s1 = None
s2 = None
s_sig = None
if mode == "1-Gene (Hi/Lo)":
    s1 = render_sidebar_1gene()
elif mode == "2-Genes (4-way)":
    auto_cut = st.session_state.get("auto_meta", {}).get("cutoff")
    s2 = render_sidebar_2gene(auto_cutoff=auto_cut)
elif mode == "Gene Signature (Multi-Gene)":
    s_sig = render_sidebar_signature()

with st.sidebar.expander("⚙️ Advanced Settings", expanded=False):
    forest_mode = st.radio(
        "Forest Plot Mode",
        ["Simple (Adjusted only)", "Compare (Unadjusted vs Adjusted)"],
        index=0
    )
    landmark_time = st.slider(
        "Landmark Analysis Cutoff (Months)",
        0, 120, 0, step=6,
        help="0 = Disabled. If > 0, performs Time-dependent Cox analysis splitting the cohort to handle crossing survival curves."
    )
    import os
    max_cores = os.cpu_count() or 4
    n_jobs = st.slider(
        "Max CPU Cores (Auto Discovery)",
        min_value=1,
        max_value=max_cores,
        value=max(1, max_cores // 2),  # Default to half the cores
        step=1,
        help="Limit the number of CPU cores used during Auto Discovery log-rank calculations. Lower this if the server is shared."
    )
    
    rmst_time = st.slider(
        "RMST Time Horizon (Months)",
        0, 120, 0, step=6,
        help="0 = Auto (minimum of max observed times). Restricted Mean Survival Time (RMST) calculates the area under the KM curve up to this time, useful when curves cross."
    )
    
    api_source = st.radio(
        "OpenTargets Data Source",
        ["Mock Database (Internal / Secure)", "Live API (External)"],
        index=0,
        help="Choose whether target profiling data should be fetched locally or via live external API requests."
    )
    use_mock = (api_source == "Mock Database (Internal / Secure)")

# Load the heavy dataset ONLY AFTER the sidebar stands a chance to render
# Thanks to @st.cache_data on _load_dataset_cached, this is instantaneous as it was pre-loaded above
with st.spinner(f"Loading dataset: {dataset_label} ..."):
    df_clin, df_exp, df_mut, df_cna = _load_dataset_cached(dataset_label, str(DATA_ROOT))

# Encode the detected categorical covariates before passing to modules
df_clin = add_covariate_columns(df_clin, detected_covariates=detected_covariates)

# =========================
# Global Guardrails
# =========================
try:
    valid_os = df_clin[["OS_MONTHS", "OS_STATUS"]].dropna()
    total_events = (pd.to_numeric(valid_os["OS_STATUS"], errors="coerce") > 0).astype(int).sum()

    current_covariates = []
    if mode == "1-Gene (Hi/Lo)" and s1:
        current_covariates = s1.covariates
    elif mode == "2-Genes (4-way)" and s2:
        current_covariates = s2.covariates

    missing_alerts = []
    for cov in current_covariates:
        col_name = cov.upper()
        if col_name in df_clin.columns:
            missing_pct = df_clin[col_name].isna().mean() * 100
            if missing_pct > 20:
                missing_alerts.append(f"{col_name} ({missing_pct:.1f}%)")

    guardrail_blocks = []
    if total_events < 10:
        guardrail_blocks.append(f"⛔ **Event DANGER:** The currently selected dataset has extremely few survival events (**{total_events}** events). Survival analysis requires at least 10-15 events to be minimally robust.")
    if missing_alerts:
        guardrail_blocks.append(f"⚠️ **Missing Data CAUTION:** The selected covariates have high missing rates: {', '.join(missing_alerts)}. This will severely reduce the sample size during PSM and Cox regression.")

    if guardrail_blocks:
        for block in guardrail_blocks:
            if "⛔" in block:
                st.error(block)
            else:
                st.warning(block)
        
        if total_events < 10:
            st.info("💡 **Hint:** Please select a different cancer type (dataset) with more recorded events to continue your mission. Analysis tabs have been safely disabled.")
            st.stop()
except Exception:
    pass

# =========================
# Main UI & Mission Routing
# =========================

st.markdown("### 🎯 Select Your Mission (Analysis Route)")
mission = st.radio(
    "Choose a guided pathway to reorder and focus the tabs for your specific goal:",
    [
        "Default (Show All Tabs)",
        "Route A: Prognostic Biomarker Discovery",
        "Route B: Therapeutic Target Discovery",
        "Route C: Drug Repurposing"
    ],
    index=0,
    horizontal=False
)

ROUTES = {
    "Default (Show All Tabs)": [
        ("🔎 Analysis", "analysis"), ("🧬 Subgroups", "subgroup"), ("🔍 Sensitivity", "sensitivity"), ("🧩 Missing data", "missing"), ("⚖️ PSM", "psm"), ("🧬 DEG & PCA", "deg"),
        ("🧪 Validation", "val"), ("🌐 Meta-Analysis", "meta"), ("🆚 Run Compare", "compare"), ("🚀 Auto Discovery", "auto"), ("📥 Import Dataset", "import"), ("❓ Help", "help")
    ],
    "Route A: Prognostic Biomarker Discovery": [
        ("🧬 1. Discovery (DEG)", "deg"), ("🚀 2. Screening (Auto)", "auto"), ("🔎 3. Profiling", "analysis"),
        ("🔍 4. Sensitivity", "sensitivity"), ("🧩 5. Missing data", "missing"), ("⚖️ 6. PSM", "psm"), ("🧪 7. Validation", "val"), ("🌐 8. Meta-Analysis", "meta"), ("❓ Help", "help")
    ],
    "Route B: Therapeutic Target Discovery": [
        ("🎯 1. Target Triage (Analysis)", "analysis"), ("🔍 2. Robustness (Sensitivity)", "sensitivity"),
        ("🧩 3. Data Quality (Missing)", "missing"), ("⚖️ 4. Clinical Adjustment (PSM)", "psm"), ("🧪 5. Efficacy Validation", "val"), ("🌐 6. Pan-Cancer Efficacy", "meta"), ("❓ Help", "help")
    ],
    "Route C: Drug Repurposing": [
        ("🧬 1. Disease Signature (DEG)", "deg"), ("💊 2. Drug Matching (Analysis)", "analysis"), 
        ("🔍 3. Sensitivity", "sensitivity"), ("🧩 4. Data Quality (Missing)", "missing"), ("🧪 5. Responder Cohort (Validation)", "val"), ("❓ Help", "help")
    ]
}

active_route = ROUTES[mission]
tab_titles = [t[0] for t in active_route]
tab_keys = [t[1] for t in active_route]

# Create dynamic tabs
rendered_tabs = st.tabs(tab_titles)

# Helper function to find the tab object for a given key
def get_tab(key: str):
    if key in tab_keys:
        return rendered_tabs[tab_keys.index(key)]
    return None

tab_analysis = get_tab("analysis")
tab_psm = get_tab("psm")
tab_deg = get_tab("deg")
tab_val = get_tab("val")
tab_meta = get_tab("meta")
tab_auto = get_tab("auto")
tab_import = get_tab("import")
tab_sensitivity = get_tab("sensitivity")
tab_missing = get_tab("missing")
tab_compare = get_tab("compare")
tab_subgroup = get_tab("subgroup")
tab_help = get_tab("help")

if tab_import:
    with tab_import:
        render_import_view(DATA_ROOT)

if tab_analysis:
    with tab_analysis:
        if mode == "1-Gene (Hi/Lo)":
            gene = s1.gene if s1 else None
            threshold_pct = s1.threshold_pct if s1 else 50
            use_auto_cutoff = s1.use_auto_cutoff if s1 else False

            deps_1 = {
                "plot_km": plot_km,
                "run_cox": run_cox,
                "plot_forest_hr": plot_forest_hr,
                "plot_compare_main_effect": plot_compare_main_effect,
                "metric_p_display": metric_p_display,
                "logrank_test": logrank_test,
                "plot_expression_by_category": plot_expression_by_category,
                "find_optimal_1gene_cutoff": find_optimal_1gene_cutoff,
                "get_survival_probability": get_survival_probability,
                # forest_mode を外から選ばせたいならここで入れる
                "forest_mode": forest_mode,
                "landmark_time": landmark_time,
                "apply_landmark": apply_landmark,
                "use_mock": use_mock,
                "run_ph_test": run_ph_test,
                "plot_schoenfeld_residuals": plot_schoenfeld_residuals,
            }
            
            from src.bootstrap import run_1gene_bootstrap, plot_bootstrap_results
            deps_1["run_1gene_bootstrap"] = run_1gene_bootstrap
            deps_1["plot_bootstrap_results"] = plot_bootstrap_results

            from src.survival import run_cox_time_varying, plot_hr_over_time, get_time_varying_snapshots
            deps_1["run_cox_time_varying"] = run_cox_time_varying
            deps_1["plot_hr_over_time"] = plot_hr_over_time
            deps_1["get_time_varying_snapshots"] = get_time_varying_snapshots
            
            deps_1["run_gsea_prerank"] = run_gsea_prerank
            deps_1["plot_gsea_enrichment"] = plot_gsea_enrichment
            deps_1["plot_top_pathways"] = plot_top_pathways
            deps_1["LIBRARY_MAPPING"] = LIBRARY_MAPPING
            
            # The druggability/known_drugs functions were already injected safely into the main deps map,
            # but 1-Gene analysis uses deps_1, so we inject them as lambdas binding the local use_mock flag
            from src.druggability import get_target_profile
            from src.known_drugs import get_known_drugs
            deps_1["get_target_profile"] = lambda g, use_mock=use_mock: get_target_profile(g, use_mock=use_mock)
            deps_1["get_known_drugs"] = lambda g, use_mock=use_mock: get_known_drugs(g, use_mock=use_mock)

            render_1gene_analysis(
                dataset_name=dataset_label ,
                df_clin=df_clin,
                df_exp=df_exp,
                df_mut=df_mut,
                df_cna=df_cna,
                gene=gene,
                threshold_pct=threshold_pct,
                use_auto_cutoff=use_auto_cutoff,
                covariates_actual = build_covariate_list(
                    "age" in covariates,
                    "gender" in covariates,
                    "stage" in covariates,
                    "msi" in covariates,
                    "tmb" in covariates,
                    dynamic_covariates=[c for c in covariates if c.endswith("_encoded")]
                ),
                covariates=covariates,
                get_gene_exp=get_gene_exp,
                add_covariate_columns=add_covariate_columns,
                deps=deps_1,
                rmst_time=rmst_time,
                calculate_rmst=calculate_rmst,
            )

        elif mode == "2-Genes (4-way)":
            gene1 = s2.gene1 if s2 else None
            gene2 = s2.gene2 if s2 else None
            th1_pct = s2.th1_pct if s2 else 50
            th2_pct = s2.th2_pct if s2 else 50
            #depsとして関数をまとめて渡す（UI側でimportさせるのではなく、必要なものだけ渡す形にする）
            deps = {
                # plotting / survival
                "plot_km": plot_km,
                "run_cox_4group": run_cox_4group,
                "run_cox_interaction": run_cox_interaction,
                "format_interaction_table_for_forest": format_interaction_table_for_forest,
                "plot_interaction_forest": plot_interaction_forest,
                "plot_forest_contrasts": plot_forest_contrasts,
                "summarize_by_cancer_type": summarize_by_cancer_type,
                "run_cox_4group_stratified": run_cox_4group_stratified,
                "run_cut_smoothness_scan": run_cut_smoothness_scan,
                "plot_cut_smoothness_curve": plot_cut_smoothness_curve,

                # stats
                "multivariate_logrank_test": multivariate_logrank_test,
                "get_survival_probability": get_survival_probability,

                # guardrails UI helpers（iwa-collections-analysis.py にあるなら渡す）
                "_assess_survival_guardrails": _assess_survival_guardrails,
                "render_guardrail_box": render_guardrail_box,
                "metric_p_display": metric_p_display,

                # quality checks
                "plot_stage_distribution": plot_stage_distribution,
                "plot_age_distribution": plot_age_distribution,
                "plot_gender_distribution": plot_gender_distribution,
                "plot_followup_distribution": plot_followup_distribution,
                "compute_missing_table": compute_missing_table,
                "plot_expression_by_category": plot_expression_by_category,
                "forest_mode": forest_mode,
                "landmark_time": landmark_time,
                "apply_landmark": apply_landmark,
                "run_ph_test": run_ph_test,
                "plot_schoenfeld_residuals": plot_schoenfeld_residuals,
                "plot_schoenfeld_residuals": plot_schoenfeld_residuals,
            }

            from src.bootstrap import run_2gene_bootstrap, plot_bootstrap_results
            deps["run_2gene_bootstrap"] = run_2gene_bootstrap
            deps["plot_bootstrap_results"] = plot_bootstrap_results

            from src.survival import run_cox_time_varying, plot_hr_over_time, get_time_varying_snapshots
            deps["run_cox_time_varying"] = run_cox_time_varying
            deps["plot_hr_over_time"] = plot_hr_over_time
            deps["get_time_varying_snapshots"] = get_time_varying_snapshots

            render_2gene_analysis(
                dataset_name=dataset_label,
                df_clin=df_clin,
                df_exp=df_exp,
                gene1=gene1,
                gene2=gene2,
                cutoff_pct=th1_pct,
                covariates_actual = build_covariate_list(
                    "age" in covariates,
                    "gender" in covariates,
                    "stage" in covariates,
                    "msi" in covariates,
                    "tmb" in covariates,
                    dynamic_covariates=[c for c in covariates if c.endswith("_encoded")]
                ),
                covariates=covariates,
                get_gene_exp=get_gene_exp,
                add_covariate_columns=add_covariate_columns,
                deps=deps,
                rmst_time=rmst_time,
                calculate_rmst=calculate_rmst,
            )

        elif mode == "Gene Signature (Multi-Gene)":
            import re
        
            genes_raw = s_sig.genes_text if s_sig else ""
            # Parse by commas or newlines
            genes_list = [g.strip().upper() for g in re.split(r'[,\n]+', genes_raw) if g.strip()]
            threshold_pct = s_sig.threshold_pct if s_sig else 50
            use_auto_cutoff = s_sig.use_auto_cutoff if s_sig else False

            deps_sig = {
                "plot_km": plot_km,
                "run_cox": run_cox,
                "plot_forest_hr": plot_forest_hr,
                "plot_compare_main_effect": plot_compare_main_effect,
                "metric_p_display": metric_p_display,
                "logrank_test": logrank_test,
                "plot_expression_by_category": plot_expression_by_category,
                "find_optimal_1gene_cutoff": find_optimal_1gene_cutoff,
                # forest_mode
                "forest_mode": forest_mode,
                "run_ph_test": run_ph_test,
                "plot_schoenfeld_residuals": plot_schoenfeld_residuals,
            }

            # We can reuse render_1gene_analysis but need a custom get_gene_exp wrapper
            # that actually returns the signature score rather than a single gene.
            def signature_exp_wrapper(df_e, pseudo_gene_name):
                # df_e is df_exp. pseudo_gene_name is "Signature"
                merged_z, meta = get_gene_signature_exp(df_e, genes_list)
                if merged_z is None:
                    return None
            
                # Show the meta info purely for visibility
                found = meta["found"]
                missing = meta["missing"]
            
                st.info(f"🧬 **Signature Score Calculated**. \n\nFound ({len(found)}): {', '.join(found)}\n\nMissing ({len(missing)}): {', '.join(missing)}")
            
                return merged_z

            title_gene = f"Signature Score ({len(genes_list)} genes)"
        
            render_1gene_analysis(
                dataset_name=dataset_label,
                df_clin=df_clin,
                df_exp=df_exp,
                df_mut=None, # Mutation/CNA complex to combine natively here, so disable.
                df_cna=None,
                gene=title_gene,
                threshold_pct=threshold_pct,
                use_auto_cutoff=use_auto_cutoff,
                covariates_actual = build_covariate_list(
                    "age" in covariates,
                    "gender" in covariates,
                    "stage" in covariates,
                    "msi" in covariates,
                    "tmb" in covariates,
                    dynamic_covariates=[c for c in covariates if c.endswith("_encoded")]
                ),
                covariates=covariates,
                get_gene_exp=signature_exp_wrapper,
                add_covariate_columns=add_covariate_columns,
                deps=deps_sig,
                rmst_time=rmst_time,
                calculate_rmst=calculate_rmst,
            )

if tab_subgroup:
    with tab_subgroup:
        render_subgroup_dashboard()

if tab_compare:
    with tab_compare:
        render_run_compare_view()

if tab_sensitivity:
    with tab_sensitivity:
        # Determine current target gene
        target_gene = ""
        if mode == "1-Gene (Hi/Lo)":
            target_gene = s1.gene if s1 else ""
        elif mode == "2-Genes (4-way)":
            # For now, sensitivity dashboard might focus on Gene1/Gene2 individually
            # or we could extend it for 4-way. Let's provide Gene1 as primary.
            target_gene = s2.gene1 if s2 else ""
        
        if target_gene:
            render_sensitivity_dashboard(
                df_clin=df_clin,
                df_exp=df_exp,
                gene=target_gene,
                get_gene_exp=get_gene_exp,
                add_covariate_columns=add_covariate_columns,
                deps={"landmark_time": landmark_time} 
            )
        else:
            st.info("Please select a target gene in the sidebar to run Sensitivity Analysis.")

if tab_missing:
    with tab_missing:
        target_gene_m = ""
        if mode == "1-Gene (Hi/Lo)":
            target_gene_m = s1.gene if s1 else ""
        elif mode == "2-Genes (4-way)":
            target_gene_m = f"{s2.gene1} + {s2.gene2}" if s2 else ""
        
        if target_gene_m:
            render_missingness_dashboard(
                df_clin=df_clin,
                df_exp=df_exp,
                gene=target_gene_m,
                get_gene_exp=get_gene_exp,
                add_covariate_columns=add_covariate_columns,
                deps={}
            )
        else:
            st.info("Please select a target gene in the sidebar to evaluate missing data impacts.")

if tab_psm:
    with tab_psm:
        deps_psm = {
            "get_gene_exp": get_gene_exp,
            "add_covariate_columns": add_covariate_columns,
            "plot_km": plot_km,
            "run_cox": run_cox,
        }
        render_psm_view(
            dataset_name=dataset_label,
            df_clin=df_clin,
            df_exp=df_exp,
            deps=deps_psm,
        )

if tab_deg:
    with tab_deg:
        deps_deg = {}
        render_deg_view(
            dataset_name=dataset_label,
            df_clin=df_clin,
            df_exp=df_exp,
            deps=deps_deg,
        )

if tab_val:
    with tab_val:
        deps_val = {
            "load_dataset_cached": _load_dataset_cached,
            "get_gene_exp": get_gene_exp,
            "plot_km": plot_km,
            "run_cox": run_cox,
            "add_covariate_columns": add_covariate_columns,
        }
        render_validation_view(
            data_root=DATA_ROOT,
            dataset_options=labels,
            current_dataset=dataset_label,
            deps=deps_val,
        )

if tab_meta:
    with tab_meta:
        deps_meta = {
            "load_dataset_cached": _load_dataset_cached,
            "get_gene_exp": get_gene_exp,
            "run_cox": run_cox,
            "add_covariate_columns": add_covariate_columns,
        }
        render_meta_analysis_view(
            data_root=str(DATA_ROOT),
            dataset_options=labels if HAS_REGISTRY else [],
            deps=deps_meta,
        )

if tab_auto:
    with tab_auto:
        deps_auto = {"run_topn_pairing_search": run_topn_pairing_search}
        render_auto_discovery(
            dataset_name=dataset_label,
            df_clin=df_clin,
            df_exp=df_exp,
            deps=deps_auto,
            n_jobs=n_jobs,
        )

if tab_help:
    with tab_help:
        DOCS_DIR = Path.home() / "iwa-collections-analysis" / "docs"
        render_help(docs_dir=DOCS_DIR)
