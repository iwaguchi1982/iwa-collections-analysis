# ui/cohort_shift_view.py
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, Any

def render_cohort_shift_dashboard(res: Dict[str, Any], df_disc: pd.DataFrame = None, df_val: pd.DataFrame = None):
    """
    Main entry point for the Cohort Shift Insight dashboard.
    """
    st.markdown("---")
    st.subheader("🔍 Cohort Shift Insight")
    st.info("💡 Recommended when validation is not reproduced. Useful to interpret cohort mismatch.")
    
    summary = res.get("summary", {})
    render_summary_card(summary)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "👥 Clinical", 
        "⏳ Outcome", 
        "🧬 Molecular", 
        "✂️ Cutoff", 
        "📊 Data Quality"
    ])
    
    with tab1:
        render_composition_plots(res.get("clinical", []))
    with tab2:
        render_outcome_plots(res.get("outcome", {}))
    with tab3:
        render_molecular_plots(res.get("molecular", {}), df_disc, df_val)
    with tab4:
        render_cutoff_plots(res.get("cutoff", {}))
    with tab5:
        render_data_quality_plots(res.get("data_quality", {}))
        
    render_interpretation_notes(summary)

def render_summary_card(summary: Dict[str, Any]):
    label = summary.get("shift_label", "Unknown")
    color = "green" if label == "Minimal" else "orange" if label == "Moderate" else "red"
    
    st.markdown(f"""
    <div style="background-color:rgba(0,0,0,0.05); padding:20px; border-radius:10px; border-left: 10px solid {color};">
        <h2 style="margin:0; color:{color};">Cohort Shift: {label}</h2>
        <p style="font-size:1.1em; margin-top:10px;">{summary.get("interpretation_text", "")}</p>
    </div>
    """, unsafe_allow_html=True)
    
    drivers = summary.get("top_shift_drivers", [])
    if drivers:
        st.markdown("**Top Contributors to Shift:**")
        cols = st.columns(len(drivers))
        for i, d in enumerate(drivers):
            cols[i].markdown(f"🚩 {d}")

def render_composition_plots(clinical: list):
    if not clinical:
        st.write("No clinical composition data available.")
        return
    
    st.markdown("#### Clinical Composition & SMD")
    
    # Forest Plot for SMD
    df_smd = pd.DataFrame(clinical)
    df_smd = df_smd.sort_values("smd", ascending=True)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_smd["smd"],
        y=df_smd["column"],
        mode='markers',
        marker=dict(size=12, color=df_smd["smd"].apply(lambda x: 'red' if x > 0.2 else 'orange' if x > 0.1 else 'royalblue')),
        error_x=None,
        hovertemplate="Variable: %{y}<br>SMD: %{x:.3f}<extra></extra>"
    ))
    fig.add_vline(x=0.1, line_dash="dash", line_color="orange", annotation_text="Moderate")
    fig.add_vline(x=0.2, line_dash="dash", line_color="red", annotation_text="Substantial")
    
    fig.update_layout(
        title="Standardized Mean Difference (SMD) Forest Plot",
        xaxis_title="SMD (Absolute)",
        yaxis_title="",
        height=300 + len(clinical)*20,
        margin=dict(l=150)
    )
    st.plotly_chart(fig, use_container_width=True)

def render_outcome_plots(outcome: Dict[str, Any]):
    if not outcome: return
    
    d = outcome.get("discovery", {})
    v = outcome.get("validation", {})
    
    st.markdown("#### Outcome Structure Comparison")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Event Rate", f"{v['event_rate']:.1%}", f"{v['event_rate']-d['event_rate']:.1%}", delta_color="inverse")
    c2.metric("Med. Follow-up", f"{v['median_followup']:.1f}m", f"{v['median_followup']-d['median_followup']:.1f}m")
    c3.metric("Censoring", f"{v['censoring_rate']:.1%}", f"{v['censoring_rate']-d['censoring_rate']:.1%}")
    
    # N counts comparison
    st.markdown("**Sample Counts Partitioning**")
    n_df = pd.DataFrame([
        {"Category": "Raw N", "Discovery": d["raw_n"], "Validation": v["raw_n"]},
        {"Category": "Eligible N", "Discovery": d["eligible_n"], "Validation": v["eligible_n"]},
        {"Category": "Events", "Discovery": d["events"], "Validation": v["events"]},
    ])
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Discovery', x=n_df["Category"], y=n_df["Discovery"]))
    fig.add_trace(go.Bar(name='Validation', x=n_df["Category"], y=n_df["Validation"]))
    fig.update_layout(barmode='group', height=300, title="Sample Size & Event counts")
    st.plotly_chart(fig, use_container_width=True)

def render_molecular_plots(molecular: Dict[str, Any], df_disc: pd.DataFrame = None, df_val: pd.DataFrame = None):
    if not molecular or not molecular.get("ok"): 
        st.write("No molecular distribution data available.")
        return
    
    d = molecular.get("discovery", {})
    v = molecular.get("validation", {})
    
    # Check for required labels
    if "median" not in d or "median" not in v:
        st.warning("Insufficient statistics to display molecular comparison.")
        return

    gene_label = molecular.get('gene', 'Target')
    st.markdown(f"#### Target Expression Distribution: {gene_label}")
    
    # 1. Overlay Plot
    if df_disc is not None and df_val is not None:
        # Prepare combined DF for plotting
        target_col = "expression" 
        d_sub = df_disc[[target_col]].copy()
        d_sub["Cohort"] = "Discovery"
        v_sub = df_val[[target_col]].copy()
        v_sub["Cohort"] = "Validation"
        
        plot_df = pd.concat([d_sub, v_sub])
        
        fig = px.violin(
            plot_df, 
            y=target_col, 
            x="Cohort", 
            color="Cohort", 
            box=True, 
            points="all",
            hover_data={"Cohort": True, target_col: ":.4f"},
            title=f"Expression distribution overlap: {gene_label}"
        )
        fig.update_layout(height=450, margin=dict(t=50))
        st.plotly_chart(fig, use_container_width=True)
    
    # 2. Summary stats table
    st.markdown("**Summary Statistics**")
    comp_df = pd.DataFrame({
        "Metric": ["Median", "IQR", "Mean", "Std"],
        "Discovery": [d["median"], d["iqr"], d["mean"], d["std"]],
        "Validation": [v["median"], v["iqr"], v["mean"], v["std"]]
    })
    st.table(comp_df.set_index("Metric"))
    
    st.info(f"Target SMD: **{molecular.get('smd', 0):.3f}** ({molecular.get('shift_severity', '')})")

def render_cutoff_plots(cutoff: Dict[str, Any]):
    if not cutoff or not cutoff.get("ok"): 
        st.write("No cutoff mapping data available.")
        return
    
    st.markdown("#### Cutoff Mapping & Group Balance")
    
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Disc. Cutoff Val", f"{cutoff['disc_cutoff_val']:.4f}")
        st.caption(f"Type: {cutoff['disc_cutoff_type']} / Percentile: {cutoff.get('disc_cutoff_percentile', 'N/A')}")
    
    with c2:
        st.metric("Val. Pop High Fraction", f"{cutoff['val_actual_high_frac']:.1%}", f"{cutoff['fraction_drift']:.1%}", delta_color="inverse")
        st.caption(f"Equivalent Percentile in Val: {cutoff['val_mapped_percentile']:.1f}%")

    st.warning(f"Cutoff Severity: **{cutoff['shift_severity']}** (Fraction drift: {cutoff['fraction_drift']:.1%})")

def render_data_quality_plots(dq: Dict[str, Any]):
    if not dq: return
    
    st.markdown("#### Data Completeness / Missingness Comparison")
    
    d_stats = dq.get("discovery_stats", {})
    v_stats = dq.get("validation_stats", {})
    
    m_df = pd.DataFrame([
        {"Cohort": "Discovery", "Missing Rate": d_stats.get("any_missing_rate", 0)},
        {"Cohort": "Validation", "Missing Rate": v_stats.get("any_missing_rate", 0)}
    ])
    
    fig = px.bar(m_df, x="Cohort", y="Missing Rate", color="Cohort", text_auto=".1%", height=300)
    fig.update_layout(title="Overall Missingness Rate Comparison")
    st.plotly_chart(fig, use_container_width=True)
    
    st.info(f"Missingness Mismatch: **{dq.get('shift_severity', '')}** (Diff: {dq.get('overall_missing_diff', 0):.1%})")

def render_interpretation_notes(summary: Dict[str, Any]):
    st.markdown("#### 💡 Interpretation Notes")
    label = summary.get("shift_label", "")
    
    if label == "Substantial":
        st.error("⚠️ **CRITICAL Mismatch Found**: The discovery and validation cohorts are significantly different in one or more key areas. Replication failure is highly likely due to these systemic differences (Cohort Shift) rather than a lack of biological effect.")
    elif label == "Moderate":
        st.warning("🔸 **Moderate Awareness**: Some settings (like cutoff value or stage balance) might need adjustment to achieve better comparability between cohorts.")
    else:
        st.success("✅ **Comparable Populations**: Your cohorts are well-aligned. Validation results should be highly representative of biological consistency.")
