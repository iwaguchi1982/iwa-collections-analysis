# ui/run_compare_view.py
import streamlit as st
import pandas as pd
from src.run_compare import RunSnapshot, compare_snapshots, generate_comparison_summary

def render_run_compare_view():
    st.title("🆚 Run Comparison Workspace")
    st.markdown("""
    Compare multiple analysis runs side-by-side to understand how changes in 
    **metadata (cutoffs, covariates, datasets)** or **model settings** affect 
    the survival outcomes and hazard ratios.
    """)

    if "saved_runs" not in st.session_state or not st.session_state.saved_runs:
        st.info("No runs saved for comparison yet. Run an analysis and click 'Save Run for Comparison' to get started.")
        return

    saved_runs = st.session_state.saved_runs
    # Use run_id as key, label+gene+time as display
    run_options = {f"[{r.created_at}] {r.label} ({r.gene})": r.run_id for r in saved_runs.values()}
    # Sort options by creation time descending
    sorted_labels = sorted(run_options.keys(), reverse=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_labels = st.multiselect(
            "Select up to 2 runs to compare:",
            options=sorted_labels,
            max_selections=2,
            help="Select exactly two runs to see the side-by-side comparison."
        )
    
    with col2:
        if st.button("Clear All Saved Runs", type="secondary"):
            st.session_state.saved_runs = {}
            st.rerun()

    if len(selected_labels) == 0:
        st.info("Please select 1 or 2 runs to view details.")
        return

    if len(selected_labels) == 1:
        # Show single run summary
        r_id = run_options[selected_labels[0]]
        s = saved_runs[r_id]
        st.markdown("---")
        st.subheader(f"Summary: {s.label}")
        col1, col2, col3 = st.columns(3)
        col1.metric("HR", f"{s.hr:.2f}" if s.hr else "N/A")
        col2.metric("P-value", f"{s.p_value:.4e}" if s.p_value is not None else "N/A")
        col3.metric("N", s.n_total)
        st.info("Select a second run to enable side-by-side comparison.")
        return

    # Comparison Mode
    r1_id = run_options[selected_labels[0]]
    r2_id = run_options[selected_labels[1]]
    
    # Order by timestamp to make comparison logical (older vs newer)
    s1 = saved_runs[r1_id]
    s2 = saved_runs[r2_id]
    if s1.created_at > s2.created_at:
        s1, s2 = s2, s1
        
    diff = compare_snapshots(s1, s2)
    summary = generate_comparison_summary(s1, s2, diff)

    st.markdown("---")
    st.subheader("💡 Automated Comparison Summary")
    st.info(summary)

    # side-by-side cards
    c1, c2 = st.columns(2)
    for i, (col, s) in enumerate(zip([c1, c2], [s1, s2])):
        with col:
            st.markdown(f"#### Run {i+1}: {s.label}")
            st.caption(f"Created at: {s.created_at}")
            
            # Metrics
            m1, m2 = st.columns(2)
            if s.hr is not None:
                m1.metric("Adjusted HR", f"{s.hr:.2f}")
                m2.metric("P-value", f"{s.p_value:.4e}" if s.p_value is not None else "N/A")
            else:
                st.error("Model problematic or insufficient data.")

    # Metadata Comparison Table
    st.markdown("---")
    st.subheader("📋 Metadata & Input Conditions")
    
    meta_df = pd.DataFrame({
        "Feature": ["Dataset", "Gene / Target", "Modality", "Endpoint", "Cutoff (%)", "Covariates", "Subgroup"],
        f"Run 1 ({s1.label})": [
            s1.dataset, s1.gene, s1.omics, s1.endpoint, 
            f"{s1.cutoff_percentile}%" if s1.cutoff_percentile is not None else "N/A",
            ", ".join(s1.covariates) if s1.covariates else "None",
            s1.subgroup or "None"
        ],
        f"Run 2 ({s2.label})": [
            s2.dataset, s2.gene, s2.omics, s2.endpoint, 
            f"{s2.cutoff_percentile}%" if s2.cutoff_percentile is not None else "N/A",
            ", ".join(s2.covariates) if s2.covariates else "None",
            s2.subgroup or "None"
        ]
    })
    st.table(meta_df)

    # Core Results Comparison Table
    st.markdown("---")
    st.subheader("📊 Core Statistical Outcomes")
    
    # Helper to calculate delta strings
    def get_delta_str(v1, v2, fmt=".2f"):
        if v1 is None or v2 is None: return "-"
        d = v2 - v1
        if d > 0: return f"+{d:{fmt}}"
        return f"{d:{fmt}}"

    def get_p_delta_str(v1, v2):
        if v1 is None or v2 is None: return "-"
        if v1 == 0: return "N/A"
        fold = v2 / v1
        return f"x{fold:.2f}"

    res_df = pd.DataFrame({
        "Metric": ["Hazard Ratio", "Log-rank P", "Cox P", "Total N", "Events"],
        f"Run 1 ({s1.label})": [
            f"{s1.hr:.2f}" if s1.hr else "N/A",
            f"{s1.logrank_p:.4e}" if s1.logrank_p else "N/A",
            f"{s1.p_value:.4e}" if s1.p_value else "N/A",
            s1.n_total,
            s1.n_events
        ],
        f"Run 2 ({s2.label})": [
            f"{s2.hr:.2f}" if s2.hr else "N/A",
            f"{s2.logrank_p:.4e}" if s2.logrank_p else "N/A",
            f"{s2.p_value:.4e}" if s2.p_value else "N/A",
            s2.n_total,
            s2.n_events
        ],
        "Difference (Δ)": [
            get_delta_str(s1.hr, s2.hr),
            get_p_delta_str(s1.logrank_p, s2.logrank_p),
            get_p_delta_str(s1.p_value, s2.p_value),
            get_delta_str(s1.n_total, s2.n_total, fmt="d"),
            get_delta_str(s1.n_events, s2.n_events, fmt="d")
        ]
    })
    st.table(res_df)

    # Add delta to metadata if applicable
    if s1.cutoff_percentile is not None and s2.cutoff_percentile is not None:
        st.caption(f"**Cutoff Change:** Δ {s2.cutoff_percentile - s1.cutoff_percentile:.1f}% percentile")

    # Issue Comparison
    st.markdown("---")
    st.subheader("⚠️ Analytical Issues & Warnings")
    
    if not s1.issues and not s2.issues:
        st.success("No analytical issues identified in either run.")
    else:
        new = diff["issue_diff"]["new"]
        resolved = diff["issue_diff"]["resolved"]
        persistent = diff["issue_diff"]["persistent"]
        
        if new:
            st.error(f"🔴 **New Issues in Run 2** ({len(new)})")
            for iss in new:
                st.markdown(f"- **{iss.get('title', 'Unknown')}** ({iss.get('severity', 'info')}): {iss.get('detail', '')}")
        
        if resolved:
            st.success(f"🟢 **Resolved Issues in Run 2** ({len(resolved)})")
            for iss in resolved:
                st.markdown(f"- **{iss.get('title', 'Unknown')}** ({iss.get('severity', 'info')})")
                
        if persistent:
            st.warning(f"🟡 **Persistent Issues** ({len(persistent)})")
            for iss in persistent:
                st.markdown(f"- **{iss.get('title', 'Unknown')}** ({iss.get('severity', 'info')}): {iss.get('detail', '')}")

    # Interpretation
    st.markdown("---")
    st.subheader("📝 Interpretation & Robustness")
    ii1, ii2 = st.columns(2)
    with ii1:
        st.markdown(f"**Run 1 ({s1.label})**")
        st.write(s1.interpretation or "No specific interpretation provided.")
    with ii2:
        st.markdown(f"**Run 2 ({s2.label})**")
        st.write(s2.interpretation or "No specific interpretation provided.")
