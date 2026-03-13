# ui/auto_discovery_view.py
from __future__ import annotations

from typing import Any, Dict, Optional
import pandas as pd
import streamlit as st

def render_auto_discovery(
    *,
    dataset_name: str,
    df_clin: pd.DataFrame,
    df_exp: pd.DataFrame,
    deps: Dict[str, Any],
    n_jobs: int = 4,
) -> None:
    """
    Auto Discovery tab UI.
    deps must contain:
      - run_topn_pairing_search (callable)
    """
    run_topn_pairing_search = deps["run_topn_pairing_search"]

    st.header("🚀 Auto Discovery")

    rank_threshold_pct = st.slider("Ranking Cutoff (%)", 5, 50, 25, key="auto_rank_cutoff")
    target_n = st.number_input("Number of genes to scan", 10, 500, 50, key="auto_target_n")
    show_only_fdr = st.checkbox(
        "Hide Red/Danger pairs (q ≥ 0.1)",
        value=True,
        key="auto_show_fdr",
    )

    if st.button("Run Top-N Pairing Search", key="auto_run_btn"):
        progress = st.progress(0.0)

        df_res = run_topn_pairing_search(
            df_clin=df_clin,
            df_exp=df_exp,
            rank_threshold_pct=int(rank_threshold_pct),
            target_n=int(target_n),
            show_only_q=bool(show_only_fdr),
            q_cutoff=0.1,
            progress_callback=progress.progress,
            n_jobs=n_jobs,
        )

        ad_meta = getattr(df_res, "attrs", {}).get("meta", {}) or {}
        progress.empty()

        st.session_state["auto_df"] = df_res
        st.session_state["auto_meta"] = {
            **ad_meta,
            "dataset": dataset_name,
            "cutoff": int(rank_threshold_pct),
            "target_n": int(target_n),
            "show_only_q": bool(show_only_fdr),
            "q_cutoff": 0.1,
        }

        st.success("✅ Auto Discovery complete")

    df_all = st.session_state.get("auto_df")
    if isinstance(df_all, pd.DataFrame) and not df_all.empty:
        meta = st.session_state.get("auto_meta") or {}
        st.caption(
            f"Stored result — dataset={meta.get('dataset')} "
            f"cutoff={meta.get('cutoff')}% target_n={meta.get('target_n')}"
        )

        df_show = df_all.copy()
        
        # Add traffic light status
        if "q-value" in df_show.columns:
            def q_status(q):
                if pd.isna(q): return "⚪ Unknown"
                if q < 0.05: return "🟢 OK"
                if q < 0.1: return "🟡 Hypothesis"
                return "🔴 Danger"
            
            # Insert status as the 3rd column for visibility
            df_show.insert(2, "Status", df_show["q-value"].apply(q_status))

        if st.session_state.get("auto_show_fdr", True) and "q-value" in df_show.columns:
            df_show = df_show[df_show["q-value"] < 0.1]

        st.dataframe(
            df_show.style.format({"p-value": "{:.3e}", "q-value": "{:.3e}"}),
            use_container_width=True,
        )

        st.markdown("**Pick a pair to apply to Analysis Settings:**")

        max_opts = min(len(df_all), 50)
        options = []
        for i in range(max_opts):
            row = df_all.iloc[i]
            q_val = row.get("q-value", float("nan"))
            icon = "🟢" if q_val < 0.05 else ("🟡" if q_val < 0.1 else "🔴")
            options.append(f"{i}: {icon} {row['Gene 1']} × {row['Gene 2']} (q={q_val:.2e})")
            
        sel = st.selectbox("Pair", options=options, index=0, key="auto_apply_sel")
        sel_idx = int(str(sel).split(":")[0])

        if st.button("Apply to Analysis Settings", key="apply_to_settings_btn"):
            row = df_all.iloc[sel_idx]
            g1 = str(row["Gene 1"]).upper()
            g2 = str(row["Gene 2"]).upper()

            auto_cut = st.session_state.get("auto_meta", {}).get("cutoff")
            th = int(auto_cut) if auto_cut is not None else int(st.session_state.get("analysis_th1_pct", 25))

            st.session_state["pending_analysis_settings"] = {
                "sidebar_mode_label": "2-Genes (4-way)",
                "analysis_gene1": g1,
                "analysis_gene2": g2,
                "analysis_th1_pct": th,
                "analysis_th2_pct": th,
                "use_auto_cutoff": True,
            }

            # ✅ ここがポイント：Analysis側のボタンを自動押下させるフラグだけ立てる
            st.session_state["auto_run_detailed"] = True

            st.success("✅ Applied to Analysis Settings and queued Detailed Analysis run.")
            st.rerun()

        if st.button("Clear stored results", key="auto_clear_btn"):
            st.session_state["auto_df"] = None
            st.session_state["auto_meta"] = {}