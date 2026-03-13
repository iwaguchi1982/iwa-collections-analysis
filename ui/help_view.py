# ui/help_view.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import streamlit as st


DEFAULT_DOCS_ORDER: List[Tuple[str, str]] = [
    ("はじめに", "start-up.md"),
    ("Overview", "overview.md"),
    ("Features & Usage", "features.md"),
    ("🎯 Mission Guidelines (GO/NOGO基準)", "mission_guidelines.md"),
    ("🔰 Beginner's Guide (解釈の注意点)", "beginner_guide.md"),
    ("🧪 Validation Mode (外部コホート検証)", "validation.md"),
    ("Performance & Tips", "performance_tips.md"),
    ("Guardrails (最重要)", "guardrails.md"),
    ("KM / Log-rank", "km_logrank.md"),
    ("Cox models", "cox_models.md"),
    ("Interaction", "interaction.md"),
    ("PanCancer bias check: by cancer type", "pancancer_bias.md"),
    ("Cut smoothness scan", "cut_smoothness.md"),
    ("🎯 Target Triage & Prioritization", "target_triage.md"),
    ("🔍 Sensitivity Analysis Guide", "sensitivity_help.md"),
    ("🧩 Missing Data Treatment Guide", "missing_data_help.md"),
    ("FAQ", "faq.md"),
    ("実装ログ", "implemented.md"),
    ("実装予定ログ","upcoming.md" ),
]


def _read_md(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        return f"**Failed to read:** `{path}`\n\n- {type(e).__name__}: {e}"


def render_help(*, docs_dir: str | Path, docs_order: List[Tuple[str, str]] = DEFAULT_DOCS_ORDER) -> None:
    docs_dir = Path(docs_dir)

    st.header("Help")
    st.caption("✅ 緑：OK / ⚠️ 黄：注意（解釈に注意） / ⛔ 赤：危険域（計算スキップ or 参考値扱い）")

    labels = [x[0] for x in docs_order]
    files = [x[1] for x in docs_order]

    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.subheader("Contents")
        with st.container(height=520):
            sel = st.radio(" ", labels, index=0, key="help_section")
        st.divider()
        st.caption(f"Docs folder: `{docs_dir}`")

    with col2:
        st.markdown("### Help")
        st.caption("✅ 緑：OK / ⚠️ 黄：注意（解釈に注意） / ⛔ 赤：危険域（計算スキップ or 参考値扱い）")
        st.divider()
        idx = labels.index(sel)
        md_path = docs_dir / files[idx]
        st.markdown(_read_md(md_path), unsafe_allow_html=False)