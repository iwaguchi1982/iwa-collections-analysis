# src/survival.py
import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test, proportional_hazard_test
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional , Tuple
from src.config import is_gpu_enabled
from src.analysis_issues import Issue

MIN_N_COX = 20
MIN_EVENTS_COX = 10

def plot_forest_hr(cph, highlight: str | None = None, title: str = "Hazard Ratios (Cox Model)"):
    s = cph.summary.copy()
    dfp = s[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].rename(columns={
        "exp(coef)": "HR",
        "exp(coef) lower 95%": "HR_L",
        "exp(coef) upper 95%": "HR_U",
        "p": "p",
    })

    # highlight first, then by p
    idx = list(dfp.index)
    if highlight in idx:
        idx.remove(highlight)
        idx = [highlight] + sorted(idx, key=lambda k: dfp.loc[k, "p"])
        dfp = dfp.loc[idx]
    else:
        dfp = dfp.sort_values("p")

    y = np.arange(len(dfp))[::-1]

    fig, ax = plt.subplots(figsize=(7.2, max(2.6, 0.45 * len(dfp) + 1.2)))

    for i, (name, row) in enumerate(dfp.iterrows()):
        yy = y[i]
        ax.plot([row["HR_L"], row["HR_U"]], [yy, yy])
        ax.plot(row["HR"], yy, marker="s")

    ax.axvline(1.0, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(dfp.index)
    ax.set_xlabel("Hazard Ratio (HR) with 95% CI")
    ax.set_title(title)

    x_right = float(dfp["HR_U"].max()) * 1.20
    ax.set_xlim(left=max(0.01, float(dfp["HR_L"].min()) * 0.85), right=x_right)

    for i, (name, row) in enumerate(dfp.iterrows()):
        yy = y[i]
        ax.text(x_right * 0.98, yy, f"p={row['p']:.2g}", va="center", ha="right")

    fig.tight_layout()
    return fig

def metric_p_display(p_val: float) -> str:
    if p_val < 0.001:
        return f"{p_val:.4e} (<0.001)"
    return f"{p_val:.4e} ({p_val:.3f})"


def plot_km(
    df: pd.DataFrame,
    group_col: str,
    title: str,
    color_map: dict | None = None,
    group_order: list[str] | None = None,
):
    if group_order is None:
        groups = sorted(df[group_col].dropna().unique())
    else:
        groups = [g for g in group_order if g in set(df[group_col].dropna().unique())]

    fig, ax = plt.subplots(figsize=(8, 6))

    for g in groups:
        mask = df[group_col] == g
        if mask.sum() <= 0:
            continue
        label = f"{g} (n={mask.sum()})"
        kmf = KaplanMeierFitter()
        kmf.fit(df.loc[mask, "OS_MONTHS"], df.loc[mask, "is_dead"], label=label)
        if color_map and g in color_map:
            kmf.plot_survival_function(ax=ax, color=color_map[g])
        else:
            kmf.plot_survival_function(ax=ax)

    ax.set_title(title)
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Survival probability")
    
    fig.tight_layout()
    return fig

def plot_compare_main_effect(unadj_res: dict, adj_res: dict, main_label: str):
    """
    Compare only main effect HR between unadjusted and adjusted models.
    Inputs are outputs from run_cox() in this project.
    """
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    ax.axvline(1.0, linestyle="--")

    rows = []
    if unadj_res and unadj_res.get("ok"):
        rows.append(("Unadjusted", unadj_res["hr"], unadj_res["hr_l"], unadj_res["hr_u"], unadj_res["p"]))
    if adj_res and adj_res.get("ok"):
        rows.append(("Adjusted", adj_res["hr"], adj_res["hr_l"], adj_res["hr_u"], adj_res["p"]))

    y = np.arange(len(rows))[::-1]

    for i, (name, hr, l, u, p) in enumerate(rows):
        yy = y[i]
        ax.plot([l, u], [yy, yy])
        ax.plot(hr, yy, marker="s")
        ax.text(u * 1.05, yy, f"p={p:.2g}", va="center")

    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel(f"{main_label} (HR with 95% CI)")
    ax.set_title("Main Effect: Unadjusted vs Adjusted")
    ax.set_ylim(-1, len(rows))
    fig.tight_layout()
    return fig

def calculate_rmst(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    group_col: str = "group",
    tau: Optional[float] = None
) -> Dict[str, Any]:
    """
    Calculate Restricted Mean Survival Time (RMST) for each group up to time `tau`.
    If tau is None, it defaults to the minimum of the maximum observed times across groups.
    """
    if group_col not in df.columns or df.empty:
        return {"ok": False, "reason": f"Missing column: {group_col} or empty df"}
        
    df_clean = df.dropna(subset=[duration_col, event_col, group_col]).copy()
    if df_clean.empty:
        return {"ok": False, "reason": "Empty dataset after dropping NAs"}

    groups = df_clean[group_col].unique()
    if len(groups) < 2:
        return {"ok": False, "reason": "Requires at least 2 groups"}

    # Determine tau if not provided
    if tau is None or tau <= 0:
        max_times = []
        for g in groups:
            max_times.append(df_clean[df_clean[group_col] == g][duration_col].max())
        tau = float(min(max_times))
        
    results = {}
    for g in groups:
        mask = df_clean[group_col] == g
        kmf = KaplanMeierFitter()
        try:
            kmf.fit(df_clean.loc[mask, duration_col], df_clean.loc[mask, event_col])
            # Calculate Area Under Curve (RMST) up to tau
            # lifelines KMF timeline and survival_function_ can be integrated
            timeline = kmf.timeline
            survival = kmf.survival_function_.iloc[:, 0].values
            
            # Restrict to tau
            valid_idx = timeline <= tau
            t_valid = timeline[valid_idx]
            s_valid = survival[valid_idx]
            
            # If tau extends beyond the last observed time, carry forward the last survival probability
            if len(t_valid) == 0:
                rmst = 0.0
            else:
                if t_valid[-1] < tau:
                    t_valid = np.append(t_valid, tau)
                    s_valid = np.append(s_valid, s_valid[-1])
                    
                # Riemann sum (step function integration)
                dt = np.diff(t_valid)
                rmst = np.sum(dt * s_valid[:-1])
                
            results[str(g)] = float(rmst)
        except Exception:
            results[str(g)] = None

    return {
        "ok": True,
        "tau": float(tau),
        "rmst_values": results
    }

def get_survival_probability(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    group_col: str = "group",
    time_points: List[float] = [36.0, 60.0]  # e.g., 3-year and 5-year survival
) -> Dict[str, Any]:
    """
    Export Kaplan-Meier estimates of survival probability at specific time points.
    Returns: {"ok": True, "prob_values": { group: { time_point: probability } }}
    """
    if group_col not in df.columns or df.empty:
        return {"ok": False, "reason": f"Missing column: {group_col} or empty df"}
        
    df_clean = df.dropna(subset=[duration_col, event_col, group_col]).copy()
    if df_clean.empty:
        return {"ok": False, "reason": "Empty dataset after dropping NAs"}

    groups = df_clean[group_col].unique()
    if len(groups) < 2:
        return {"ok": False, "reason": "Requires at least 2 groups"}
        
    results = {}
    for g in groups:
        mask = df_clean[group_col] == g
        kmf = KaplanMeierFitter()
        results[str(g)] = {}
        try:
            kmf.fit(df_clean.loc[mask, duration_col], df_clean.loc[mask, event_col])
            
            for t in time_points:
                # predict(t) returns the survival probability at time t
                # If t is beyond max observed time, it carries forward the last value,
                # but we should probably cap it or return None if it's too far out.
                # lifelines `predict` handles this naturally (step function).
                prob = kmf.predict(t)
                results[str(g)][t] = float(prob)
                
        except Exception:
            for t in time_points:
                results[str(g)][t] = None

    return {
        "ok": True,
        "prob_values": results
    }

def run_cox(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    main_var: str,
    covariates: list[str],
    min_n: int = MIN_N_COX,
    min_events: int = MIN_EVENTS_COX,
):
    """Fit CoxPH on df using columns [main_var + covariates + duration + event]."""
    covariates = covariates or []
    covariates = [c for c in covariates if c in df.columns]
    use_cols = [main_var] + covariates + [duration_col, event_col]
    df_fit = df.dropna(subset=use_cols).copy()
    
    # --- Drop zero-variance covariates to prevent lin-alg convergence errors ---
    valid_covs = []
    for c in covariates:
        if df_fit[c].nunique(dropna=True) > 1:
            valid_covs.append(c)
            
    if len(valid_covs) != len(covariates):
        use_cols = [main_var] + valid_covs + [duration_col, event_col]
        df_fit = df_fit[use_cols]
    
    if is_gpu_enabled():
        pass # Placeholder for using cuML Cox models in the future

    issues = []
    if len(df_fit) < min_n or df_fit[event_col].sum() < min_events:
        issues.append(Issue(
            category="data sufficiency",
            severity="critical",
            kind="error",
            title="Fatal data insufficiency",
            detail=f"Sample size (N={len(df_fit)}) or event count ({int(df_fit[event_col].sum())}) is below the absolute minimum required for Cox modeling.",
            evidence=f"N={len(df_fit)}, events={int(df_fit[event_col].sum())}",
            recommendation="Increase sample size or use a simpler descriptive analysis.",
            source_module="survival.run_cox"
        ))
        return {
            "ok": False,
            "reason": f"Insufficient data for Cox (N={len(df_fit)}, events={int(df_fit[event_col].sum())}).",
            "model": None,
            "issues": issues
        }

    cph = CoxPHFitter()
    
    # 1. Data Sufficiency Checks
    n_fit = len(df_fit)
    events_fit = int(df_fit[event_col].sum())
    
    if events_fit < 15:
        issues.append(Issue(
            category="data sufficiency",
            severity="warning",
            kind="caution",
            title="Relatively few events",
            detail=f"Only {events_fit} events observed. Estimates may be unstable.",
            evidence=f"events={events_fit}",
            recommendation="Consider reducing covariates or using a larger dataset.",
            source_module="survival.run_cox"
        ))
        
    epv = events_fit / (len(covariates) + 1)
    if epv < 5:
        issues.append(Issue(
            category="data sufficiency",
            severity="warning",
            kind="caution",
            title="Low Events Per Variable (EPV)",
            detail=f"EPV={epv:.1f} is below the recommended threshold (10).",
            evidence=f"EPV={epv:.1f}",
            recommendation="Reduce the number of covariates to avoid overfitting.",
            source_module="survival.run_cox"
        ))

    try:
        cph.fit(df_fit[use_cols], duration_col=duration_col, event_col=event_col)
        summary = cph.summary.loc[main_var]
        
        # 2. Model Assumption Checks (PH violation)
        try:
            ph_test = proportional_hazard_test(cph, df_fit[use_cols], time_transform='rank')
            p_ph = ph_test.p_value.min()
            if p_ph < 0.05:
                issues.append(Issue(
                    category="model assumption",
                    severity="warning",
                    kind="caution",
                    title="Proportional Hazards (PH) violation detected",
                    detail="One or more variables appear to violate the PH assumption.",
                    evidence=f"PH test p-min={p_ph:.3f}",
                    recommendation="Review crossing KM curves or use time-varying Cox / RMST.",
                    source_module="survival.run_cox"
                ))
        except Exception:
            pass # PH test might fail for small datasets

        return {
            "ok": True,
            "model": cph,
            "data": df_fit[use_cols],
            "n": n_fit,
            "events": events_fit,
            "hr": float(summary["exp(coef)"]),
            "hr_l": float(summary["exp(coef) lower 95%"]),
            "hr_u": float(summary["exp(coef) upper 95%"]),
            "p": float(summary["p"]),
            "issues": issues
        }
    except Exception as e:
        return {"ok": False, "reason": f"Cox model error: {e}", "model": None, "issues": issues}

def run_cox_4group(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    group_col: str = "group",
    ref_group: str = "LoLo",
    covariates: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    4群（カテゴリ）を同一Coxモデルに入れて、ref_group基準で各群のHRを同時に推定する。
    - group_col: "HiHi/HiLo/LoHi/LoLo" を想定
    - ref_group: 基準群（例：LoLo）
    """
    covariates = [c for c in (covariates or []) if c in df.columns]

    if group_col not in df.columns:
        return {"ok": False, "reason": f"Missing column: {group_col}"}
    if duration_col not in df.columns or event_col not in df.columns:
        return {"ok": False, "reason": "Missing survival columns"}

    # groupをカテゴリ化（ref_group を先頭に）
    levels = [ref_group] + [g for g in ["HiHi", "HiLo", "LoHi", "LoLo"] if g != ref_group]
    # データ内に存在するものだけに絞る（将来拡張で）
    present = [g for g in levels if g in set(df[group_col].astype(str))]
    
    warnings_list = []
    if ref_group not in present:
        if not present:
            return {"ok": False, "reason": "No valid groups found (all empty)."}
        # Fallback to the most populated group
        counts = df[group_col].value_counts()
        actual_ref = counts.idxmax()
        warnings_list.append(f"Reference group '{ref_group}' not found (0 patients). Automatically changed reference to '{actual_ref}'.")
        ref_group = actual_ref
        levels = [ref_group] + [g for g in ["HiHi", "HiLo", "LoHi", "LoLo"] if g != ref_group]
        present = [g for g in levels if g in set(df[group_col].astype(str))]

    work = df.copy()
    work[group_col] = work[group_col].astype(str)
    work[group_col] = pd.Categorical(work[group_col], categories=present, ordered=True)

    # ダミー化：refを落として、他群の指標を作る
    dummies = pd.get_dummies(work[group_col], prefix=group_col, drop_first=True)

    # フィット用df
    use_cols = [duration_col, event_col] + covariates
    fit = work[use_cols].join(dummies)

    # 数値化＆欠損落とし
    fit[duration_col] = pd.to_numeric(fit[duration_col], errors="coerce")
    fit[event_col] = pd.to_numeric(fit[event_col], errors="coerce")

    # covariatesも念のため数値化（カテゴリは preprocess 側で数値化済み想定）
    for c in covariates:
        fit[c] = pd.to_numeric(fit[c], errors="coerce")

    fit = fit.dropna(axis=0, subset=[duration_col, event_col] + covariates + list(dummies.columns))
    
    # --- Drop zero-variance covariates to prevent lin-alg convergence errors ---
    valid_covs = []
    for c in covariates:
        if fit[c].nunique(dropna=True) > 1:
            valid_covs.append(c)
            
    fit = fit[[duration_col, event_col] + valid_covs + list(dummies.columns)]
    n = int(fit.shape[0])
    if n < 30:
        return {"ok": False, "reason": f"Too few samples after filtering: N={n}"}

    events = int(fit[event_col].sum()) if np.issubdtype(fit[event_col].dtype, np.number) else None

    try:
        cph = CoxPHFitter()
        cph.fit(fit, duration_col=duration_col, event_col=event_col)

        # group係数だけ抜き出して整形
        s = cph.summary.copy()
        g_rows = s.loc[s.index.str.startswith(f"{group_col}_")].copy()
        if g_rows.empty:
            return {"ok": False, "reason": "No group terms in model (unexpected)"}

        g_rows["HR"] = np.exp(g_rows["coef"])
        g_rows["HR_L"] = np.exp(g_rows["coef lower 95%"])
        g_rows["HR_U"] = np.exp(g_rows["coef upper 95%"])
        g_rows["p"] = g_rows["p"]

        # 表示用ラベル（group_HiHi -> HiHi vs LoLo）
        def _label(ix: str) -> str:
            # ix: group_HiHi
            tgt = ix.split("_", 1)[1] if "_" in ix else ix
            return f"{tgt} vs {ref_group}"

        out = pd.DataFrame({
            "contrast": [_label(ix) for ix in g_rows.index],
            "term": g_rows.index,
            "hr": g_rows["HR"].to_numpy(),
            "hr_l": g_rows["HR_L"].to_numpy(),
            "hr_u": g_rows["HR_U"].to_numpy(),
            "p": g_rows["p"].to_numpy(),
        }).sort_values("contrast")

        return {
            "ok": True,
            "model": cph,
            "data": fit,
            "table": out.reset_index(drop=True),
            "n": n,
            "events": events,
            "ref_group": ref_group,
        }

    except Exception as e:
        return {"ok": False, "reason": f"Cox failed: {e}"}

# ---------------------------------------------
#
# ---------------------------------------------

def plot_forest_contrasts(
    df,
    title="Cox Forest",
    xlabel="Hazard Ratio (HR) with 95% CI",
    p_digits=3,
    use_logx=False,
):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 4))

    y_pos = np.arange(len(df))

    # 有意判定
    significant = df["p"] < 0.05

    colors = ["#1f77b4" if sig else "#888888" for sig in significant]

    # エラーバー
    for i, row in df.iterrows():
        ax.plot(
            [row["hr_l"], row["hr_u"]],
            [i, i],
            color=colors[i],
            linewidth=2,
        )

        ax.scatter(
            row["hr"],
            i,
            color=colors[i],
            s=60,
            zorder=3,
        )

        # p値表示
        ax.text(
            ax.get_xlim()[1] * 0.98,
            i,
            f"p={row['p']:.{p_digits}g}",
            va="center",
            ha="right",
            fontsize=9,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["contrast"])
    ax.axvline(1.0, linestyle="--", linewidth=1.5, color="black")

    ax.set_xlabel(xlabel)
    ax.set_title(title)

    if use_logx:
        ax.set_xscale("log")

    ax.invert_yaxis()
    ax.grid(axis="x", linestyle=":", alpha=0.5)

    plt.tight_layout()
    return fig

def run_cox_interaction(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    x1_col: str,
    x2_col: str,
    covariates: Optional[List[str]] = None,
    zscore: bool = True,
) -> Dict[str, Any]:
    """
    Cox with continuous x1, x2 and interaction (x1*x2).
    - zscore=True: standardize x1/x2 on the fitted subset to reduce scaling issues.
    """
    covariates = covariates or []
    covariates = [c for c in covariates if c in df.columns]

    use_cols = [duration_col, event_col, x1_col, x2_col] + covariates
    missing = [c for c in use_cols if c not in df.columns]
    if missing:
        return {"ok": False, "reason": f"missing columns: {missing}"}

    df_fit = df[use_cols].copy()
    df_fit[duration_col] = pd.to_numeric(df_fit[duration_col], errors="coerce")
    df_fit[event_col] = pd.to_numeric(df_fit[event_col], errors="coerce")
    df_fit[x1_col] = pd.to_numeric(df_fit[x1_col], errors="coerce")
    df_fit[x2_col] = pd.to_numeric(df_fit[x2_col], errors="coerce")

    df_fit = df_fit.dropna(subset=[duration_col, event_col, x1_col, x2_col]).copy()

    # covariatesも numeric / category は upstream で整ってる前提（add_covariate_columns）
    # ここでは欠測だけ落とす
    if covariates:
        df_fit = df_fit.dropna(subset=covariates).copy()

    # --- Drop zero-variance covariates to prevent lin-alg convergence errors ---
    valid_covs = []
    for c in covariates:
        if df_fit[c].nunique(dropna=True) > 1:
            valid_covs.append(c)
    covariates = valid_covs

    n = len(df_fit)
    if n < 30:
        return {"ok": False, "reason": f"too few samples after dropna: n={n}"}

    events = int(df_fit[event_col].astype(int).sum())
    if events < 10:
        return {"ok": False, "reason": f"too few events: events={events}"}

    # z-score (center + scale)
    if zscore:
        x1_mean = df_fit[x1_col].mean()
        x2_mean = df_fit[x2_col].mean()
        x1_std = df_fit[x1_col].std(ddof=0)
        x2_std = df_fit[x2_col].std(ddof=0)

        # avoid zero std
        if x1_std == 0 or x2_std == 0 or np.isnan(x1_std) or np.isnan(x2_std):
            return {"ok": False, "reason": "zero/NaN std in x1 or x2 (cannot z-score)"}

        df_fit["x1_z"] = (df_fit[x1_col] - x1_mean) / x1_std
        df_fit["x2_z"] = (df_fit[x2_col] - x2_mean) / x2_std
        x1_use, x2_use = "x1_z", "x2_z"
    else:
        x1_use, x2_use = x1_col, x2_col

    df_fit["x1_x2"] = df_fit[x1_use] * df_fit[x2_use]

    cols_model = [duration_col, event_col, x1_use, x2_use, "x1_x2"] + covariates
    df_model = df_fit[cols_model].copy()

    try:
        cph = CoxPHFitter()
        cph.fit(df_model, duration_col=duration_col, event_col=event_col)

        summ = cph.summary.reset_index()

        # lifelinesのバージョン差を吸収（index名が covariate / index / など）
        if "covariate" in summ.columns:
            summ = summ.rename(columns={"covariate": "term"})
        elif "index" in summ.columns:
            summ = summ.rename(columns={"index": "term"})
        elif "variable" in summ.columns:
            summ = summ.rename(columns={"variable": "term"})
        else:
            # 最後の保険：先頭列をterm扱い（ほぼここには来ない想定）
            summ = summ.rename(columns={summ.columns[0]: "term"})

        # exp(coef) 系の列名もバージョン差があるので、存在チェックして拾う
        def _pick(cols):
            for c in cols:
                if c in summ.columns:
                    return c
            return None

        col_hr  = _pick(["exp(coef)", "exp(coef_)"])
        col_l   = _pick(["exp(coef) lower 95%", "exp(coef) lower 0.95", "exp(coef) lower 95"])
        col_u   = _pick(["exp(coef) upper 95%", "exp(coef) upper 0.95", "exp(coef) upper 95"])
        col_p   = _pick(["p", "p-value", "p_value"])

        if col_hr is None or col_l is None or col_u is None or col_p is None:
            return {"ok": False, "reason": f"unexpected lifelines summary columns: {list(summ.columns)}"}

        out = pd.DataFrame({
            "term": summ["term"].astype(str),
            "hr": summ[col_hr].astype(float),
            "hr_l": summ[col_l].astype(float),
            "hr_u": summ[col_u].astype(float),
            "p": summ[col_p].astype(float),
        })

        rename_map = {
            x1_use: f"{x1_col} (z)" if zscore else x1_col,
            x2_use: f"{x2_col} (z)" if zscore else x2_col,
            "x1_x2": f"{x1_col}×{x2_col} (interaction)",
        }
        out["label"] = out["term"].map(rename_map).fillna(out["term"])
        out = out[["label", "term", "hr", "hr_l", "hr_u", "p"]].sort_values("p")

        return {
            "ok": True,
            "model": cph,
            "data": df_model,
            "table": out.reset_index(drop=True),
            "n": n,
            "events": events,
            "zscore": zscore,
        }

    except Exception as ex:
        return {"ok": False, "reason": f"cox interaction failed: {ex}"}

def format_interaction_table_for_forest(
    table: pd.DataFrame,
    gene1_label: str,
    gene2_label: str,
) -> pd.DataFrame:
    """
    Input table columns: label, term, hr, hr_l, hr_u, p
    terms: x1_z, x2_z, x1_x2
    """
    if table is None or table.empty:
        return pd.DataFrame(columns=["term", "label", "hr", "hr_l", "hr_u", "p", "is_interaction"])

    df = table.copy()

    # pick only main + interaction terms (your convention)
    keep_terms = {"x1_z", "x2_z", "x1_x2"}
    df = df[df["term"].isin(keep_terms)].copy()

    # relabel for display (教育的に “gene名” を前面に)
    def _disp_label(term: str) -> str:
        if term == "x1_z":
            return f"{gene1_label} (z)"
        if term == "x2_z":
            return f"{gene2_label} (z)"
        if term == "x1_x2":
            return f"{gene1_label}×{gene2_label} ⭐"
        return term

    out = pd.DataFrame({
        "term": df["term"].astype(str),
        "label": df["term"].astype(str).map(_disp_label),
        "hr": df["hr"],
        "hr_l": df["hr_l"],
        "hr_u": df["hr_u"],
        "p": df["p"],
    })
    out["is_interaction"] = out["term"].eq("x1_x2")

    # order: x1, x2, interaction
    order = {"x1_z": 0, "x2_z": 1, "x1_x2": 2}
    out["__order"] = out["term"].map(order).fillna(9)
    out = out.sort_values("__order").drop(columns="__order").reset_index(drop=True)

    return out

def plot_interaction_forest(forest_df: pd.DataFrame, title: str):
    if forest_df is None or forest_df.empty:
        fig, ax = plt.subplots(figsize=(7, 2))
        ax.axis("off")
        ax.text(0, 0.5, "No interaction results to plot.", fontsize=11)
        return fig

    df = forest_df.copy()
    y = np.arange(len(df))[::-1]

    fig_h = max(2.6, 0.6 * len(df) + 1.2)
    fig, ax = plt.subplots(figsize=(8.5, fig_h))

    ax.axvline(1.0, linestyle="--", linewidth=1)

    for i, row in df.iterrows():
        yi = y[i]
        hr, lo, hi = row["hr"], row["hr_l"], row["hr_u"]
        is_int = bool(row["is_interaction"])

        lw = 2.3 if is_int else 1.2
        ms = 9 if is_int else 6

        ax.plot([lo, hi], [yi, yi], linewidth=lw)
        ax.plot(hr, yi, marker="o", markersize=ms)

        p = row.get("p", np.nan)
        if pd.notna(p):
            ax.text(
                1.02, yi, f"p={p:.3g}",
                transform=ax.get_yaxis_transform(),
                va="center",
                fontsize=10,
                fontweight="bold" if is_int else "normal"
            )

    ax.set_yticks(y)
    ax.set_yticklabels(df["label"].tolist(), fontsize=11)
    ax.set_xscale("log")
    ax.set_xlabel("Hazard Ratio (log scale)")
    ax.set_title(title)

    xmin = np.nanmin(df["hr_l"].values)
    xmax = np.nanmax(df["hr_u"].values)
    if np.isfinite(xmin) and np.isfinite(xmax) and xmin > 0:
        ax.set_xlim(xmin * 0.8, xmax * 1.25)

    ax.grid(True, axis="x", linewidth=0.4)
    fig.tight_layout()
    return fig


# =========================
# Cut Smoothness Curve
# =========================
def run_cut_smoothness_scan(
    merged: pd.DataFrame,
    duration_col: str = "OS_MONTHS",
    event_col: str = "is_dead",
    x1_col: str = "expression_1",
    x2_col: str = "expression_2",
    min_pct: int = 10,
    max_pct: int = 60,
    step: int = 5,
) -> Dict[str, Any]:
    from lifelines.statistics import multivariate_logrank_test

    results = {"pct_cutoffs": [], "p_values": [], "min_p": np.nan, "max_p": np.nan, "ok": True}

    # guardrails
    if merged is None or merged.empty:
        results["ok"] = False
        results["reason"] = "merged is empty"
        return results

    need = [duration_col, event_col, x1_col, x2_col]
    missing = [c for c in need if c not in merged.columns]
    if missing:
        results["ok"] = False
        results["reason"] = f"Missing columns: {missing}"
        return results

    # ✅ 必要列だけに絞る（ここから下は tmp0 だけを使う）
    tmp0 = merged[[duration_col, event_col, x1_col, x2_col]].copy()

    # 型を安全化
    tmp0[duration_col] = pd.to_numeric(tmp0[duration_col], errors="coerce")
    tmp0[event_col] = pd.to_numeric(tmp0[event_col], errors="coerce")

    # event を 0/1 に寄せる（True/Falseや2以上が混ざってもOK）
    tmp0[event_col] = (tmp0[event_col] > 0).astype(int)

    tmp0 = tmp0.dropna()
    if tmp0.empty:
        results["ok"] = False
        results["reason"] = "No valid rows after dropna/type coercion"
        return results

    # pct list
    pcts = list(range(min_pct, max_pct + 1, step))
    if max_pct not in pcts:
        pcts.append(max_pct)

    # scan
    for pct in pcts:
        q = pct / 100.0
        c1 = tmp0[x1_col].quantile(q)
        c2 = tmp0[x2_col].quantile(q)

        g1_hi = tmp0[x1_col] >= c1
        g2_hi = tmp0[x2_col] >= c2

        tmp_group = np.where(g1_hi & g2_hi, "HiHi",
                     np.where(g1_hi & ~g2_hi, "HiLo",
                     np.where(~g1_hi & g2_hi, "LoHi", "LoLo")))

        try:
            lr = multivariate_logrank_test(
                tmp0[duration_col],
                tmp_group,
                tmp0[event_col],
            )
            p_val = float(lr.p_value)
        except Exception:
            p_val = np.nan

        results["pct_cutoffs"].append(int(pct))
        results["p_values"].append(p_val)

    p_vals = [p for p in results["p_values"] if pd.notna(p)]
    if p_vals:
        results["min_p"] = float(min(p_vals))
        results["max_p"] = float(max(p_vals))

    return results

def plot_cut_smoothness_curve(
    scan_results: Dict[str, Any],
    title: str = "Cut Smoothness Curve: p-value Stability",
    xlabel: str = "Percentile Cutoff (%)",
):
    pcts = scan_results.get("pct_cutoffs", [])
    p_vals = np.array(scan_results.get("p_values", []), dtype=float)

    # log軸の安全策：0以下はNaNへ（またはepsで丸め）
    p_vals[p_vals <= 0] = np.nan

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(pcts, p_vals, marker="o", linewidth=2, markersize=6, label="log-rank p-value")

    ax.axhline(0.05, linestyle="--", linewidth=1.5, label="p=0.05", alpha=0.7)
    ax.axhline(0.1, linestyle="--", linewidth=1.0, label="p=0.10", alpha=0.5)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Log-rank p-value", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)

    fig.tight_layout()
    return fig

def run_cox_4group_stratified(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    group_col: str = "group",
    ref_group: str = "LoLo",
    covariates: Optional[List[str]] = None,
    strata_col: str = "CANCER_TYPE",
    min_n: int = 80,
    min_events: int = 20,
) -> Dict[str, Any]:
    """
    4群Cox（同一モデル） + strata（例: CANCER_TYPE）でベースラインを癌種ごとに分ける。
    """
    covariates = [c for c in covariates if c in df.columns]
    need = [duration_col, event_col, group_col, strata_col] + covariates
    missing = [c for c in need if c not in df.columns]
    if missing:
        return {"ok": False, "reason": f"missing columns: {missing}"}

    work = df[need].copy()
    work[duration_col] = pd.to_numeric(work[duration_col], errors="coerce")
    work[event_col] = (pd.to_numeric(work[event_col], errors="coerce") > 0).astype(int)
    for c in covariates:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna()

    n = int(work.shape[0])
    events = int(work[event_col].sum())
    if n < min_n or events < min_events:
        return {"ok": False, "reason": f"Insufficient data: N={n}, events={events}"}

    # groupをカテゴリ化（ref_groupを先頭に）
    levels = [ref_group] + [g for g in ["HiHi", "HiLo", "LoHi", "LoLo"] if g != ref_group]
    present = [g for g in levels if g in set(work[group_col].astype(str))]
    if ref_group not in present:
        return {"ok": False, "reason": f"Reference group '{ref_group}' not found"}

    work[group_col] = work[group_col].astype(str)
    work[group_col] = pd.Categorical(work[group_col], categories=present, ordered=True)

    dummies = pd.get_dummies(work[group_col], prefix=group_col, drop_first=True)

    use_cols = [duration_col, event_col, strata_col] + covariates
    fit = work[use_cols].join(dummies)

    # lifelines strata はカテゴリ/文字列でもOKだが、念のため文字列化
    fit[strata_col] = fit[strata_col].astype(str)

    try:
        cph = CoxPHFitter()
        cph.fit(
            fit,
            duration_col=duration_col,
            event_col=event_col,
            strata=[strata_col],
        )

        s = cph.summary.copy()
        g_rows = s.loc[s.index.str.startswith(f"{group_col}_")].copy()
        if g_rows.empty:
            return {"ok": False, "reason": "No group terms in model"}

        g_rows["HR"] = np.exp(g_rows["coef"])
        g_rows["HR_L"] = np.exp(g_rows["coef lower 95%"])
        g_rows["HR_U"] = np.exp(g_rows["coef upper 95%"])
        g_rows["p"] = g_rows["p"]

        def _label(ix: str) -> str:
            tgt = ix.split("_", 1)[1] if "_" in ix else ix
            return f"{tgt} vs {ref_group}"

        out = pd.DataFrame({
            "contrast": [_label(ix) for ix in g_rows.index],
            "term": g_rows.index,
            "hr": g_rows["HR"].to_numpy(),
            "hr_l": g_rows["HR_L"].to_numpy(),
            "hr_u": g_rows["HR_U"].to_numpy(),
            "p": g_rows["p"].to_numpy(),
        }).sort_values("contrast")

        return {
            "ok": True,
            "model": cph,
            "table": out.reset_index(drop=True),
            "n": n,
            "events": events,
            "strata_col": strata_col,
            "ref_group": ref_group,
        }

    except Exception as e:
        return {"ok": False, "reason": f"Cox stratified failed: {e}"}


def summarize_by_cancer_type(
    merged: pd.DataFrame,
    *,
    duration_col: str = "OS_MONTHS",
    event_col: str = "is_dead",
    group_col: str = "group",
    cancer_col: str = "CANCER_TYPE",
    covariates: Optional[List[str]] = None,
    min_n: int = 80,
    min_events: int = 20,
) -> pd.DataFrame:
    """
    Cancer typeごとに、KM log-rank p と 4群Cox(共変量入り)の各コントラスト
    (HiHi/HiLo/LoHi vs LoLo) を要約して返す。
    """
    covariates = covariates or []
    covariates = [c for c in covariates if c in merged.columns]

    if cancer_col not in merged.columns:
        return pd.DataFrame()

    out: List[Dict[str, Any]] = []

    for ct, df_ct in merged.groupby(cancer_col):
        df_ct = df_ct.copy()

        # sanitize
        df_ct[duration_col] = pd.to_numeric(df_ct[duration_col], errors="coerce")
        df_ct[event_col] = (pd.to_numeric(df_ct[event_col], errors="coerce") > 0).astype(int)
        df_ct = df_ct.dropna(subset=[duration_col, event_col, group_col])

        n = int(df_ct.shape[0])
        events = int(df_ct[event_col].sum())

        row: Dict[str, Any] = {
            cancer_col: str(ct),
            "N": n,
            "events": events,
            "km_logrank_p": np.nan,
            "cox_HiHi_vs_LoLo_HR": np.nan,
            "cox_HiHi_p": np.nan,
            "cox_HiLo_vs_LoLo_HR": np.nan,
            "cox_HiLo_p": np.nan,
            "cox_LoHi_vs_LoLo_HR": np.nan,
            "cox_LoHi_p": np.nan,
            "note": "",
        }

        if n < min_n or events < min_events:
            row["note"] = f"skip (N<{min_n} or events<{min_events})"
            out.append(row)
            continue

        # KM log-rank (4 groups)
        try:
            # df_ct は cancer subset（merged を groupby したときの df_ct）
            vc = df_ct[group_col].astype(str).value_counts()

            # 2群以上が十分な人数（>=10）を満たす時だけ
            ok_groups = vc[vc >= 10].index.tolist()
            if len(ok_groups) >= 2:
                dkm = df_ct[df_ct[group_col].astype(str).isin(ok_groups)].copy()

                # 全体イベントが十分ある時だけ
                if int(dkm[event_col].sum()) >= 10:
                    lr = multivariate_logrank_test(
                        dkm[duration_col],
                        dkm[group_col].astype(str),
                        dkm[event_col],
                    )
                    row["km_logrank_p"] = float(lr.p_value)
                else:
                    row["km_logrank_p"] = np.nan
            else:
                row["km_logrank_p"] = np.nan
        except Exception as e:
            # デバッグ ---------------------# 
            print("KM failed:", ct, e)
            row["km_logrank_p"] = np.nan
            row["note"] = f"KM_fail: {type(e).__name__}"
            # Debug End--------------------#
            row["km_logrank_p"] = np.nan

        # Cox 4-group within cancer type
        keep_cov = [c for c in covariates if c in df_ct.columns]
        keep_cols = [duration_col, event_col, group_col] + keep_cov
        df_fit = df_ct[keep_cols].dropna().copy()

        cox4 = run_cox_4group(
            df_fit,
            duration_col=duration_col,
            event_col=event_col,
            group_col=group_col,
            ref_group="LoLo",
            covariates=keep_cov,
        )

        if cox4.get("ok"):
            tbl = cox4["table"].copy()
            # ensure numeric
            tbl["hr"] = pd.to_numeric(tbl["hr"], errors="coerce")
            tbl["p"] = pd.to_numeric(tbl["p"], errors="coerce")

            def _pick(contrast_prefix: str) -> Tuple[float, float]:
                hit = tbl[tbl["contrast"].str.startswith(contrast_prefix)]
                if len(hit) == 0:
                    return (np.nan, np.nan)
                r0 = hit.iloc[0]
                return (float(r0["hr"]), float(r0["p"]))

            row["cox_HiHi_vs_LoLo_HR"], row["cox_HiHi_p"] = _pick("HiHi vs")
            row["cox_HiLo_vs_LoLo_HR"], row["cox_HiLo_p"] = _pick("HiLo vs")
            row["cox_LoHi_vs_LoLo_HR"], row["cox_LoHi_p"] = _pick("LoHi vs")

        out.append(row)

    df_out = pd.DataFrame(out)
    if df_out.empty:
        return df_out

    # force numeric (avoid None showing up in UI)
    for c in [
        "km_logrank_p",
        "cox_HiHi_vs_LoLo_HR", "cox_HiHi_p",
        "cox_HiLo_vs_LoLo_HR", "cox_HiLo_p",
        "cox_LoHi_vs_LoLo_HR", "cox_LoHi_p",
    ]:
        df_out[c] = pd.to_numeric(df_out[c], errors="coerce")

    # sort: LoHi p first (because often it is the driver), then KM p
    df_out = df_out.sort_values(
        ["cox_LoHi_p", "cox_HiHi_p", "cox_HiLo_p", "km_logrank_p"],
        na_position="last",
    ).reset_index(drop=True)

    return df_out

def find_optimal_1gene_cutoff(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    expr_col: str = "expression",
    min_pct: int = 20,
    max_pct: int = 80,
    step: int = 1,
) -> Dict[str, Any]:
    from lifelines.statistics import logrank_test
    import numpy as np

    results = {"optimal_pct": 50, "optimal_cutoff": np.nan, "min_p": 1.0, "ok": False}
    if df is None or df.empty:
        return results
    
    need = [duration_col, event_col, expr_col]
    missing = [c for c in need if c not in df.columns]
    if missing:
        return results

    tmp = df[need].copy()
    tmp[duration_col] = pd.to_numeric(tmp[duration_col], errors="coerce")
    tmp[event_col] = (pd.to_numeric(tmp[event_col], errors="coerce") > 0).astype(int)
    tmp[expr_col] = pd.to_numeric(tmp[expr_col], errors="coerce")
    tmp = tmp.dropna()

    if len(tmp) < 20 or tmp[event_col].sum() < 5:
        return results
        
    best_p = 1.0
    best_pct = 50
    best_cut = tmp[expr_col].median()
    
    for pct in range(min_pct, max_pct + 1, step):
        cut = tmp[expr_col].quantile(pct / 100.0)
        idx_high = tmp[expr_col] >= cut
        
        # Ensure we have at least some samples in both groups
        if idx_high.sum() < 5 or (~idx_high).sum() < 5:
            continue
            
        try:
            lr = logrank_test(
                tmp.loc[idx_high, duration_col],
                tmp.loc[~idx_high, duration_col],
                tmp.loc[idx_high, event_col],
                tmp.loc[~idx_high, event_col],
            )
            p_val = float(lr.p_value)
            if p_val < best_p:
                best_p = p_val
                best_pct = pct
                best_cut = cut
        except Exception:
            pass

    results["optimal_pct"] = int(best_pct)
    results["optimal_cutoff"] = float(best_cut)
    results["min_p"] = float(best_p)
    results["ok"] = True
    return results

def apply_landmark(df: pd.DataFrame, duration_col: str, event_col: str, landmark_time: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applies landmark analysis to survival data.
    Returns early_df, late_df.
    Early events are those <= landmark_time. Events after are censored at landmark_time.
    Late cohort only includes patients who survived to landmark_time. Their duration is shifted by -landmark_time.
    """
    if df is None or df.empty or landmark_time <= 0:
        return df, None
        
    early_df = df.copy()
    late_df = df[df[duration_col] >= landmark_time].copy()
    
    # Early cohort: events after landmark are censored at landmark
    early_df.loc[early_df[duration_col] > landmark_time, event_col] = 0
    early_df[duration_col] = early_df[duration_col].clip(upper=landmark_time)
    
    # Late cohort: duration is shifted relative to the landmark time
    late_df[duration_col] = late_df[duration_col] - landmark_time
    
    return early_df, late_df
def run_1gene_cut_smoothness_scan(
    merged: pd.DataFrame,
    duration_col: str = "OS_MONTHS",
    event_col: str = "is_dead",
    x_col: str = "expression",
    min_pct: int = 10,
    max_pct: int = 90,
    step: int = 2,
) -> Dict[str, Any]:
    from lifelines.statistics import multivariate_logrank_test
    import numpy as np
    import pandas as pd

    results = {"pct_cutoffs": [], "p_values": [], "min_p": np.nan, "max_p": np.nan, "ok": True}

    if merged is None or merged.empty:
        results["ok"] = False
        results["reason"] = "merged is empty"
        return results

    need = [duration_col, event_col, x_col]
    missing = [c for c in need if c not in merged.columns]
    if missing:
        results["ok"] = False
        results["reason"] = f"Missing columns: {missing}"
        return results

    tmp0 = merged[[duration_col, event_col, x_col]].copy()
    tmp0[duration_col] = pd.to_numeric(tmp0[duration_col], errors="coerce")
    tmp0[event_col] = pd.to_numeric(tmp0[event_col], errors="coerce")
    tmp0[event_col] = (tmp0[event_col] > 0).astype(int)
    tmp0 = tmp0.dropna()
    
    if tmp0.empty:
        results["ok"] = False
        results["reason"] = "No valid rows"
        return results

    pcts = list(range(min_pct, max_pct + 1, step))
    if max_pct not in pcts:
        pcts.append(max_pct)

    for pct in pcts:
        q = pct / 100.0
        c1 = tmp0[x_col].quantile(q)
        g_hi = tmp0[x_col] >= c1
        tmp_group = np.where(g_hi, "High", "Low")

        try:
            lr = multivariate_logrank_test(tmp0[duration_col], tmp_group, tmp0[event_col])
            p_val = float(lr.p_value)
        except Exception:
            p_val = np.nan

        results["pct_cutoffs"].append(int(pct))
        results["p_values"].append(p_val)

    p_vals = [p for p in results["p_values"] if pd.notna(p)]
    if p_vals:
        results["min_p"] = float(min(p_vals))
        results["max_p"] = float(max(p_vals))

    return results

# ---------------------------------------------
# Diagnostics
# ---------------------------------------------

def run_ph_test(cph: CoxPHFitter, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Runs the proportional hazards assumption test using Schoenfeld residuals.
    Returns the p-values for all covariates. A p-value < 0.05 indicates a violation
    of the proportional hazards assumption for that covariate.
    """
    try:
        results = proportional_hazard_test(cph, df, time_transform='rank')
        # lifelines proportional_hazard_test returns a StatisticalResult object.
        # It contains a summary dataframe with p-values per variable.
        return {
            "ok": True,
            "p_values": results.summary['p'].to_dict(), # {var_name: p_value}
            "summary": results.summary
        }
    except Exception as e:
        return {"ok": False, "reason": str(e)}

def plot_schoenfeld_residuals(cph: CoxPHFitter, df: pd.DataFrame, var_name: str, title: Optional[str] = None):
    """
    Plots the scaled Schoenfeld residuals for a specific variable against time.
    Provides a visual check of the PH assumption.
    """
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        cph.check_assumptions(df, p_value_threshold=0.05, show_plots=True, columns=[var_name], axes=ax)
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"Schoenfeld Residuals: {var_name}")
            
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Error plotting Schoenfeld residuals: {e}")
        return None

def get_at_risk_df(df, group_col, duration_col="OS_MONTHS", event_col="is_dead", group_order=None, limit_ticks=6):
    """
    Generate a Number-at-Risk table as a pandas DataFrame.
    Samples the counts at 'limit_ticks' evenly spaced time points up to max_time.
    """
    from lifelines import KaplanMeierFitter
    from matplotlib.ticker import MaxNLocator
    import pandas as pd
    import numpy as np

    if group_order is None:
        groups = sorted(df[group_col].dropna().unique())
    else:
        groups = [g for g in group_order if g in set(df[group_col].dropna().unique())]
        
    if df.empty or len(groups) == 0:
        return pd.DataFrame()

    max_time = df[duration_col].max()
    locator = MaxNLocator(limit_ticks)
    ticks = locator.tick_values(0, max_time)
    ticks = [t for t in ticks if 0 <= t <= max_time]
    if len(ticks) == 0:
        ticks = [0, max_time]

    rows = []
    kmf = KaplanMeierFitter()
    for g in groups:
        mask = df[group_col] == g
        if mask.sum() == 0: continue
        kmf.fit(df.loc[mask, duration_col], df.loc[mask, event_col])
        ev = kmf.event_table
        
        at_risk = []
        censored = []
        observed = []
        
        for t in ticks:
            # At risk just prior to time t
            v = ev.index[ev.index >= t]
            if len(v) > 0:
                ar = ev.loc[v[0], 'at_risk']
            else:
                ar = 0
            at_risk.append(ar)
            
            # Cumulative events and censors up to time t
            past = ev.index[ev.index <= t]
            if len(past) > 0:
                obs = ev.loc[past, 'observed'].sum()
                cen = ev.loc[past, 'censored'].sum()
            else:
                obs = 0
                cen = 0
            observed.append(obs)
            censored.append(cen)
            
        group_label = f"{g} (n={mask.sum()})"
        rows.append({"Group": group_label, "Metric": "At risk", **{f"{t:g}": c for t, c in zip(ticks, at_risk)}})
        rows.append({"Group": group_label, "Metric": "Censored", **{f"{t:g}": c for t, c in zip(ticks, censored)}})
        rows.append({"Group": group_label, "Metric": "Events", **{f"{t:g}": c for t, c in zip(ticks, observed)}})
        
    return pd.DataFrame(rows)

from lifelines import CoxTimeVaryingFitter
from lifelines.utils import to_episodic_format

def run_cox_time_varying(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    main_var: str,
    covariates: list[str] | None = None,
    time_transform: str = 'log'
) -> dict:
    """
    Fits a Time-Varying Coefficient Cox model to evaluate non-proportional hazards.
    """
    if df.empty:
        return {"ok": False, "reason": "Empty dataset"}
        
    cols_to_keep = [duration_col, event_col, main_var]
    if covariates:
        cols_to_keep.extend(covariates)
    
    # Pre-clean: only keep necessary columns to avoid passing strings (like exact STAGE) to CoxTimeVaryingFitter
    df_clean = df[cols_to_keep].dropna().copy()
    
    # ⚠️ lifelines to_episodic_format crashes with "IndexError: index -1 is out of bounds for axis 0 with size 0" if duration <= 0
    df_clean = df_clean[df_clean[duration_col] > 0].copy()
    
    if df_clean.empty:
        return {"ok": False, "reason": "Dataset empty after dropping NaNs and non-positive durations"}

    # to_episodic_format expects an ID column
    df_clean['__id'] = np.arange(len(df_clean))
    
    try:
        # Convert to episodic (long) format, breaking timelines at events
        df_long = to_episodic_format(df_clean, duration_col=duration_col, event_col=event_col, id_col='__id')
        
        # Create interaction term (Z * f(t))
        # df_long['stop'] behaves as the time `t` for the interval ending
        stop_time = df_long['stop'].astype(float)
        main_val = df_long[main_var].astype(float)
        
        if time_transform == 'log':
            # log(time + 1) to avoid log(0) just in case
            df_long[f"{main_var}_t"] = main_val * np.log1p(stop_time)
        elif time_transform == 'linear':
            df_long[f"{main_var}_t"] = main_val * stop_time
        else:
            return {"ok": False, "reason": f"Unknown time_transform: {time_transform}"}
            
        ctv = CoxTimeVaryingFitter(penalizer=0.01) # Small L2 penalty for stability
        
        ctv.fit(
            df_long,
            id_col='__id',
            event_col=event_col,
            start_col='start',
            stop_col='stop',
            show_progress=False
        )
        
        summary = ctv.summary.copy()
        
        # Extract main and time coefficients safely
        if main_var not in summary.index or f"{main_var}_t" not in summary.index:
            return {"ok": False, "reason": "Missing interaction variables in model output"}
            
        res = {
            "ok": True,
            "model": ctv,
            "summary": summary,
            "main_var": main_var,
            "time_transform": time_transform,
            "beta_main": summary.loc[main_var, "coef"],
            "se_main": summary.loc[main_var, "se(coef)"],
            "beta_time": summary.loc[f"{main_var}_t", "coef"],
            "se_time": summary.loc[f"{main_var}_t", "se(coef)"],
            "p_time": summary.loc[f"{main_var}_t", "p"],
            "covariates_used": [c for c in (covariates or []) if c in summary.index],
            "max_time": df_clean[duration_col].max(),
            "n": ctv._n_examples,
            "events": df_clean[event_col].sum()
        }
        return res
        
    except Exception as e:
        return {"ok": False, "reason": f"CoxTimeVarying fitting failed: {str(e)}"}

def get_time_varying_snapshots(res: dict, timepoints: list[float] = [12, 24, 36, 60]) -> pd.DataFrame:
    """
    Returns point estimates of HR(t) at specific months.
    """
    if not res.get("ok"):
        return pd.DataFrame()
        
    b_main = res["beta_main"]
    b_time = res["beta_time"]
    se_main = res["se_main"]
    se_time = res["se_time"]
    transform = res["time_transform"]
    
    # We must properly compute variance of the combined estimate over time.
    # var(b_main + f(t)*b_time) = var(b_main) + f(t)^2 * var(b_time) + 2*f(t)*cov(b_main, b_time)
    # Using the fitter variance_matrix_
    cov_matrix = res["model"].variance_matrix_
    
    # In some lifelines versions, variance_matrix_ has string index but integer columns
    if isinstance(cov_matrix.columns, pd.RangeIndex):
        cov_matrix.columns = cov_matrix.index
        
    main_var = res["main_var"]
    time_var = f"{main_var}_t"
    
    var_m = cov_matrix.loc[main_var, main_var]
    var_t = cov_matrix.loc[time_var, time_var]
    cov_mt = cov_matrix.loc[main_var, time_var]
    
    rows = []
    for t in timepoints:
        if t > res["max_time"]:
            continue
            
        if transform == 'log':
            ft = np.log1p(t)
        else:
            ft = t
            
        beta_t = b_main + ft * b_time
        var_beta_t = var_m + (ft**2)*var_t + 2*ft*cov_mt
        se_beta_t = np.sqrt(var_beta_t)
        
        hr = np.exp(beta_t)
        hr_l = np.exp(beta_t - 1.96 * se_beta_t)
        hr_u = np.exp(beta_t + 1.96 * se_beta_t)
        
        rows.append({
            "Month": float(t),
            "HR(t)": hr,
            "95% CI Lower": hr_l,
            "95% CI Upper": hr_u
        })
        
    return pd.DataFrame(rows)

def plot_hr_over_time(res: dict, title: str = "Time-Varying Hazard Ratio HR(t)") -> plt.Figure:
    """
    Plots the HR(t) trajectory over the entire duration of follow-up.
    """
    if not res.get("ok"):
        return plt.subplots(figsize=(6, 4))[0]

    max_t = res["max_time"]
    t_vals = np.linspace(0, max_t, 100)
    
    b_main = res["beta_main"]
    b_time = res["beta_time"]
    transform = res["time_transform"]
    
    cov_matrix = res["model"].variance_matrix_
    if isinstance(cov_matrix.columns, pd.RangeIndex):
        cov_matrix.columns = cov_matrix.index
        
    main_var = res["main_var"]
    time_var = f"{main_var}_t"
    var_m = cov_matrix.loc[main_var, main_var]
    var_t = cov_matrix.loc[time_var, time_var]
    cov_mt = cov_matrix.loc[main_var, time_var]

    hr_vals = []
    lb_vals = []
    ub_vals = []
    
    for t in t_vals:
        if transform == 'log':
            ft = np.log1p(t)
        else:
            ft = t
            
        beta_t = b_main + ft * b_time
        var_beta_t = var_m + (ft**2)*var_t + 2*ft*cov_mt
        
        # Safeguard against negative variance due to floating precision issues
        var_beta_t = max(var_beta_t, 1e-12)
        se_beta_t = np.sqrt(var_beta_t)
        
        hr_vals.append(np.exp(beta_t))
        lb_vals.append(np.exp(beta_t - 1.96 * se_beta_t))
        ub_vals.append(np.exp(beta_t + 1.96 * se_beta_t))
        
    fig, ax = plt.subplots(figsize=(7, 4))
    
    ax.axhline(1.0, color='grey', linestyle='--', alpha=0.8)
    ax.plot(t_vals, hr_vals, color='firebrick', lw=2, label=f"HR(t)")
    ax.fill_between(t_vals, lb_vals, ub_vals, color='firebrick', alpha=0.15, label="95% CI")
    
    ax.set_yscale('log')
    ax.set_ylabel("Hazard Ratio (log scale)")
    ax.set_xlabel("Time (Months)")
    ax.set_title(title)
    
    # Indicate significance of the interaction term
    p_time = res["p_time"]
    sig_text = "Significant" if p_time < 0.05 else "Non-significant"
    ax.text(0.02, 0.95, f"Interaction p-value: {p_time:.3e} ({sig_text})", 
            transform=ax.transAxes, va='top', ha='left',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
            
    fig.tight_layout()
    return fig
