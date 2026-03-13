# src/preprocess.py
import re
import pandas as pd


# -------------------------
# clinical mapping helpers
# -------------------------
def _first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def _normalize_sex_to_is_male(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    # cBioPortal / TCGAでよくある表記揺れを吸収
    return s.isin(["male", "m", "1"]).astype("Int64")


def _extract_stage_roman(text: str) -> str | None:
    if text is None:
        return None
    t = str(text).upper()

    # よくある "STAGE IIA", "Stage III", "IV", "T1N0..." などから
    # まずはローマ数字のI/II/III/IVだけを拾う（新人向けはここで十分）
    m = re.search(r"\b(IV|III|II|I)\b", t)
    if m:
        return m.group(1)

    # "STAGE IA"みたいに境界が怪しい場合の保険（IA, IIBなど）
    m = re.search(r"(IV|III|II|I)[A-C]?\b", t)
    if m:
        return m.group(1)

    return None


def _stage_num_from_series(stage_series: pd.Series) -> pd.Series:
    roman = stage_series.apply(_extract_stage_roman)
    stage_map = {"I": 1, "II": 2, "III": 3, "IV": 4}
    return roman.map(stage_map).astype("Float64")


def normalize_clinical(df_clin_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize clinical df into common columns used by the app.
    Output guarantees:
      - OS_MONTHS (float)
      - OS_STATUS (string, best-effort)
      - is_dead (0/1)
      - AGE (float) if detectable else NA
      - SEX (string) if detectable else NA
      - is_male (0/1/NA)
      - AJCC_PATHOLOGIC_TUMOR_STAGE (string) if detectable else NA
      - stage_num (1-4/NA)
      - PATIENT_ID must exist (else raises)
    """
    df = df_clin_raw.copy()

    # --- PATIENT_ID ---
    pid_col = _first_existing_col(df, ["PATIENT_ID", "PATIENT", "patient_id", "CASE_ID", "case_id"])
    if pid_col is None:
        raise ValueError("Clinical file: cannot find patient id column (PATIENT_ID/CASE_ID).")
    if pid_col != "PATIENT_ID":
        df["PATIENT_ID"] = df[pid_col]

    # --- Survival time (months) ---
    # cBioPortal/TCGAでありがちな候補を広めに
    time_col = _first_existing_col(
        df,
        [
            "OS_MONTHS", "OS_MONTH",
            "OS.time", "OS_TIME", "OS_TIME_MONTHS",
            "Overall Survival (Months)",
            "OS_DAYS", "OS_DAY", "OS_TIME_DAYS",
        ],
    )
    if time_col is None:
        raise ValueError("Clinical file: cannot find survival time column (e.g., OS_MONTHS / OS_DAYS).")

    # OS_DAYS系なら months に変換
    if "DAYS" in time_col.upper():
        df["OS_MONTHS"] = pd.to_numeric(df[time_col], errors="coerce") / 30.4375
    else:
        df["OS_MONTHS"] = pd.to_numeric(df[time_col], errors="coerce")

    # --- Survival status -> is_dead ---
    status_col = _first_existing_col(
        df,
        [
            "OS_STATUS", "OS_STATUS_TEXT", "OS",
            "OS.event", "OS_EVENT", "OS_STATUS_BINARY",
            "Overall Survival Status",
            "DECEASED", "DEAD",
        ],
    )

    # いったんOS_STATUSは残す（表示/デバッグ用）
    if status_col is not None and status_col != "OS_STATUS":
        df["OS_STATUS"] = df[status_col]
    elif status_col is None:
        df["OS_STATUS"] = pd.NA

    # is_dead 推定：文字列型（DECEASED/DEAD/1など）に対応
    s = df["OS_STATUS"].astype(str).str.upper()
    # cBioPortalは "DECEASED"/"LIVING" が多い
    is_dead = s.str.contains("DECEASED") | s.str.contains("DEAD")
    # 数値っぽいもの（1=event）も拾う
    s_num = pd.to_numeric(df["OS_STATUS"], errors="coerce")
    is_dead = is_dead | (s_num == 1)

    df["is_dead"] = is_dead.astype(int)

    # --- AGE ---
    age_col = _first_existing_col(df, ["AGE", "AGE_AT_DIAGNOSIS", "AGE_AT_INITIAL_PATHOLOGIC_DIAGNOSIS", "age"])
    if age_col is not None and age_col != "AGE":
        df["AGE"] = df[age_col]
    df["AGE"] = pd.to_numeric(df.get("AGE", pd.NA), errors="coerce")

    # --- SEX ---
    sex_col = _first_existing_col(df, ["SEX", "GENDER", "sex", "gender"])
    if sex_col is not None and sex_col != "SEX":
        df["SEX"] = df[sex_col]
    df["SEX"] = df.get("SEX", pd.NA)
    df["is_male"] = _normalize_sex_to_is_male(df["SEX"]) if "SEX" in df.columns else pd.NA

    # --- Stage ---
    stage_col = _first_existing_col(
        df,
        [
            "AJCC_PATHOLOGIC_TUMOR_STAGE",
            "PATHOLOGIC_STAGE",
            "TUMOR_STAGE",
            "STAGE",
            "ajcc_pathologic_tumor_stage",
        ],
    )
    if stage_col is not None and stage_col != "AJCC_PATHOLOGIC_TUMOR_STAGE":
        df["AJCC_PATHOLOGIC_TUMOR_STAGE"] = df[stage_col]
    df["AJCC_PATHOLOGIC_TUMOR_STAGE"] = df.get("AJCC_PATHOLOGIC_TUMOR_STAGE", pd.NA)
    df["stage_num"] = _stage_num_from_series(df["AJCC_PATHOLOGIC_TUMOR_STAGE"])

    # --- MSI and TMB ---
    # Attempt to extract MSI from MANTIS or SENSOR scores
    msi_col = _first_existing_col(df, ["MSI_SCORE_MANTIS", "MSI_SENSOR_SCORE"])
    if msi_col is not None:
        df["MSI"] = pd.to_numeric(df[msi_col], errors="coerce")
    else:
        df["MSI"] = pd.NA

    # Extract TMB
    tmb_col = _first_existing_col(df, ["TMB_NONSYNONYMOUS"])
    if tmb_col is not None:
        df["TMB"] = pd.to_numeric(df[tmb_col], errors="coerce")
    else:
        df["TMB"] = pd.NA

    # --- basic cleanup ---
    df = df.dropna(subset=["OS_MONTHS"])
    return df


def detect_additional_covariates(df_clin: pd.DataFrame, min_not_null_pct: float = 0.2) -> dict[str, str]:
    """
    データフレームを走査し、生存解析で交絡因子になり得る有用な列（喫煙歴、腫瘍純度など）を自動抽出する。
    
    Returns:
        Dict: { '表示名': '実際の列名' } の辞書
    """
    candidates = {
        "Smoking History": ["SMOKING_HISTORY", "TOBACCO_SMOKING_HISTORY", "SMOKING", "TOBACCO_SMOKING_HISTORY_INDICATOR"],
        "Tumor Purity": ["TUMOR_PURITY", "PURITY", "ESTIMATE_TUMOR_PURITY"],
        "Race": ["RACE", "RACE_CATEGORY"],
        "Ethnicity": ["ETHNICITY"],
        "Prior Therapy": ["PRIOR_DX", "RADIATION_THERAPY", "HISTORY_NEOADJUVANT_TRTMT", "PRIOR_MALIGNANCY"],
        "Tumor Site": ["TUMOR_SAMPLE_HISTOLOGY", "TUMOR_SITE", "SITE_OF_RESECTION_OR_BIOPSY"],
        "Sample Type": ["SAMPLE_TYPE", "TISSUE_SOURCE_SITE"]
    }
    
    detected = {}
    for label, col_names in candidates.items():
        # 既に見つかったラベルはスキップ
        if label in detected: continue
        
        found_col = _first_existing_col(df_clin, col_names)
        if found_col is not None:
            # 欠損値の割合をチェック
            not_null_pct = df_clin[found_col].notna().mean()
            if not_null_pct >= min_not_null_pct:
                detected[label] = found_col
                
    return detected

def add_covariate_columns(df: pd.DataFrame, detected_covariates: dict[str, str] = None) -> pd.DataFrame:
    """
    互換性のため残しつつ、normalize_clinicalが作る列があればそのまま使う。
    （将来的に iwa-collections-analysis.py 側の呼び出しを normalize_clinical に寄せてもOK）
    """
    out = df.copy()
    # ここは normalize_clinical が作っている前提だが、
    # 直呼びのために最低限の保険だけ入れる。
    if "AGE" in out.columns:
        out["AGE"] = pd.to_numeric(out["AGE"], errors="coerce")
    if "SEX" in out.columns and "is_male" not in out.columns:
        out["is_male"] = _normalize_sex_to_is_male(out["SEX"])
    if "AJCC_PATHOLOGIC_TUMOR_STAGE" in out.columns and "stage_num" not in out.columns:
        out["stage_num"] = _stage_num_from_series(out["AJCC_PATHOLOGIC_TUMOR_STAGE"])
        
    # detected_covariatesのダミー変数化（Cox用数値化）
    if detected_covariates:
        for label, col in detected_covariates.items():
            if col in out.columns:
                # 数値型ならそのまま
                if pd.api.types.is_numeric_dtype(out[col]):
                    out[col] = pd.to_numeric(out[col], errors="coerce")
                else:
                    # 文字列/カテゴリ変数の場合
                    # 最頻値を0にするようにダミー変数化
                    s = out[col].astype(str).replace("nan", pd.NA)
                    if s.notna().any():
                        mode_val = s.mode()[0]
                        # 最頻値=0, それ以外=1 のシンプルな二値化ルール
                        # (多値カテゴリの厳密なOne-Hot展開は複雑になりすぎるため、一番多いマジョリティとその他で分ける)
                        out[col + "_encoded"] = (s != mode_val).astype("Int64")
                        # オリジナルの値がNAならNAに戻す
                        out.loc[s.isna(), col + "_encoded"] = pd.NA
                    else:
                        out[col + "_encoded"] = pd.NA
                        
    return out


def build_covariate_list(adjust_age: bool, adjust_sex: bool, adjust_stage: bool, adjust_msi: bool = False, adjust_tmb: bool = False, dynamic_covariates: list[str] = None) -> list[str]:
    covs = []
    if adjust_age:
        covs.append("AGE")
    if adjust_sex:
        covs.append("is_male")
    if adjust_stage:
        covs.append("stage_num")
    if adjust_msi:
        covs.append("MSI")
    if adjust_tmb:
        covs.append("TMB")
        
    if dynamic_covariates:
        for c in dynamic_covariates:
            covs.append(c)
            
    return covs