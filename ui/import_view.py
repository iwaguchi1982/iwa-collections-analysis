import streamlit as st
import pandas as pd
from pathlib import Path

def render_import_view(data_root: Path):
    st.header("📥 Import Custom Dataset")
    st.markdown("Upload your own clinical and expression data to use across all analysis modules.")
    
    with st.form("import_form"):
        dataset_name = st.text_input("Dataset Name (e.g. My_RNAseq_Cohort)", help="A new folder 'Custom_<Name>' will be created.")
        
        st.subheader("1. Clinical Data")
        clin_file = st.file_uploader("Upload Clinical Data (CSV/TSV)", type=["csv", "tsv", "txt"])
        
        st.subheader("2. Expression Data")
        exp_file = st.file_uploader("Upload Expression Data (CSV/TSV)", type=["csv", "tsv", "txt"])
        
        submitted = st.form_submit_button("Preview Data")

    if not submitted and "import_clin_df" not in st.session_state:
        return

    # Once files are uploaded, parse them into session state
    if submitted:
        if not dataset_name:
            st.error("Please provide a dataset name.")
            return
        if clin_file is None or exp_file is None:
            st.error("Please upload both clinical and expression files.")
            return
        
        # Read clin
        try:
            sep = "," if clin_file.name.endswith(".csv") else "\t"
            df_clin = pd.read_csv(clin_file, sep=sep)
            st.session_state["import_clin_df"] = df_clin
        except Exception as e:
            st.error(f"Error reading clinical file: {e}")
            return
            
        # Read exp
        try:
            sep = "," if exp_file.name.endswith(".csv") else "\t"
            df_exp = pd.read_csv(exp_file, sep=sep)
            st.session_state["import_exp_df"] = df_exp
        except Exception as e:
            st.error(f"Error reading expression file: {e}")
            return

        st.session_state["import_dataset_name"] = dataset_name.strip()
        st.session_state["import_step"] = "mapping"

    if st.session_state.get("import_step") == "mapping":
        _render_mapping_ui(data_root)

def _render_mapping_ui(data_root: Path):
    df_clin = st.session_state["import_clin_df"]
    df_exp = st.session_state["import_exp_df"]
    dataset_name = st.session_state["import_dataset_name"]
    
    st.markdown("---")
    st.subheader("Mapping & Quality Control")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Clinical Data Preview**")
        st.dataframe(df_clin.head(), use_container_width=True)
    with col2:
        st.write("**Expression Data Preview**")
        st.dataframe(df_exp.head(), use_container_width=True)
        
    st.markdown("### 1. Map Clinical Columns")
    clin_cols = [""] + list(df_clin.columns)
    
    c1, c2, c3 = st.columns(3)
    # Try to guess
    guess_id = next((c for c in clin_cols if c.upper() in ["PATIENT_ID", "PATIENT", "ID", "SAMPLE"]), "")
    guess_time = next((c for c in clin_cols if "OS" in c.upper() or "TIME" in c.upper() or "MONTH" in c.upper() or "DAY" in c.upper()), "")
    guess_stat = next((c for c in clin_cols if "STATUS" in c.upper() or "EVENT" in c.upper() or "OS" in c.upper()), "")

    pat_col = c1.selectbox("Patient ID Column", clin_cols, index=clin_cols.index(guess_id) if guess_id else 0)
    time_col = c2.selectbox("Survival Time Column", clin_cols, index=clin_cols.index(guess_time) if guess_time else 0)
    stat_col = c3.selectbox("Survival Event Column", clin_cols, index=clin_cols.index(guess_stat) if guess_stat else 0)
    
    c1b, c2b, _ = st.columns(3)
    time_unit = c2b.radio("Time Unit", ["Months", "Days"])
    event_val = ""
    if stat_col:
        unique_vals = list(df_clin[stat_col].dropna().astype(str).unique())
        event_val = c3.selectbox(f"Which value in '{stat_col}' indicates an EVENT (Death)?", [""] + unique_vals)
    
    st.markdown("### 2. Expression Data Format")
    exp_format = st.radio("Expression Matrix Layout", ["Genes as Rows (Typical GEO/TCGA - samples are columns)", "Genes as Columns (Wide matrix - samples are rows)"])
    if exp_format.startswith("Genes as Rows"):
        gene_col = st.selectbox("Column containing Gene Symbols", [""] + list(df_exp.columns))
        patient_col_exp = None
    else:
        gene_col = None
        patient_col_exp = st.selectbox("Column containing Patient IDs in Expression Data", [""] + list(df_exp.columns))
        
    if st.button("Run QC & Register Dataset", type="primary"):
        if not pat_col or not time_col or not stat_col or not event_val:
            st.error("Please complete all clinical mapping fields.")
            return
        if exp_format.startswith("Genes as Rows") and not gene_col:
            st.error("Please select the gene symbol column.")
            return
        if exp_format.startswith("Genes as Columns") and not patient_col_exp:
            st.error("Please select the patient ID column in expression data.")
            return
            
        _process_and_save(data_root, dataset_name, df_clin, df_exp, pat_col, time_col, time_unit, stat_col, event_val, exp_format, gene_col, patient_col_exp)
        
def _process_and_save(data_root, dataset_name, df_clin, df_exp, pat_col, time_col, time_unit, stat_col, event_val, exp_format, gene_col, patient_col_exp):
    # 1. Format Clinical
    c = df_clin.copy()
    c = c.rename(columns={pat_col: "PATIENT_ID", stat_col: "OS_STATUS"})
    
    # Handle time conversion
    c["_raw_time"] = pd.to_numeric(c[time_col], errors="coerce")
    if time_unit == "Days":
        c["OS_MONTHS"] = c["_raw_time"] / 30.4375
    else:
        c["OS_MONTHS"] = c["_raw_time"]
    c = c.drop(columns=["_raw_time"])
        
    c["PATIENT_ID"] = c["PATIENT_ID"].astype(str)
    c["is_dead"] = (c["OS_STATUS"].astype(str).str.strip() == str(event_val).strip()).astype(int)
    
    # Check intersecting patients before saving
    all_pats_clin = set(c["PATIENT_ID"].unique())
    
    if exp_format.startswith("Genes as Rows"):
        all_pats_exp = set(df_exp.columns)
    else:
        all_pats_exp = set(df_exp[patient_col_exp].astype(str).unique())
        
    intersect = all_pats_clin.intersection(all_pats_exp)
    if not intersect:
        st.error(f"Failed: No overlapping patients found between clinical ({len(all_pats_clin)}) and expression data ({len(all_pats_exp)}). Please check your IDs.")
        return
        
    # 2. Format Expression
    e = df_exp.copy()
    if exp_format.startswith("Genes as Rows"):
        e = e.rename(columns={gene_col: "Hugo_Symbol"})
        # Drop empty symbols
        e = e.dropna(subset=["Hugo_Symbol"])
        e = e.drop_duplicates(subset=["Hugo_Symbol"])
    else:
        e = e.rename(columns={patient_col_exp: "PATIENT_ID"})
        e["PATIENT_ID"] = e["PATIENT_ID"].astype(str)

    # 3. Save
    out_dir = Path(data_root) / f"Custom_{dataset_name.replace(' ', '_')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        c.to_parquet(out_dir / "clinical.parquet", engine="pyarrow")
        e.to_parquet(out_dir / "expression.parquet", engine="pyarrow")
        
        st.success(f"✅ Successfully registered dataset '{dataset_name}' with {len(intersect)} overlapping patients!")
        st.info("The new dataset will now appear in the Sidebar dropdown. You may need to select it or reload the page.")
        
        # Clear state
        if "import_clin_df" in st.session_state: del st.session_state["import_clin_df"]
        if "import_exp_df" in st.session_state: del st.session_state["import_exp_df"]
        if "import_dataset_name" in st.session_state: del st.session_state["import_dataset_name"]
        if "import_step" in st.session_state: del st.session_state["import_step"]
        
    except Exception as err:
        st.error(f"Failed to save parquet files: {err}")
