import os
import sys
from pathlib import Path
import pandas as pd

# ---------------------------------------------------------
# Set up paths to import src modules if needed, or define constants here.
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

CLINICAL_CANDIDATES = [
    "data_clinical_patient.txt",
    "data_clinical_patient.tsv",
    "data_clinical_patient.csv",
    "data_clinical.txt",
]

EXPR_CANDIDATES = [
    "data_mrna_seq_v2_rsem.txt",
    "data_mrna_seq_v2_rsem_zscores_ref_diploid_samples.txt",
    "data_rna_seq_v2_mrna_median_Zscores.txt",
    "data_mrna_seq_v2_zscores_ref_all_samples.txt",
    "data_expression.txt",
]

def _find_first_existing(dirpath: Path, filenames: list[str]) -> Path | None:
    for fn in filenames:
        p = dirpath / fn
        if p.exists():
            return p
    return None

def _read_table(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path, comment="#")
    return pd.read_csv(path, sep="\t", comment="#")

def convert_dataset_to_parquet(dataset_dir: Path):
    print(f"Processing dataset: {dataset_dir.name}")
    
    # 1. Convert Clinical
    clin_parquet_path = dataset_dir / "clinical.parquet"
    if not clin_parquet_path.exists():
        clin_path = _find_first_existing(dataset_dir, CLINICAL_CANDIDATES)
        if clin_path:
            print(f"  - Reading clinical from {clin_path.name} ...")
            try:
                df_clin = _read_table(clin_path)
                df_clin.to_parquet(clin_parquet_path, engine="pyarrow", index=False)
                print(f"  -> Saved {clin_parquet_path.name}")
            except Exception as e:
                print(f"  -> Error converting clinical: {e}")
    else:
        print(f"  - {clin_parquet_path.name} already exists. Skipping.")

    # 2. Convert Expression
    expr_parquet_path = dataset_dir / "expression.parquet"
    if not expr_parquet_path.exists():
        exp_path = _find_first_existing(dataset_dir, EXPR_CANDIDATES)
        if exp_path:
            print(f"  - Reading expression from {exp_path.name} ...")
            try:
                df_exp = _read_table(exp_path)
                df_exp.to_parquet(expr_parquet_path, engine="pyarrow", index=False)
                print(f"  -> Saved {expr_parquet_path.name}")
            except Exception as e:
                print(f"  -> Error converting expression: {e}")
    else:
        print(f"  - {expr_parquet_path.name} already exists. Skipping.")

def main():
    if not DATA_ROOT.exists() or not DATA_ROOT.is_dir():
        print(f"Data root not found: {DATA_ROOT}")
        sys.exit(1)
        
    for item in DATA_ROOT.iterdir():
        if item.is_dir():
            # Check if it looks like a dataset (contains clinical/expr text files)
            has_text_data = _find_first_existing(item, CLINICAL_CANDIDATES) or _find_first_existing(item, EXPR_CANDIDATES)
            if has_text_data:
                convert_dataset_to_parquet(item)

    print("\nAll conversions finished!")

if __name__ == "__main__":
    main()
