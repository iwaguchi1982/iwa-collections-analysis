import argparse
import sys
from pathlib import Path
import pandas as pd
import gzip

def main():
    parser = argparse.ArgumentParser(description="Parse a GEO Series Matrix file into clinical and expression TSVs.")
    parser.add_argument("input_file", help="Path to the GSE..._series_matrix.txt(.gz) file.")
    parser.add_argument("--outdir", "-o", default=".", help="Output directory to save the TSV files.")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist.")
        sys.exit(1)
        
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Parsing {input_path} manually (fast mode)...")
    
    open_func = gzip.open if input_path.suffix == '.gz' else open
    mode = 'rt' if input_path.suffix == '.gz' else 'r'
    
    clinical_data = {}
    expr_lines = []
    in_matrix = False
    
    with open_func(input_path, mode, encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            if in_matrix:
                if line == "!series_matrix_table_end":
                    break
                expr_lines.append(line)
                continue
                
            if line == "!series_matrix_table_begin":
                in_matrix = True
                continue
                
            if line.startswith("!Sample_"):
                parts = line.split('\t')
                key = parts[0].strip('!"')
                values = [v.strip('"') for v in parts[1:]]
                
                if key in clinical_data:
                    idx = 1
                    while f"{key}_{idx}" in clinical_data:
                        idx += 1
                    key = f"{key}_{idx}"
                clinical_data[key] = values

    if clinical_data:
        max_len = max(len(v) for v in clinical_data.values())
        for k in list(clinical_data.keys()):
            if len(clinical_data[k]) < max_len:
                clinical_data[k].extend([None] * (max_len - len(clinical_data[k])))
                
        df_clin = pd.DataFrame(clinical_data)
        
        # Expand "key: value" pairs from characteristics
        expanded_cols = {}
        for c in df_clin.columns:
            if "characteristics" in c.lower():
                for idx, val in df_clin[c].items():
                    if val and ":" in str(val):
                        parts = str(val).split(":", 1)
                        if len(parts) == 2:
                            k, v = parts[0].strip(), parts[1].strip()
                            if k not in expanded_cols:
                                expanded_cols[k] = [None] * len(df_clin)
                            expanded_cols[k][idx] = v
        
        if expanded_cols:
            df_expanded = pd.DataFrame(expanded_cols)
            df_clin = pd.concat([df_clin, df_expanded], axis=1)

        clin_out = out_dir / "clinical.tsv"
        df_clin.to_csv(clin_out, sep="\t", index=False)
        print(f"Saved clinical data (samples={len(df_clin)}, features={len(df_clin.columns)}) to {clin_out}")
    else:
        print("Warning: No clinical metadata found.")
        
    if expr_lines:
        from io import StringIO
        print("Loading expression matrix into Pandas...")
        csv_data = "\n".join(expr_lines)
        df_exp = pd.read_csv(StringIO(csv_data), sep="\t", low_memory=False)
        df_exp.columns = [str(c).strip('"') for c in df_exp.columns]
        
        exp_out = out_dir / "expression.tsv"
        df_exp.to_csv(exp_out, sep="\t", index=False)
        print(f"Saved expression matrix (features/probes={len(df_exp)}, samples={len(df_exp.columns)-1}) to {exp_out}")
    else:
        print("Warning: No expression matrix found.")

if __name__ == "__main__":
    main()
