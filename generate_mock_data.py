import pandas as pd
import json
import os

coad_clin = pd.read_csv('data/coadread_tcga_pan_can_atlas_2018/data_clinical_patient.txt', sep='\t', comment='#')
coad_exp = pd.read_csv('data/coadread_tcga_pan_can_atlas_2018/data_mrna_seq_v2_rsem.txt', sep='\t', comment='#')

# Extract ABCC6
abcc6 = coad_exp[coad_exp['Hugo_Symbol'] == 'ABCC6']

# Create mock expression file
mock_exp = abcc6.melt(id_vars=['Hugo_Symbol', 'Entrez_Gene_Id'], var_name='PATIENT_ID', value_name='expression')
mock_exp['PATIENT_ID'] = mock_exp['PATIENT_ID'].astype(str).str[:12]
mock_exp = mock_exp[['PATIENT_ID', 'Hugo_Symbol', 'expression']].dropna()

# Create matched clinical file
def simple_norm(df):
    out = df.copy()
    out = out.rename(columns={'PATIENT_ID': 'PATIENT_ID', 'OS_MONTHS': 'OS_MONTHS', 'OS_STATUS': 'OS_STATUS'})
    out['is_dead'] = out['OS_STATUS'].astype(str).str.contains('1|DECEASED', case=False, na=False).astype(int)
    return out

norm_clin = simple_norm(coad_clin)
norm_clin['PATIENT_ID'] = norm_clin['PATIENT_ID'].astype(str).str[:12]

# Simplify and write
norm_clin[['PATIENT_ID', 'OS_MONTHS', 'is_dead']].to_csv('mock_coadread_clin.csv', index=False)
mock_exp.to_csv('mock_coadread_exp.csv', index=False)
print('Mock files created: mock_coadread_clin.csv, mock_coadread_exp.csv')
