import hashlib
from typing import List, Dict, Any

# =====================================================================
# Mock OpenTargets Database for Known Drugs / Drug Repurposing
# =====================================================================

MOCK_KNOWN_DRUGS = {
    "EGFR": [
        {"name": "Gefitinib", "phase": "Approved", "disease": "Non-small cell lung cancer", "type": "Small molecule"},
        {"name": "Osimertinib", "phase": "Approved", "disease": "Non-small cell lung cancer", "type": "Small molecule"},
        {"name": "Cetuximab", "phase": "Approved", "disease": "Colorectal cancer / Head and neck cancer", "type": "Antibody"},
    ],
    "ERBB2": [
        {"name": "Trastuzumab", "phase": "Approved", "disease": "Breast cancer / Gastric cancer", "type": "Antibody"},
        {"name": "Trastuzumab deruxtecan", "phase": "Approved", "disease": "Breast cancer", "type": "Antibody Drug Conjugate"},
        {"name": "Lapatinib", "phase": "Approved", "disease": "Breast cancer", "type": "Small molecule"},
        {"name": "Tucatinib", "phase": "Approved", "disease": "Breast cancer", "type": "Small molecule"},
    ],
    "CD274": [ # PD-L1
        {"name": "Atezolizumab", "phase": "Approved", "disease": "Urothelial / NSCLC / TNBC", "type": "Antibody"},
        {"name": "Durvalumab", "phase": "Approved", "disease": "Urothelial / NSCLC", "type": "Antibody"},
        {"name": "Avelumab", "phase": "Approved", "disease": "Merkel cell / Urothelial / RCC", "type": "Antibody"},
    ],
    "PDCD1": [ # PD-1
        {"name": "Pembrolizumab", "phase": "Approved", "disease": "Melanoma / NSCLC / Head and neck etc.", "type": "Antibody"},
        {"name": "Nivolumab", "phase": "Approved", "disease": "Melanoma / NSCLC / RCC etc.", "type": "Antibody"},
    ],
    "BRAF": [
        {"name": "Vemurafenib", "phase": "Approved", "disease": "Melanoma (V600E)", "type": "Small molecule"},
        {"name": "Dabrafenib", "phase": "Approved", "disease": "Melanoma / NSCLC", "type": "Small molecule"},
        {"name": "Encorafenib", "phase": "Approved", "disease": "Melanoma / Colorectal", "type": "Small molecule"},
    ],
    "TP53": [
        {"name": "Eprenetapopt (APR-246)", "phase": "Phase III", "disease": "MDS / AML", "type": "Small molecule"},
        {"name": "Advexin", "phase": "Phase III", "disease": "Head and neck cancer", "type": "Gene Therapy"},
    ],
    "KRAS": [
        {"name": "Sotorasib", "phase": "Approved", "disease": "NSCLC (G12C)", "type": "Small molecule"},
        {"name": "Adagrasib", "phase": "Approved", "disease": "NSCLC (G12C)", "type": "Small molecule"},
    ]
}

import requests

def get_known_drugs(gene_symbol: str, use_mock: bool = True) -> List[Dict[str, Any]]:
    """
    Returns a list of known drugs targeting the given gene.
    If use_mock is True, uses hardcoded realistic drugs or deterministic fallback.
    If False, attempts to query the live OpenTargets API.
    """
    g = gene_symbol.strip().upper()
    if not g:
        return []

    if not use_mock:
        try:
            # 1. Search for Ensembl ID
            search_query = '''
            query searchTarget($queryString: String!) {
              search(queryString: $queryString, entityNames: ["target"], page: {index: 0, size: 1}) {
                hits { id entity }
              }
            }
            '''
            r = requests.post("https://api.opentargets.io/api/v4/graphql", 
                              json={"query": search_query, "variables": {"queryString": g}}, 
                              timeout=5)
            r.raise_for_status()
            hits = r.json().get("data", {}).get("search", {}).get("hits", [])
            
            if hits and hits[0].get("entity") == "target":
                ensembl_id = hits[0]["id"]
                
                # 2. Fetch known drugs
                d_query = '''
                query targetDrugs($ensemblId: String!) {
                  target(ensemblId: $ensemblId) {
                    knownDrugs(size: 15) {
                      rows {
                        drug { name drugType maximumClinicalTrialPhase }
                        phase
                        status
                        disease { name }
                      }
                    }
                  }
                }
                '''
                r2 = requests.post("https://api.opentargets.io/api/v4/graphql", 
                                   json={"query": d_query, "variables": {"ensemblId": ensembl_id}}, 
                                   timeout=10)
                r2.raise_for_status()
                
                api_drugs = []
                rows = r2.json().get("data", {}).get("target", {}).get("knownDrugs", {}).get("rows", [])
                
                for row in rows:
                    drug_info = row.get("drug", {})
                    disease_info = row.get("disease", {})
                    
                    phase_num = row.get("phase", 0)
                    phase_str = "Approved" if phase_num == 4 else f"Phase {phase_num}"
                    
                    api_drugs.append({
                        "name": drug_info.get("name", "Unknown"),
                        "phase": phase_str,
                        "disease": disease_info.get("name", "Unknown"),
                        "type": drug_info.get("drugType", "Unknown")
                    })
                    
                # Deduplicate and sort
                # Unique by name and disease
                unique_drugs = {f"{d['name']}_{d['disease']}": d for d in api_drugs}.values()
                drugs_list = list(unique_drugs)
                
                # Sort so Approved comes first
                def sort_key(x):
                    p = x["phase"]
                    if p == "Approved": return 4
                    if "Phase 4" in p: return 4
                    if "Phase 3" in p: return 3
                    if "Phase 2" in p: return 2
                    if "Phase 1" in p: return 1
                    return 0
                
                drugs_list.sort(key=sort_key, reverse=True)
                return drugs_list
                
        except Exception as e:
            print(f"OpenTargets API Error (Drugs): {e}")
            pass

    # Mock Path
    if g in MOCK_KNOWN_DRUGS:
        return MOCK_KNOWN_DRUGS[g]

    return _generate_mock_drugs(g)

def _generate_mock_drugs(gene: str) -> List[Dict[str, Any]]:
    m = hashlib.md5()
    m.update(gene.encode('utf-8'))
    hash_hex = m.hexdigest()

    # Use first char to determine number of drugs (0 to 3)
    num_val = int(hash_hex[0], 16)
    
    if num_val < 8:
        # 50% chance of no known drugs
        return []
    elif num_val < 12:
        num_drugs = 1
    elif num_val < 14:
        num_drugs = 2
    else:
        num_drugs = 3

    phases = ["Phase I", "Phase I/II", "Phase II", "Phase III"]
    diseases = [
        "Solid Tumors", "Advanced Malignancies", "Hematologic Malignancies", 
        "NSCLC", "Breast Cancer", "Colorectal Cancer", "Hepatocellular Carcinoma"
    ]
    types = ["Small molecule", "Antibody", "PROTAC / Degrader", "Peptide", "RNAi"]
    
    prefix = ["Nivo", "Pembro", "Vemu", "Soto", "Adagra", "Atezo", "Ipi", "Osi", "Gefi", "Vande", "Pazo"]
    suffix = ["nib", "mab", "ximab", "tinib", "rafenib", "zumab", "stat", "limab", "deg"]

    drugs = []
    for i in range(num_drugs):
        # Use subsequent segments of the hash to pick attributes
        idx = (i + 1) * 2
        hx = int(hash_hex[idx:idx+2], 16)
        
        phase = phases[hx % len(phases)]
        disease = diseases[(hx // 2) % len(diseases)]
        drug_type = types[(hx // 3) % len(types)]
        
        name_p = prefix[(hx // 5) % len(prefix)]
        name_s = suffix[(hx // 7) % len(suffix)]
        
        # If it's a small molecule and gets a -mab suffix, fix it
        if drug_type == "Small molecule" and name_s in ["mab", "ximab", "zumab", "limab"]:
            name_s = "nib"
        # If antibody and gets a -nib suffix, fix it
        if drug_type == "Antibody" and name_s in ["nib", "tinib", "rafenib", "stat"]:
            name_s = "mab"
            
        drug_name = f"Investigational {name_p}{name_s}"

        drugs.append({
            "name": drug_name,
            "phase": phase,
            "disease": disease,
            "type": drug_type
        })

    # Sort so Phase III comes before Phase I
    drugs.sort(key=lambda x: x["phase"], reverse=True)
    return drugs
