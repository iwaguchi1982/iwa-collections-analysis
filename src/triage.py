# src/triage.py
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import math

class TargetTriageEngine:
    """
    Engine to compute the Translational Priority core of a target based on 
    Prognostic Evidence, Druggability, Safety, and Translation Strategy.
    """
    
    def __init__(self, mode: str = "Balanced"):
        """
        mode: "Repurposing-first", "Novel-target-first", or "Balanced"
        """
        valid_modes = ["Repurposing-first", "Novel-target-first", "Balanced"]
        if mode not in valid_modes:
            mode = "Balanced"
        self.mode = mode

    def evaluate(self, target_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates the target and returns a comprehensive structured dictionary.
        
        Expected structure of `target_data` (all fields optional, missing means 'uncertain'):
        {
            "target_symbol": "EGFR",
            "hr": 2.5,
            "p_value": 0.001,
            "q_value": 0.04,
            "psm_maintained": True,            # Did HR stay significant after PSM?
            "time_varying_stable": True,       # Is the effect stable over time (PH test passed/effect persists)?
            "gsea_hallmark_hit": True,         # Was there a significant Hallmark pathway hit?
            "accessibility": "Membrane",       # e.g., "Membrane", "Secreted", "Nucleus", "Cytosol"
            "target_class": "Kinase",          # e.g., "Kinase", "Transcription Factor", "Receptor"
            "is_essential": False,             # CRISPR essentiality
            "normal_tissue_expression": "Low", # "High", "Medium", "Low"
            "known_drugs": {
                "has_approved": True,
                "has_clinical": True
            }
        }
        """
        res_evidence = self._score_evidence(target_data)
        res_druggability = self._score_druggability(target_data)
        res_safety = self._score_safety(target_data)
        res_translation = self._score_translation_strategy(target_data)

        total_score = (
            res_evidence["score"] + 
            res_druggability["score"] + 
            res_safety["score"] + 
            res_translation["score"]
        )
        
        # Determine priority string based on total score (heuristic)
        if total_score >= 70:
            priority = "High"
        elif total_score >= 50:
            priority = "Medium"
        else:
            priority = "Low"
            
        # Compile missing evidence
        missing_count = sum(len(x.get("missing", [])) for x in [res_evidence, res_druggability, res_safety, res_translation])
        if missing_count == 0:
            confidence = "High evidence completeness"
        elif missing_count <= 3:
            confidence = "Moderate evidence completeness"
        else:
            confidence = "Low evidence / Missing data"

        # Generate Pros & Cons
        pros = []
        cons = []
        for cat_res in [res_evidence, res_druggability, res_safety, res_translation]:
            pros.extend(cat_res.get("pros", []))
            cons.extend(cat_res.get("cons", []))

        return {
            "target_symbol": target_data.get("target_symbol", "Unknown"),
            "mode": self.mode,
            "total_score": max(0, min(100, total_score)),
            "priority": priority,
            "confidence": confidence,
            "breakdown": {
                "Evidence": res_evidence,
                "Druggability": res_druggability,
                "Safety": res_safety,
                "Translation Strategy": res_translation
            },
            "pros": pros,
            "cons": cons,
            "missing_data": missing_count
        }

    def _score_evidence(self, data: Dict[str, Any]) -> Dict[str, Any]:
        score = 0
        max_score = 40
        pros = []
        cons = []
        missing = []
        breakdown_table = []

        # 1. Prognostic Strength (15 pts)
        hr = data.get("hr")
        pval = data.get("p_value")
        prog_score = 0
        if hr is not None and pval is not None:
            if pval < 0.05:
                # Strong effect: HR > 2.0 or HR < 0.5
                if hr > 2.0 or hr < 0.5:
                    prog_score = 15
                    pros.append("Strong prognostic effect size (HR > 2.0 or < 0.5)")
                # Moderate effect
                elif hr > 1.5 or hr < 0.66:
                    prog_score = 10
                    pros.append("Moderate prognostic effect size")
                else:
                    prog_score = 5
                    cons.append("Weak prognostic effect size despite significance")
                
                breakdown_table.append({"Subdomain": "Prognostic strength", "Score": f"{prog_score}/15", "Evidence": f"HR={hr:.2f}, p={pval:.2e}"})
            else:
                cons.append("No significant prognostic effect detected")
                breakdown_table.append({"Subdomain": "Prognostic strength", "Score": f"0/15", "Evidence": f"Not significant (p={pval:.2e})"})
        else:
            missing.append("Prognostic Metrics")
            breakdown_table.append({"Subdomain": "Prognostic strength", "Score": "Uncertain", "Evidence": "Missing HR/P-value"})

        score += prog_score

        # 2. Robustness / Reproducibility (15 pts)
        # PSM (10 pts)
        rob_score = 0
        psm = data.get("psm_maintained")
        if psm is not None:
            if psm:
                rob_score += 10
                pros.append("Prognostic effect robust against covariates (PSM)")
            else:
                cons.append("Effect lost after Propensity Score Matching")
        else:
            missing.append("PSM Results")
        
        # Time-varying / landmark (5 pts)
        tv = data.get("time_varying_stable")
        if tv is not None:
            if tv:
                rob_score += 5
            else:
                cons.append("Effect violates proportional hazards (time-dependent)")
        else:
            missing.append("Time-varying stability")
            
        score += rob_score
        breakdown_table.append({"Subdomain": "Robustness", "Score": f"{rob_score}/15", 
                                "Evidence": f"PSM: {'Maintained' if psm else 'Lost/Missing'}, Time-Stable: {'Yes' if tv else 'No/Missing'}"})

        # 3. Mechanistic Support (10 pts)
        mech_score = 0
        gsea = data.get("gsea_hallmark_hit")
        if gsea is not None:
            if gsea:
                mech_score += 10
                pros.append("Clear pathway mechanisms supported by GSEA/ORA")
            else:
                cons.append("No clear Hallmark/KEGG pathway associations")
        else:
            missing.append("Pathway Analysis (GSEA/ORA)")
            
        score += mech_score
        breakdown_table.append({"Subdomain": "Mechanistic support", "Score": f"{mech_score}/10", "Evidence": "Pathway hits found" if gsea else "No clear pathways / Missing"})

        return {
            "score": score,
            "max": max_score,
            "pros": pros,
            "cons": cons,
            "missing": missing,
            "details": breakdown_table
        }

    def _score_druggability(self, data: Dict[str, Any]) -> Dict[str, Any]:
        score = 0
        max_score = 30
        pros = []
        cons = []
        missing = []
        breakdown_table = []

        # 1. Accessibility (10 pts)
        acc = (data.get("accessibility") or "").lower()
        acc_score = 0
        if acc:
            if "membrane" in acc or "secreted" in acc or "surface" in acc:
                acc_score = 10
                pros.append("Highly accessible (Membrane/Secreted) - viable for Antibodies/ADCs")
                ev_str = "Membrane/Secreted"
            elif "cytosol" in acc or "cytoplasm" in acc:
                acc_score = 5
                cons.append("Intracellular target - restricted to small molecules/PROTACs")
                ev_str = "Intracellular"
            elif "nucleus" in acc:
                acc_score = 2
                cons.append("Nuclear target - traditionally hard to drug")
                ev_str = "Nuclear"
            else:
                ev_str = acc
            breakdown_table.append({"Subdomain": "Accessibility", "Score": f"{acc_score}/10", "Evidence": ev_str})
        else:
            missing.append("Subcellular Location")
            breakdown_table.append({"Subdomain": "Accessibility", "Score": "Uncertain", "Evidence": "Location unknown"})
        score += acc_score

        # 2. Modality fit & Target class tractability (20 pts)
        tc = (data.get("target_class") or "").lower()
        tc_score = 0
        if tc:
            if "kinase" in tc or "receptor" in tc or "enzyme" in tc or "ion channel" in tc:
                tc_score = 20
                pros.append(f"Highly tractable target class ({data.get('target_class')})")
            elif "transcription factor" in tc or "scaffold" in tc or "epigenetic" in tc:
                tc_score = 5
                cons.append(f"Challenging target class ({data.get('target_class')})")
            else:
                tc_score = 10
            breakdown_table.append({"Subdomain": "Modality & Tractability", "Score": f"{tc_score}/20", "Evidence": data.get("target_class")})
        else:
            missing.append("Target Class")
            breakdown_table.append({"Subdomain": "Modality & Tractability", "Score": "Uncertain", "Evidence": "Target class unknown"})
        
        score += tc_score

        return {
            "score": score,
            "max": max_score,
            "pros": pros,
            "cons": cons,
            "missing": missing,
            "details": breakdown_table
        }

    def _score_safety(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Safety is deduction based. Max is 15. We start at 15 and subtract.
        score = 15
        max_score = 15
        pros = []
        cons = []
        missing = []
        breakdown_table = []

        # 1. Essentiality Risk (deduct up to 10)
        ess = data.get("is_essential")
        if ess is not None:
            if ess:
                score -= 10
                cons.append("High safety risk: Gene is a CRISPR pan-essential survival gene")
                breakdown_table.append({"Subdomain": "Essentiality risk", "Score": "-10", "Evidence": "Pan-essential"})
            else:
                pros.append("Not a pan-essential gene (lower baseline toxicity risk)")
                breakdown_table.append({"Subdomain": "Essentiality risk", "Score": "0", "Evidence": "Not essential"})
        else:
            missing.append("CRISPR Essentiality")
            breakdown_table.append({"Subdomain": "Essentiality risk", "Score": "Uncertain", "Evidence": "Missing DepMap data"})

        # 2. Normal Tissue Risk (deduct up to 5)
        nt = (data.get("normal_tissue_expression") or "").lower()
        if nt:
            if nt == "high":
                score -= 5
                cons.append("High predicted toxicity: Broad or high expression in normal / vital organs")
                breakdown_table.append({"Subdomain": "Normal tissue risk", "Score": "-5", "Evidence": "High vital organ expression"})
            elif nt == "low":
                pros.append("Favorable safety profile: Low baseline expression in healthy tissues")
                breakdown_table.append({"Subdomain": "Normal tissue risk", "Score": "0", "Evidence": "Low normal expression"})
            else:
                breakdown_table.append({"Subdomain": "Normal tissue risk", "Score": "-2", "Evidence": "Moderate normal expression"})
                score -= 2
        else:
            missing.append("Normal Tissue Expression")
            breakdown_table.append({"Subdomain": "Normal tissue risk", "Score": "Uncertain", "Evidence": "Missing normal tissue data"})

        return {
            "score": max(0, score), # Prevention against going negative
            "max": max_score,
            "pros": pros,
            "cons": cons,
            "missing": missing,
            "details": breakdown_table
        }

    def _score_translation_strategy(self, data: Dict[str, Any]) -> Dict[str, Any]:
        score = 0
        max_score = 15
        pros = []
        cons = []
        missing = []
        breakdown_table = []

        kd = data.get("known_drugs", {})
        has_appr = kd.get("has_approved", False)
        has_clin = kd.get("has_clinical", False)
        
        ev_str = "No known trials"
        if has_appr:
            ev_str = "Approved drugs exist"
        elif has_clin:
            ev_str = "Clinical stage drugs exist"

        if self.mode == "Repurposing-first":
            # Rewards existence of drugs
            if has_appr:
                score = 15
                pros.append("Excellent repurposing candidate (Approved drugs exist)")
            elif has_clin:
                score = 10
                pros.append("Strong translational candidate (Clinical Phase drugs exist)")
            else:
                score = 0
                cons.append("Poor repurposing candidate (No known compounds in trials)")
                
        elif self.mode == "Novel-target-first":
            # Rewards lack of drugs (Novelty / Uncrowded IP space)
            if has_appr:
                score = 0
                cons.append("Highly highly crowded space (Approved drugs already exist)")
            elif has_clin:
                score = 5
                cons.append("Moderately crowded space (Compounds in clinical trials)")
            else:
                score = 15
                pros.append("High Novelty: Pristine competitive landscape (First-in-class opportunity)")
                
        else: # "Balanced"
            if has_appr:
                score = 10
                pros.append("Validated target (Approved drugs exist)")
            elif has_clin:
                score = 10
                pros.append("Validated target (Clinical compounds exist)")
            else:
                score = 10
                pros.append("Novel target space (Differentiation potential)")

        breakdown_table.append({"Subdomain": f"Strategy ({self.mode})", "Score": f"{score}/15", "Evidence": ev_str})

        return {
            "score": score,
            "max": max_score,
            "pros": pros,
            "cons": cons,
            "missing": missing,
            "details": breakdown_table
        }
