# Sensitivity Analysis Guide

The **Sensitivity Analysis Dashboard** is designed to evaluate whether your prognostic findings are "robust" (stable across different settings) or "fragile" (dependent on specific settings).

## 📊 Cutoff Percentile Scan
This plot shows how the **Hazard Ratio (HR)** and **P-value** change as you slide the high/low expression threshold (from the 20th to the 80th percentile).

- **Stable Target**: Shows a broad "Significant Region" (shaded in green) where the direction of the effect remains consistent.
- **Fragile Target**: Shows only a single narrow peak of significance, suggesting the result might be a statistical fluke.

## 🌲 Adjustment Robustness
This forest plot compares the Hazard Ratio across three levels of adjustment:
1. **Unadjusted**: Raw association.
2. **Basic**: Adjusted for Age and Sex.
3. **Full**: Adjusted for Age, Sex, and Stage.

If the HR "drifts" significantly toward 1.0 or loses significance entirely upon adjustment, the gene may not be an independent prognostic factor.

## ⚖️ Analytic Method Consistency
We compare results across three distinct analytic frameworks:
1. **Cox Proportional Hazards**: The standard regression model.
2. **Restricted Mean Survival Time (RMST)**: A model-free calculation of survival years/months.
3. **Landmark Analysis (Late phase)**: Focuses on survival differences *after* a specific time point, useful for detecting late-diverging effects.

- **Consistent**: All methods agree on the direction of effect.
- **Conflicting**: Significant disagreement between methods, often indicating complex survival patterns or data instability.

## 🛡️ Robustness Classification
The engine automatically classifies the target into one of three categories:
- ✅ **Stable**: Found significant across most cutoffs and adjustment models.
- ⚠️ **Moderately Sensitive**: Generally robust, but loses significance in certain specific edge cases.
- ❌ **Fragile**: Highly dependent on specific settings; interpret with caution.
