# Missing Data Treatment Framework Guide

## 🧩 Overview
Clinical datasets often contain missing values in covariates like Age, Sex, or Stage. The **Missing Data Treatment Framework** allows you to evaluate how different strategies for handling these missing values impact your prognostic findings.

## 📊 Missingness Overview
The first section provides a statistical summary of missing values across your clinical covariates.
- **Heatmap**: Visualizes the pattern of missingness across rows (patients) and columns (variables).
- **Severity Labels**: Categorizes variables as having Low, Moderate, or Severe missingness.

## ⚖️ Handling Strategy Comparison
We compare results across three standard strategies:
1.  **Complete-case**: Excludes any patient with at least one missing clinical value. This is the most transparent but can significantly reduce sample size (N).
2.  **Simple Imputation**: Fills missing continuous values with the *Median* and categorical values with the *Mode*.
3.  **Missing Category Encoding**: Treats "Missing" as an independent category for categorical variables (like Stage).

## 🛡️ Robustness Classification
The engine automatically classifies the finding based on its sensitivity to missing data treatment:
- **Robust**: Conclusion (direction and significance) remains stable across all strategies.
- **Moderately sensitive**: Direction is consistent, but significance or effect size (HR) depends on the handling method.
- **Fragile**: Conclusions fluctuate significantly; results should be interpreted with high caution.

## 📉 Sample Retention
This plot shows the proportion of patients retained vs. dropped for each strategy. A high "Dropped" rate in complete-case analysis may indicate that your results are biased toward a specific sub-population.
