# Local Execution & Verification Report

## Environment Setup & Pipeline Testing
- **Python Version:** 3.10+
- **Execution Command:** `python3 src/main.py`
- **Execution Status:** Verified locally. All pipeline outputs generated within 0.7s.

## Verification of Terminal Output Metrics
- **Dataset Scope:** 26,759 race results across 1,125 races (1950–2024).
- **Baseline Model (grid <= 3):** Test Accuracy: 0.897 | Precision: 0.657 | Recall: 0.652 | ROC-AUC: 0.911
- **Logistic Regression Model:** Test Accuracy: 0.901 | Precision: 0.720 | Recall: 0.558 | ROC-AUC: 0.928

## Generated Asset Audit
Verified generation and rendering of 9 visualization plots in `outputs/`:
1. `podium_rate_by_grid.png`
2. `top_drivers_podiums.png`
3. `top_constructors_podiums.png`
4. `grid_dependence_index.png`
5. `confusion_matrix.png`
6. `roc_curve.png`
7. `calibration_curve.png`
8. `feature_coefficients.png`
9. `backtest_top3.png`

*Report compiled and verified by Monika Singh (Documentation & Testing Lead).*
