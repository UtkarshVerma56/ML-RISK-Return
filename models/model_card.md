# Model Card — Return Risk Scorer

## Overview
Predicts the probability that an e-commerce order will be returned, using order, customer, and product features available at checkout time. Outputs a risk score, a bounded risk flag, and the top contributing factors (via SHAP).

## Intended Use
- **Primary use case:** Flag high-return-risk orders at checkout for review, targeted friction (e.g., stricter verification), or exclusion from aggressive promotions.
- **Not intended for:** Automated order rejection without human review; use on customer populations very different from the training distribution (see Limitations).

## Training Data
- **Source:** Synthetically generated (not real transactions), ~1800 orders, built with deliberate causal correlations (category, discount level, customer history, payment method) based on documented e-commerce return-rate patterns.
- **Class balance:** ~15-20% positive (returned) class.
- **Note:** Since data is synthetic, absolute performance numbers should not be interpreted as real-world guarantees — the value of this project is in demonstrating the correct ML lifecycle, not in the specific metric values.

## Model
- **Algorithm:** XGBoost Classifier
- **Tuning:** RandomizedSearchCV, 30 iterations, 5-fold stratified CV, optimized for PR-AUC
- **Class imbalance handling:** `scale_pos_weight` set from training class ratio

## Performance (held-out test set)
See `test_metrics.json` for full numbers. Key metrics:
- ROC-AUC and PR-AUC reported (PR-AUC prioritized given class imbalance)
- Confusion matrix and classification report included
- Calibration curve checked — see EDA/evaluation notebook

## Decision Threshold
- Default 0.5 threshold was NOT used for production decisions.
- Threshold selected via cost-sensitive analysis: assumed cost of a missed return (₹350, reflecting reverse logistics) vs. a false alarm (₹50, minor friction cost).
- See `cost_analysis.json` for the full threshold sweep and reasoning.

## Explainability
- SHAP (TreeExplainer) used for both global feature importance and per-prediction explanations.
- Every prediction served via the API includes the top 3 contributing factors in plain language.

## Known Limitations
- **Synthetic data:** Real merchant data would include noise, missing values, and label lag (returns aren't known immediately) not present here.
- **No temporal validation:** Data isn't split by time; a real deployment should validate on more recent orders, not just a random held-out split.
- **Small scale:** ~1800 rows is enough for a credible pipeline demonstration, not for claims of production-grade statistical power.
- **Static model:** No automated retraining; drift monitoring script (see `monitoring/`) is a manual/scheduled check, not a live pipeline.
- **Fairness/bias:** Not audited for disparate impact across customer segments — a real deployment would need this before production use.

## Monitoring
Basic drift detection (KS-test for numeric features, PSI for categorical features) implemented in `monitoring/drift_check.py`. Run periodically against new data batches to detect distribution shift.
