# Return Risk Scorer — End-to-End ML Pipeline

Predicts the probability an e-commerce order will be returned, using a cost-aware, explainable ML pipeline — from synthetic data generation through to a deployed, live-serving API.

**Live API:** https://ml-risk-return.onrender.com/docs
*(Free-tier hosting — first request after inactivity may take ~30-50s to wake up.)*

## What this demonstrates
- Full ML lifecycle: data generation → EDA → feature engineering → training → evaluation → explainability → deployment → monitoring
- Class imbalance handled explicitly (not ignored)
- PR-AUC and calibration used alongside accuracy, given imbalanced classes
- **Cost-sensitive threshold selection** — not the default 0.5, but one chosen to minimize real business cost (false positive vs. false negative cost tradeoff)
- SHAP-based explainability — every prediction includes plain-language reasoning
- Deployed as a live FastAPI service on Render
- Basic drift monitoring (KS-test, PSI) to detect when the model may be going stale

## Project structure
