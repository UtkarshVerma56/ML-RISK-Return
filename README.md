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

    data/                      - synthetic dataset + EDA summary
    models/                    - trained model, preprocessor, SHAP explainer, metrics, model card
    serving/app.py             - FastAPI serving app
    monitoring/drift_check.py  - drift detection script
    requirements.txt
    render.yaml                - deployment config

## Key results
| Metric | Value | Context |
|---|---|---|
| PR-AUC (test set) | 0.312 | ~1.6-1.7x better than random baseline (~0.18-0.20, the test set's return rate) |
| ROC-AUC (test set) | 0.711 | Solid discriminative power for a tabular, moderately-noisy problem |
| Cost-optimal threshold | 0.37 | Chosen via cost-sensitive analysis (missed return costs ~7x more than a false alarm) |
| Cost savings vs. default 0.5 threshold | ₹2,650 (24.8%) | Real, measured reduction in total business cost on the test batch |

*Note: dataset is synthetic (see Limitations in `models/model_card.md`); absolute metric values should be read as a demonstration of methodology, not as claims about real-world merchant performance.*

## Try it

```bash
curl -X POST https://ml-risk-return.onrender.com/predict -H "Content-Type: application/json" -d '{"category": "apparel", "payment_method": "cod", "order_amount": 1200.0, "discount_pct": 40.0, "customer_past_orders": 1, "customer_past_returns": 1, "account_age_days": 20, "day_of_week": 5, "hour_of_day": 22, "address_mismatch": 1}'
```

## Full documentation
See `models/model_card.md` for model details, intended use, and honest limitations.

## Run locally

```bash
pip install -r requirements.txt
uvicorn serving.app:app --reload
```
# Testing auto-deploy
