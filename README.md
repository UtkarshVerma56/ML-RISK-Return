# Return Risk Scorer — End-to-End ML Pipeline

Predicts the probability an e-commerce order will be returned, using a cost-aware, explainable ML pipeline — from synthetic data generation through to a deployed, live-serving API and interactive dashboard.

**Live API:** https://ml-risk-return.onrender.com/docs
**Live Dashboard:** https://return-risk-scorer.streamlit.app
*(Both on free-tier hosting — first request after inactivity may take ~30-50s to wake up.)*

## What this demonstrates
- Full ML lifecycle: data generation → EDA → feature engineering → training → evaluation → explainability → deployment → monitoring
- Trained on 18,000 synthetic orders with realistic, causally-structured correlations
- Class imbalance handled explicitly (not ignored)
- PR-AUC and calibration used alongside accuracy, given imbalanced classes
- **Cost-sensitive threshold selection** — chosen to minimize real business cost (false positive vs. false negative tradeoff), not the default 0.5
- SHAP-based explainability — every prediction includes plain-language reasoning
- Deployed as a live FastAPI service on Render, with an interactive Streamlit dashboard on top
- Version-locked dependencies with a startup check that fails loudly on mismatch (prevents silent model corruption)
- CI pipeline (GitHub Actions) — automatically tests model loading and API health on every push
- Basic drift monitoring (KS-test, PSI) to detect when the model may be going stale

## Project structure

    data/                       - synthetic dataset + processed splits
    models/                     - trained model, preprocessor, SHAP explainer, metrics, model card
    scripts/
      generate_data.py          - synthetic data generation (parametrized)
      train_pipeline.py         - full training pipeline: features -> tuning -> eval -> SHAP
    serving/
      app.py                    - FastAPI serving app
      version_check.py          - startup version verification
    dashboard/
      app.py                    - Streamlit interactive dashboard
    monitoring/
      drift_check.py            - drift detection script
    .github/workflows/ci.yml    - CI: auto-test on every push
    version_lock.txt            - single source of truth for library versions
    requirements.txt
    render.yaml                 - deployment config

## Key results
| Metric | Value | Context |
|---|---|---|
| PR-AUC (test set, 18k rows) | 0.331 | ~1.9x better than random baseline (~0.17, the test set's return rate) |
| ROC-AUC (test set) | 0.706 | Solid discriminative power for a tabular, moderately-noisy problem |
| Cost-optimal threshold | 0.41 | Chosen via cost-sensitive analysis (missed return costs ~7x more than a false alarm) |
| Cost savings vs. default 0.5 threshold | ₹3,500 (3.7%) | Real, measured reduction in total business cost on the test batch |

*Note: dataset is synthetic (see Limitations in `models/model_card.md`); absolute metric values should be read as a demonstration of methodology, not as claims about real-world merchant performance.*

## Try it

**Dashboard (recommended):** https://return-risk-scorer.streamlit.app — fill in order details, get an instant risk score with explanation.

**API directly:**
```bash
curl -X POST https://ml-risk-return.onrender.com/predict -H "Content-Type: application/json" -d '{"category": "apparel", "payment_method": "cod", "order_amount": 1200.0, "discount_pct": 40.0, "customer_past_orders": 1, "customer_past_returns": 1, "account_age_days": 20, "day_of_week": 5, "hour_of_day": 22, "address_mismatch": 1}'
```

## Full documentation
See `models/model_card.md` for model details, intended use, and honest limitations.

## Run locally
```bash
pip install -r requirements.txt
python scripts/generate_data.py --n_rows 18000
python scripts/train_pipeline.py
uvicorn serving.app:app --reload
```
