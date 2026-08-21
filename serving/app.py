from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import json
import os

from serving.version_check import check_versions
check_versions()

app = FastAPI(title="Return Risk Scorer API", version="1.0.0")

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
model = joblib.load(os.path.join(MODEL_DIR, 'return_risk_model.pkl'))
preprocessor = joblib.load(os.path.join(MODEL_DIR, 'preprocessor.pkl'))
explainer = joblib.load(os.path.join(MODEL_DIR, 'shap_explainer.pkl'))

with open(os.path.join(MODEL_DIR, 'feature_names.json')) as f:
    feature_names = json.load(f)
with open(os.path.join(MODEL_DIR, 'cost_analysis.json')) as f:
    cost_analysis = json.load(f)

CHOSEN_THRESHOLD = cost_analysis['chosen_threshold']['value']


class OrderInput(BaseModel):
    category: str = Field(..., example="apparel")
    payment_method: str = Field(..., example="cod")
    order_amount: float = Field(..., example=1200.0)
    discount_pct: float = Field(..., example=30.0)
    customer_past_orders: int = Field(..., example=2)
    customer_past_returns: int = Field(..., example=1)
    account_age_days: int = Field(..., example=90)
    day_of_week: int = Field(..., ge=0, le=6, example=3)
    hour_of_day: int = Field(..., ge=0, le=23, example=14)
    address_mismatch: int = Field(..., ge=0, le=1, example=0)


class PredictionResponse(BaseModel):
    risk_score: float
    risk_flag: bool
    threshold_used: float
    top_factors: list[str]


@app.get("/")
def root():
    return {"message": "Return Risk Scorer API is running", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(order: OrderInput):
    try:
        customer_return_rate = (
            order.customer_past_returns / order.customer_past_orders
            if order.customer_past_orders > 0 else 0.0
        )
        raw_row = pd.DataFrame([{
            'order_amount': order.order_amount, 'discount_pct': order.discount_pct,
            'customer_past_orders': order.customer_past_orders,
            'customer_past_returns': order.customer_past_returns,
            'customer_return_rate': customer_return_rate,
            'account_age_days': order.account_age_days, 'day_of_week': order.day_of_week,
            'hour_of_day': order.hour_of_day, 'address_mismatch': order.address_mismatch,
            'category': order.category, 'payment_method': order.payment_method
        }])

        processed = preprocessor.transform(raw_row)
        processed_df = pd.DataFrame(processed, columns=feature_names)

        risk_score = float(model.predict_proba(processed_df)[:, 1][0])
        risk_flag = risk_score >= CHOSEN_THRESHOLD

        sv = explainer.shap_values(processed_df)[0]
        contributions = pd.DataFrame({'feature': feature_names, 'shap_value': sv}) \
            .sort_values('shap_value', key=abs, ascending=False).head(3)

        top_factors = []
        for _, row in contributions.iterrows():
            direction = "increased" if row['shap_value'] > 0 else "decreased"
            top_factors.append(f"{row['feature']} {direction} risk")

        return PredictionResponse(
            risk_score=round(risk_score, 4), risk_flag=risk_flag,
            threshold_used=CHOSEN_THRESHOLD, top_factors=top_factors
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
