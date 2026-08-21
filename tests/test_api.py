import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi.testclient import TestClient
from serving.app import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200

def test_predict_valid_input():
    payload = {
        "category": "apparel", "payment_method": "cod", "order_amount": 1200.0,
        "discount_pct": 40.0, "customer_past_orders": 1, "customer_past_returns": 1,
        "account_age_days": 20, "day_of_week": 5, "hour_of_day": 22, "address_mismatch": 1
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "risk_score" in data
    assert 0.0 <= data["risk_score"] <= 1.0

def test_predict_invalid_type():
    payload = {
        "category": 123, "payment_method": "cod", "order_amount": 1200.0,
        "discount_pct": 40.0, "customer_past_orders": 1, "customer_past_returns": 1,
        "account_age_days": 20, "day_of_week": 5, "hour_of_day": 22, "address_mismatch": 1
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_missing_field():
    payload = {"category": "apparel", "payment_method": "cod"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
