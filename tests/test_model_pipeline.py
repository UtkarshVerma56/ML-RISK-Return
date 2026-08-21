import joblib
import pandas as pd
import pytest
import json

MODEL_DIR = 'models'

@pytest.fixture
def model():
    return joblib.load(f'{MODEL_DIR}/return_risk_model.pkl')

@pytest.fixture
def preprocessor():
    return joblib.load(f'{MODEL_DIR}/preprocessor.pkl')

@pytest.fixture
def feature_names():
    with open(f'{MODEL_DIR}/feature_names.json') as f:
        return json.load(f)

def test_model_loads(model):
    assert model is not None

def test_preprocessor_loads(preprocessor):
    assert preprocessor is not None

def test_prediction_shape(model, preprocessor, feature_names):
    sample = pd.DataFrame([{
        'order_amount': 1200.0, 'discount_pct': 40.0, 'customer_past_orders': 1,
        'customer_past_returns': 1, 'customer_return_rate': 1.0,
        'account_age_days': 20, 'day_of_week': 5, 'hour_of_day': 22,
        'address_mismatch': 1, 'category': 'apparel', 'payment_method': 'cod'
    }])
    processed = preprocessor.transform(sample)
    processed_df = pd.DataFrame(processed, columns=feature_names)
    pred = model.predict_proba(processed_df)
    assert pred.shape == (1, 2)

def test_prediction_range(model, preprocessor, feature_names):
    sample = pd.DataFrame([{
        'order_amount': 1200.0, 'discount_pct': 40.0, 'customer_past_orders': 1,
        'customer_past_returns': 1, 'customer_return_rate': 1.0,
        'account_age_days': 20, 'day_of_week': 5, 'hour_of_day': 22,
        'address_mismatch': 1, 'category': 'apparel', 'payment_method': 'cod'
    }])
    processed = preprocessor.transform(sample)
    processed_df = pd.DataFrame(processed, columns=feature_names)
    risk_score = model.predict_proba(processed_df)[:, 1][0]
    assert 0.0 <= risk_score <= 1.0

def test_high_risk_scores_higher_than_low_risk(model, preprocessor, feature_names):
    high_risk = pd.DataFrame([{
        'order_amount': 1200.0, 'discount_pct': 45.0, 'customer_past_orders': 1,
        'customer_past_returns': 1, 'customer_return_rate': 1.0,
        'account_age_days': 5, 'day_of_week': 5, 'hour_of_day': 22,
        'address_mismatch': 1, 'category': 'apparel', 'payment_method': 'cod'
    }])
    low_risk = pd.DataFrame([{
        'order_amount': 1200.0, 'discount_pct': 0.0, 'customer_past_orders': 10,
        'customer_past_returns': 0, 'customer_return_rate': 0.0,
        'account_age_days': 800, 'day_of_week': 2, 'hour_of_day': 14,
        'address_mismatch': 0, 'category': 'books', 'payment_method': 'UPI'
    }])
    high_processed = pd.DataFrame(preprocessor.transform(high_risk), columns=feature_names)
    low_processed = pd.DataFrame(preprocessor.transform(low_risk), columns=feature_names)
    high_score = model.predict_proba(high_processed)[:, 1][0]
    low_score = model.predict_proba(low_processed)[:, 1][0]
    assert high_score > low_score, "Model sanity check failed: high-risk scored lower than low-risk"
