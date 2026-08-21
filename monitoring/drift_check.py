"""
Basic drift monitoring for the Return Risk Scorer model.

Compares the distribution of a new incoming data batch against
the original training data distribution, using the Kolmogorov-Smirnov
test for numeric features and a Population Stability Index (PSI)
style check for categorical features.
"""

import pandas as pd
import numpy as np
from scipy import stats
import json


NUMERIC_COLS = ['order_amount', 'discount_pct', 'customer_past_orders',
                 'customer_past_returns', 'customer_return_rate',
                 'account_age_days', 'day_of_week', 'hour_of_day']

CATEGORICAL_COLS = ['category', 'payment_method']

KS_PVALUE_THRESHOLD = 0.05   # below this -> distributions are significantly different
PSI_THRESHOLD = 0.2          # above this -> meaningful categorical shift


def check_numeric_drift(train_series, new_series, feature_name):
    """Kolmogorov-Smirnov test: are these two distributions different?"""
    statistic, p_value = stats.ks_2samp(train_series, new_series)
    drifted = p_value < KS_PVALUE_THRESHOLD
    return {
        'feature': feature_name,
        'test': 'KS',
        'statistic': round(float(statistic), 4),
        'p_value': round(float(p_value), 4),
        'drifted': bool(drifted)
    }


def calculate_psi(train_series, new_series, buckets=10):
    """Population Stability Index for a numeric or ordinal-encoded feature."""
    breakpoints = np.linspace(0, 100, buckets + 1)
    bucket_edges = np.percentile(train_series, breakpoints)
    bucket_edges[0] = -np.inf
    bucket_edges[-1] = np.inf

    train_counts = np.histogram(train_series, bins=bucket_edges)[0]
    new_counts = np.histogram(new_series, bins=bucket_edges)[0]

    train_pct = np.where(train_counts == 0, 0.0001, train_counts / len(train_series))
    new_pct = np.where(new_counts == 0, 0.0001, new_counts / len(new_series))

    psi = np.sum((new_pct - train_pct) * np.log(new_pct / train_pct))
    return float(psi)


def check_categorical_drift(train_series, new_series, feature_name):
    """Compares category proportions between train and new data."""
    train_dist = train_series.value_counts(normalize=True)
    new_dist = new_series.value_counts(normalize=True)

    all_categories = set(train_dist.index) | set(new_dist.index)
    train_pct = np.array([train_dist.get(c, 0.0001) for c in all_categories])
    new_pct = np.array([new_dist.get(c, 0.0001) for c in all_categories])

    psi = np.sum((new_pct - train_pct) * np.log(new_pct / train_pct))
    drifted = psi > PSI_THRESHOLD

    return {
        'feature': feature_name,
        'test': 'PSI',
        'psi_score': round(float(psi), 4),
        'drifted': bool(drifted)
    }


def run_drift_report(train_df, new_df):
    """Runs drift checks across all features, returns a full report."""
    results = []

    for col in NUMERIC_COLS:
        results.append(check_numeric_drift(train_df[col], new_df[col], col))

    for col in CATEGORICAL_COLS:
        results.append(check_categorical_drift(train_df[col], new_df[col], col))

    n_drifted = sum(r['drifted'] for r in results)

    report = {
        'total_features_checked': len(results),
        'features_drifted': n_drifted,
        'overall_drift_detected': n_drifted > 0,
        'details': results
    }
    return report


if __name__ == "__main__":
    train_df = pd.read_csv('data/raw_orders.csv')

    # Simulate a "new batch" with a deliberate shift for demonstration:
    # more high-discount apparel orders (a realistic post-launch promo scenario)
    new_batch = train_df.sample(300, random_state=99).copy()
    new_batch['discount_pct'] = new_batch['discount_pct'] * 1.6  # inflate discounts
    new_batch.loc[new_batch.sample(frac=0.3, random_state=1).index, 'category'] = 'apparel'

    report = run_drift_report(train_df, new_batch)

    print(json.dumps(report, indent=2))

    with open('monitoring/drift_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n{'⚠️  DRIFT DETECTED' if report['overall_drift_detected'] else '✅ No significant drift'}")
    print(f"Features drifted: {report['features_drifted']}/{report['total_features_checked']}")
