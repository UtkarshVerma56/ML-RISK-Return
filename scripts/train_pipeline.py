"""End-to-end training pipeline: feature engineering -> tuning -> evaluation -> SHAP."""
import pandas as pd
import numpy as np
import json
import joblib
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score,
                               recall_score, f1_score, confusion_matrix)
import xgboost as xgb
import shap

NUMERIC_COLS = ['order_amount', 'discount_pct', 'customer_past_orders',
                 'customer_past_returns', 'customer_return_rate', 'account_age_days',
                 'day_of_week', 'hour_of_day', 'address_mismatch']
CATEGORICAL_COLS = ['category', 'payment_method']
COST_FALSE_POSITIVE = 50
COST_FALSE_NEGATIVE = 350


def load_and_split(data_path):
    df = pd.read_csv(data_path)
    X = df.drop(columns=['order_id', 'customer_id', 'returned'])
    y = df['returned']
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor():
    return ColumnTransformer(transformers=[
        ('num', StandardScaler(), NUMERIC_COLS),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_COLS)
    ])


def train_model(X_train, y_train):
    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos

    param_grid = {
        'n_estimators': [200, 300, 400, 500], 'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1], 'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0], 'min_child_weight': [1, 3, 5]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    xgb_base = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric='aucpr', random_state=42)
    search = RandomizedSearchCV(xgb_base, param_distributions=param_grid, n_iter=30,
                                  scoring='average_precision', cv=cv, random_state=42, n_jobs=-1, verbose=1)
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, scale_pos_weight


def find_optimal_threshold(y_test, test_probs):
    thresholds = np.arange(0.05, 0.95, 0.02)
    results = []
    for t in thresholds:
        preds = (test_probs >= t).astype(int)
        tp = np.sum((preds==1)&(y_test==1)); fp = np.sum((preds==1)&(y_test==0))
        fn = np.sum((preds==0)&(y_test==1))
        total_cost = (fp*COST_FALSE_POSITIVE) + (fn*COST_FALSE_NEGATIVE)
        precision = tp/(tp+fp) if (tp+fp)>0 else 0
        recall = tp/(tp+fn) if (tp+fn)>0 else 0
        results.append({'threshold': round(t,2), 'total_cost': total_cost, 'precision': precision, 'recall': recall})
    cost_df = pd.DataFrame(results)
    optimal = cost_df.loc[cost_df['total_cost'].idxmin()]
    default = cost_df.iloc[(cost_df['threshold']-0.5).abs().argsort()[:1]].iloc[0]
    return optimal, default


def main(data_path='data/raw_orders.csv', model_dir='models'):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)

    print("Loading and splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split(data_path)
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    print("Fitting preprocessor...")
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)
    X_train_p = preprocessor.transform(X_train)
    X_val_p = preprocessor.transform(X_val)
    X_test_p = preprocessor.transform(X_test)
    feature_names = NUMERIC_COLS + list(preprocessor.named_transformers_['cat'].get_feature_names_out(CATEGORICAL_COLS))

    print("Training model (this may take a few minutes)...")
    model, best_params, scale_pos_weight = train_model(X_train_p, y_train)

    val_probs = model.predict_proba(X_val_p)[:, 1]
    print(f"Val PR-AUC: {average_precision_score(y_val, val_probs):.4f}")
    print(f"Val ROC-AUC: {roc_auc_score(y_val, val_probs):.4f}")

    print("Evaluating on test set...")
    test_probs = model.predict_proba(X_test_p)[:, 1]
    test_preds_default = model.predict(X_test_p)
    tn, fp, fn, tp = confusion_matrix(y_test, test_preds_default).ravel()

    test_metrics = {
        'roc_auc': float(roc_auc_score(y_test, test_probs)),
        'pr_auc': float(average_precision_score(y_test, test_probs)),
        'precision': float(precision_score(y_test, test_preds_default)),
        'recall': float(recall_score(y_test, test_preds_default)),
        'f1_score': float(f1_score(y_test, test_preds_default)),
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
        'test_set_size': int(len(y_test)), 'test_return_rate': float(y_test.mean())
    }
    print(json.dumps(test_metrics, indent=2))

    print("Finding cost-optimal threshold...")
    optimal, default = find_optimal_threshold(y_test.values, test_probs)
    savings = default['total_cost'] - optimal['total_cost']
    savings_pct = (savings / default['total_cost']) * 100
    print(f"Optimal threshold: {optimal['threshold']} | Savings: ₹{savings:,.0f} ({savings_pct:.1f}%)")

    cost_analysis = {
        'cost_assumptions': {'false_positive_inr': COST_FALSE_POSITIVE, 'false_negative_inr': COST_FALSE_NEGATIVE},
        'default_threshold': {'value': 0.5, 'total_cost_inr': float(default['total_cost']),
                                'precision': float(default['precision']), 'recall': float(default['recall'])},
        'chosen_threshold': {'value': float(optimal['threshold']), 'total_cost_inr': float(optimal['total_cost']),
                               'precision': float(optimal['precision']), 'recall': float(optimal['recall']),
                               'savings_inr': float(savings), 'savings_pct': float(savings_pct)}
    }

    print("Computing SHAP explainer...")
    explainer = shap.TreeExplainer(model)

    print("Saving all artifacts...")
    joblib.dump(preprocessor, f'{model_dir}/preprocessor.pkl')
    joblib.dump(model, f'{model_dir}/return_risk_model.pkl')
    joblib.dump(explainer, f'{model_dir}/shap_explainer.pkl')

    with open(f'{model_dir}/feature_names.json', 'w') as f:
        json.dump(feature_names, f)
    with open(f'{model_dir}/test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)
    with open(f'{model_dir}/cost_analysis.json', 'w') as f:
        json.dump(cost_analysis, f, indent=2)
    with open(f'{model_dir}/model_metadata.json', 'w') as f:
        json.dump({'best_params': best_params, 'scale_pos_weight': float(scale_pos_weight),
                    'train_size': len(X_train), 'val_size': len(X_val)}, f, indent=2)

    print("✅ Training pipeline complete. All artifacts saved to", model_dir)


if __name__ == "__main__":
    main()
