"""Verifies installed library versions match what the model was trained with.
Fails fast and loud at startup instead of allowing silent prediction corruption."""
import sys

REQUIRED_VERSIONS = {
    'sklearn': '1.5.2',
    'xgboost': '2.1.1',
    'shap': '0.51.0',
}

def check_versions():
    errors = []
    import sklearn
    if sklearn.__version__ != REQUIRED_VERSIONS['sklearn']:
        errors.append(f"scikit-learn: expected {REQUIRED_VERSIONS['sklearn']}, got {sklearn.__version__}")
    import xgboost
    if xgboost.__version__ != REQUIRED_VERSIONS['xgboost']:
        errors.append(f"xgboost: expected {REQUIRED_VERSIONS['xgboost']}, got {xgboost.__version__}")
    import shap
    if shap.__version__ != REQUIRED_VERSIONS['shap']:
        errors.append(f"shap: expected {REQUIRED_VERSIONS['shap']}, got {shap.__version__}")

    if errors:
        print("=" * 60)
        print("FATAL: Library version mismatch detected.")
        for e in errors:
            print(f"  - {e}")
        print("=" * 60)
        sys.exit(1)

    print("✅ All library versions match training environment")
