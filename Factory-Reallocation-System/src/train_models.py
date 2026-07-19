"""
Model Training & Evaluation for Lead Time Prediction
=====================================================
Trains Linear Regression, Random Forest, and Gradient Boosting models.
Evaluates on RMSE, MAE, R². Selects and saves the best model.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

def load_data():
    """Load the pre-processed train/test splits."""
    processed_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
    X_test  = pd.read_csv(os.path.join(processed_dir, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(processed_dir, 'y_train.csv')).squeeze()
    y_test  = pd.read_csv(os.path.join(processed_dir, 'y_test.csv')).squeeze()
    return X_train, X_test, y_train, y_test

def evaluate(model, X_test, y_test):
    """Return RMSE, MAE, R² for a fitted model."""
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)
    return rmse, mae, r2

def main():
    # ── Load data ──────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = load_data()
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples:     {len(X_test)}\n")

    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)

    # ── Define models ──────────────────────────────────────────────────
    models = {
        'Linear Regression':     LinearRegression(),
        'Random Forest':         RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
        'Gradient Boosting':     GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
    }

    # ── Train & evaluate ───────────────────────────────────────────────
    results = {}
    trained = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        rmse, mae, r2 = evaluate(model, X_test, y_test)
        results[name] = {'RMSE': rmse, 'MAE': mae, 'R²': r2}
        trained[name] = model
        print(f"  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")

    # ── Comparison table ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    comparison_df = pd.DataFrame(results).T
    comparison_df.index.name = 'Model'
    print(comparison_df.to_string())

    # ── Select best model (lowest RMSE, tie-break highest R²) ─────────
    best_name = min(results, key=lambda n: (results[n]['RMSE'], -results[n]['R²']))
    print(f"\n✓ Best model: {best_name} (RMSE={results[best_name]['RMSE']:.4f}, R²={results[best_name]['R²']:.4f})")

    # ── Save all models ───────────────────────────────────────────────
    model_filenames = {
        'Linear Regression':  'linear_regression.joblib',
        'Random Forest':      'random_forest.joblib',
        'Gradient Boosting':  'gradient_boosting.joblib',
    }
    for name, model in trained.items():
        path = os.path.join(models_dir, model_filenames[name])
        joblib.dump(model, path)
        print(f"  Saved {name} → {path}")

    best_path = os.path.join(models_dir, 'best_model.joblib')
    joblib.dump(trained[best_name], best_path)
    print(f"  Saved best model → {best_path}")

    # ── Feature importances (tree-based models) ───────────────────────
    feature_names = X_train.columns.tolist()
    for name in ['Random Forest', 'Gradient Boosting']:
        model = trained[name]
        importances = model.feature_importances_
        fi_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        print(f"\n── {name}: Feature Importances ──")
        print(fi_df.to_string(index=False))

    # ── Sanity check: 10 sample predictions vs actual ─────────────────
    best_model = trained[best_name]
    preds = best_model.predict(X_test)
    sample_df = pd.DataFrame({
        'Actual':    y_test.values[:10],
        'Predicted': np.round(preds[:10], 2),
        'Error':     np.round(preds[:10] - y_test.values[:10], 2)
    })
    print(f"\n── Sanity Check: 10 Sample Predictions ({best_name}) ──")
    print(sample_df.to_string(index=False))

    return results, trained, best_name

if __name__ == '__main__':
    main()
