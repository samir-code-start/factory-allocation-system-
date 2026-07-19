"""
Feature Engineering for Lead Time Prediction
=============================================
Loads clean_orders.csv, encodes categorical features using LabelEncoder
(optimal for tree-based models), splits into train/test, and saves artifacts.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def main():
    # ── Load data ──────────────────────────────────────────────────────
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'clean_orders.csv')
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows from clean_orders.csv")

    # ── Select features & target ───────────────────────────────────────
    feature_cols = ['Product Name', 'Factory', 'Region', 'Ship Mode',
                    'Sales', 'Units', 'Cost', 'Gross Profit', 'distance']
    target_col = 'lead_time_days'

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # ── Encode categoricals with LabelEncoder ──────────────────────────
    # LabelEncoder is preferred over one-hot for tree-based models:
    # • Avoids high-dimensional sparse matrices (Product Name has many levels)
    # • Tree-based models handle ordinal-encoded features natively
    categorical_cols = ['Product Name', 'Factory', 'Region', 'Ship Mode']
    encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le
        print(f"  Encoded '{col}': {len(le.classes_)} unique values")

    # ── Save encoders ──────────────────────────────────────────────────
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    encoder_path = os.path.join(models_dir, 'encoders.joblib')
    joblib.dump(encoders, encoder_path)
    print(f"\nEncoders saved to {encoder_path}")

    # ── Train/Test split (80/20) ───────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set:  {X_test.shape[0]} samples")

    # ── Save processed arrays ──────────────────────────────────────────
    processed_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    X_train.to_csv(os.path.join(processed_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(processed_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(processed_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(processed_dir, 'y_test.csv'), index=False)
    print(f"Saved X_train, X_test, y_train, y_test to {processed_dir}")

    # Also save the feature column names for later use
    joblib.dump(feature_cols, os.path.join(models_dir, 'feature_columns.joblib'))
    print("Saved feature column names to models/feature_columns.joblib")

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    main()
