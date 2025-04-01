# FINAL WORKING model_training.py
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    f1_score
)
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
import joblib
import json
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load and convert data to numpy arrays immediately"""
    print("Loading and converting data...")
    try:
        # Load data
        X_train = pd.read_csv('data/processed/X_train.csv')
        y_train = pd.read_csv('data/processed/y_train.csv')
        X_test = pd.read_csv('data/processed/X_test.csv')
        y_test = pd.read_csv('data/processed/y_test.csv')
        
        # Convert to numpy arrays
        X_train = X_train.values.astype(np.float32)
        y_train = y_train.values.ravel().astype(np.float32)
        X_test = X_test.values.astype(np.float32)
        y_test = y_test.values.ravel().astype(np.float32)
        
        print("Data conversion complete:")
        print(f"- X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
        print(f"- y_train shape: {y_train.shape}, dtype: {y_train.dtype}")
        
        return X_train, y_train, X_test, y_test
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def handle_class_imbalance(X_train, y_train):
    """Apply SMOTE to handle class imbalance"""
    print("\nChecking class distribution...")
    unique, counts = np.unique(y_train, return_counts=True)
    print(dict(zip(unique, counts)))
    
    if len(unique) > 1:
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        print("\nAfter SMOTE:")
        unique, counts = np.unique(y_res, return_counts=True)
        print(dict(zip(unique, counts)))
        return X_res, y_res
    else:
        print("Only one class present - skipping SMOTE")
        return X_train, y_train

def train_model(X_train, y_train):
    """Train XGBoost with numpy arrays only"""
    print("\nStarting model training...")
    
    # Model configuration
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1
    )
    
    # Conservative parameter grid
    param_grid = {
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
        'n_estimators': [100, 150],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    # Grid search with error handling
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        verbose=2,
        error_score='raise',
        n_jobs=-1
    )
    
    print("\nTraining with GridSearchCV...")
    grid_search.fit(X_train, y_train)
    
    print("\nBest parameters found:")
    print(grid_search.best_params_)
    
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'f1_score': f1_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    
    # Print results
    print("\n" + "="*50)
    print("Evaluation Metrics")
    print("="*50)
    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return metrics

def save_artifacts(model, metrics):
    """Save model and metrics"""
    print("\nSaving artifacts...")
    os.makedirs('model', exist_ok=True)
    
    # Save model
    joblib.dump(model, 'model/risk_model.pkl')
    
    # Save metrics
    with open('model/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("Artifacts saved to 'model' directory")

if __name__ == "__main__":
    print("Starting model training pipeline...")
    
    try:
        # 1. Load and convert data
        X_train, y_train, X_test, y_test = load_data()
        
        # 2. Handle class imbalance
        X_res, y_res = handle_class_imbalance(X_train, y_train)
        
        # 3. Train model
        model = train_model(X_res, y_res)
        
        # 4. Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        
        # 5. Save artifacts
        save_artifacts(model, metrics)
        
        print("\nTraining completed successfully!")
        
    except Exception as e:
        print(f"\nError in training pipeline: {str(e)}")
        raise