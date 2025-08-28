import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, roc_auc_score, average_precision_score, f1_score, precision_score, recall_score

from .features import load_data, split_X_y
from .pipeline import make_pipeline
from .utils import save_json

def get_models(random_state=42):
    models = {
        'log_reg_balanced': LogisticRegression(max_iter=1000, class_weight='balanced', n_jobs=-1),
        'random_forest': RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=20, min_samples_leaf=10,
            n_jobs=-1, random_state=random_state
        ),
        'xgb': XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, n_jobs=-1, random_state=random_state, tree_method='hist', eval_metric='logloss'
        )
    }
    return models

def main(data_csv, model_out_dir):
    df = load_data(data_csv)
    X, y = split_X_y(df)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scorers = {
        'roc_auc': 'roc_auc',
        'pr_auc': 'average_precision',
        'f1': make_scorer(f1_score),
        'precision': make_scorer(precision_score, zero_division=0),
        'recall': make_scorer(recall_score, zero_division=0),
    }

    results = {}
    best_name, best_score = None, -np.inf

    print(f"Starting cross-validation on {len(X)} samples...")
    print(f"Training {len(get_models())} models with 5-fold CV...")
    
    for i, (name, est) in enumerate(get_models().items(), 1):
        print(f"\n[{i}/3] Training {name}...")
        # SMOTE is useful for tree & linear
        pipe = make_pipeline(est, use_smote=True)
        cv_res = cross_validate(pipe, X, y, cv=skf, scoring=scorers, return_estimator=True, n_jobs=-1)
        metrics = {k: float(np.mean(v)) for k, v in cv_res.items() if k.startswith('test_')}
        results[name] = metrics
        
        print(f"  ✓ {name} - PR-AUC: {metrics['test_pr_auc']:.4f}, ROC-AUC: {metrics['test_roc_auc']:.4f}")

        # select by PR-AUC (better for rare positives)
        if metrics['test_pr_auc'] > best_score:
            best_score = metrics['test_pr_auc']
            best_name = name
            best_estimator = cv_res['estimator'][np.argmax([r for r in cv_res['test_pr_auc']])]

    Path(model_out_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(best_estimator, f'{model_out_dir}/best_model.joblib')
    save_json(results, f'{model_out_dir}/cv_results.json')

    print("Saved best model:", best_name, "with PR-AUC:", best_score)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_csv', default='data/raw/creditcard.csv')
    parser.add_argument('--model_out_dir', default='models')
    args = parser.parse_args()
    main(args.data_csv, args.model_out_dir)
