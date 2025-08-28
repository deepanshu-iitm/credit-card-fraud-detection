import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, precision_recall_curve
)

from .features import load_data, split_X_y
from .pipeline import make_pipeline
from .train import get_models
from .utils import save_json

def main(data_csv, model_out_dir, report_out_dir):
    df = load_data(data_csv)
    X, y = split_X_y(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Train the best family (XGB here) using full train
    model = get_models()['xgb']
    pipe = make_pipeline(model, use_smote=True)
    pipe.fit(X_train, y_train)

    # Save final model
    Path(model_out_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, f'{model_out_dir}/final_model.joblib')

    # Evaluate on test
    y_prob = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        'roc_auc': float(roc_auc_score(y_test, y_prob)),
        'pr_auc': float(average_precision_score(y_test, y_prob)),
        'report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }

    Path(report_out_dir).mkdir(parents=True, exist_ok=True)
    save_json(metrics, f'{report_out_dir}/metrics.json')

    # Optional: Choose threshold to maximize F1 or recall
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    best_f1, best_t = -1, 0.5
    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        f1 = 2*p*r/(p+r+1e-9)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    metrics['best_threshold_by_f1'] = float(best_t)
    save_json(metrics, f'{report_out_dir}/metrics.json')

    print("Test ROC-AUC:", metrics['roc_auc'])
    print("Test PR-AUC :", metrics['pr_auc'])
    print("Best threshold by F1:", metrics['best_threshold_by_f1'])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_csv', default='data/raw/creditcard.csv')
    parser.add_argument('--model_out_dir', default='models')
    parser.add_argument('--report_out_dir', default='reports')
    args = parser.parse_args()
    main(args.data_csv, args.model_out_dir, args.report_out_dir)
