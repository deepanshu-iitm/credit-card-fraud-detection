# Credit Card Fraud Detection

A production-ready machine learning system for detecting fraudulent credit card transactions using advanced ensemble methods and class imbalance handling techniques.

## 🎯 Project Overview

This project implements a comprehensive fraud detection pipeline that achieves **86.5% PR-AUC** and **98.1% ROC-AUC** on real-world credit card transaction data. The system uses multiple ML algorithms with SMOTE oversampling to handle the inherent class imbalance in fraud detection.

## 📊 Performance Results

| Model | ROC-AUC | PR-AUC | Status |
|-------|---------|--------|---------|
| **XGBoost** ⭐ | 98.06% | **86.53%** | Selected Best Model |
| Random Forest | 98.37% | 82.83% | Strong Performance |
| Logistic Regression | 97.78% | 74.04% | Baseline |

- **Test ROC-AUC**: 98.06% (Excellent discrimination)
- **Test PR-AUC**: 86.53% (Outstanding for imbalanced data)
- **Optimal Threshold**: 0.9857 (F1-score optimized)


## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <https://github.com/deepanshu-iitm/credit-card-fraud-detection.git>
cd credit-card-fraud-detection

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Train Models

```bash
# Train all models with cross-validation
python -m src.train --data_csv data/raw/creditcard.csv --model_out_dir models
```

```bash
# Evaluate on test set and generate reports
python -m src.evaluate --data_csv data/raw/creditcard.csv --model_out_dir models --report_out_dir reports
```

## 🔧 Technical Details

### Dataset Features
- **V1-V28**: PCA-transformed features (anonymized)
- **Amount**: Transaction amount
- **Time**: Time elapsed since first transaction
- **Class**: Target variable (0=Normal, 1=Fraud)
- **Size**: 284,807 transactions, 492 fraudulent (0.17%)

### ML Pipeline Components

1. **Preprocessing**:
   - StandardScaler for Amount and Time features
   - V1-V28 features pass through (already PCA-normalized)

2. **Class Imbalance Handling**:
   - SMOTE (Synthetic Minority Oversampling Technique)
   - Generates synthetic fraud samples for training

3. **Model Selection**:
   - 5-fold stratified cross-validation
   - Selection based on PR-AUC (optimal for imbalanced data)
   - Multiple metrics evaluation (ROC-AUC, PR-AUC, F1, Precision, Recall)
