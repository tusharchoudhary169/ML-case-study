# Machine Learning Case Studies: Healthcare & Financial Risk Modeling

This repository contains two end-to-end machine learning case studies focusing on risk prediction in highly regulated domains: healthcare (hospital readmissions) and finance (credit card fraud detection).

---

## 📋 Table of Contents
1. [Case Study 1: Hospital Readmission Prediction](#-case-study-1-hospital-readmission-prediction)
2. [Case Study 2: Credit Card Fraud Detection](#-case-study-2-credit-card-fraud-detection)
3. [Case Study Comparison Matrix](#-case-study-comparison-matrix)
4. [Google Colab Setup & Instructions](#-google-colab-setup--instructions)

---

## 🏥 Case Study 1: Hospital Readmission Prediction

### 📌 Overview
Predicting 30-day hospital readmission risk for patients using patient records (vitals, demographics, prior visits) to enable early medical intervention and reduce financial penalties.

### 📊 Dataset
* **Dataset:** Diabetes 130-US hospitals (1999–2008)
* **File:** `diabetic_data.csv` (Kaggle)
* **Target Variable:** `readmitted` (Binary: `1` for `<30` days, `0` otherwise)

### ⚙️ Technical Pipeline
1. **Preprocessing:** Cleaned missing entries, handled categorical columns (`age`, `gender`) using One-Hot Encoding with `drop_first=True` to prevent multicollinearity.
2. **Stratified Splitting:** 80/20 train-test split maintaining class proportions (`stratify=y`).
3. **Feature Scaling:** Applied `StandardScaler` to prevent feature magnitude bias in L2 regularization.
4. **Model Training:** Trained a **Logistic Regression** model with L2 Regularization (Ridge penalty).
5. **Evaluation:** Evaluated model using ROC-AUC curves and Confusion Matrices.

### ⚕️ Clinical Risk Analysis
* **False Positives (FP):** Predicting a healthy patient will be readmitted. Leads to unnecessary follow-up calls and minor resource allocation.
* **False Negatives (FN):** Failing to identify a high-risk patient who gets readmitted. Leads to severe medical complications and heavy hospital financial penalties.
* **Verdict:** **False Negatives are significantly more costly.** Production models should lower the probability threshold below `0.5` to prioritize **Recall**.

---

## 💳 Case Study 2: Credit Card Fraud Detection

### 📌 Overview
Detecting fraudulent transactions in an extremely imbalanced dataset using XGBoost, synthetic oversampling (SMOTE), decision threshold optimization, and feature importance interpretation.

### 📊 Dataset
* **Dataset:** Credit Card Fraud Detection (MLG-ULB)
* **File:** `creditcard.csv` (Kaggle)
* **Class Ratio:** ~0.17% Fraudulent (`1`), 99.83% Normal (`0`)

### ⚙️ Technical Pipeline
1. **Data Preparation:** Dropped the non-predictive `Time` column and missing rows (`dropna()`).
2. **Oversampling:** Used **SMOTE** on the training set only to balance class distributions without data leakage.
3. **Model Training:** Trained an **XGBoost Classifier** (`XGBClassifier`) on the resampled data.
4. **Threshold Tuning:** Derived optimal classification thresholds using `precision_recall_curve` to maximize the F1-Score instead of relying on the default `0.5` cutoff.
5. **Feature Importance:** Plotted top features using XGBoost's `plot_importance` to identify key fraud drivers.

### 📈 Financial Risk Analysis
* **False Positives (FP):** Declining a legitimate customer's card. Causes user friction and reduced customer trust.
* **False Negatives (FN):** Allowing a fraudulent transaction. Results in direct monetary loss and chargeback fees.
* **Verdict:** Optimal performance requires tuning the threshold along the **Precision-Recall Curve** to achieve a business-aligned balance between fraud capture (Recall) and user friction (Precision).

---

## 📊 Case Study Comparison Matrix

| Metric / Dimension | Case Study 1: Readmission | Case Study 2: Fraud Detection |
| :--- | :--- | :--- |
| **Domain** | Healthcare | Financial Tech |
| **Primary Algorithm** | Logistic Regression (L2) | XGBoost Classifier |
| **Data Imbalance** | Moderate (~11% positive) | Extreme (~0.17% positive) |
| **Imbalance Handling** | Stratified Splitting | SMOTE Oversampling |
| **Primary Evaluation** | ROC-AUC Score | PR-AUC / F1-Score |
| **Most Critical Error** | False Negative (Patient Risk) | False Negative (Monetary Loss) |
| **Key Optimization** | Feature Scaling & Standardization | Decision Threshold Tuning |

---

## 🚀 Google Colab Setup & Instructions

### Prerequisites
Make sure you have downloaded both CSV files from Kaggle:
1. `diabetic_data.csv`
2. `creditcard.csv`

### Running the Notebooks
1. Open [Google Colab](https://colab.research.google.com/).
2. Click the **Folder Icon** in the left sidebar and upload the corresponding `.csv` file.
3. Run all code cells sequentially.

```bash
# Required Dependencies (Pre-installed on Colab)
pip install xgboost imbalanced-learn scikit-learn pandas matplotlib seaborn