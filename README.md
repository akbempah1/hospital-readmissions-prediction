# Hospital Readmissions Prediction — Diabetic Patients

![Python](https://img.shields.io/badge/Python-3.10-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange) ![XGBoost](https://img.shields.io/badge/XGBoost-1.7-green) ![SMOTE](https://img.shields.io/badge/imbalanced--learn-SMOTE-red) ![Data](https://img.shields.io/badge/Data-UCI%20ML%20Repository-lightblue)

**End-to-end ML pipeline predicting 30-day hospital readmissions — 101,766 patient records, 3-model comparison, GridSearchCV tuning, and a critical lesson on data leakage.**

> Author: Afriyie Karikari Bempah, PharmD | [LinkedIn](https://linkedin.com/in/afriyiekarikaribempah) | [GitHub](https://github.com/akbempah1)

---

## Overview

30-day hospital readmissions cost the US healthcare system over $26 billion annually. This project builds and compares three machine learning models to identify high-risk diabetic patients at discharge — enabling targeted clinical interventions.

The dataset contains 101,766 patient encounters from 130 US hospitals (1999–2008) with 50 clinical and demographic features.

---

## Key Findings

| Finding | Implication |
|---|---|
| **Prior inpatient visits → 45% readmission rate** | History of hospitalization dominates all other risk signals |
| **Logistic Regression outperforms RF and XGBoost** | Linear relationships dominate in administrative healthcare data |
| **SMOTE improved recall from 1% to 42%** | Class imbalance handling is critical for rare clinical outcomes |
| **Data leakage inflated AUC from 0.54 to 0.71** | Pipeline-based SMOTE is required for honest cross-validation |
| **Test AUC = 0.543** | Administrative data alone is insufficient — clinical notes needed |

---

## Technical Approach

### Models Compared
- Logistic Regression (best performer — AUC 0.634 CV)
- Random Forest (AUC 0.595 CV)
- XGBoost (AUC 0.614 CV)

### Key Methodological Decisions
- **SMOTE inside Pipeline** — prevents data leakage during cross-validation
- **StratifiedKFold (k=5)** — preserves class distribution across folds
- **GridSearchCV** — systematic hyperparameter search over C and penalty
- **Stratified train/test split** — ensures representative test set

### Critical ML Lesson
Applying SMOTE *before* cross-validation causes data leakage, inflating CV AUC by 0.17. The correct approach wraps SMOTE in an imblearn Pipeline so it is applied only to each training fold — never touching validation data.

---

## ML Concepts Demonstrated

- Exploratory data analysis on clinical data
- Handling class imbalance (SMOTE)
- Multi-model comparison
- Stratified k-fold cross-validation
- Hyperparameter tuning with GridSearchCV
- Data leakage detection and correction
- ROC-AUC, confusion matrix, precision-recall interpretation
- Logistic regression coefficient analysis

---

## How to Run

```bash
git clone https://github.com/akbempah1/hospital-readmissions-prediction.git
pip install pandas numpy matplotlib scikit-learn xgboost imbalanced-learn jupyter
jupyter notebook readmissions_prediction.ipynb
```

Dataset: [UCI ML Repository — Diabetes 130-US Hospitals](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

---

## Part of a Larger Portfolio

- [Project 1 — Medicare Drug Spending Analysis](https://github.com/akbempah1/medicare-drug-spending-analysis)
- [Project 2 — FDA Adverse Event Analysis](https://github.com/akbempah1/fda-adverse-events-analysis)
- **Project 3 — Hospital Readmissions Prediction** ← you are here

---

**Data Source:** UCI Machine Learning Repository  
*Analysis and interpretations are independent and not affiliated with UCI or any hospital system.*
