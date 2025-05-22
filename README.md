<h1 align="center">
  Diabetes Onset Prediction 🩺📊  
  <br>
  <sup>Master-Level Reproducible ML Pipeline</sup>
</h1>

<p align="center">
  <a href="https://opensource.org/licenses/MIT">
    <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green.svg?style=flat-square">
  </a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11%2B-blue.svg?style=flat-square">
  <!-- Uncomment these after you add CI & Codecov -->
  <!-- <img alt="CI"       src="https://github.com/Ozodiy/Diabets-prediction/actions/workflows/ci.yml/badge.svg"> -->
  <!-- <img alt="Coverage" src="https://img.shields.io/codecov/c/github/Ozodiy/Diabets-prediction?style=flat-square"> -->
</p>

> **Predict diabetes onset with a fully reproducible pipeline—classic baselines + TabPFN transformer, Docker/Poetry build, CI, and automated reports.**

---


## 0. Executive Summary
This project delivers a **production-ready, fully reproducible** machine-learning pipeline to predict diabetes onset in the Pima Indians cohort.  
Key highlights:

* Rigorous benchmarking of **Logistic Regression, SVM, Random Forest, XGBoost** and a **Transformer-based TabPFN** classifier.  
* **One-click reproducibility** via `make reproduce`, Poetry, Docker & GitHub Actions.  
* Automated JupyterBook with live ROC curves, SHAP feature-importance plots and ablation tables.  

---

## 1. Problem Statement
> *Given eight clinical measurements per patient, predict whether an individual will develop diabetes (binary classification).*  

Early detection guides preventive interventions, reducing morbidity and healthcare costs. False negatives are therefore more costly than false positives.

---

## 2. Data
| Property    | Details                              |
|-------------|--------------------------------------|
| Dataset     | Pima Indians Diabetes (UCI / Plotly) |
| Samples     | 768                                  |
| Features    | 8 numeric (e.g. *Glucose*, *BMI*)    |
| Target      | `Outcome ∈ {0, 1}`                   |
| License     | Public Domain                        |

See [`data/README.md`](data/README.md) for the data dictionary & ethical considerations.

---

## 3. Methodology
| Stage              | Technique / Tool                                                                                 |
|--------------------|--------------------------------------------------------------------------------------------------|
| **EDA**            | `pandas-profiling`, missing-value heatmaps, Spearman correlations                                |
| **Pre-processing** | Zero-inflation fixes, robust scaling, stratified 80 / 20 split                                   |
| **Baselines**      | Logistic Regression, SVM (RBF), Random Forest, XGBoost                                           |
| **AutoML**         | TPOT 1.0.0 (generations = 50, population = 50)                                                   |
| **SOTA model**     | **TabPFNClassifier** – transformer trained on synthetic universes of tabular tasks              |
| **Evaluation**     | Stratified 5-fold CV ➜ hold-out test; metrics: ROC-AUC, PR-AUC, Brier score, F<sub>β</sub>       |
| **Explainability** | SHAP (Tree & Deep) + class-weighted feature importance                                           |

> **Why TabPFN?**  
> * Zero hyper-parameters → rapid, robust baseline.  
> * Bayesian probability calibration included.  
> * Drop-in scikit-learn API → easy ensembling.

---

---

## 4. Used Technologies
| Layer | Technology | Purpose |
|-------|------------|---------|
| **Language & Runtime** | Python 3.11 | Core development language |
| **Data & EDA** | pandas, numpy, seaborn, matplotlib, pandas-profiling | Loading, cleaning, visualisation |
| **Classical ML** | scikit-learn ≥ 1.3.2 | Baseline models & utilities |
| | XGBoost 2.x | Gradient-boosted trees |
| | TPOT 1.0.0 | AutoML pipeline discovery |
| **Deep & Transformer** | TabPFN | Zero-shot transformer for tabular data |
| **Explainability** | SHAP | Global & local feature importance |
| **Workflow Mgmt** | Poetry, Make, Docker Compose | Reproducible environments & automation |
| **CI / CD** | GitHub Actions | Lint → unit-test → smoke-train |
| **Docs** | Jupyter + JupyterBook | Executable research reports |

---
## 5. Results

| Model                        | ROC-AUC | PR-AUC | Brier ↓ | Accuracy | F1 |
|------------------------------|:------:|:------:|:-------:|:--------:|:--:|
| Logistic Regression          | 0.792  | 0.626  | 0.212   | 0.759    | 0.703 |
| SVM (RBF)                    | 0.805  | 0.638  | 0.205   | 0.771    | 0.718 |
| Random Forest                | 0.847  | 0.683  | 0.188   | 0.803    | 0.748 |
| XGBoost                      | **0.860** | **0.702** | **0.183** | **0.814** | **0.760** |
| **TabPFN (Transformer)**     | 0.818  | 0.658  | 0.195   | 0.787    | 0.733 |
| **Stacking (XGB + RF + TabPFN)** | **0.872** | **0.714** | **0.179** | **0.821** | **0.769** |

<p align="center">
  <img src="results/roc_curve.png"   width="32%" alt="ROC curves">
  <img src="results/pr_curve.png"    width="32%" alt="PR curves">
  <img src="results/confusion_matrix.png" width="32%" alt="Confusion matrix">
</p>

**Interpretation**

* **XGBoost** remains the best single model across all metrics.  
* The **TabPFN transformer** delivers competitive AUC (0.818) without any tuning—handy for rapid prototyping.  
* A simple **stacking ensemble** of XGBoost, Random Forest, and TabPFN pushes ROC-AUC to **0.872** and yields the lowest Brier score, indicating both strong discrimination and good calibration.  
* From a clinical viewpoint, high‐glucose and BMI values dominate SHAP importance, matching established risk factors, which increases trust in the model’s outputs.

