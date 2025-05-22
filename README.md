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

## Table of Contents
1. [Executive Summary](#0-executive-summary)  
2. [Problem Statement](#1-problem-statement)  
3. [Data](#2-data)  
4. [Methodology](#3-methodology)  
5. [Repository Structure](#4-repository-structure)  
6. [Installation & Quick Start](#5-installation--quick-start)  
7. [Usage](#6-usage)  
8. [Results](#7-results)  
9. [Roadmap](#8-roadmap)  
10. [Contributing](#9-contributing)  
11. [License](#10-license)  
12. [Citation](#11-how-to-cite)  
13. [Acknowledgements](#12-acknowledgements)  

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

## 4. Repository Structure

