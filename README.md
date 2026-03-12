# EE4685 Assignment 2 — Credit Card Fraud Detection

## Project Overview
This project addresses **credit card fraud detection** as a machine learning problem. The objective is to distinguish fraudulent transactions from legitimate ones using a combination of **non-Bayesian** and **Bayesian** approaches, in line with the requirements of Assignment 2 for EE4685.

Fraud detection is a practically relevant and technically challenging task. In real payment systems, fraudulent transactions are rare, which makes this an **extremely imbalanced classification problem**. As a result, standard accuracy is not a reliable indicator of model quality, and more suitable metrics such as precision-recall based measures are needed.

In this project, we aim to compare a small but well-justified set of models and evaluate them not only in terms of predictive performance, but also with respect to interpretability, uncertainty handling, and practical relevance.

## Dataset
We use the **Credit Card Fraud Detection** dataset available on Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The dataset contains transactions made by European cardholders in **September 2013** over a period of **two days**. It includes:

- **284,807 total transactions**
- **492 fraudulent transactions**
- a fraud rate of **0.172%**

This extreme class imbalance is one of the central challenges of the project.

The dataset is not included in this repository because of its size. To run the notebook locally, please download the dataset from the Kaggle link above and place the file `creditcard.csv` in the `data/` folder.

### Features
The dataset consists of numerical variables only:

- **V1 to V28**: anonymized PCA-transformed features
- **Time**: seconds elapsed between each transaction and the first transaction
- **Amount**: transaction amount
- **Class**: target variable, where `1` denotes fraud and `0` denotes a legitimate transaction

Because the original feature meanings are hidden for confidentiality reasons, interpretability is limited. This is an important limitation that we take into account in the analysis and discussion.

## Project Goal
The main goal of the project is to compare **Bayesian** and **non-Bayesian** approaches for fraud detection in a highly imbalanced setting.

More specifically, the project is structured around three questions:

1. How well do standard supervised classifiers perform on this task?
2. What changes when we move from a non-Bayesian to a Bayesian formulation?
3. Can fraud also be approached from an anomaly-detection perspective?

## Planned Model Comparison
Rather than testing many models superficially, we focus on a smaller number of methods that support a meaningful comparison.

Our planned model lineup is:

- **Logistic Regression** as a non-Bayesian supervised baseline
- **Bayesian Logistic Regression** as the Bayesian supervised counterpart
- **One anomaly-detection method**, such as One-Class SVM or PCA-based reconstruction error

This setup allows us to compare:
- **Bayesian vs non-Bayesian**
- **supervised classification vs anomaly detection**

while keeping the project manageable and analytically focused.

## Evaluation Strategy
Because the dataset is highly imbalanced, plain classification accuracy is not informative. A classifier that predicts every transaction as legitimate would still achieve very high accuracy while being useless in practice.

For that reason, the project focuses primarily on:

- **Area Under the Precision-Recall Curve (AUPRC)**
- **Precision**
- **Recall**
- **F1-score**
- **ROC-AUC**
- **Confusion matrices**

We also analyze decision thresholds, since the balance between false positives and false negatives is especially important in fraud detection.

## Repository Structure
```text
.
├── README.md
├── notebooks/
│   └── assignment2_fraud_detection.ipynb
├── data/
│   └── creditcard.csv
├── figures/
├── src/
└── requirements.txt