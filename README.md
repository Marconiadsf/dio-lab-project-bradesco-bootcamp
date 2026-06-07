# Detecção de Anomalias em Transações em Python

Lab Project do **Bootcamp Bradesco - GenAI, Dados & Cyber** (DIO)  
Trilha: *Análise de Dados com Python: Da Preparação à Aplicação com Segurança*

## Objetivo

Construir um pipeline de detecção de fraudes em transações de cartão de crédito, abordando o desafio central de **classificação com classes extremamente desbalanceadas** (~0.17% de fraudes).

O projeto cobre as três etapas do lab:

1. **Primeiros passos** — EDA, train/test split estratificado, feature engineering (log1p + StandardScaler sem data leakage)
2. **Avaliação e Técnicas de Balanceamento** — Regressão Logística com Undersampling e SMOTE, ajuste de threshold
3. **Modelos Avançados e Explicabilidade** — Random Forest e XGBoost com GridSearchCV e SHAP

## Dataset

[Credit Card Fraud Detection](https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv) — 284.807 transações, features V1–V28 (PCA anonimizado), Amount, Time e Class (0 = legítima, 1 = fraude).

## Resultados

| Modelo | Precision | Recall | F1 |
|---|:---:|:---:|:---:|
| LR + SMOTE (threshold 0.5) | 0.15 | 0.85 | 0.25 |
| XGBoost GridSearch (`scale_pos_weight=57`) | 0.55 | **0.83** | 0.66 |
| XGBoost padrão (`scale_pos_weight=10`) | 0.93 | 0.78 | **0.85** |
| LR + SMOTE + threshold ótimo (~1.0) | 0.87 | 0.76 | 0.81 |
| Random Forest GridSearch + threshold (0.6447) | 0.89 | 0.74 | 0.81 |

> Ranking completo e análise por modelo no notebook.

## Stack

- Python · pandas · NumPy · Matplotlib · Seaborn
- scikit-learn (LogisticRegression, RandomForest, GridSearchCV)
- XGBoost · imbalanced-learn (SMOTE) · SHAP

## Estrutura

```
src/
└── Lab_Project_DIO_Detecção_de_Anomalias.ipynb
data/       # vazio — dataset carregado via URL no notebook
tests/
utils/
```

## Como executar

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn shap
jupyter notebook src/Lab_Project_DIO_Detecção_de_Anomalias.ipynb
```
