# 📊 Classification Bank Churn

## 🏆 Objectif

L'objectif de cette [compétition Kaggle](https://www.kaggle.com/competitions/mlpro-classification-bank-churn-2025) est de prédire si un client va continuer à utiliser les services d'une banque ou s'il va clôturer son compte. Cette prédiction permet de mettre en place des stratégies de rétention plus efficaces.

## 📂 Données

Nous utilisons un ensemble de données clients contenant des informations :

- **Démographiques** (âge, sexe, pays)
- **Financières** (score de crédit, solde du compte, nombre de produits bancaires, possession d'une carte de crédit, salaire estimé)
- **Comportementales** (ancienneté, activité)

Une variable cible nous permet de dire si le client est parti (`Exited = 1`) ou resté (`Exited = 0`).

## 📄 Structure des fichiers

- `data/`
  - `submissions/` : Fichiers de prédictions
  - `train_data.csv` : Données avec labels
  - `test_data.csv` : Données sans labels
- `notebooks/`
  - `eda.ipynb` : Analyse exploratoire des données
  - `model.ipynb` : Préprocessing et entraînement du modèle
- `models/` : Sauvegarde des modèles
- `mlruns/` : Suivi des expériences avec MLflow

## 🛠️ Technologies utilisées

- **Python** (Pandas, NumPy, Matplotlib, Seaborn)
- **Machine Learning** : Scikit-Learn, XGBoost
- **Tracking des modèles** : MLflow
- **Jupyter Notebook** pour l'expérimentation
