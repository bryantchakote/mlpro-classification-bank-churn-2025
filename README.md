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

```text
- data/
  - submissions/ : Fichiers de prédictions
  - train_data.csv : Données avec labels
  - test_data.csv : Données sans labels (Kaggle test)
- notebooks/
  - eda.ipynb : Analyse exploratoire des données
  - model.ipynb : Préprocessing et entraînement du modèle
- .python-version, .poetry.lock, .pyproject.toml : Fichiers de gestion de l'environnement
```

## 🔧 Setup du projet

1. **Installation des dépendances**

   ```sh
   poetry install
   ```

2. **Activation de l'environnement**

   ```sh
   poetry shell
   ```

3. **Lancement du serveur MLflow**

   Pour suivre les expériences avec MLflow, il faut d'abord activer le serveur :

   ```sh
   mlflow ui -p 8080
   ```

   Si vous choisisssez un autre port, veillez à modifier le notebook également : `mlflow.set_tracking_uri(uri="http://localhost:PORT")`.

4. **Réinitialisation de l'environnement MLflow**

   Si besoin, vous pouvez réinitialiser votre environnement MLflow en supprimant les fichiers de suivi :

   ```sh
   rm -rf mlartifacts mlruns
   ```

  Cela permettra de repartir sur une base propre pour l'expérimentation. Relancez également le serveur local pour nettoyer l'environnement sur l'interface graphique.

## 🛠️ Technologies utilisées

- **Python** (Pandas, NumPy, Matplotlib, Seaborn)
- **Machine Learning** : Scikit-Learn, XGBoost, Hyperopt
- **Tracking des modèles** : MLflow
- **Jupyter Notebook** pour l'expérimentation
