# Utilisez ce script pour entraîner le modèle final sur toutes les données
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

# Chargement des données
data_folder = Path("../data")
df = pd.read_csv(data_folder / "train_data.csv")
X = df.drop(columns="Exited")
y = df["Exited"]


# Création de nouvelles variables potentiellement pertinenentes après analyse
# Les intervalles ont été créés après essais-erreurs en combinant observation
# des données et techniques de clustering (k-means)
def create_columns(df):
    X = df.copy()

    X["IsNewClient"] = X["Tenure"] == 0

    X["HasNullBalance"] = X["Balance"] == 0

    X["NumOfProducts_2"] = X["NumOfProducts"].replace({4: 3})

    X["EstimatedSalary_2"] = pd.cut(
        x=X["EstimatedSalary"],
        bins=[-np.inf, 39500, 78260, 115400, 154430, np.inf],
        labels=[0, 1, 2, 3, 4],
    )

    X["Balance_2"] = pd.cut(
        x=X["Balance"],
        bins=[-np.inf, 50000, 100000, 150000, 200000, np.inf],
        labels=[0, 1, 2, 3, 4],
    )

    X["CreditScore_2"] = pd.cut(
        x=X["CreditScore"],
        bins=[-np.inf, 545, 612, 673, 744, np.inf],
        labels=[0, 1, 2, 3, 4],
    )

    return X


# Preprocessing
preprocessor = Pipeline(
    steps=[
        ("CreateColumns", FunctionTransformer(create_columns)),
        (
            "Transformer",
            ColumnTransformer(
                transformers=[
                    (
                        "OneHotEncoder",
                        OneHotEncoder(drop="first", handle_unknown="error"),
                        ["Gender", "Geography"],
                    ),
                    (
                        "MinMaxScaler",
                        MinMaxScaler(),
                        [
                            "Age",
                            "NumOfProducts_2",
                            "NumOfProducts",
                            "Balance_2",
                            "Balance",
                            "CreditScore",
                            "CreditScore_2",
                            "EstimatedSalary",
                            "EstimatedSalary_2",
                            "Tenure",
                        ],
                    ),
                    (
                        "Passthrough",
                        "passthrough",
                        [
                            "IsActiveMember",
                            "HasNullBalance",
                            "HasCrCard",
                            "IsNewClient",
                        ],
                    ),
                ]
            ),
        ),
    ]
)


# Modèle
params = {
    "colsample_bytree": 0.75,
    "gamma": 3.2399903571543387,
    "learning_rate": 0.22,
    "max_delta_step": 7.155584591130996,
    "max_depth": 6,
    "min_child_weight": 3.8392819888268797,
    "n_estimators": 660,
    "ranom_state": 42,
    "reg_alpha": 4.419633127363078,
    "reg_lambda": 3.100594988383205,
    "subsample": 0.9,
}

model = Pipeline(
    steps=[
        ("Preprocessor", preprocessor),
        ("Classifier", XGBClassifier(**params)),
    ]
)
model.fit(X, y)
