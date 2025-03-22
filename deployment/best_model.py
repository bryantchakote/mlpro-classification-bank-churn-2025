# Script permettant de retrouver le meilleur modèle d'une expérience MLflow
import os

experiment_id = 910302910665913739
model_name = "bank-churn-model-xgb-hyperopt"

# Métriques de tous les runs
f1_score = dict()

for run_in in os.listdir(f"../mlruns/{experiment_id}"):
    if run_in == "meta.yaml":
        continue
    with open(f"../mlruns/{experiment_id}/{run_in}/metrics/f1_score") as file:
        score = file.read()
        score = float(score.split()[1])
    f1_score[run_in] = score

# Meilleur run
best_run_id = max(f1_score, key=f1_score.get)

# Meilleur modèle
model_path = f"../mlartifacts/{experiment_id}/{best_run_id}/artifacts/{model_name}"

print(f"Path : {model_path}")
print(f"Score : {f1_score[best_run_id]}")
