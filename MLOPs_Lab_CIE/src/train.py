import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import json
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("data/training_data.csv")

X = df.drop("land_stability_score", axis=1)
y = df["land_stability_score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("geosurvey-land-stability-score")

def evaluate(model, name):
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

        mlflow.log_params(model.get_params())
        mlflow.log_metrics({"mae": mae, "rmse": rmse, "r2": r2, "mape": mape})
        mlflow.set_tag("experiment_type", "baseline_comparison")

        mlflow.sklearn.log_model(model, "model")

        return {"name": name, "mae": mae, "rmse": rmse, "r2": r2, "mape": mape, "model": model}

lr = evaluate(LinearRegression(), "LinearRegression")
rf = evaluate(RandomForestRegressor(random_state=42), "RandomForest")

models = [lr, rf]
best = min(models, key=lambda x: x["rmse"])

joblib.dump(best["model"], "models/best_model.pkl")

output = {
    "experiment_name": "geosurvey-land-stability-score",
    "models": [
        {k: v for k, v in lr.items() if k != "model"},
        {k: v for k, v in rf.items() if k != "model"}
    ],
    "best_model": best["name"],
    "best_metric_name": "rmse",
    "best_metric_value": best["rmse"]
}

os.makedirs("results", exist_ok=True)

with open("results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 1 DONE")