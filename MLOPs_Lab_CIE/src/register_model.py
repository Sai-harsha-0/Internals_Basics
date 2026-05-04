import mlflow
from mlflow.tracking import MlflowClient
import json

mlflow.set_tracking_uri("mlruns")
client = MlflowClient()

experiment = client.get_experiment_by_name("geosurvey-land-stability-score")
runs = client.search_runs([experiment.experiment_id], order_by=["metrics.rmse ASC"])

best_run = runs[0]
run_id = best_run.info.run_id

model_uri = f"runs:/{run_id}/model"
result = mlflow.register_model(model_uri, "geosurvey-land-stability-score-predictor")

output = {
    "registered_model_name": "geosurvey-land-stability-score-predictor",
    "version": result.version,
    "run_id": run_id,
    "source_metric": "rmse",
    "source_metric_value": best_run.data.metrics["rmse"]
}

with open("results/step3_s6.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 3 DONE")
