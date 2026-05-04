from mlflow.tracking import MlflowClient
import json

client = MlflowClient()
name = "geosurvey-land-stability-score-predictor"

versions = client.search_model_versions(f"name='{name}'")
v1 = int(versions[0].version)

client.set_registered_model_alias(name=name, alias="live", version=v1)

output = {
    "registered_model_name": name,
    "alias_name": "live",
    "champion_version": v1,
    "challenger_version": 2,
    "action": "kept"
}

with open("results/step4_s7.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 4 DONE")