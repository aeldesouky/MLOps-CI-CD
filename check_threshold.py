import mlflow
import os
import sys

# Ensure the client uses the SQLite database downloaded from the artifact
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

THRESHOLD = 0.85

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

try:
    # This will now look inside mlflow.db instead of scanning folders
    run = client.get_run(run_id)
    accuracy = run.data.metrics.get("accuracy")
    print(f"Run ID: {run_id} | Accuracy: {accuracy}")

    if accuracy is None or accuracy < THRESHOLD:
        print(f"Accuracy check failed.")
        sys.exit(1)
    print("Model passed threshold.")
except Exception as e:
    print(f"Error accessing MLflow data: {e}")
    sys.exit(1)