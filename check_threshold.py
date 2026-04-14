import mlflow
import os
import sys

# Explicitly point to the SQLite DB
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

THRESHOLD = 0.85

if not os.path.exists("model_info.txt"):
    print("Error: model_info.txt not found. Did training run?")
    sys.exit(1)

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

try:
    run = client.get_run(run_id)
    accuracy = run.data.metrics.get("accuracy")
    print(f"Run ID: {run_id} | Accuracy: {accuracy}")

    if accuracy is None:
        print("Metric 'accuracy' not found in MLflow.")
        sys.exit(1)

    if accuracy < THRESHOLD:
        print(f"FAILED: Accuracy {accuracy:.4f} is below threshold {THRESHOLD}")
        sys.exit(1)

    print("SUCCESS: Accuracy passed threshold.")
except Exception as e:
    print(f"System Error: {e}")
    sys.exit(1)