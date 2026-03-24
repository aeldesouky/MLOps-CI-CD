import mlflow
import sys
import os

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(tracking_uri)

THRESHOLD = 0.85

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

client = mlflow.tracking.MlflowClient()
run = client.get_run(run_id)

accuracy = run.data.metrics.get("accuracy")

print(f"Run ID: {run_id}")
print(f"Accuracy: {accuracy}")

if accuracy is None:
    print("No accuracy found. Failing.")
    sys.exit(1)

if accuracy < THRESHOLD:
    print(f"Accuracy below threshold ({THRESHOLD}). Failing pipeline.")
    sys.exit(1)

print("Model passed threshold.")