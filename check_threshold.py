import mlflow
import os
import sys

# Explicitly set tracking URI from environment for the client
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(tracking_uri)

THRESHOLD = 0.85

if not os.path.exists("model_info.txt"):
    print("model_info.txt not found!")
    sys.exit(1)

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

client = mlflow.tracking.MlflowClient()

try:
    run = client.get_run(run_id)
    accuracy = run.data.metrics.get("accuracy")
    print(f"Run ID: {run_id} | Accuracy: {accuracy}")
    
    if accuracy is None:
        print("No accuracy metric found.")
        sys.exit(1)
    if accuracy < THRESHOLD:
        print(f"Accuracy {accuracy} below threshold {THRESHOLD}.")
        sys.exit(1)
    print("Threshold passed!")
except Exception as e:
    print(f"Failed to retrieve run {run_id}: {e}")
    # Debug: List the directories to see what MLflow sees
    print("Directory structure of mlruns:")
    os.system("ls -R mlruns")
    sys.exit(1)