import mlflow
import sys

THRESHOLD = 0.85

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

client = mlflow.tracking.MlflowClient()

run = client.get_run(run_id)

accuracy = run.data.metrics.get("accuracy", 0)

print(f"Accuracy: {accuracy}")

if accuracy < THRESHOLD:
    print("Model failed threshold check")
    sys.exit(1)
else:
    print("Model passed threshold check")