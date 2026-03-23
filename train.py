with mlflow.start_run() as run:
    run_id = run.info.run_id

    # your training code here

    with open("model_info.txt", "w") as f:
        f.write(run_id)