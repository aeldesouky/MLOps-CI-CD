# MLOps CI/CD Project

This repository demonstrates a simple machine learning workflow integrated with continuous integration using GitHub Actions.

The project trains a small convolutional neural network on the MNIST dataset using PyTorch and tracks experiments using MLflow.

## Components

Training script:
train_mlflow_pytorch.py

Dependency specification:
requirements.txt

CI pipeline:
.github/workflows/ml-pipeline.yml

## CI Pipeline

The GitHub Actions workflow performs the following tasks:

1. Sets up a Python 3.10 environment
2. Installs project dependencies
3. Runs a linter check
4. Performs a simple environment validation test using PyTorch
5. Uploads the README.md file as an artifact

## MLflow Tracking

Experiments are tracked using MLflow. The training script logs:

Parameters
learning_rate
batch_size
epochs

Metrics
loss
accuracy

Model artifact
final trained model

## Repository

https://github.com/aeldesouky/MLOps-CI-CD# Minor update
