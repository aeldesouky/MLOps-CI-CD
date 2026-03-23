FROM python:3.10-slim

WORKDIR /app

ARG RUN_ID

COPY . .

CMD echo "Downloading model for Run ID: $RUN_ID"