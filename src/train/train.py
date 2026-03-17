"""
ML Laboratory: Training Script.
Supports LogisticRegression and RandomForestClassifier.
Can automatically move the '@Production' alias.
"""

import os
import argparse
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlflow.models.signature import infer_signature
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Setup MLflow Tracking URI to point to local server
os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"

def train(model_type="logistic", auto_publish=False):
    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_experiment("Iris_Lab")

    print(f"Starting MLflow run for {model_type}...")
    with mlflow.start_run() as run:
        # Choose model
        if model_type == "logistic":
            model = LogisticRegression(max_iter=200)
            params = {"model_type": "LogisticRegression"}
        else:
            model = RandomForestClassifier(n_estimators=100)
            params = {"model_type": "RandomForestClassifier"}
        
        mlflow.log_params(params)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        print(f"Model trained. Accuracy: {accuracy:.4f}")

        # Signature
        signature = infer_signature(X_train, y_pred)

        # Log model
        model_name = "IrisClassifier"
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
        )
        print("Model logged to MLflow.")

        # Registration
        model_uri = f"runs:/{run.info.run_id}/model"
        client = mlflow.MlflowClient()
        
        # Ensure model exists in registry
        try:
            client.create_registered_model(model_name)
        except mlflow.exceptions.MlflowException:
            pass

        # Create version
        mv = client.create_model_version(model_name, model_uri, run.info.run_id)
        version = mv.version
        print(f"Model version {version} created.")

        # Automation: set alias if requested
        if auto_publish:
            client.set_registered_model_alias(model_name, "Production", version)
            print(f"Alias '@Production' automatically moved to version {version}.")
        else:
            print(f"Note: Alias '@Production' NOT moved. Target version: {version}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="logistic", choices=["logistic", "forest"])
    parser.add_argument("--auto-publish", action="store_true")
    args = parser.parse_args()

    train(model_type=args.model, auto_publish=args.auto_publish)
