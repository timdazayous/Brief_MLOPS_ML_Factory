"""
MLflow Training and Registration Script.
Trains a RandomForestClassifier on the Iris dataset, logs parameters and metrics to MLflow,
and registers the best model with the '@Production' alias.
"""

import os
import mlflow
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlflow.models.signature import infer_signature

from router.ai_router import AIRouter
from router.skill_classifier import TaskType

# Setup MLflow Tracking URI to point to local server
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"

def train_and_register():
    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Set experiment
    mlflow.set_experiment("Iris_Classification")

    # Ask AI Router for advice
    router = AIRouter()
    advice = router.route(
        "What are good hyperparameters for a RandomForestClassifier on the Iris dataset?", 
        task_hint=TaskType.REASONING
    )
    print(f"AI Advice: {advice.response}")

    print("Starting MLflow run...")
    with mlflow.start_run() as run:
        # Define hyperparams (could be influenced by AI advice)
        params = {"n_estimators": 100, "random_state": 42, "max_depth": 5}
        mlflow.log_params(params)

        # Train model
        rf = RandomForestClassifier(**params)
        rf.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)

        print(f"Model trained. Accuracy: {accuracy:.4f}")

        # Infer signature (input/output schema)
        signature = infer_signature(X_train, y_pred)

        # Log model artifact (without registered_model_name to avoid issues)
        mlflow.sklearn.log_model(
            sk_model=rf,
            artifact_path="random_forest_model",
            signature=signature,
        )
        print("Model logged to MLflow.")

        # Explicitly register the model via the client API
        model_uri = f"runs:/{run.info.run_id}/random_forest_model"

    client = mlflow.MlflowClient()
    model_name = "IrisClassifier"

    # Create registered model if it doesnʼt exist
    try:
        client.create_registered_model(model_name)
        print(f"Registered model '{model_name}' created.")
    except mlflow.exceptions.MlflowException:
        print(f"Registered model '{model_name}' already exists.")

    # Create a new version from the run artifact
    mv = client.create_model_version(model_name, model_uri, run.info.run_id)
    print(f"Model version {mv.version} created.")

    # Apply @Production alias
    client.set_registered_model_alias(model_name, "Production", mv.version)
    print(f"Alias '@Production' set on {model_name} version {mv.version}.")

if __name__ == "__main__":
    train_and_register()
