"""
MLflow Model Loader with API Caching.
Dynamically loads the '@Production' model from MLflow and maps it in a global cache,
avoiding unnecessary re-downloads if the version hasn't changed.
"""

import os
import mlflow
from mlflow.tracking import MlflowClient

# Setup environment variables inside the module or rely on external setup
os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
os.environ["MLFLOW_S3_IGNORE_TLS"] = os.getenv("MLFLOW_S3_IGNORE_TLS", "true")

# Global Cache
model_cache = {
    "model": None,
    "version": None
}

def load_production_model(model_name="IrisClassifier"):
    """
    Checks the alias '@Production' in MLflow and loads the model if the version changed.
    """
    client = MlflowClient()
    try:
        model_version_info = client.get_model_version_by_alias(model_name, "Production")
        current_prod_version = model_version_info.version
    except Exception as e:
        print(f"Could not retrieve Production alias for {model_name}: {e}")
        return None

    if model_cache["version"] != current_prod_version:
        print(f"Loading model {model_name} version {current_prod_version} from MLflow...")
        try:
            model_uri = f"models:/{model_name}@Production"
            loaded_model = mlflow.sklearn.load_model(model_uri)
            model_cache["model"] = loaded_model
            model_cache["version"] = current_prod_version
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    else:
        print(f"Using cached model {model_name} version {model_cache['version']}")

    return model_cache["model"]

if __name__ == "__main__":
    # Example usage
    model = load_production_model()
    if model:
        print("Model loaded successfully.")
        print(f"Version: {model_cache['version']}")
