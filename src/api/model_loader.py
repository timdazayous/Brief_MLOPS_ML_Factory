import os
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Setup environment variables
os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"

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
        # If no production model found, return current cache or None
        return model_cache["model"]

    if model_cache["version"] != current_prod_version:
        print(f"Update detected: Loading version {current_prod_version} from registry...")
        try:
            model_uri = f"models:/{model_name}@Production"
            loaded_model = mlflow.sklearn.load_model(model_uri)
            model_cache["model"] = loaded_model
            model_cache["version"] = current_prod_version
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    return model_cache["model"]
