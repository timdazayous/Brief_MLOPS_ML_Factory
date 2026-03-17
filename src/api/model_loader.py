import os
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Setup environment variables
# Setup environment variables
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
os.environ["MLFLOW_TRACKING_URI"] = TRACKING_URI
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"

import mlflow
mlflow.set_tracking_uri(TRACKING_URI)

import sys

def debug_log(msg):
    sys.stdout.write(f"DEBUG: {msg}\n")
    sys.stdout.flush()

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
        # MLflow aliases are case-sensitive in some versions/backends
        try:
            model_version_info = client.get_model_version_by_alias(model_name, "Production")
        except:
            model_version_info = client.get_model_version_by_alias(model_name, "production")
            
        current_prod_version = model_version_info.version
    except Exception as e:
        debug_log(f"Error getting version for alias 'Production/production': {e}")
        # If no production model found, return current cache or None
        return model_cache["model"]

    debug_log(f"Registry check - @Production is version {current_prod_version} (Cache: {model_cache['version']})")

    if model_cache["version"] != current_prod_version:
        debug_log(f"Update detected: Loading version {current_prod_version} from registry...")
        try:
            # Try both cases for URI as well
            try:
                model_uri = f"models:/{model_name}@production"
                loaded_model = mlflow.sklearn.load_model(model_uri)
            except:
                model_uri = f"models:/{model_name}@Production"
                loaded_model = mlflow.sklearn.load_model(model_uri)
                
            model_cache["model"] = loaded_model
            model_cache["version"] = current_prod_version
            debug_log(f"Successfully loaded version {current_prod_version}")
        except Exception as e:
            debug_log(f"Error loading model: {e}")
            return None
    
    return model_cache["model"]
