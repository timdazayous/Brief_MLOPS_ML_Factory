from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import sys
import os

# Align path for internal imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.model_loader import load_production_model, model_cache

app = FastAPI(title="ML Serving Factory")

IRIS_CLASSES = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

class IrisInput(BaseModel):
    features: list[float] = Field(..., min_items=4, max_items=4)

@app.get("/health")
def health():
    return {"status": "up", "model_version": model_cache["version"]}

@app.post("/predict")
def predict(data: IrisInput):
    model = load_production_model()
    if not model:
        raise HTTPException(status_code=503, detail="No production model available.")
    
    try:
        prediction = model.predict([data.features])
        # Get probabilities if the model supports it
        try:
            probabilities = model.predict_proba([data.features])[0].tolist()
        except:
            probabilities = None
            
        predicted_class = int(prediction[0])
        
        return {
            "prediction": predicted_class,
            "class_name": IRIS_CLASSES[predicted_class],
            "probabilities": probabilities,
            "model_version": model_cache["version"],
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
