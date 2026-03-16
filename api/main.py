"""
FastAPI application for ML predictions.
Uses load_production_model to serve predictions dynamically.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import sys
import os
from typing import Annotated

# Ensure the root directory is in the path to import router components later
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.model_loader import load_production_model
from router.ai_router import AIRouter
from router.skill_classifier import TaskType

app = FastAPI(
    title="ML Factory API",
    description="""
API d'inférence ML avec routeur IA intelligent.

## Fonctionnement

1. **Chargement automatique** : Au démarrage, le modèle taggé `@Production` dans le Model Registry MLflow est chargé en mémoire.
2. **Inférence dynamique** : Si l'alias `@Production` change (nouveau modèle entraîné), le modèle est rechargé automatiquement.
3. **Explication IA** : Chaque prédiction est accompagnée d'une explication générée par le routeur IA (par défaut simulé, branchable sur Claude/GPT/Ollama).

## Services requis

- **MLflow** : `http://localhost:5000`
- **MinIO** : `http://localhost:9000`
""",
    version="1.0.0",
    contact={"name": "ML Factory", "url": "http://localhost:5000"},
    license_info={"name": "MIT"},
)

ai_router = AIRouter(budget_usd=5.0)

IRIS_CLASSES = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}


class IrisFeatures(BaseModel):
    features: Annotated[
        list[float],
        Field(
            description="Liste de 4 features numériques dans l'ordre : sepal_length, sepal_width, petal_length, petal_width (en cm)",
            min_length=4,
            max_length=4,
            examples=[[5.1, 3.5, 1.4, 0.2], [6.3, 3.3, 4.7, 1.6]],
        ),
    ]


class PredictionResponse(BaseModel):
    prediction: int = Field(description="Classe prédite (0=Setosa, 1=Versicolor, 2=Virginica)")
    class_name: str = Field(description="Nom de l'espèce prédite")
    explanation: str = Field(description="Explication générée par le routeur IA")
    router_cost_usd: float = Field(description="Coût estimé de l'appel au routeur IA (en USD)")
    model_version: str | None = Field(description="Version du modèle en cache")
    message: str = Field(description="Statut de la prédiction")


@app.on_event("startup")
def startup_event():
    print("Initializing model on startup...")
    load_production_model()


@app.get(
    "/",
    summary="Page d'accueil",
    description="Retourne la liste des endpoints disponibles.",
    tags=["Infos"],
)
def root():
    return {
        "service": "ML Factory API",
        "status": "running",
        "endpoints": {
            "POST /predict": "Prédiction Iris (4 features)",
            "GET /health": "Vérification de santé",
            "GET /docs": "Documentation Swagger interactive",
        },
    }


@app.get(
    "/health",
    summary="Vérification de santé",
    description="Vérifie si le modèle de production est chargé et disponible.",
    tags=["Infos"],
)
def health():
    from api.model_loader import model_cache
    model = load_production_model()
    return {
        "status": "healthy" if model else "degraded",
        "model_loaded": model is not None,
        "cached_version": model_cache.get("version"),
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Prédire l'espèce d'Iris",
    description="""
Prédit l'espèce d'une fleur d'Iris à partir de ses 4 mensurations.

**Ordre des features :**
1. `sepal_length` — Longueur du sépale (cm)
2. `sepal_width` — Largeur du sépale (cm)
3. `petal_length` — Longueur du pétale (cm)
4. `petal_width` — Largeur du pétale (cm)

**Classes possibles :**
- `0` → Iris Setosa
- `1` → Iris Versicolor
- `2` → Iris Virginica

Le résultat inclut une **explication en langage naturel** générée par le Routeur IA (actuellement simulé).
Le modèle est rechargé automatiquement si l'alias `@Production` change dans MLflow.
""",
    tags=["Prédiction"],
)
def predict(request: IrisFeatures):
    if len(request.features) != 4:
        raise HTTPException(status_code=400, detail="Iris dataset requires exactly 4 features.")

    model = load_production_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized or unavailable.")

    try:
        from api.model_loader import model_cache
        prediction = model.predict([request.features])
        predicted_class = int(prediction[0])

        # Ask AI router for an explanation
        prompt = (
            f"The model predicted class {predicted_class} ({IRIS_CLASSES[predicted_class]}) "
            f"for features {request.features} in the classical Iris dataset. "
            "Provide a short explanation for a non-expert."
        )
        explanation = ai_router.route(prompt, task_hint=TaskType.VALIDATION)

        return PredictionResponse(
            prediction=predicted_class,
            class_name=IRIS_CLASSES[predicted_class],
            explanation=explanation.response,
            router_cost_usd=explanation.cost,
            model_version=model_cache.get("version"),
            message="Prediction successful.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
