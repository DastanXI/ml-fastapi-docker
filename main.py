"""FastAPI application that serves predictions from the trained model."""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("ml-api")

MODEL_PATH = Path(__file__).parent / "model.joblib"

# The loaded model lives here after startup
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once when the server starts (not on every request)."""
    if not MODEL_PATH.exists():
        raise RuntimeError(f"{MODEL_PATH.name} not found. Run 'python train.py' first.")
    artifact = joblib.load(MODEL_PATH)
    if artifact.get("sklearn_version") != sklearn.__version__:
        logger.warning(
            "Model was trained with scikit-learn %s but %s is installed. "
            "Re-run train.py if you see errors.",
            artifact.get("sklearn_version"),
            sklearn.__version__,
        )
    state.update(artifact)
    yield
    state.clear()


app = FastAPI(
    title="Iris ML API",
    description="Serves a scikit-learn classifier that predicts the Iris flower species.",
    version="1.0.0",
    lifespan=lifespan,
)


class IrisFeatures(BaseModel):
    """Input features (all measurements in centimetres)."""

    sepal_length: float = Field(..., gt=0, le=20, description="Sepal length in cm")
    sepal_width: float = Field(..., gt=0, le=20, description="Sepal width in cm")
    petal_length: float = Field(..., gt=0, le=20, description="Petal length in cm")
    petal_width: float = Field(..., gt=0, le=20, description="Petal width in cm")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
            ]
        }
    }


class PredictionResponse(BaseModel):
    prediction: str
    class_id: int
    probabilities: dict[str, float]


@app.get("/")
def root():
    """Confirms that the API is running."""
    return {"message": "ML API is running"}


@app.get("/health")
def health():
    """Extra endpoint: used by Docker/monitoring to check the service."""
    return {"status": "ok", "model_loaded": "model" in state}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: IrisFeatures):
    """Accept input features, run the trained model, and return the prediction."""
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    # Keep the order of the features exactly as in training
    row = np.array([[getattr(features, name) for name in state["feature_names"]]])

    model = state["model"]
    class_id = int(model.predict(row)[0])
    probabilities = model.predict_proba(row)[0]

    return PredictionResponse(
        prediction=state["target_names"][class_id],
        class_id=class_id,
        probabilities={
            name: round(float(p), 4) for name, p in zip(state["target_names"], probabilities)
        },
    )
