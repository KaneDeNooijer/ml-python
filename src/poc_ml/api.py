from datetime import datetime, timezone
from functools import lru_cache

from fastapi import FastAPI

from poc_ml.classifier import ProductCategoryClassifier
from poc_ml.schemas import (
    AlternativePrediction,
    ModelMetadata,
    PredictionRequest,
    PredictionResponse,
)


app = FastAPI(title="LiveList ML", version="0.1.0")


@lru_cache
def classifier() -> ProductCategoryClassifier:
    return ProductCategoryClassifier()


@app.get("/health")
def health() -> dict[str, str]:
    model = classifier()

    return {"status": "ok", "model": model.name, "version": model.version}


@app.post("/v1/category-predictions", response_model=PredictionResponse)
def predict_category(request: PredictionRequest) -> PredictionResponse:
    model = classifier()
    prediction = model.predict(request.product, request.expected_category)

    return PredictionResponse(
        predicted_category=prediction.category,
        expected_category=request.expected_category,
        matches_expected=prediction.category == request.expected_category,
        confidence=prediction.confidence,
        alternatives=[
            AlternativePrediction(category=category, confidence=confidence)
            for category, confidence in prediction.alternatives
        ],
        model=ModelMetadata(name=model.name, version=model.version),
        used_fields=prediction.used_fields,
        predicted_at=datetime.now(timezone.utc),
    )
