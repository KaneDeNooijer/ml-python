from datetime import datetime

from pydantic import BaseModel, Field


class ProductInput(BaseModel):
    barcode: str
    product_name: str = ""
    brands: str = ""
    categories: list[str] = Field(default_factory=list)
    ingredients_text: str = ""


class PredictionRequest(BaseModel):
    product: ProductInput
    expected_category: str


class AlternativePrediction(BaseModel):
    category: str
    confidence: float


class ModelMetadata(BaseModel):
    name: str
    version: str


class PredictionResponse(BaseModel):
    predicted_category: str
    expected_category: str
    matches_expected: bool
    confidence: float
    alternatives: list[AlternativePrediction]
    model: ModelMetadata
    used_fields: list[str]
    estimated: bool = True
    predicted_at: datetime
