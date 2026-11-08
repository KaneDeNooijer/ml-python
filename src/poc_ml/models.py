"""Typed products and API responses."""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from poc_ml.config import NOVA_GROUPS


class Product(BaseModel):
    """The Open Food Facts fields the NOVA estimate uses."""

    code: str
    product_name: str = ""
    product_name_nl: str = ""
    brands: str = ""
    categories_tags: list[str] = Field(default_factory=list)
    ingredients_tags: list[str] = Field(default_factory=list)
    additives_n: int | None = None
    nova_group: int | None = None

    @field_validator("brands", mode="before")
    @classmethod
    def brands_as_text(cls, value: Any) -> str:
        """The search index sends a list, the product API a comma-separated string."""
        if isinstance(value, list):
            return ", ".join(str(brand) for brand in value)
        return str(value or "")

    @field_validator("nova_group", mode="before")
    @classmethod
    def known_nova_group(cls, value: Any) -> int | None:
        """Anything outside groups 1 to 4 counts as unknown."""
        try:
            group = int(value)
        except (TypeError, ValueError):
            return None
        return group if group in NOVA_GROUPS else None

    @property
    def name(self) -> str:
        return (self.product_name_nl or self.product_name).strip()


class Nova(BaseModel):
    """The NOVA group of a product, measured or estimated."""

    group: int | None
    estimated: bool
    model: str | None = None
    confidence: float | None = None
    reasons: list[str] = Field(default_factory=list)
    trained_on: int = 0
    category: str | None = None


class ProductResponse(BaseModel):
    barcode: str
    name: str
    brands: str
    categories: list[str]
    nova: Nova


class ModelScore(BaseModel):
    model: str
    accuracy: float


class EvaluationResponse(BaseModel):
    category: str
    samples: int
    folds: int
    scores: list[ModelScore]
    chosen_model: str
    status: Literal["ok", "too_few_samples"]
