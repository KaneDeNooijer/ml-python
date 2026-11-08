"""HTTP routes of the proof of concept; Laravel is the only client."""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Path, Query, Request
from fastapi.responses import JSONResponse

from poc_ml.models import EvaluationResponse, ProductResponse
from poc_ml.open_food_facts import OpenFoodFacts, OpenFoodFactsError
from poc_ml.service import ProductService

app = FastAPI(title="LLMaaltijd ML proof of concept", version="0.2.0")


@lru_cache
def get_service() -> ProductService:
    return ProductService(OpenFoodFacts())


ServiceDependency = Annotated[ProductService, Depends(get_service)]


@app.exception_handler(OpenFoodFactsError)
def open_food_facts_unavailable(request: Request, error: OpenFoodFactsError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": str(error)})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/products/{barcode}", response_model=ProductResponse)
def product(barcode: Annotated[str, Path(pattern=r"^\d{8,14}$")], service: ServiceDependency) -> ProductResponse:
    """A product with its NOVA group; estimated by the chosen model when Open Food Facts lacks it."""
    result = service.product(barcode)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Product {barcode} not found")
    return result


@app.get("/evaluation", response_model=EvaluationResponse)
def evaluation(
    service: ServiceDependency,
    category: Annotated[str, Query(pattern=r"^[a-z]{2}:[a-z0-9-]+$")] = "en:yogurts",
) -> EvaluationResponse:
    """Cross-validated accuracy of every model against the majority baseline."""
    return service.evaluate(category)
