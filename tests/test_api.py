"""API tests against a fake Open Food Facts, so they never use the network."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from poc_ml.api import app, get_service
from poc_ml.config import CHOSEN_MODEL
from poc_ml.models import Product
from poc_ml.nova import MODELS, create_model, tokens
from poc_ml.open_food_facts import OpenFoodFactsError
from poc_ml.service import ProductService


class FakeOpenFoodFacts:
    def __init__(self) -> None:
        self.products: dict[str, Product] = {}
        self.categories: dict[str, list[Product]] = {}
        self.fail = False

    def product(self, barcode: str) -> Product | None:
        if self.fail:
            raise OpenFoodFactsError("Could not reach Open Food Facts: down")
        return self.products.get(barcode)

    def category(self, category: str) -> list[Product]:
        return self.categories.get(category, [])


def yoghurts() -> list[Product]:
    """Plain yoghurts are NOVA 1, flavoured ones with additives NOVA 4: a pattern a model can learn."""
    plain = [
        Product(
            code=f"1000{i:04d}",
            product_name=f"Yoghurt naturel {i}",
            categories_tags=["en:dairies", "en:yogurts"],
            ingredients_tags=["en:milk"],
            additives_n=0,
            nova_group=1,
        )
        for i in range(15)
    ]
    flavoured = [
        Product(
            code=f"2000{i:04d}",
            product_name=f"Aardbeienyoghurt {i}",
            categories_tags=["en:dairies", "en:yogurts"],
            ingredients_tags=["en:milk", "en:sugar", "en:flavouring"],
            additives_n=3,
            nova_group=4,
        )
        for i in range(15)
    ]
    return plain + flavoured


@pytest.fixture
def source() -> FakeOpenFoodFacts:
    return FakeOpenFoodFacts()


@pytest.fixture
def client(source: FakeOpenFoodFacts) -> Iterator[TestClient]:
    app.dependency_overrides[get_service] = lambda: ProductService(source)
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_known_nova_group_is_returned_as_measured(client: TestClient, source: FakeOpenFoodFacts) -> None:
    source.products["87100001"] = Product(code="87100001", product_name="Melk", nova_group=1)

    nova = client.get("/products/87100001").json()["nova"]

    assert nova == {
        "group": 1,
        "estimated": False,
        "model": None,
        "confidence": None,
        "reasons": [],
        "trained_on": 0,
        "category": None,
    }


def test_missing_nova_group_is_estimated_from_the_same_category(client: TestClient, source: FakeOpenFoodFacts) -> None:
    source.categories["en:yogurts"] = yoghurts()
    source.products["87100002"] = Product(
        code="87100002",
        product_name="Bosvruchtenyoghurt",
        categories_tags=["en:dairies", "en:yogurts"],
        ingredients_tags=["en:milk", "en:sugar", "en:flavouring"],
        additives_n=3,
    )

    nova = client.get("/products/87100002").json()["nova"]

    assert nova["group"] == 4
    assert nova["estimated"] is True
    assert nova["model"] == CHOSEN_MODEL
    assert nova["category"] == "en:yogurts"
    assert nova["trained_on"] == 30
    assert 0.5 < nova["confidence"] <= 1
    assert nova["reasons"]


def test_nova_stays_unknown_without_enough_comparable_products(client: TestClient, source: FakeOpenFoodFacts) -> None:
    source.categories["en:yogurts"] = yoghurts()[:5]
    source.products["87100003"] = Product(code="87100003", product_name="Yoghurt", categories_tags=["en:yogurts"])

    nova = client.get("/products/87100003").json()["nova"]

    assert nova["group"] is None
    assert nova["estimated"] is False


def test_unknown_product_is_404_and_bad_barcode_is_422(client: TestClient) -> None:
    assert client.get("/products/87100009").status_code == 404
    assert client.get("/products/abc").status_code == 422


def test_open_food_facts_failure_is_502(client: TestClient, source: FakeOpenFoodFacts) -> None:
    source.fail = True

    assert client.get("/products/87100001").status_code == 502


def test_evaluation_scores_every_model_against_the_baseline(client: TestClient, source: FakeOpenFoodFacts) -> None:
    source.categories["en:yogurts"] = yoghurts()

    body = client.get("/evaluation", params={"category": "en:yogurts"}).json()

    assert body["status"] == "ok"
    assert {score["model"] for score in body["scores"]} == set(MODELS)
    accuracy = {score["model"]: score["accuracy"] for score in body["scores"]}
    assert accuracy[CHOSEN_MODEL] > accuracy["majority_baseline"]


def test_evaluation_needs_enough_labelled_products(client: TestClient, source: FakeOpenFoodFacts) -> None:
    source.categories["en:yogurts"] = yoghurts()[:5]

    assert client.get("/evaluation", params={"category": "en:yogurts"}).json()["status"] == "too_few_samples"


@pytest.mark.parametrize("name", list(MODELS))
def test_every_model_can_replace_another(name: str) -> None:
    products = yoghurts()
    model = create_model(name)

    model.fit([tokens(product) for product in products], [product.nova_group or 0 for product in products])
    estimate = model.estimate(tokens(products[0]))

    assert estimate.group in (1, 4)
    assert 0 <= estimate.confidence <= 1
