from fastapi.testclient import TestClient

from poc_ml.api import app


client = TestClient(app)


def test_health_exposes_loaded_model() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model"] == "open-food-facts-taxonomy-tfidf"


def test_predicts_a_yoghurt_product() -> None:
    response = client.post(
        "/v1/category-predictions",
        json={
            "expected_category": "en:yogurts",
            "product": {
                "barcode": "8710000000001",
                "product_name": "Griekse yoghurt naturel",
                "brands": "Demo Dairy",
                "categories": ["en:dairies", "en:yogurts"],
                "ingredients_text": "melk, yoghurtculturen",
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_category"] == "en:yogurts"
    assert payload["matches_expected"] is True
    assert payload["estimated"] is True
    assert payload["model"]["version"]


def test_matches_an_arbitrary_open_food_facts_category() -> None:
    response = client.post(
        "/v1/category-predictions",
        json={
            "expected_category": "en:rices",
            "product": {
                "barcode": "8710000000002",
                "product_name": "Basmati rice",
                "brands": "Demo Foods",
                "categories": ["en:plant-based-foods", "en:cereals", "en:rices"],
                "ingredients_text": "basmati rice",
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_category"] == "en:rices"
    assert payload["matches_expected"] is True
    assert payload["confidence"] == 0.99


def test_matches_a_custom_ingredient_against_open_food_facts_product_text() -> None:
    response = client.post(
        "/v1/category-predictions",
        json={
            "expected_category": "custom:taco-shells",
            "product": {
                "barcode": "8710000000003",
                "product_name": "Crunchy taco shells",
                "brands": "Demo Foods",
                "categories": ["en:corn-products", "en:meal-kits"],
                "ingredients_text": "corn flour, sunflower oil, salt",
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_category"] == "custom:taco-shells"
    assert payload["matches_expected"] is True
