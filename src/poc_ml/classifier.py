import hashlib
import re
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from poc_ml.schemas import ProductInput


MODEL_SIGNATURE = b"open-food-facts-taxonomy-tfidf-v1"


@dataclass(frozen=True)
class Prediction:
    category: str
    confidence: float
    alternatives: list[tuple[str, float]]
    used_fields: list[str]


def normalize_category(category: str) -> str:
    without_language = re.sub(r"^[a-z]+:", "", category.lower())

    return without_language.replace("-", " ").replace("_", " ").strip()


def product_text(product: ProductInput) -> tuple[str, list[str]]:
    values = {
        "product_name": product.product_name,
        "brands": product.brands,
        "categories": " ".join(normalize_category(category) for category in product.categories),
        "ingredients_text": product.ingredients_text,
    }
    used_fields = [key for key, value in values.items() if value.strip()]
    text = " ".join(value for value in values.values() if value.strip())

    return text.lower(), used_fields


class ProductCategoryClassifier:
    name = "open-food-facts-taxonomy-tfidf"
    version = hashlib.sha256(MODEL_SIGNATURE).hexdigest()[:12]

    def predict(self, product: ProductInput, expected_category: str) -> Prediction:
        text, used_fields = product_text(product)
        categories = list(dict.fromkeys(product.categories))
        expected_phrase = normalize_category(expected_category)
        custom_phrase_match = (
            expected_category.startswith("custom:")
            and expected_phrase in f"{product.product_name} {product.ingredients_text}".lower()
        )

        similarities = self._similarities(expected_category, text, categories)
        text_similarity = similarities[0]
        ranked_categories = sorted(
            zip(categories, similarities[1:], strict=True),
            key=lambda item: item[1],
            reverse=True,
        )

        if expected_category in categories:
            predicted_category = expected_category
            confidence = 0.99
        elif custom_phrase_match:
            predicted_category = expected_category
            confidence = 0.92
        elif ranked_categories:
            closest_category, category_similarity = ranked_categories[0]
            semantic_similarity = max(text_similarity, category_similarity)
            predicted_category = (
                expected_category if semantic_similarity >= 0.62 else closest_category
            )
            confidence = (
                min(0.94, 0.55 + (semantic_similarity * 0.4))
                if predicted_category == expected_category
                else min(0.9, max(0.05, category_similarity))
            )
        else:
            predicted_category = expected_category if text_similarity >= 0.62 else "unknown"
            confidence = (
                min(0.9, 0.5 + (text_similarity * 0.4))
                if predicted_category == expected_category
                else max(0.0, text_similarity)
            )

        alternatives = [
            (category, round(float(score), 4))
            for category, score in ranked_categories
            if category != predicted_category
        ][:3]

        return Prediction(
            category=predicted_category,
            confidence=round(float(confidence), 4),
            alternatives=alternatives,
            used_fields=used_fields,
        )

    def _similarities(
        self,
        expected_category: str,
        text: str,
        categories: list[str],
    ) -> list[float]:
        documents = [
            normalize_category(expected_category) or "unknown category",
            text or "unknown product",
            *(normalize_category(category) for category in categories),
        ]
        vectors = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            strip_accents="unicode",
            sublinear_tf=True,
        ).fit_transform(documents)

        return cosine_similarity(vectors[0:1], vectors[1:]).flatten().tolist()
