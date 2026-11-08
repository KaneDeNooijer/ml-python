"""NOVA group estimation with interchangeable models (Strategy pattern).

Every model learns from the same features: the tokens of a product's name, categories,
ingredients and number of additives. `MODELS` maps a name to a model class, so the
service, the evaluation and the tests can swap models without changing any other code.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from poc_ml.config import REASON_COUNT
from poc_ml.models import Product


def tokens(product: Product) -> list[str]:
    """The features of a product: words of the name, category and ingredient tags, additive count."""
    name_words = [f"name:{word}" for word in product.name.lower().split() if len(word) > 2]
    categories = [f"category:{tag}" for tag in product.categories_tags]
    ingredients = [f"ingredient:{tag}" for tag in product.ingredients_tags]
    additives = [] if product.additives_n is None else [f"additives:{min(product.additives_n, 5)}"]
    return name_words + categories + ingredients + additives


def identity(values: list[str]) -> list[str]:
    """Tokens are made by `tokens`; the vectorizer must not split them again."""
    return values


def vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(analyzer=identity)


@dataclass(frozen=True)
class Estimate:
    group: int
    confidence: float
    reasons: list[str] = field(default_factory=list)


class NovaModel(Protocol):
    """The strategy interface: learn from labelled products, then estimate one product."""

    name: str

    def fit(self, documents: list[list[str]], groups: list[int]) -> None: ...

    def estimate(self, document: list[str]) -> Estimate: ...


class MajorityBaseline:
    """Always the most common group in the training data: the benchmark a real model must beat."""

    name = "majority_baseline"

    def __init__(self) -> None:
        self._group = 4
        self._share = 0.0

    def fit(self, documents: list[list[str]], groups: list[int]) -> None:
        group, count = Counter(groups).most_common(1)[0]
        self._group, self._share = group, count / len(groups)

    def estimate(self, document: list[str]) -> Estimate:
        return Estimate(self._group, round(self._share, 2))


class SklearnModel:
    """Shared fit/estimate for the scikit-learn models: TF-IDF features, then a classifier."""

    name = "sklearn"

    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None
        self._single_group: int | None = None

    def classifier(self) -> object:
        raise NotImplementedError

    def fit(self, documents: list[list[str]], groups: list[int]) -> None:
        if len(set(groups)) < 2:
            self._single_group = groups[0]
            return
        self._pipeline = Pipeline([("features", vectorizer()), ("classifier", self.classifier())])
        self._pipeline.fit(documents, groups)

    def estimate(self, document: list[str]) -> Estimate:
        if self._pipeline is None:
            return Estimate(self._single_group or 4, 1.0)
        probabilities = self._pipeline.predict_proba([document])[0]
        best = int(np.argmax(probabilities))
        group = int(self._pipeline.classes_[best])
        return Estimate(group, round(float(probabilities[best]), 2), self.reasons(document, best))

    def reasons(self, document: list[str], class_index: int) -> list[str]:
        return []


class LogisticRegressionModel(SklearnModel):
    """Linear model: fast, gives probabilities, and its weights explain each estimate."""

    name = "logistic_regression"

    def classifier(self) -> object:
        return LogisticRegression(max_iter=1000)

    def reasons(self, document: list[str], class_index: int) -> list[str]:
        """The tokens of this product that pushed the estimate most towards the chosen group."""
        assert self._pipeline is not None
        features: TfidfVectorizer = self._pipeline.named_steps["features"]
        model: LogisticRegression = self._pipeline.named_steps["classifier"]
        weights = features.transform([document]).toarray()[0]
        coefficients = (
            model.coef_[class_index] if model.coef_.shape[0] > 1 else model.coef_[0] * (1 if class_index else -1)
        )
        contributions = weights * coefficients
        names = features.get_feature_names_out()
        ranked = sorted(
            (index for index in np.nonzero(weights)[0] if contributions[index] > 0),
            key=lambda index: contributions[index],
            reverse=True,
        )
        return [str(names[index]) for index in ranked[:REASON_COUNT]]


class NaiveBayesModel(SklearnModel):
    """Counts how often tokens occur per group: very fast and works with little data."""

    name = "naive_bayes"

    def classifier(self) -> object:
        return MultinomialNB()


class NearestNeighboursModel(SklearnModel):
    """Takes the group of the most similar products (cosine similarity of the tokens)."""

    name = "k_nearest_neighbours"

    def __init__(self) -> None:
        super().__init__()
        self._documents: list[list[str]] = []
        self._groups: list[int] = []

    def classifier(self) -> object:
        return KNeighborsClassifier(n_neighbors=5, metric="cosine", weights="distance")

    def fit(self, documents: list[list[str]], groups: list[int]) -> None:
        self._documents, self._groups = documents, groups
        super().fit(documents, groups)

    def reasons(self, document: list[str], class_index: int) -> list[str]:
        """The tokens this product shares with its nearest neighbours in the chosen group, rarest first."""
        assert self._pipeline is not None
        features: TfidfVectorizer = self._pipeline.named_steps["features"]
        model: KNeighborsClassifier = self._pipeline.named_steps["classifier"]
        group = int(self._pipeline.classes_[class_index])
        neighbours = model.kneighbors(features.transform([document]), return_distance=False)[0]
        shared = {
            token
            for index in neighbours
            if self._groups[index] == group
            for token in self._documents[index]
            if token in document
        }
        weights = dict(zip(features.get_feature_names_out(), features.idf_, strict=True))
        ranked = sorted(shared, key=lambda token: weights.get(token, 0.0), reverse=True)
        return ranked[:REASON_COUNT]


MODELS: dict[str, type[NovaModel]] = {
    model.name: model for model in (MajorityBaseline, NaiveBayesModel, NearestNeighboursModel, LogisticRegressionModel)
}


def create_model(name: str) -> NovaModel:
    """The model with this name (a KeyError for an unknown name is a programming error)."""
    return MODELS[name]()
