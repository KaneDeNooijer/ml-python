"""What the API does with a product: fetch it and, when its NOVA group is missing, estimate it."""

from sklearn.model_selection import KFold

from poc_ml.config import (
    CATEGORY_LEVELS,
    CHOSEN_MODEL,
    EVALUATION_FOLDS,
    MIN_TRAINING_SAMPLES,
    RANDOM_STATE,
)
from poc_ml.models import EvaluationResponse, ModelScore, Nova, Product, ProductResponse
from poc_ml.nova import MODELS, create_model, tokens
from poc_ml.open_food_facts import ProductSource


class ProductService:
    def __init__(self, source: ProductSource, model_name: str = CHOSEN_MODEL) -> None:
        self._source = source
        self._model_name = model_name

    def product(self, barcode: str) -> ProductResponse | None:
        product = self._source.product(barcode)
        if product is None:
            return None
        return ProductResponse(
            barcode=product.code,
            name=product.name,
            brands=product.brands,
            categories=product.categories_tags,
            nova=self._nova(product),
        )

    def evaluate(self, category: str) -> EvaluationResponse:
        """Cross-validated accuracy of every model on the labelled products of one category."""
        labelled = [product for product in self._source.category(category) if product.nova_group is not None]
        if len(labelled) < MIN_TRAINING_SAMPLES:
            return EvaluationResponse(
                category=category,
                samples=len(labelled),
                folds=EVALUATION_FOLDS,
                scores=[],
                chosen_model=self._model_name,
                status="too_few_samples",
            )

        documents = [tokens(product) for product in labelled]
        groups = [product.nova_group or 0 for product in labelled]
        folds = KFold(n_splits=EVALUATION_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scores = []
        for name in MODELS:
            correct = 0
            for train, test in folds.split(documents):
                model = create_model(name)
                model.fit([documents[i] for i in train], [groups[i] for i in train])
                correct += sum(model.estimate(documents[i]).group == groups[i] for i in test)
            scores.append(ModelScore(model=name, accuracy=round(correct / len(documents), 3)))

        return EvaluationResponse(
            category=category,
            samples=len(labelled),
            folds=EVALUATION_FOLDS,
            scores=sorted(scores, key=lambda score: score.accuracy, reverse=True),
            chosen_model=self._model_name,
            status="ok",
        )

    def _nova(self, product: Product) -> Nova:
        if product.nova_group is not None:
            return Nova(group=product.nova_group, estimated=False)

        category, training = self._training_set(product)
        if not training:
            return Nova(group=None, estimated=False)

        model = create_model(self._model_name)
        model.fit([tokens(item) for item in training], [item.nova_group or 0 for item in training])
        estimate = model.estimate(tokens(product))
        return Nova(
            group=estimate.group,
            estimated=True,
            model=model.name,
            confidence=estimate.confidence,
            reasons=estimate.reasons,
            trained_on=len(training),
            category=category,
        )

    def _training_set(self, product: Product) -> tuple[str | None, list[Product]]:
        """Labelled products from the most specific category that has enough of them."""
        for category in list(reversed(product.categories_tags))[:CATEGORY_LEVELS]:
            labelled = [
                item
                for item in self._source.category(category)
                if item.nova_group is not None and item.code != product.code
            ]
            if len(labelled) >= MIN_TRAINING_SAMPLES:
                return category, labelled
        return None, []
