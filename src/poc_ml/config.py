"""All constants of the proof of concept in one place."""

from typing import Final

OFF_PRODUCT_URL: Final = "https://world.openfoodfacts.org/api/v2/product/{barcode}"
OFF_SEARCH_URL: Final = "https://search.openfoodfacts.org/search"
OFF_FIELDS: Final = "code,product_name,product_name_nl,brands,categories_tags,ingredients_tags,additives_n,nova_group"
OFF_LANGUAGE: Final = "nl"
OFF_PAGE_SIZE: Final = 100
OFF_TIMEOUT_SECONDS: Final = 20
OFF_CACHE_SECONDS: Final = 3600
USER_AGENT: Final = "LLMaaltijd-PoC/0.1 (AdA Informatica student project)"

NOVA_GROUPS: Final = (1, 2, 3, 4)
# The model learns from Dutch products in the same category; it needs this many with a known NOVA group.
MIN_TRAINING_SAMPLES: Final = 20
# How many category levels to try, from the most specific category to broader ones.
CATEGORY_LEVELS: Final = 3
# How many tokens are shown as the reason for an estimate.
REASON_COUNT: Final = 3

# The model the service uses; chosen with the evaluation in docs/analyse.md.
CHOSEN_MODEL: Final = "k_nearest_neighbours"
EVALUATION_FOLDS: Final = 5
RANDOM_STATE: Final = 42
