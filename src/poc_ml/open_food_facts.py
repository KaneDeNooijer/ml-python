"""Gateway to Open Food Facts (Adapter): the rest of the code only sees the ProductSource protocol."""

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Protocol

from poc_ml.config import (
    OFF_CACHE_SECONDS,
    OFF_FIELDS,
    OFF_LANGUAGE,
    OFF_PAGE_SIZE,
    OFF_PRODUCT_URL,
    OFF_SEARCH_URL,
    OFF_TIMEOUT_SECONDS,
    USER_AGENT,
)
from poc_ml.models import Product


class OpenFoodFactsError(Exception):
    """Open Food Facts could not be reached or answered with an error."""


class ProductSource(Protocol):
    def product(self, barcode: str) -> Product | None: ...

    def category(self, category: str) -> list[Product]: ...


class OpenFoodFacts:
    """Product API for one product, Search-a-licious for a category of Dutch products (cached)."""

    def __init__(self) -> None:
        self._cache: dict[str, tuple[float, Any]] = {}

    def product(self, barcode: str) -> Product | None:
        body = self._get(OFF_PRODUCT_URL.format(barcode=barcode), {"fields": OFF_FIELDS})
        if body is None or body.get("status") != 1 or not isinstance(body.get("product"), dict):
            return None
        product = body["product"]
        return Product.model_validate({**product, "code": str(body.get("code") or barcode)})

    def category(self, category: str) -> list[Product]:
        query = f'categories_tags:"{category}" lang:{OFF_LANGUAGE}'
        body = self._get(OFF_SEARCH_URL, {"q": query, "page_size": OFF_PAGE_SIZE, "fields": OFF_FIELDS})
        hits = body.get("hits", []) if body else []
        return [Product.model_validate(hit) for hit in hits if isinstance(hit, dict) and hit.get("code")]

    def _get(self, url: str, params: dict[str, Any]) -> dict[str, Any] | None:
        address = f"{url}?{urllib.parse.urlencode(params)}"
        cached = self._cache.get(address)
        if cached is not None and time.monotonic() - cached[0] < OFF_CACHE_SECONDS:
            return cached[1]  # type: ignore[no-any-return]

        request = urllib.request.Request(address, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=OFF_TIMEOUT_SECONDS) as response:
                body = json.loads(response.read())
        except urllib.error.HTTPError as error:
            if error.code == 404:
                body = None
            else:
                raise OpenFoodFactsError(f"Open Food Facts answered with status {error.code}") from error
        except (urllib.error.URLError, TimeoutError, ValueError) as error:
            raise OpenFoodFactsError(f"Could not reach Open Food Facts: {error}") from error

        self._cache[address] = (time.monotonic(), body)
        return body if isinstance(body, dict) else None
