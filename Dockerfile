# syntax=docker/dockerfile:1.7
# Local development image of the ML proof of concept (not for production).

FROM ghcr.io/astral-sh/uv:0.12.16 AS uv

FROM python:3.13-slim-bookworm AS dev
WORKDIR /app
COPY --from=uv /uv /usr/local/bin/uv
# The virtual environment lives outside /app, so the mounted checkout (and your own .venv) never clashes with it.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/opt/venv/bin:$PATH"
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --group dev --no-install-project
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --group dev
EXPOSE 8100
CMD ["uvicorn", "poc_ml.api:app", "--host", "0.0.0.0", "--port", "8100", "--reload"]
