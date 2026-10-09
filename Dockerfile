FROM ghcr.io/astral-sh/uv:0.9.30-python3.12-bookworm-slim@sha256:e5b65587bce7de595f299855d7385fe7fca39b8a74baa261ba1b7147afa78e58 AS builder

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY pyproject.toml uv.lock README.md ./
COPY LICENSE ./LICENSE
RUN uv sync --locked --no-dev --no-install-project

COPY src/ src/
COPY config/ config/
RUN uv sync --locked --no-dev --no-editable

FROM python:3.14-slim-bookworm@sha256:48b13b003dda20b16f9442b8475aa05fe21bf6579a8c881db92ffb4d8fd20f83 AS runtime

RUN apt-get update \
    && apt-get install --no-install-recommends -y libexpat1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 app
WORKDIR /app
ENV PATH="/app/.venv/bin:${PATH}" \
    UV_CACHE_DIR=/tmp/uv

COPY --from=builder --chown=app:app /app/.venv /app/.venv
VOLUME ["/work"]
USER app

ENTRYPOINT ["osm-polygon-eunis"]
CMD ["--help"]
