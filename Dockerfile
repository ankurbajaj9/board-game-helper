FROM python:3.14-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /usr/local/bin/uv

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/models/huggingface \
    FASTEMBED_CACHE_PATH=/models/fastembed

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev

CMD ["board-game-rules-watch", "watch"]
