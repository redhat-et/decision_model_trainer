FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    HOME=/tmp

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        python3 python3-venv python3-dev ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.1 /uv /uvx /usr/local/bin/

# Install project dependencies first (huge and slow, doesn't change often)
WORKDIR /app
COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-cache --no-install-project --extra cuda --extra pipeline --no-dev \
    && mkdir -p /workspace /opt/venv \
    && chgrp -R 0 /app /opt/venv /workspace \
    && chmod -R g=u /app /opt/venv /workspace

# Install project after (changes often, installs fast)
COPY README.md ./
COPY src ./src
RUN uv sync --frozen --no-cache --extra cuda --extra pipeline --no-dev

ENV PATH="/opt/venv/bin:${PATH}" \
    HF_HOME=/workspace/cache/huggingface \
    CC=/usr/bin/gcc

USER 1001
ENTRYPOINT ["decision-model-trainer"]
