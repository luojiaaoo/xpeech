# syntax=docker/dockerfile:1
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.27
ARG PYTHON_IMAGE=python:3.12-slim-bookworm
ARG NODE_IMAGE=node:24.4.1-bookworm-slim
ARG GO_IMAGE=golang:1.23-bookworm

FROM ${UV_IMAGE} AS uv

FROM ${NODE_IMAGE} AS node-base

FROM node-base AS frontend
WORKDIR /frontend
COPY xpeech/channel/web_client/frontend/ ./
RUN npm install --no-audit --no-fund && npm run build

FROM ${GO_IMAGE} AS lark-cli-builder
ENV GOPROXY=https://goproxy.cn,direct UV_HTTP_TIMEOUT=120
COPY --from=uv /uv /uvx /usr/local/bin/
WORKDIR /src/lark-cli
ARG LARK_CLI_REPOSITORY=https://gitee.com/luojiaaoo/lark-cli.git
RUN git clone --filter=blob:none --no-checkout "$LARK_CLI_REPOSITORY" . && git checkout --detach v1.0.89
COPY feishu.lark-cli/main.go ./main.go
COPY feishu.lark-cli/generate.py /build/generate.py
COPY feishu.lark-cli/mycred/mycred.go.tmpl /build/mycred.go.tmpl
COPY feishu.lark-cli/oauth/main.go.tmpl /build/oauth-main.go.tmpl
COPY feishu.lark-cli/oauth/main_test.go ./cmd/xpeech-lark-cli-auth/main_test.go
RUN --mount=type=bind,source=conf.toml,target=/build/conf.toml,readonly \
    uv run --no-project --python 3.12 /build/generate.py \
        --config /build/conf.toml --template /build/mycred.go.tmpl --output ./mycred/mycred.go && \
    uv run --no-project --python 3.12 /build/generate.py \
        --config /build/conf.toml --template /build/oauth-main.go.tmpl --output ./cmd/xpeech-lark-cli-auth/main.go
RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    CGO_ENABLED=0 go test ./cmd/xpeech-lark-cli-auth && \
    CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/lark-cli . && \
    CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/lark-cli-auth ./cmd/xpeech-lark-cli-auth

FROM ${PYTHON_IMAGE} AS python-base
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy UV_HTTP_TIMEOUT=120 UV_PYTHON_DOWNLOADS=never
COPY --from=uv /uv /uvx /usr/local/bin/
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY runtime/pyproject.toml ./runtime/pyproject.toml

# Backend builds do not depend on the agent image, Node CLI or Go tooling.
FROM python-base AS backend
RUN uv sync --frozen --no-dev --package xpeech --no-install-workspace
COPY xpeech/ ./xpeech/
COPY assets/ ./assets/
RUN uv sync --frozen --no-dev --package xpeech --no-editable
COPY --from=frontend /frontend/dist/ ./xpeech/channel/web_client/frontend/dist/
ENTRYPOINT ["/app/.venv/bin/python", "-m", "xpeech"]

FROM python-base AS runtime-deps
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        bash ca-certificates curl ffmpeg git jq zip tzdata libstdc++6 && \
    rm -rf /var/lib/apt/lists/*
COPY --from=node-base /usr/local/bin/ /usr/local/bin/
COPY --from=node-base /usr/local/lib/node_modules/ /usr/local/lib/node_modules/
RUN uv sync --frozen --no-dev --package xpeech-runtime --no-install-workspace
RUN npm install --global agent-browser && /app/.venv/bin/python --version

FROM runtime-deps AS runtime
COPY xpeech/ ./xpeech/
COPY assets/ ./assets/
COPY custom_tools/ ./custom_tools/
COPY runtime/src/ ./runtime/src/
RUN uv sync --frozen --no-dev --package xpeech-runtime --no-editable
COPY --from=lark-cli-builder /out/lark-cli /usr/local/bin/lark-cli
COPY --from=lark-cli-builder /out/lark-cli-auth /usr/local/bin/lark-cli-auth
# Task-local uv must never discover the application's project as an ancestor.
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken-cache LITELLM_LOCAL_MODEL_COST_MAP=true
# Token counting must not download its vocabulary during a sandbox run.
RUN mkdir -p /opt/xpeech-build && mv /app/pyproject.toml /app/uv.lock /opt/xpeech-build/ && \
    /app/.venv/bin/python -c "import tiktoken; tiktoken.get_encoding('o200k_base')"
ENTRYPOINT ["/app/.venv/bin/python", "-m", "xpeech_runtime"]
