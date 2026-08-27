FROM docker.1ms.run/library/golang:1.23-bookworm AS lark-cli-builder

ENV GOPROXY=https://goproxy.cn,direct \
    UV_PYTHON=3.12 \
    UV_HTTP_TIMEOUT=200 \
    UV_PYTHON_INSTALL_MIRROR=https://registry.npmmirror.com/-/binary/python-build-standalone/ \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_INSTALL_DIR=/usr/local/bin \
    UV_NO_MODIFY_PATH=1

RUN curl -fsSL https://gitee.com/luojiaaoo/uv/releases/download/0.12.5/uv-installer.sh | sh && \
    uv --version && \
    uv python install

WORKDIR /src/lark-cli

RUN git clone --filter=blob:none --no-checkout https://gitee.com/luojiaaoo/lark-cli.git . && \
    git checkout --detach v1.0.89
COPY feishu.lark-cli/main.go ./main.go
COPY feishu.lark-cli/generate.py /build/generate.py
COPY feishu.lark-cli/mycred/mycred.go.tmpl /build/mycred.go.tmpl
COPY feishu.lark-cli/oauth/main.go.tmpl /build/oauth-main.go.tmpl
COPY feishu.lark-cli/oauth/main_test.go ./cmd/xpeech-lark-oauth/main_test.go
RUN --mount=type=bind,source=conf.toml,target=/build/conf.toml,readonly \
    uv run --no-project python /build/generate.py \
        --config /build/conf.toml \
        --template /build/mycred.go.tmpl \
        --output ./mycred/mycred.go && \
    uv run --no-project python /build/generate.py \
        --config /build/conf.toml \
        --template /build/oauth-main.go.tmpl \
        --output ./cmd/xpeech-lark-oauth/main.go
RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    CGO_ENABLED=0 go test ./cmd/xpeech-lark-oauth && \
    CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/lark-cli . && \
    CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/lark-oauth ./cmd/xpeech-lark-oauth

FROM docker.1panel.live/library/ubuntu:22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple \
    UV_PYTHON=3.12 \
    UV_HTTP_TIMEOUT=200 \
    UV_PYTHON_INSTALL_MIRROR=https://registry.npmmirror.com/-/binary/python-build-standalone/ \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_INSTALL_DIR=/usr/local/bin \
    UV_NO_MODIFY_PATH=1

RUN sed -i \
        -e 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' \
        -e 's|http://security.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' \
        /etc/apt/sources.list && \
    apt-get update && \
    apt-get -o Acquire::https::Verify-Peer=false update && \
    apt-get -o Acquire::https::Verify-Peer=false install -y --no-install-recommends ca-certificates && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends bubblewrap cron curl ffmpeg git jq zip tzdata && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "aarch64" ]; then NODE_ARCH="arm64"; else NODE_ARCH="x64"; fi && \
    curl -fsSL "https://registry.npmmirror.com/-/binary/node/v24.4.1/node-v24.4.1-linux-${NODE_ARCH}.tar.gz" | tar -xz -C /usr/local --strip-components=1 && \
    node --version && \
    npm --version

RUN curl -fsSL https://gitee.com/luojiaaoo/uv/releases/download/0.12.5/uv-installer.sh | sh && \
    uv --version && \
    uv python install

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY xpeech/ ./xpeech/
COPY custom_tools/ ./custom_tools/
COPY assets/ ./assets/
RUN uv sync --frozen --no-dev

RUN npm config set registry https://registry.npmmirror.com/

RUN cd xpeech/channel/web_client/frontend && \
    npm install --no-audit --no-fund && \
    npm run build && \
    rm -rf node_modules

RUN npm i -g agent-browser

COPY --from=lark-cli-builder /out/lark-cli /usr/local/bin/lark-cli
COPY --from=lark-cli-builder /out/lark-oauth /usr/local/bin/lark-oauth

RUN printf '0 0 * * * find /app/data/cache -type f -mmin +1440 -exec rm -f {} + > /dev/null 2>&1\n' | crontab -

ENTRYPOINT ["/bin/sh", "-c", "/etc/init.d/cron start && exec uv run -m xpeech \"$@\"", "--"]
