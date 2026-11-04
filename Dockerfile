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

ARG LARK_CLI_REF=v1.0.89
RUN git clone --filter=blob:none --no-checkout https://gitee.com/luojiaaoo/lark-cli.git . && \
    git checkout --detach "${LARK_CLI_REF}"
COPY feishu.lark-cli/main.go ./main.go
COPY feishu.lark-cli/generate.py /build/generate.py
COPY feishu.lark-cli/mycred/mycred.go.tmpl /build/mycred.go.tmpl
COPY feishu.lark-cli/oauth/main.go.tmpl /build/oauth-main.go.tmpl
COPY feishu.lark-cli/oauth/main_test.go ./cmd/xpeech-lark-cli-auth/main_test.go
RUN --mount=type=bind,source=conf.toml,target=/build/conf.toml,readonly \
    uv run --no-project python /build/generate.py \
        --config /build/conf.toml \
        --template /build/mycred.go.tmpl \
        --output ./mycred/mycred.go && \
    uv run --no-project python /build/generate.py \
        --config /build/conf.toml \
        --template /build/oauth-main.go.tmpl \
        --output ./cmd/xpeech-lark-cli-auth/main.go
RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    CGO_ENABLED=0 go test ./cmd/xpeech-lark-cli-auth && \
    CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/lark-cli . && \
    CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/lark-cli-auth ./cmd/xpeech-lark-cli-auth

# Buildx exports this target directly to a host directory. Neither application
# nor execution images depend on the CLI build stage.
FROM scratch AS lark-cli-export
COPY --from=lark-cli-builder /out/lark-cli /lark-cli
COPY --from=lark-cli-builder /out/lark-cli-auth /lark-cli-auth

FROM docker.1panel.live/library/ubuntu:22.04 AS os-base

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
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends bubblewrap cron curl ffmpeg git jq zip tzdata xz-utils && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# A standalone Miniforge runtime for user workloads.
FROM os-base AS sandbox
ARG MINIFORGE_VERSION=26.9.4-1
RUN MINIFORGE_ARCH="$(uname -m)" && \
    case "$MINIFORGE_ARCH" in x86_64|aarch64) ;; *) exit 1 ;; esac && \
    MINIFORGE_ASSET="Miniforge3-${MINIFORGE_VERSION}-Linux-${MINIFORGE_ARCH}.sh" && \
    MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}" && \
    curl -fsSL --retry 3 "${MINIFORGE_URL}/${MINIFORGE_ASSET}" -o /tmp/miniforge.sh && \
    curl -fsSL --retry 3 "${MINIFORGE_URL}/${MINIFORGE_ASSET}.sha256" -o /tmp/miniforge.sha256 && \
    printf '%s  /tmp/miniforge.sh\n' "$(cut -d ' ' -f1 /tmp/miniforge.sha256)" | sha256sum -c - && \
    bash /tmp/miniforge.sh -b -p /opt/miniforge && \
    rm -f /tmp/miniforge.sh /tmp/miniforge.sha256
COPY docker/sandbox/condarc /opt/miniforge/.condarc
RUN /opt/miniforge/bin/mamba create --yes --prefix /opt/miniforge/envs/runtime \
        python=3.12 pip nodejs=24 uv && \
    /opt/miniforge/bin/mamba clean --all --yes && \
    ln -s /opt/miniforge/envs/runtime/bin/python /usr/local/bin/python-runtime && \
    printf '[global]\nindex-url = https://pypi.tuna.tsinghua.edu.cn/simple\n' > /etc/pip.conf && \
    /opt/miniforge/envs/runtime/bin/npm config --global set registry https://registry.npmmirror.com/
COPY docker/sandbox/run_command.py /opt/xpeech/run_command.py
ENV PATH=/opt/miniforge/envs/runtime/bin:/opt/miniforge/condabin:/opt/miniforge/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    UV_PYTHON=/usr/local/bin/python-runtime \
    HOME=/root
WORKDIR /workspace
CMD ["tail", "-f", "/dev/null"]

FROM os-base AS tool-runtime
ARG DCG_VERSION=0.16.0
RUN DCG_ARCH="$(uname -m)" && \
    case "$DCG_ARCH" in \
        x86_64) DCG_TARGET="x86_64-unknown-linux-musl" ;; \
        aarch64) DCG_TARGET="aarch64-unknown-linux-gnu" ;; \
        *) echo "Unsupported dcg architecture: $DCG_ARCH" >&2; exit 1 ;; \
    esac && \
    DCG_ASSET="dcg-${DCG_TARGET}.tar.xz" && \
    DCG_RELEASE_URL="https://gitee.com/luojiaaoo/destructive_command_guard/releases/download/v${DCG_VERSION}" && \
    DCG_TMP_DIR="$(mktemp -d)" && \
    curl -fsSL --retry 3 "${DCG_RELEASE_URL}/${DCG_ASSET}" -o "${DCG_TMP_DIR}/${DCG_ASSET}" && \
    curl -fsSL --retry 3 "${DCG_RELEASE_URL}/${DCG_ASSET}.sha256" -o "${DCG_TMP_DIR}/${DCG_ASSET}.sha256" && \
    (cd "$DCG_TMP_DIR" && sha256sum -c "${DCG_ASSET}.sha256") && \
    tar -xJf "${DCG_TMP_DIR}/${DCG_ASSET}" -C "$DCG_TMP_DIR" && \
    install -m 0755 "${DCG_TMP_DIR}/dcg" /usr/local/bin/dcg && \
    rm -rf "$DCG_TMP_DIR" && \
    dcg --version

RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "aarch64" ]; then NODE_ARCH="arm64"; else NODE_ARCH="x64"; fi && \
    curl -fsSL "https://registry.npmmirror.com/-/binary/node/v24.4.1/node-v24.4.1-linux-${NODE_ARCH}.tar.gz" | tar -xz -C /usr/local --strip-components=1 && \
    node --version && \
    npm --version

RUN curl -fsSL https://gitee.com/luojiaaoo/uv/releases/download/0.12.5/uv-installer.sh | sh && \
    uv --version && \
    uv python install

# Keep the backend as the default target. Its runtimes are still needed by
# local media tools, configured command templates, and stdio MCP.
FROM tool-runtime AS backend
# Compose passes the real host project directory so __file__-based built-in
# skill paths are valid bind sources for the host Docker daemon too.
ARG APP_ROOT=/app
WORKDIR ${APP_ROOT}

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY xpeech/ ./xpeech/
COPY custom_tools/ ./custom_tools/
COPY assets/ ./assets/
RUN uv sync --frozen --no-dev

RUN npm config set registry https://registry.npmmirror.com/

RUN cd xpeech/subapp/channel/web_client/frontend && \
    npm install --no-audit --no-fund && \
    npm run build && \
    rm -rf node_modules

RUN printf '0 0 * * * find "%s/docker_data/cache" -type f -mmin +1440 -exec rm -f {} + > /dev/null 2>&1\n' "${APP_ROOT}" | crontab -

ENTRYPOINT ["/bin/sh", "-c", "/etc/init.d/cron start && exec uv run python -m xpeech.container_entrypoint \"$@\"", "--"]
