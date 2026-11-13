ARG GO_VERSION=1.27.3
FROM docker.1ms.run/library/golang:${GO_VERSION}-bookworm AS lark-cli-builder

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

FROM docker.1panel.live/library/ubuntu:22.04

COPY --from=lark-cli-builder /usr/local/go /usr/local/go

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple \
    UV_PYTHON=3.12 \
    UV_HTTP_TIMEOUT=200 \
    UV_PYTHON_INSTALL_MIRROR=https://registry.npmmirror.com/-/binary/python-build-standalone/ \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_INSTALL_DIR=/usr/local/bin \
    UV_NO_MODIFY_PATH=1 \
    PATH=/usr/local/go/bin:${PATH}

RUN go version

RUN export DEBIAN_FRONTEND=noninteractive && \
    # 软件源与 CA 证书：供后续 apt、curl、Git 等工具校验 HTTPS 证书。
    sed -i \
        -e 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' \
        -e 's|http://security.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' \
        /etc/apt/sources.list && \
    apt-get -o Acquire::https::Verify-Peer=false update && \
    apt-get -o Acquire::https::Verify-Peer=false install -y --no-install-recommends ca-certificates && \
    # 基础工具：定时清理、网络下载、Git 仓库操作、JSON 处理、归档和时区支持。
    apt-get install -y --no-install-recommends \
        cron curl git jq zip tzdata xz-utils && \
    # Shell 沙盒：bubblewrap 隔离进程和文件系统，slirp4netns 提供出站网络。
    apt-get install -y --no-install-recommends \
        bubblewrap slirp4netns && \
    # C/C++ 开发：编译、构建、依赖探测、静态检查、调试与代码格式化。
    apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build pkg-config cppcheck gdb clang-format && \
    # 音视频处理：转码、提取音轨及媒体文件分析。
    apt-get install -y --no-install-recommends \
        ffmpeg && \
    # 图形与 CAD 运行库：为依赖 OpenGL/GLU 的图形处理工具提供动态库。
    apt-get install -y --no-install-recommends \
        libgl1 libglu1-mesa && \
    # 网络报文分析：解析 PCAP/PCAPNG，导出协议字段和 JSON。
    apt-get install -y --no-install-recommends \
        tshark && \
    # Office 文档：无界面处理和转换文字文档、演示文稿、表格及绘图文件。
    apt-get install -y --no-install-recommends \
        libreoffice-writer libreoffice-impress libreoffice-calc libreoffice-draw && \
    # PDF 与文档转换：PDF 文本提取、页面渲染，以及 Markdown/HTML 等格式转换。
    apt-get install -y --no-install-recommends \
        poppler-utils pandoc && \
    # OCR：识别图片和扫描件中的英文、简体中文文字。
    apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-eng tesseract-ocr-chi-sim && \
    # 系统字体：常用中西文字体、楷体和 Emoji，统一通过 apt 安装。
    apt-get install -y --no-install-recommends \
        fontconfig fonts-dejavu-core fonts-noto-cjk fonts-wqy-microhei \
        fonts-arphic-ukai fonts-liberation2 fonts-noto-color-emoji && \
    fc-cache -f && \
    test -n "$(fc-list :lang=zh file)" && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

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

WORKDIR /app

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

COPY --from=lark-cli-builder /out/lark-cli /usr/local/bin/lark-cli
COPY --from=lark-cli-builder /out/lark-cli-auth /usr/local/bin/lark-cli-auth

RUN printf '0 0 * * * find /app/data/cache -type f -mmin +1440 -exec rm -f {} + > /dev/null 2>&1\n' | crontab -

ENTRYPOINT ["/bin/sh", "-c", "/etc/init.d/cron start && exec uv run -m xpeech \"$@\"", "--"]
