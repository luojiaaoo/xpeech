FROM docker.1panel.live/library/ubuntu:22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple \
    UV_PYTHON=3.12 \
    UV_HTTP_TIMEOUT=60 \
    UV_PYTHON_INSTALL_MIRROR=https://registry.npmmirror.com/-/binary/python-build-standalone/ \
    UV_DOWNLOAD_URL=https://mirrors.ustc.edu.cn/github-release/astral-sh/uv/LatestRelease/ \
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
    apt-get install -y --no-install-recommends bubblewrap cron curl ffmpeg git zip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "aarch64" ]; then NODE_ARCH="arm64"; else NODE_ARCH="x64"; fi && \
    curl -fsSL "https://registry.npmmirror.com/-/binary/node/v24.4.1/node-v24.4.1-linux-${NODE_ARCH}.tar.gz" | tar -xz -C /usr/local --strip-components=1 && \
    node --version && \
    npm --version

RUN curl -sL https://mirrors.ustc.edu.cn/github-release/astral-sh/uv/LatestRelease/uv-installer.sh | sh && \
    uv --version && \
    uv python install

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY xpeech/ ./xpeech/
COPY custom_tools/ ./custom_tools/
RUN uv sync --frozen --no-dev

RUN npm config set registry https://registry.npmmirror.com/ && \
    npm i -g agent-browser

RUN mkdir /app/sandbox-home-defaults
COPY data/sandbox-home/.npmrc /app/sandbox-home-defaults/.npmrc
COPY data/sandbox-home/.pip/pip.conf /app/sandbox-home-defaults/.pip/pip.conf
COPY data/sandbox-home/.config/uv/uv.toml /app/sandbox-home-defaults/.config/uv/uv.toml

RUN printf '0 0 * * * find /app/data/cache -type f -mmin +1440 -exec rm -f {} +\n' | crontab -

ENTRYPOINT ["/bin/sh", "-c", "mkdir -p /app/data/sandbox-home /app/data/cache && cp -a /app/sandbox-home-defaults/. /app/data/sandbox-home/ && /etc/init.d/cron start && exec uv run -m xpeech \"$@\"", "--"]
