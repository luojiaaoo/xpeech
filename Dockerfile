FROM docker.1panel.live/library/ubuntu:22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple \
    UV_PYTHON=3.12 \
    UV_PYTHON_INSTALL_MIRROR=https://registry.npmmirror.com/-/binary/python-build-standalone \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_INSTALL_DIR=/usr/local/bin \
    UV_NO_MODIFY_PATH=1

RUN sed -i \
        -e 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' \
        -e 's|http://security.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' \
        /etc/apt/sources.list \
    && apt-get -o Acquire::https::Verify-Peer=false update \
    && apt-get -o Acquire::https::Verify-Peer=false install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        bubblewrap \
        curl \
        ffmpeg \
        git \
        nodejs \
        npm \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && uv --version

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY xpeech/ ./xpeech/
COPY custom_tools/ ./custom_tools/
COPY data/sandbox-home/.npmrc ./sandbox-home/.npmrc
COPY data/sandbox-home/.pip/pip.conf ./sandbox-home/.pip/pip.conf
COPY data/sandbox-home/.config/uv/uv.toml ./sandbox-home/.config/uv/uv.toml
RUN uv sync --frozen --no-dev

ENTRYPOINT ["uv", "run", "-m", "xpeech"]
