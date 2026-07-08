#!/bin/bash
set -e

echo "========================================"
echo "  Xpeech Desktop - PyInstaller 打包脚本"
echo "========================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── 1. 构建前端 ──────────────────────────
echo "[1/4] 构建前端..."
cd "$SCRIPT_DIR/xpeech/channel/desktop_client/frontend"
if [ ! -d "node_modules" ]; then
    npm install
fi
npm run build
cd "$SCRIPT_DIR"
echo "[OK] 前端构建完成"
echo ""

# ── 2. 安装 PyInstaller ──────────────────
echo "[2/4] 安装 PyInstaller..."
uv pip install pyinstaller
echo "[OK] PyInstaller 安装完成"
echo ""

# ── 3. PyInstaller 打包 ──────────────────
echo "[3/4] PyInstaller 打包中..."

# 根据平台选择隐藏导入和路径分隔符
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    HIDDEN_IMPORTS="--hidden-import webview.platforms.gtk"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    HIDDEN_IMPORTS="--hidden-import webview.platforms.cocoa"
else
    HIDDEN_IMPORTS="--hidden-import clr --hidden-import webview.platforms.winforms"
fi

uv run pyinstaller \
    --name "DesktopClient" \
    --onefile \
    --windowed \
    --clean \
    --noconfirm \
    --icon "./xpeech/channel/desktop_client/favicon.ico" \
    --add-data "./xpeech/channel/desktop_client/favicon.ico:xpeech/channel/desktop_client/" \
    --add-data "./xpeech/channel/desktop_client/frontend/dist:xpeech/channel/desktop_client/frontend/dist/" \
    $HIDDEN_IMPORTS \
    desktop_entry.py

echo "[OK] PyInstaller 打包完成"
echo ""

# ── 4. 清理构建产物 ──────────────────────
echo "[4/4] 清理构建产物..."
cd "$SCRIPT_DIR"
if [ -d "build" ]; then
    rm -rf build
    echo "[OK] build 目录已删除"
fi
if [ -f "DesktopClient.spec" ]; then
    rm -f DesktopClient.spec
    echo "[OK] DesktopClient.spec 已删除"
fi
echo ""

echo "========================================"
echo "  打包完成！"
echo "========================================"
echo ""
