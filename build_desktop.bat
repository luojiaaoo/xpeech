@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo   Xpeech Desktop - PyInstaller 打包脚本
echo ========================================
echo.

cd /d "%~dp0"

REM ── 1. 构建前端 ──────────────────────────
echo [1/3] 构建前端...
cd /d "%~dp0xpeech\channel\desktop_client\frontend"
if not exist "node_modules" (
    call npm install
    if errorlevel 1 (
        echo [ERROR] npm install 失败
        pause & exit /b 1
    )
)
call npm run build
if errorlevel 1 (
    echo [ERROR] 前端构建失败
    pause & exit /b 1
)
cd /d "%~dp0"
echo [OK] 前端构建完成
echo.

REM ── 2. 安装 PyInstaller ──────────────────
echo [2/3] 安装 PyInstaller...
uv pip install pyinstaller
if errorlevel 1 (
    echo [ERROR] PyInstaller 安装失败
    pause & exit /b 1
)
echo [OK] PyInstaller 安装完成
echo.

REM ── 3. PyInstaller 打包 ──────────────────
echo [3/3] PyInstaller 打包中...
uv run pyinstaller ^
    --name "DesktopClient" ^
    --onefile ^
    --windowed ^
    --clean ^
    --noconfirm ^
    --icon "./xpeech/channel/desktop_client/favicon.ico" ^
    --add-data "./xpeech/channel/desktop_client/favicon.ico;xpeech/channel/desktop_client/" ^
    --add-data "./xpeech/channel/desktop_client/frontend/dist;xpeech/channel/desktop_client/frontend/dist/" ^
    --hidden-import clr ^
    --hidden-import webview.platforms.winforms ^
    desktop_entry.py

if errorlevel 1 (
    echo [ERROR] PyInstaller 打包失败
    pause & exit /b 1
)

echo.

REM ── 4. 清理构建产物 ──────────────────
echo [4/4] 清理构建产物...
cd /d "%~dp0"
if exist "build" (
    rmdir /s /q "build"
    echo [OK] build 目录已删除
)
if exist "DesktopClient.spec" (
    del /q "DesktopClient.spec"
    echo [OK] DesktopClient.spec 已删除
)
echo.

echo ========================================
echo   打包完成！
echo ========================================
echo.
pause
