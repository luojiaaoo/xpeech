#!/usr/bin/env python3
"""
verify.py — 通过远程 CDP 浏览器批量验证 HTML

Usage:
    python verify.py <preview-url>
    python verify.py <preview-url> --viewports 1920x1080,375x667
    python verify.py <preview-url> --slides 10
    python verify.py <preview-url> --output ./screenshots/

依赖：
    pip install playwright

注意：
    target 必须是 create_browser_preview 返回的 HTTP(S) URL，不接受本地文件路径。
    Playwright 仅作为 CDP 客户端；脚本不会安装或启动本地浏览器。
"""

import argparse
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit


def parse_viewport(s):
    w, h = s.split('x')
    return {'width': int(w), 'height': int(h)}


def verify_html(preview_url, viewports=None, slides=0, output_dir=None, wait=2000):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("ERROR: playwright未安装。")
        print("运行: pip install playwright")
        sys.exit(1)

    parsed_url = urlsplit(preview_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        print("ERROR: target 必须是 create_browser_preview 返回的 HTTP(S) URL")
        sys.exit(1)
    cdp_url = os.environ.get("XPEECH_TOOL__CDP_URL")
    if not cdp_url:
        print("ERROR: 请通过环境变量 XPEECH_TOOL__CDP_URL 提供远程 CDP 地址")
        sys.exit(1)

    if output_dir is None:
        output_dir = Path.cwd() / 'screenshots'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(parsed_url.path).stem or 'preview'

    if viewports is None:
        viewports = [{'width': 1440, 'height': 900}]

    console_errors = []
    page_errors = []

    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(cdp_url)
        try:
            contexts = browser.contexts
            context = contexts[0] if contexts else browser.new_context()

            for viewport in viewports:
                page = context.new_page()
                page.set_viewport_size(viewport)

                page.on("console", lambda msg: console_errors.append(f"[{msg.type}] {msg.text}") if msg.type in ("error", "warning") else None)
                page.on("pageerror", lambda err: page_errors.append(str(err)))

                print(f"\n→ 打开 {preview_url} @ {viewport['width']}x{viewport['height']}")
                page.goto(preview_url, wait_until='networkidle')
                page.wait_for_timeout(wait)

                if slides > 0:
                    for i in range(slides):
                        screenshot_path = output_dir / f"{stem}-slide-{str(i + 1).zfill(2)}.png"
                        page.screenshot(path=str(screenshot_path), full_page=False)
                        print(f"  ✓ slide {i+1} → {screenshot_path.name}")

                        if i < slides - 1:
                            page.keyboard.press('ArrowRight')
                            page.wait_for_timeout(500)
                else:
                    suffix = f"-{viewport['width']}x{viewport['height']}" if len(viewports) > 1 else ""
                    screenshot_path = output_dir / f"{stem}{suffix}.png"
                    page.screenshot(path=str(screenshot_path), full_page=False)
                    print(f"  ✓ 截图 → {screenshot_path.name}")

                    full_path = output_dir / f"{stem}{suffix}-full.png"
                    page.screenshot(path=str(full_path), full_page=True)
                    print(f"  ✓ 完整页 → {full_path.name}")

                page.close()
        finally:
            browser.close()

    print("\n" + "=" * 50)
    print("验证报告")
    print("=" * 50)

    if page_errors:
        print(f"\n❌ Page Errors ({len(page_errors)}):")
        for e in page_errors:
            print(f"  - {e}")
    else:
        print("\n✅ 无JavaScript错误")

    if console_errors:
        print(f"\n⚠️  Console Errors/Warnings ({len(console_errors)}):")
        for e in console_errors[:20]:
            print(f"  - {e}")
        if len(console_errors) > 20:
            print(f"  ... 还有{len(console_errors) - 20}条")
    else:
        print("✅ Console干净")

    print(f"\n📸 截图保存至: {output_dir}")

    return 0 if not page_errors else 1


def main():
    parser = argparse.ArgumentParser(
        description="Verify previewed HTML through a remote CDP browser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("preview_url", help="create_browser_preview 返回的 HTTP(S) URL")
    parser.add_argument("--viewports", default="1440x900",
                        help="逗号分隔的viewport列表，格式 WxH（默认 1440x900）")
    parser.add_argument("--slides", type=int, default=0,
                        help="幻灯片模式：截取前N张（需要HTML支持ArrowRight翻页）")
    parser.add_argument("--output", default=None,
                        help="输出目录（默认当前目录的 screenshots/）")
    parser.add_argument("--wait", type=int, default=2000,
                        help="打开页面后等待的毫秒数（默认2000）")

    args = parser.parse_args()

    viewports = [parse_viewport(v) for v in args.viewports.split(",")]

    return verify_html(
        preview_url=args.preview_url,
        viewports=viewports,
        slides=args.slides,
        output_dir=args.output,
        wait=args.wait,
    )


if __name__ == "__main__":
    sys.exit(main())
