"""Internal service dispatcher used by the Docker image entrypoint."""

from __future__ import annotations

import argparse
import os
import platform
from urllib.parse import urlsplit

# 取消 litellm 的本地模型成本映射
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "true"


def _normalize_http_url(url: str) -> str:
    normalized = url.rstrip("/")
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise argparse.ArgumentTypeError("URL must be an absolute HTTP(S) URL")
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Docker Compose Xpeech service.")
    subparsers = parser.add_subparsers(dest="service", required=True)

    api_parser = subparsers.add_parser("api", help="Run the Xpeech API service.")
    api_parser.add_argument("--host", default="0.0.0.0")
    api_parser.add_argument("--port", type=int, default=7878)
    api_parser.add_argument("--scheduler-url", type=_normalize_http_url, required=True)

    feishu_parser = subparsers.add_parser("feishu", help="Run the Feishu bridge.")
    feishu_parser.add_argument("--chat-url", type=_normalize_http_url, required=True)

    web_client_parser = subparsers.add_parser(
        "web_client", help="Run the authenticated web client.",
    )
    web_client_parser.add_argument("--host", default="0.0.0.0")
    web_client_parser.add_argument("--port", type=int, default=7939)
    web_client_parser.add_argument("--backend-url", type=_normalize_http_url, required=True)

    token_parser = subparsers.add_parser(
        "token_manager", help="Run the Lark token manager.",
    )
    token_parser.add_argument("--host", default="0.0.0.0")
    token_parser.add_argument("--port", type=int, default=7883)

    scheduler_parser = subparsers.add_parser(
        "scheduler", help="Run the single scheduler service.",
    )
    scheduler_parser.add_argument("--host", default="0.0.0.0")
    scheduler_parser.add_argument("--port", type=int, default=8090)
    scheduler_parser.add_argument("--backend-url", type=_normalize_http_url, required=True)

    args = parser.parse_args()
    service = args.service

    if service == "api":
        from .agent.server.app import run

        if platform.system() != "Linux":
            raise RuntimeError("Xpeech API is only supported on Linux.")
        run(host=args.host, port=args.port, scheduler_url=args.scheduler_url)
        return

    if service == "feishu":
        from .subapp.channel.feishu.bridge import run

        run(chat_url=args.chat_url)
        return

    if service == "web_client":
        from .subapp.channel.web_client.app import run

        run(host=args.host, port=args.port, backend_url=args.backend_url)
        return

    if service == "token_manager":
        from .subapp.lark_token_manager.app import run

        run(host=args.host, port=args.port)
        return

    if service == "scheduler":
        from .subapp.scheduler.app import run

        run(host=args.host, port=args.port, backend_url=args.backend_url)
        return

    parser.error(f"Unknown service: {service}")


if __name__ == "__main__":
    main()
