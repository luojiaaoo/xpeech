from __future__ import annotations

import os
import platform
import argparse

# 取消 litellm 的本地模型成本映射
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "true"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Xpeech services.")
    subparsers = parser.add_subparsers(dest="service")

    api_parser = subparsers.add_parser("api", help="Run the Xpeech API service.")
    api_parser.add_argument("--host", default="0.0.0.0")
    api_parser.add_argument("--port", type=int, default=7878)
    api_parser.set_defaults(service="api")

    feishu_parser = subparsers.add_parser("feishu", help="Run the Feishu bridge.")
    feishu_parser.add_argument("--chat-url", default="http://127.0.0.1:7878", help="Xpeech /chat endpoint.")
    feishu_parser.set_defaults(service="feishu")

    web_parser = subparsers.add_parser("web", help="Run the authenticated Xpeech web client.")
    web_parser.add_argument("--host", default="0.0.0.0")
    web_parser.add_argument("--port", type=int, default=7880)
    web_parser.add_argument("--backend-url", default="http://127.0.0.1:7878")
    web_parser.add_argument("--database", default="data/web/users.db")
    web_parser.set_defaults(service="web")

    args = parser.parse_args()
    service = args.service or "api"

    if service == "api":
        from .agent.server.app import run

        if platform.system() != "Linux":
            raise RuntimeError("Xpeech API is only supported on Linux.")
        run(host=getattr(args, "host", "0.0.0.0"), port=getattr(args, "port", 7878))
        return

    if service == "feishu":
        from .channel.feishu import run

        run(chat_url=args.chat_url)
        return

    if service == "web":
        from .channel.web_client.app import run

        run(
            host=args.host,
            port=args.port,
            backend_url=args.backend_url,
            database_path=args.database,
        )
        return

    parser.error(f"Unknown service: {service}")


if __name__ == "__main__":
    main()
