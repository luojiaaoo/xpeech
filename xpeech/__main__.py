from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Xpeech services.")
    subparsers = parser.add_subparsers(dest="service")

    api_parser = subparsers.add_parser("api", help="Run the Xpeech API service.")
    api_parser.add_argument("--host", default="0.0.0.0")
    api_parser.add_argument("--port", type=int, default=7878)
    api_parser.set_defaults(service="api")

    feishu_parser = subparsers.add_parser("feishu", help="Run the Feishu bridge.")
    feishu_parser.add_argument("--chat-url", default="http://127.0.0.1:7878/chat", help="Xpeech /chat endpoint.")
    feishu_parser.add_argument("--app-id", default="cli_aa8fcbaccae41cba", help="Feishu app ID.")
    feishu_parser.add_argument("--app-secret", default="a4TR1TMqrk1ZJPl3TV9QffgyQiWzu5YD", help="Feishu app secret.")
    feishu_parser.set_defaults(service="feishu")

    args = parser.parse_args()
    service = args.service or "api"

    if service == "api":
        from .agent.server.app import run

        run(host=getattr(args, "host", "0.0.0.0"), port=getattr(args, "port", 7878))
        return

    if service == "feishu":
        from .channel.feishu import run

        run(chat_url=args.chat_url, app_id=args.app_id, app_secret=args.app_secret)
        return

    parser.error(f"Unknown service: {service}")


if __name__ == "__main__":
    main()
