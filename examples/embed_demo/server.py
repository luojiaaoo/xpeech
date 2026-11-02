"""Local embed demo. Run: python3 examples/embed_demo/server.py

Configuration: data/embed-demo/config.json (ignored by Git).
The token endpoint always signs the configured demo user, never browser input.
This is a local development demo, not a production login service.
"""

import argparse
import base64
import hashlib
import hmac
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[2]


def encode(value):
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def issue_jwt(config):
    algorithm = config.get("algorithm", "HS256")
    digest = {"HS256": hashlib.sha256, "HS384": hashlib.sha384, "HS512": hashlib.sha512}[algorithm]
    header = {"alg": algorithm, "typ": "JWT", "kid": config["channel"]}
    # Omit iat to tolerate small clock differences between the two servers.
    payload = {"sub": config["user_id"], "exp": int(time.time()) + 60}
    message = ".".join(encode(json.dumps(part, separators=(",", ":")).encode()) for part in (header, payload))
    signature = hmac.new(config["secret_key"].encode(), message.encode(), digest).digest()
    return message + "." + encode(signature)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--config", type=Path, default=ROOT / "data/embed-demo/config.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    html = Path(__file__).with_name("index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, body, content_type="application/json; charset=utf-8"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            # Restrict this unauthenticated development issuer to local origins.
            host = urlsplit("http://" + self.headers.get("Host", "")).hostname
            if host not in {"localhost", "127.0.0.1"}:
                self.send(403, b'{"error":"Local access only"}')
                return
            path = urlsplit(self.path).path
            if path == "/":
                self.send(200, html, "text/html; charset=utf-8")
            elif path in {"/api/config", "/api/assistant/jwt"}:
                origin = self.headers.get("Origin")
                if self.headers.get("Sec-Fetch-Site") == "cross-site" or (
                    origin and origin != "http://" + self.headers.get("Host")
                ):
                    self.send(403, b'{"error":"Same-origin requests required"}')
                    return
                result = (
                    {"jwt": issue_jwt(config)} if path.endswith("/jwt") else
                    {
                        **{key: config[key] for key in ("assistant_url", "channel", "user_id")},
                        "sender_name": config.get("sender_name", config["user_id"]),
                        "title": config.get("title", "AI 助手"),
                        "session_metadata": config.get("session_metadata", {}),
                    }
                )
                self.send(200, json.dumps(result).encode())
            else:
                self.send(404, b'{"error":"Not found"}')

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Embed demo running at http://localhost:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
