"""Local embed demo. Run: python3 examples/embed_demo/server.py

Configuration: docker_data/embed-demo/config.json (ignored by Git).
The token endpoint always signs the configured demo user, never browser input.
Channel and signing-key overrides from the browser are test-only convenience:
the key round-trips through this backend, which signs and returns the JWT.
Production must obtain JWTs from the business backend; never expose signing
keys to the frontend. This is a local development demo, not a production login
service.
"""

import argparse
import base64
import hashlib
import hmac
import json
import re
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


def create_server(config, port=8080):
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

        def allowed(self, *, same_origin=False):
            # Restrict this unauthenticated development issuer to local origins.
            host = urlsplit("http://" + self.headers.get("Host", "")).hostname
            if host not in {"localhost", "127.0.0.1"}:
                self.send(403, b'{"error":"Local access only"}')
                return False
            if same_origin:
                origin = self.headers.get("Origin")
                if self.headers.get("Sec-Fetch-Site") == "cross-site" or (
                    origin and origin != "http://" + self.headers.get("Host")
                ):
                    self.send(403, b'{"error":"Same-origin requests required"}')
                    return False
            return True

        def do_GET(self):
            path = urlsplit(self.path).path
            if not self.allowed(same_origin=path.startswith("/api/")):
                return
            if path == "/":
                self.send(200, html, "text/html; charset=utf-8")
            elif path in {"/api/config", "/api/assistant/jwt"}:
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

        def do_POST(self):
            if not self.allowed(same_origin=True):
                return
            if urlsplit(self.path).path != "/api/assistant/jwt":
                self.send(404, b'{"error":"Not found"}')
                return
            if self.headers.get_content_type() != "application/json":
                self.send(415, b'{"error":"Expected application/json"}')
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16384:
                    raise ValueError("Invalid request length")
                values = json.loads(self.rfile.read(length))
                if not isinstance(values, dict):
                    raise ValueError("Expected a JSON object")
                channel = values.get("channel", config["channel"])
                if not isinstance(channel, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,63}", channel):
                    raise ValueError("Invalid channel")
                if channel in {"internal", "web_client", "feishu"}:
                    raise ValueError("Use a custom channel")
                secret = values.get("secret_key", "")
                if not isinstance(secret, str):
                    raise ValueError("Key must be a string")
                if not secret and channel == config["channel"]:
                    secret = config.get("secret_key", "")
                if not secret.strip():
                    raise ValueError("Enter a key for this channel")
            except (ValueError, UnicodeDecodeError) as error:
                self.send(400, json.dumps({"error": str(error)}).encode())
                return
            # Test-only overrides: never accept browser-supplied identity or persist the key.
            token = issue_jwt({**config, "channel": channel, "secret_key": secret})
            self.send(200, json.dumps({"jwt": token}).encode())

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--config", type=Path, default=ROOT / "docker_data/embed-demo/config.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    server = create_server(config, args.port)
    print(f"Embed demo running at http://localhost:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
