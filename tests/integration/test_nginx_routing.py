"""Run with NGINX_BINARY=/path/to/nginx to test the real gateway and DNS peers."""

import asyncio
import json
import os
import socket
import struct
from collections import Counter
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("NGINX_BINARY"), reason="NGINX_BINARY is not set")


class ReplicaDNS(asyncio.DatagramProtocol):
    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, address):
        offset = 12
        labels = []
        while data[offset]:
            size = data[offset]
            labels.append(data[offset + 1:offset + size + 1].decode())
            offset += size + 1
        question = data[12:offset + 5]
        hostname = ".".join(labels)
        addresses = [f"127.0.0.{index}" for index in range(2, 6)] if hostname == "backend" else []
        records = b"".join(
            b"\xc0\x0c" + struct.pack("!HHIH", 1, 1, 1, 4) + socket.inet_aton(ip)
            for ip in addresses
        )
        header = data[:2] + struct.pack("!HHHHH", 0x8180, 1, len(addresses), 0, 0)
        self.transport.sendto(header + question + records, address)


@pytest.mark.asyncio
async def test_gateway_hashes_chat_and_scheduled_runs_and_sends_polls_to_backends(tmp_path):
    async def handle_request(reader, writer):
        try:
            headers = await reader.readuntil(b"\r\n\r\n")
            length = next(
                (int(line.split(b":", 1)[1]) for line in headers.split(b"\r\n") if line.lower().startswith(b"content-length:")),
                0,
            )
            if length:
                await reader.readexactly(length)
            body = json.dumps({"replica": writer.get_extra_info("sockname")[0]}).encode()
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n"
                + f"Content-Length: {len(body)}\r\n\r\n".encode() + body,
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    servers = []
    dns_transport = None
    process = None
    try:
        first = await asyncio.start_server(handle_request, "127.0.0.2", 0)
        servers.append(first)
        backend_port = first.sockets[0].getsockname()[1]
        for index in range(3, 6):
            servers.append(await asyncio.start_server(handle_request, f"127.0.0.{index}", backend_port))
        dns_transport, _ = await asyncio.get_running_loop().create_datagram_endpoint(
            ReplicaDNS, local_addr=("127.0.0.1", 0),
        )
        dns_port = dns_transport.get_extra_info("sockname")[1]
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            gateway_port = reservation.getsockname()[1]

        config = (Path(__file__).resolve().parents[2] / "docker/nginx.conf").read_text()
        config = config.replace("worker_processes auto;", "worker_processes 2;")
        config = config.replace("127.0.0.11", f"127.0.0.1:{dns_port}")
        config = config.replace("backend:8080", f"backend:{backend_port}")
        config = config.replace("listen 7878;", f"listen 127.0.0.1:{gateway_port};")
        paths = "\n".join(
            f"{directive}_temp_path {tmp_path / directive};"
            for directive in ("client_body", "proxy", "fastcgi", "uwsgi", "scgi")
        )
        config = config.replace("http {", f"http {{\naccess_log off;\n{paths}")
        config = f"daemon off;\nerror_log stderr;\npid {tmp_path / 'nginx.pid'};\n{config}"
        config_path = tmp_path / "nginx.conf"
        config_path.write_text(config)
        process = await asyncio.create_subprocess_exec(
            os.environ["NGINX_BINARY"], "-e", "stderr", "-p", f"{tmp_path}/", "-c", str(config_path),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{gateway_port}", trust_env=False) as client:
            for _ in range(100):
                if process.returncode is not None:
                    _, stderr = await process.communicate()
                    pytest.fail(stderr.decode())
                try:
                    response = await client.get("/health", timeout=0.2)
                    if response.status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.05)
            else:
                pytest.fail("nginx did not become ready")

            distribution = Counter()
            for index in range(100):
                headers = {"x-user-id": f"user-{index}"}
                replicas = set()
                for path in ("/chat", "/answer_question", "/internal/scheduled-run", "/chat"):
                    response = await client.post(path, headers=headers, json={"task": "test"})
                    response.raise_for_status()
                    replicas.add(response.json()["replica"])
                assert len(replicas) == 1, f"User {index} reached multiple replicas: {replicas}"
                distribution.update(replicas)
            assert len(distribution) == 4, distribution

            response = await client.get("/background_message?channel=feishu")
            response.raise_for_status()
            assert response.json()["replica"] in distribution
    finally:
        if process is not None and process.returncode is None:
            process.terminate()
            await process.communicate()
        if dns_transport is not None:
            dns_transport.close()
        for server in servers:
            server.close()
            await server.wait_closed()
