from types import SimpleNamespace

from xpeech.agent.tools import web


def test_web_fetch_sets_network_timeouts_and_closes_response(monkeypatch):
    calls = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            calls.append("response_closed")

        def raise_for_status(self):
            calls.append("status_checked")

    class FakeSession:
        def __init__(self):
            self.headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            calls.append("session_closed")

        def get(self, url, **kwargs):
            calls.append((url, kwargs))
            return FakeResponse()

    class FakeMarkItDown:
        def convert(self, response):
            assert isinstance(response, FakeResponse)
            return SimpleNamespace(title="Example", text_content="body")

    monkeypatch.setattr(web.requests, "Session", FakeSession)
    monkeypatch.setattr(web, "MarkItDown", FakeMarkItDown)

    result = web._fetch_and_convert("https://example.com/page")

    assert result == "[TITLE: Example]\n\nbody"
    assert calls == [
        (
            "https://example.com/page",
            {"stream": True, "timeout": (10, 60)},
        ),
        "status_checked",
        "response_closed",
        "session_closed",
    ]
