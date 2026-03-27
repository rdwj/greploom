from __future__ import annotations

import json

import httpx
import pytest

from greploom.index.embedder import EmbeddingClient

DIM = 4  # small dimension keeps fixture data concise


def make_mock_transport(
    *,
    dim: int = DIM,
    status: int = 200,
    body: dict | None = None,
) -> httpx.MockTransport:
    """Return a MockTransport that echoes back one embedding vector per input text."""

    def handler(request: httpx.Request) -> httpx.Response:
        if status != 200:
            return httpx.Response(status, text=body.get("text", "error") if body else "error")
        payload = json.loads(request.content)
        n = len(payload["input"])
        embeddings = [[0.1 * (i + 1)] * dim for i in range(n)]
        response_body = body if body is not None else {"embeddings": embeddings}
        return httpx.Response(200, json=response_body)

    return httpx.MockTransport(handler)


def client_with(transport: httpx.MockTransport) -> EmbeddingClient:
    ec = EmbeddingClient()
    ec._client = httpx.Client(transport=transport)
    return ec


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_embed_one_returns_single_vector() -> None:
    ec = client_with(make_mock_transport())
    vec = ec.embed_one("hello world")
    assert len(vec) == DIM
    assert all(isinstance(v, float) for v in vec)
    ec.close()


@pytest.mark.parametrize(
    "texts, batch_size, expected_count",
    [
        (["a", "b", "c"], 32, 3),   # single batch
        (["a", "b", "c", "d"], 2, 4),  # two batches of 2
        (["x"] * 10, 3, 10),        # uneven batches
    ],
)
def test_embed_batch(texts: list[str], batch_size: int, expected_count: int) -> None:
    ec = client_with(make_mock_transport())
    vecs = ec.embed(texts, batch_size=batch_size)
    assert len(vecs) == expected_count
    for vec in vecs:
        assert len(vec) == DIM
    ec.close()


def test_connection_error_raises_connection_error() -> None:
    def failing_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    ec = client_with(httpx.MockTransport(failing_handler))
    with pytest.raises(ConnectionError, match="Is ollama running"):
        ec.embed_one("test")
    ec.close()


def test_bad_status_raises_runtime_error() -> None:
    ec = client_with(make_mock_transport(status=500, body={"text": "internal error"}))
    with pytest.raises(RuntimeError, match="500"):
        ec.embed_one("test")
    ec.close()


def test_missing_embeddings_key_raises_runtime_error() -> None:
    ec = client_with(make_mock_transport(body={"result": []}))
    with pytest.raises(RuntimeError, match="missing 'embeddings' key"):
        ec.embed_one("test")
    ec.close()


def test_context_manager_closes_client() -> None:
    transport = make_mock_transport()
    with EmbeddingClient() as ec:
        ec._client = httpx.Client(transport=transport)
        vec = ec.embed_one("context manager test")
    assert len(vec) == DIM
    # Verify client is closed — further use should raise
    assert ec._client.is_closed
