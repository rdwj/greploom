from __future__ import annotations

import httpx


class EmbeddingClient:
    """Sync HTTP client for embedding text via ollama or any OpenAI-compatible endpoint."""

    def __init__(
        self,
        url: str = "http://localhost:11434",
        model: str = "nomic-embed-code",
    ) -> None:
        self._url = url.rstrip("/")
        self._model = model
        self._client = httpx.Client(timeout=60.0)

    def embed(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Embed a list of texts, batching requests for efficiency."""
        results: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            results.extend(self._post_batch(batch))
        return results

    def embed_one(self, text: str) -> list[float]:
        """Embed a single text string."""
        return self.embed([text])[0]

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> EmbeddingClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _post_batch(self, batch: list[str]) -> list[list[float]]:
        endpoint = f"{self._url}/api/embed"
        try:
            response = self._client.post(
                endpoint,
                json={"model": self._model, "input": batch},
            )
        except httpx.ConnectError:
            raise ConnectionError(
                f"Cannot connect to embedding service at {self._url}. Is ollama running?"
            )
        except httpx.TimeoutException:
            raise ConnectionError(
                f"Request to embedding service at {self._url} timed out."
            )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Embedding request failed with status {exc.response.status_code}: "
                f"{exc.response.text}"
            ) from exc

        data = response.json()
        if "embeddings" not in data:
            raise RuntimeError(
                f"Unexpected response from embedding service — missing 'embeddings' key. "
                f"Got: {list(data.keys())}"
            )

        return data["embeddings"]
