from __future__ import annotations

import httpx


class EmbeddingClient:
    """Sync HTTP client for embedding text via ollama or any OpenAI-compatible endpoint."""

    def __init__(
        self,
        url: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
        provider: str = "ollama",
    ) -> None:
        self._url = url.rstrip("/")
        self._model = model
        self._provider = provider
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
        if self._provider == "openai":
            endpoint = f"{self._url}/v1/embeddings"
        else:
            endpoint = f"{self._url}/api/embed"

        try:
            response = self._client.post(
                endpoint,
                json={"model": self._model, "input": batch},
            )
        except httpx.ConnectError:
            raise ConnectionError(
                f"Cannot connect to embedding server at {self._url}."
            )
        except httpx.TimeoutException:
            raise ConnectionError(
                f"Request to embedding server at {self._url} timed out."
            )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Embedding request failed with status {exc.response.status_code}: "
                f"{exc.response.text}"
            ) from exc

        data = response.json()

        if self._provider == "openai":
            if "data" not in data:
                raise RuntimeError(
                    f"Unexpected response from embedding server — missing 'data' key. "
                    f"Got: {list(data.keys())}"
                )
            try:
                return [item["embedding"] for item in data["data"]]
            except (KeyError, TypeError) as exc:
                raise RuntimeError(
                    f"Malformed embedding response — expected 'embedding' key in each item. "
                    f"First item: {data['data'][0] if data['data'] else '<empty>'}"
                ) from exc
        else:
            if "embeddings" not in data:
                raise RuntimeError(
                    f"Unexpected response from embedding server — missing 'embeddings' key. "
                    f"Got: {list(data.keys())}"
                )
            return data["embeddings"]
