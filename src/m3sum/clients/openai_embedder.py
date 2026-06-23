from __future__ import annotations

from openai import OpenAI

from m3sum.config import ApiCredentials
from m3sum.clients.api_client import build_openai_client


class OpenAIEmbedder:
    def __init__(self, model: str, credentials: ApiCredentials):
        self.model = model
        self.client: OpenAI = build_openai_client(credentials)

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        if not texts:
            return []
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            resp = self.client.embeddings.create(model=self.model, input=chunk)
            all_vectors.extend(item.embedding for item in resp.data)
        return all_vectors

    def embed_one(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]
