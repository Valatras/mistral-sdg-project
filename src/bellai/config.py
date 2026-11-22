"""Mistral configuration shared by the CLI and the agents."""

import os
from functools import cached_property

from mistralai.client import Mistral


class Models:
    def __init__(self, embedding_model: str, chat_model: str):
        self.embedding_model = embedding_model
        self.chat_model = chat_model

    @cached_property
    def client(self) -> Mistral:
        key = os.getenv("MISTRAL_API_KEY", "").strip()
        if not key:
            raise ValueError("Set MISTRAL_API_KEY in .env or your environment.")
        return Mistral(api_key=key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        client = self.client
        try:
            response = client.embeddings.create(model=self.embedding_model, inputs=texts)
            rows = sorted(response.data, key=lambda row: row.index)
            if [row.index for row in rows] != list(range(len(texts))):
                raise ValueError("Unexpected embedding response indices.")
            return [list(row.embedding) for row in rows]
        except Exception as exc:
            raise ValueError(f"Mistral embedding request failed ({type(exc).__name__}).") from None

    def chat(self, system: str, payload: str) -> str:
        client = self.client
        try:
            response = client.chat.complete(
                model=self.chat_model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": payload}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            if not isinstance(content, str):
                raise ValueError("Unexpected response content.")
            return content
        except Exception as exc:
            raise ValueError(f"Mistral chat request failed ({type(exc).__name__}).") from None


def configured_models() -> Models:
    return Models(
        os.getenv("MISTRAL_EMBED_MODEL", "mistral-embed"),
        os.getenv("MISTRAL_CHAT_MODEL", "mistral-small-latest"),
    )
