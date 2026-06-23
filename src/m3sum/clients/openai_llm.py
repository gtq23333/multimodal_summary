from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from m3sum.config import ApiCredentials
from m3sum.clients.api_client import build_openai_client


class OpenAILLMClient:
    def __init__(self, model: str, credentials: ApiCredentials):
        self.model = model
        self.client: OpenAI = build_openai_client(credentials)

    def chat_json(self, system: str, user: str) -> dict[str, Any]:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        content = resp.choices[0].message.content or "{}"
        return json.loads(content)

    def chat_text(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content or ""
