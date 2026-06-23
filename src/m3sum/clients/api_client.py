from __future__ import annotations

from openai import OpenAI

from m3sum.config import ApiCredentials


def build_openai_client(creds: ApiCredentials) -> OpenAI:
    return OpenAI(api_key=creds.api_key, base_url=creds.base_url)
