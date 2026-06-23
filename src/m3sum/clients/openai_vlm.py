from __future__ import annotations

import base64
from pathlib import Path

from openai import OpenAI

from m3sum.config import ApiCredentials
from m3sum.clients.api_client import build_openai_client


class OpenAIVLMClient:
    def __init__(self, model: str, credentials: ApiCredentials):
        self.model = model
        self.client: OpenAI = build_openai_client(credentials)

    def describe_image(self, image_path: str, caption: str = "") -> str:
        path = Path(image_path)
        if not path.is_file():
            return f"[图片不可用: {path.name}]"

        data = base64.b64encode(path.read_bytes()).decode("utf-8")
        suffix = path.suffix.lower().lstrip(".")
        mime = "jpeg" if suffix in ("jpg", "jpeg") else suffix

        prompt = (
            "请用中文简洁描述这张数模论文图表的核心信息（趋势、结构或结论），"
            "50-120字，不要重复图注字面内容。"
        )
        if caption:
            prompt += f"\n图注参考：{caption}"

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/{mime};base64,{data}"},
                        },
                    ],
                }
            ],
            max_tokens=300,
            temperature=0.2,
        )
        return (resp.choices[0].message.content or "").strip()
