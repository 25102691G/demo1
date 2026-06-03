from __future__ import annotations

import json
import os
import re
from typing import Any


class DeepSeekClient:
    """Small OpenAI-compatible DeepSeek client returning parsed JSON objects."""

    def __init__(self) -> None:
        self.model = os.getenv("DEEPSEEK_MODEL")
        self.base_url = os.getenv("DEEPSEEK_BASE_URL")
        self.api_key = os.getenv("DEEPSEEK_API_KEY")

        if not self.model or not self.base_url or not self.api_key:
            raise ValueError("Missing DEEPSEEK_MODEL / DEEPSEEK_BASE_URL / DEEPSEEK_API_KEY")

        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - depends on optional local package
            raise RuntimeError("openai Python SDK is required to use DeepSeekClient.") from exc

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:
            raise RuntimeError(f"DeepSeek API request failed: {exc}") from exc

        try:
            content = response.choices[0].message.content
        except Exception as exc:
            raise ValueError("DeepSeek response did not include message content.") from exc

        if not content or not content.strip():
            raise ValueError("DeepSeek returned an empty response.")

        data = _parse_json_object(content)
        if not isinstance(data, dict):
            raise ValueError("DeepSeek JSON response must be an object.")
        return data


def _parse_json_object(content: str) -> Any:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"DeepSeek returned non-JSON content: {exc}") from exc
