from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol


class JsonChatClient(Protocol):
    def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class LlmConfig:
    api_key: str
    base_url: str
    model: str
    temperature: float = 0.0


def load_llm_config_from_env(
    *,
    api_key_envs: Sequence[str] = ("DEEPSEEK_API_KEY", "OPENAI_API_KEY"),
    base_url_env: str = "DEEPSEEK_BASE_URL",
    model_env: str = "DEEPSEEK_MODEL",
    default_base_url: str = "https://api.deepseek.com",
    default_model: str = "deepseek-chat",
    temperature: float = 0.0,
) -> LlmConfig:
    api_key = _first_env(api_key_envs)
    base_url = os.environ.get(base_url_env) or default_base_url
    model = os.environ.get(model_env) or default_model
    if not api_key:
        names = " / ".join(api_key_envs)
        raise ValueError(f"Missing LLM API key. Set one of: {names}")
    return LlmConfig(api_key=api_key, base_url=base_url, model=model, temperature=temperature)


def load_deepseek_config_from_env(*, temperature: float = 0.0) -> LlmConfig:
    model = os.getenv("DEEPSEEK_MODEL")
    base_url = os.getenv("DEEPSEEK_BASE_URL")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not model or not base_url or not api_key:
        raise ValueError("Missing DEEPSEEK_MODEL / DEEPSEEK_BASE_URL / DEEPSEEK_API_KEY")
    return LlmConfig(api_key=api_key, base_url=base_url, model=model, temperature=temperature)


class OpenAICompatibleJsonChatClient:
    """OpenAI-compatible JSON chat client shared by project LLM callers."""

    def __init__(self, config: LlmConfig) -> None:
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - depends on optional local package
            raise RuntimeError("openai Python SDK is required to use the LLM client.") from exc

        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.config.temperature,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            raise RuntimeError(f"LLM API request failed: {exc}") from exc

        content = _response_message_content(response)
        if not content or not content.strip():
            return {}
        return loads_json_object(content)


def loads_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned non-JSON content: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("LLM JSON response must be an object.")
    return data


def _response_message_content(response: Any) -> str | None:
    choices = getattr(response, "choices", None)
    if not choices:
        return None
    try:
        first_choice = choices[0]
    except (IndexError, TypeError):
        return None
    message = getattr(first_choice, "message", None)
    if message is None:
        return None
    return getattr(message, "content", None)


def _first_env(names: Sequence[str]) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None
