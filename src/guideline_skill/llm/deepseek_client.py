from __future__ import annotations

from typing import Any

from skill_engine.llm_client import OpenAICompatibleJsonChatClient, load_deepseek_config_from_env


class DeepSeekClient:
    """Small OpenAI-compatible DeepSeek client returning parsed JSON objects."""

    def __init__(self) -> None:
        self.config = load_deepseek_config_from_env()
        self.model = self.config.model
        self.base_url = self.config.base_url
        self.api_key = self.config.api_key
        self.client = OpenAICompatibleJsonChatClient(self.config)

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        return self.client.chat_json(system_prompt, user_prompt)
