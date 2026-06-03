from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from guideline_skill.llm.deepseek_client import DeepSeekClient


def test_deepseek_client_reports_missing_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    with pytest.raises(ValueError, match="Missing DEEPSEEK_MODEL / DEEPSEEK_BASE_URL / DEEPSEEK_API_KEY"):
        DeepSeekClient()


def test_deepseek_client_parses_plain_json(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_openai(monkeypatch, '{"ok": true, "value": 1}')
    _set_deepseek_env(monkeypatch)

    client = DeepSeekClient()

    assert client.chat_json("system", "user") == {"ok": True, "value": 1}


def test_deepseek_client_parses_fenced_json(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_openai(monkeypatch, '```json\n{"ok": true}\n```')
    _set_deepseek_env(monkeypatch)

    client = DeepSeekClient()

    assert client.chat_json("system", "user") == {"ok": True}


def test_deepseek_client_raises_for_non_json_response(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_openai(monkeypatch, "not json")
    _set_deepseek_env(monkeypatch)
    client = DeepSeekClient()

    with pytest.raises(ValueError, match="non-JSON"):
        client.chat_json("system", "user")


def _set_deepseek_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-test")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")


def _install_fake_openai(monkeypatch: pytest.MonkeyPatch, content: str) -> None:
    class FakeCompletions:
        def create(self, **kwargs: object) -> object:
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=content),
                    )
                ]
            )

    class FakeChat:
        completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            self.chat = FakeChat()

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
