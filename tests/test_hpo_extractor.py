from __future__ import annotations

import json

from skill_engine.hpo_extractor import (
    HPO_EXTRACTION_SYSTEM_PROMPT,
    HpoExtractor,
    HpoResources,
    _load_definition2id,
    phenotypes_to_positive_features,
)


class MockDeepSeekClient:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls: list[tuple[str, str]] = []

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, object]:
        self.calls.append((system_prompt, user_prompt))
        return self.payload


def test_hpo_extractor_import_does_not_require_torch_or_transformers() -> None:
    assert HpoExtractor is not None
    assert HpoResources is not None


def test_extract_phenotypes_uses_deepseek_client() -> None:
    client = MockDeepSeekClient(
        {
            "phenotypes": [
                {"phenotype": "腹痛"},
                {"phenotype": "腹泻"},
                {"phenotype": "腹痛"},
            ]
        }
    )
    extractor = HpoExtractor(
        HpoResources(
            model=None,
            tokenizer=None,
            definition2id={},
            definition_embeddings=None,
            definition_keys=[],
        )
    )

    phenotypes = extractor.extract_phenotypes("患者腹痛、腹泻。", client)

    assert phenotypes == ["腹痛", "腹泻"]
    assert client.calls
    assert client.calls[0][0] == HPO_EXTRACTION_SYSTEM_PROMPT
    assert json.loads(client.calls[0][1])["clinical_text"] == "患者腹痛、腹泻。"


def test_extract_from_text_returns_empty_result_without_phenotypes() -> None:
    client = MockDeepSeekClient({"phenotypes": []})
    extractor = HpoExtractor(
        HpoResources(
            model=None,
            tokenizer=None,
            definition2id={},
            definition_embeddings=None,
            definition_keys=[],
        )
    )

    result = extractor.extract_from_text("无明显症状。", client)

    assert result == {
        "phenotypes": [],
        "hpo_codes": [],
        "hpo_descriptions": [],
        "hpo_mappings": [],
    }


def test_phenotypes_to_positive_features_uses_symptom_features() -> None:
    result = phenotypes_to_positive_features(["鑵圭棝", "鑵规郴", "鑵圭棝", ""])

    assert result == {
        "symptoms": [
            {"name": "鑵圭棝", "weight": 0.2},
            {"name": "鑵规郴", "weight": 0.2},
        ]
    }


def test_load_definition2id_accepts_list_and_string_values(tmp_path) -> None:
    path = tmp_path / "definition2id.json"
    path.write_text(
        json.dumps(
            {
                "腹痛": ["HP:0002027"],
                "腹泻": "HP:0002014",
                "空值": [],
                "无值": None,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    assert _load_definition2id(path) == {
        "腹痛": "HP:0002027",
        "腹泻": "HP:0002014",
    }
