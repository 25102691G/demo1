from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .card_store import CardStore, CardStoreError
from .schemas import SchemaValidationError, load_json_schema, validate_json
from .utils import clean_text


class SkillLoadError(ValueError):
    """Raised when a skill pack cannot be loaded or validated."""


@dataclass(frozen=True)
class SkillPack:
    skill_dir: Path
    skill: dict[str, Any]
    skill_id: str
    disease_name: str
    cards: list[dict[str, Any]]
    cards_by_id: dict[str, dict[str, Any]]

    @property
    def card_store(self) -> CardStore:
        return CardStore(cards=self.cards, cards_by_id=self.cards_by_id)


def discover_skill_dirs(skills_dir: Path, *, skill_filename: str = "skill.yaml") -> list[Path]:
    root = Path(skills_dir)
    if not root.exists():
        return []
    return sorted(path.parent for path in root.glob(f"*/{skill_filename}") if path.is_file())


def load_skill_pack(
    skill_dir: Path,
    skill_schema_path: Path,
    *,
    skill_filename: str = "skill.yaml",
) -> SkillPack:
    directory = Path(skill_dir)
    skill_path = directory / skill_filename
    if not skill_path.exists():
        raise SkillLoadError(f"{directory}: missing {skill_filename}")
    try:
        with skill_path.open("r", encoding="utf-8-sig") as handle:
            skill = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise SkillLoadError(f"{skill_path}: invalid YAML: {exc}") from exc
    if not isinstance(skill, dict):
        raise SkillLoadError(f"{skill_path}: {skill_filename} must be an object")

    skill_id = clean_text((skill.get("metadata") or {}).get("skill_id")) or directory.name
    try:
        validate_json(skill, load_json_schema(Path(skill_schema_path)), label=f"skill_pack:{skill_id}")
    except SchemaValidationError as exc:
        raise SkillLoadError(f"{skill_id} ({skill_path}): {exc}") from exc

    cards_path_value = clean_text((skill.get("knowledge_base") or {}).get("cards_path"))
    cards_path = directory / cards_path_value
    try:
        store = CardStore.from_jsonl(cards_path)
    except CardStoreError as exc:
        raise SkillLoadError(f"{skill_id} ({cards_path}): {exc}") from exc

    disease_name = clean_text((skill.get("metadata") or {}).get("disease_name"))
    pack = SkillPack(
        skill_dir=directory,
        skill=skill,
        skill_id=skill_id,
        disease_name=disease_name,
        cards=store.cards,
        cards_by_id=store.cards_by_id,
    )
    _validate_cross_references(pack)
    return pack


def load_skill_packs(
    skills_dir: Path,
    skill_schema_path: Path,
    *,
    strict: bool = False,
    skill_filename: str = "skill.yaml",
) -> tuple[list[SkillPack], list[str]]:
    packs: list[SkillPack] = []
    errors: list[str] = []
    for skill_dir in discover_skill_dirs(Path(skills_dir), skill_filename=skill_filename):
        try:
            packs.append(load_skill_pack(skill_dir, skill_schema_path, skill_filename=skill_filename))
        except SkillLoadError as exc:
            if strict:
                raise
            errors.append(str(exc))
    if not packs and not errors:
        errors.append(f"{skills_dir}: no {skill_filename} found")
    return packs, errors


def _validate_cross_references(pack: SkillPack) -> None:
    skill = pack.skill
    errors: list[str] = []
    workflow = skill.get("workflow") or {}
    steps = workflow.get("steps") or []
    step_ids = {clean_text(step.get("step_id")) for step in steps if isinstance(step, dict)}
    entrypoint = clean_text(workflow.get("entrypoint"))
    if entrypoint not in step_ids:
        errors.append(f"workflow.entrypoint {entrypoint!r} does not exist")

    subskills = skill.get("subskills") or []
    subskill_ids = {
        clean_text(subskill.get("subskill_id")) for subskill in subskills if isinstance(subskill, dict)
    }
    template_ids = set((skill.get("output_templates") or {}).keys())
    card_ids = set(pack.cards_by_id.keys())

    for step in steps:
        if not isinstance(step, dict):
            continue
        step_id = clean_text(step.get("step_id"))
        for transition in step.get("transitions") or []:
            if not isinstance(transition, dict):
                continue
            target = clean_text(transition.get("to"))
            if target and target not in step_ids:
                errors.append(f"workflow step {step_id!r} transition.to {target!r} does not exist")
        config = step.get("config") or {}
        if not isinstance(config, dict):
            continue
        subskill_ref = clean_text(config.get("subskill_ref"))
        if subskill_ref and subskill_ref not in subskill_ids:
            errors.append(
                f"workflow step {step_id!r} config.subskill_ref {subskill_ref!r} does not exist"
            )
        output_template = clean_text(config.get("output_template"))
        if output_template and output_template not in template_ids:
            errors.append(
                f"workflow step {step_id!r} config.output_template {output_template!r} does not exist"
            )

    for subskill in subskills:
        if not isinstance(subskill, dict):
            continue
        subskill_id = clean_text(subskill.get("subskill_id"))
        selection = subskill.get("card_selection") or {}
        if not isinstance(selection, dict):
            continue
        for field in ("required", "optional"):
            for card_id in selection.get(field) or []:
                if clean_text(card_id) not in card_ids:
                    errors.append(
                        f"subskill {subskill_id!r} card_selection.{field} "
                        f"references unknown card_id {card_id!r}"
                    )
    if errors:
        message = "\n".join(errors)
        raise SkillLoadError(f"{pack.skill_id} ({pack.skill_dir}): {message}")
