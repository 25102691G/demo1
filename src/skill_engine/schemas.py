from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError


class SchemaValidationError(ValueError):
    """Raised when an instance fails JSON Schema validation."""


def load_json_schema(path: Path) -> dict[str, Any]:
    schema_path = Path(path)
    try:
        with schema_path.open("r", encoding="utf-8-sig") as handle:
            schema = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{schema_path}: invalid JSON schema: {exc.msg}") from exc
    if not isinstance(schema, dict):
        raise ValueError(f"{schema_path}: expected JSON object schema")
    Draft202012Validator.check_schema(schema)
    return schema


def validate_json(instance: dict[str, Any], schema: dict[str, Any], *, label: str) -> None:
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(instance), key=lambda error: list(error.path))
    if not errors:
        return
    raise SchemaValidationError(_format_validation_error(errors[0], label=label))


def build_required_defaults(schema: Mapping[str, Any]) -> dict[str, Any]:
    """根据 JSON Schema 的 required 字段生成默认字典。"""
    properties = schema.get("properties", {})
    defaults: dict[str, Any] = {}

    for field in schema.get("required", []):
        field_schema = properties.get(field, {})
        defaults[field] = copy.deepcopy(field_schema.get("default"))

    return defaults


def _format_validation_error(error: ValidationError, *, label: str) -> str:
    return (
        f"{label} schema validation failed: {error.message}; "
        f"json path: {_format_path(error.path)}; "
        f"schema path: {_format_path(error.schema_path)}"
    )


def _format_path(parts: Iterable[Any]) -> str:
    tokens = [str(part) for part in parts]
    return "$" if not tokens else "$." + ".".join(tokens)
