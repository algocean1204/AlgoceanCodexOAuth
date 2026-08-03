"""JSON Schema helpers for Codex structured output."""

from __future__ import annotations

from typing import Any


# Strict structured output rejects these; Pydantic emits `default` for every
# field with a default value, including Optional ones.
_UNSUPPORTED_KEYWORDS = ("default",)


def normalize_codex_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Ensure schema satisfies Codex/OpenAI structured output requirements.

    Strict mode demands `additionalProperties: false` and — the part Pydantic
    breaks — *every* property listed in `required`. An Optional field left out
    of `required` makes the backend reject the whole request as
    `invalid_json_schema`.
    """

    def walk(node: Any) -> Any:
        if isinstance(node, list):
            return [walk(item) for item in node]
        if not isinstance(node, dict):
            return node

        updated = {
            key: walk(value) for key, value in node.items() if key not in _UNSUPPORTED_KEYWORDS
        }

        if updated.get("type") == "object" or "properties" in updated:
            updated.setdefault("type", "object")
            updated["additionalProperties"] = False
            properties = updated.get("properties")
            if isinstance(properties, dict):
                updated["required"] = list(properties)

        return updated

    return walk(dict(schema))
