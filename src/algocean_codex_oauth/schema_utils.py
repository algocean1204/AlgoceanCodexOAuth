"""JSON Schema helpers for Codex structured output."""

from __future__ import annotations

from typing import Any


def normalize_codex_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Ensure schema satisfies Codex/OpenAI structured output requirements."""

    def walk(node: Any) -> Any:
        if isinstance(node, list):
            return [walk(item) for item in node]
        if not isinstance(node, dict):
            return node

        updated = {key: walk(value) for key, value in node.items()}

        if updated.get("type") == "object" or "properties" in updated:
            updated.setdefault("type", "object")
            updated["additionalProperties"] = False

        return updated

    return walk(dict(schema))
