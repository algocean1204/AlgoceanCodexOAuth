"""OpenAI-style tool calling over `codex exec --output-schema`.

The Codex CLI has no tool-call API, so `bind_tools` is emulated: the tool
specs go into the prompt, a fixed output schema forces the model to answer in
`{content, tool_calls}` form, and the result is decoded back into
``AIMessage.tool_calls``. That keeps LangGraph agents working unchanged.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from langchain_core.utils.function_calling import convert_to_openai_tool

from .errors import AlgoceanCodexOAuthError

#: Response shape the model must emit when tools are bound.
TOOL_CALL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "content": {
            "type": "string",
            "description": "Reply for the user. Empty string when calling tools.",
        },
        "tool_calls": {
            "type": "array",
            "description": "Tools to call now. Empty array when answering directly.",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Exact tool name"},
                    "arguments": {
                        "type": "string",
                        "description": "Arguments as a JSON object string",
                    },
                },
                "required": ["name", "arguments"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["content", "tool_calls"],
    "additionalProperties": False,
}


def normalize_tools(tools: Sequence[Any]) -> list[dict[str, Any]]:
    """Accept anything ChatOpenAI.bind_tools accepts → OpenAI tool dicts."""
    return [convert_to_openai_tool(tool) for tool in tools]


def build_tool_preamble(
    tools: list[dict[str, Any]],
    tool_choice: Any = None,
) -> str:
    """Prompt block describing the bound tools and the required answer shape."""
    lines = [
        "You can call tools. Answer ONLY with the JSON object required by the "
        "output schema: {\"content\": string, \"tool_calls\": array}.",
        "",
        "Available tools:",
    ]

    for tool in tools:
        function = tool.get("function", tool)
        name = function.get("name", "")
        description = (function.get("description") or "").strip()
        parameters = function.get("parameters") or {"type": "object", "properties": {}}
        lines.append(f"- {name}: {description}")
        lines.append(f"  parameters: {json.dumps(parameters, ensure_ascii=False)}")

    lines.extend(
        [
            "",
            "Rules:",
            "- To call tools, put them in tool_calls and leave content empty.",
            "- `arguments` must be a JSON object encoded as a string, matching that tool's parameters.",
            "- When you already have every tool result you need, answer in content and leave tool_calls empty.",
            "- `content` holds the answer text itself — never put another JSON object inside it.",
            "- Never invent a tool name that is not listed above.",
        ]
    )

    requirement = _tool_choice_requirement(tool_choice, tools)
    if requirement:
        lines.append(f"- {requirement}")

    return "\n".join(lines)


def _tool_choice_requirement(tool_choice: Any, tools: list[dict[str, Any]]) -> str:
    if tool_choice in (None, "auto"):
        return ""
    if tool_choice == "none":
        return "Do NOT call any tool this turn; answer in content."
    if tool_choice in ("any", "required"):
        return "You MUST call at least one tool this turn."

    name = tool_choice
    if isinstance(tool_choice, dict):
        name = tool_choice.get("function", {}).get("name") or tool_choice.get("name")

    known = {t.get("function", t).get("name") for t in tools}
    if name in known:
        return f"You MUST call the tool `{name}` this turn."
    raise AlgoceanCodexOAuthError(f"tool_choice={tool_choice!r} names a tool that was not bound.")


def parse_tool_response(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Decode the schema-constrained reply into (content, LangChain tool_calls)."""
    stripped = (text or "").strip()
    if not stripped:
        return "", []

    decoded = _decode_payload(stripped)
    if decoded is None:
        # Schema enforcement slipped (or the model answered in plain prose).
        # Prefer degrading to a normal answer over blowing up the graph.
        return stripped, []

    content, tool_calls = decoded

    # Models sometimes wrap the payload inside `content` a second time, which
    # would surface raw JSON to the user. Peel the extra layers off.
    depth = 0
    while not tool_calls and depth < 3:
        inner = _decode_payload(content.strip())
        if inner is None:
            break
        content, tool_calls = inner
        depth += 1

    return content, tool_calls


def _decode_payload(text: str) -> tuple[str, list[dict[str, Any]]] | None:
    """Return (content, tool_calls) when `text` is the wrapper object, else None."""
    if not text.startswith("{"):
        return None

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict) or not ("content" in payload or "tool_calls" in payload):
        return None

    raw_calls = payload.get("tool_calls") or []
    if not isinstance(raw_calls, list):
        raw_calls = []

    tool_calls: list[dict[str, Any]] = []
    for index, call in enumerate(raw_calls):
        if not isinstance(call, dict):
            continue
        name = call.get("name")
        if not name:
            continue
        tool_calls.append(
            {
                "name": str(name),
                "args": _decode_arguments(call.get("arguments")),
                "id": f"call_{index}",
                "type": "tool_call",
            }
        )

    return str(payload.get("content") or ""), tool_calls


def _decode_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str) or not arguments.strip():
        return {}
    try:
        decoded = json.loads(arguments)
    except json.JSONDecodeError:
        return {"__raw__": arguments}
    return decoded if isinstance(decoded, dict) else {"__raw__": arguments}
