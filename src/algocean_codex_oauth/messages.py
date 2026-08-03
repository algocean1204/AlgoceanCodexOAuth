"""Convert LangChain messages into Codex exec prompts."""

from __future__ import annotations

import json

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


def messages_to_prompt(
    messages: list[BaseMessage],
    *,
    preamble: str | None = None,
) -> str:
    """Serialize full chat history into a single prompt (ChatOpenAI-style context)."""
    blocks: list[str] = []

    if preamble and preamble.strip():
        blocks.append(f"[System]\n{preamble.strip()}")

    for message in messages:
        block = _render(message)
        if block:
            blocks.append(block)

    if not blocks:
        raise ValueError("At least one non-empty message is required.")

    return "\n\n".join(blocks)


def _render(message: BaseMessage) -> str:
    role = _role_name(message)
    content = stringify_content(message.content).strip()

    if isinstance(message, ToolMessage):
        name = getattr(message, "name", None) or "tool"
        label = f"[Tool result: {name}]"
        if message.tool_call_id:
            label = f"[Tool result: {name} (id={message.tool_call_id})]"
        return f"{label}\n{content}" if content else f"{label}\n(empty)"

    # An assistant turn that only calls tools has empty content — dropping it
    # would erase the call from history and the agent would loop forever.
    tool_calls = getattr(message, "tool_calls", None) if isinstance(message, AIMessage) else None
    if tool_calls:
        rendered = ", ".join(
            f"{call.get('name')}({json.dumps(call.get('args') or {}, ensure_ascii=False)})"
            for call in tool_calls
        )
        body = f"{content}\n" if content else ""
        return f"[{role}]\n{body}Called tools: {rendered}"

    if not content:
        return ""
    return f"[{role}]\n{content}"


def messages_to_delta_prompt(messages: list[BaseMessage], previous_count: int) -> str:
    """Serialize only new messages since the previous invoke (codex exec resume)."""
    if previous_count < 0 or previous_count > len(messages):
        raise ValueError(f"invalid previous_count={previous_count} for {len(messages)} messages")

    delta = messages[previous_count:]
    if not delta:
        last = messages[-1]
        return messages_to_prompt([last])
    return messages_to_prompt(delta)


def _role_name(message: BaseMessage) -> str:
    if isinstance(message, SystemMessage):
        return "System"
    if isinstance(message, HumanMessage):
        return "User"
    if isinstance(message, AIMessage):
        return "Assistant"
    if isinstance(message, ToolMessage):
        return "Tool"
    return message.type.capitalize()


def stringify_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                else:
                    parts.append(str(block))
            else:
                parts.append(str(block))
        return "\n".join(part for part in parts if part)
    return str(content)
