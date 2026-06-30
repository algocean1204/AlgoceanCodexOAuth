"""Convert LangChain messages into Codex exec prompts."""

from __future__ import annotations

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
        role = _role_name(message)
        content = stringify_content(message.content)
        if not content.strip():
            continue
        blocks.append(f"[{role}]\n{content}")

    if not blocks:
        raise ValueError("At least one non-empty message is required.")

    return "\n\n".join(blocks)


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
