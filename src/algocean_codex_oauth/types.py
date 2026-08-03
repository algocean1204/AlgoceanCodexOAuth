"""Result types for Codex CLI invocations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CodexExecResult:
    text: str
    usage: dict[str, Any] | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    thread_id: str | None = None


def to_usage_metadata(usage: dict[str, Any] | None) -> dict[str, Any] | None:
    """Codex `turn.completed` usage → LangChain ``AIMessage.usage_metadata``.

    Without this, token accounting in LangGraph, LangSmith and callbacks stays
    empty where ChatOpenAI reports numbers.
    """
    if not usage:
        return None

    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    cached = int(usage.get("cached_input_tokens") or 0)
    reasoning = int(usage.get("reasoning_output_tokens") or 0)

    metadata: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    if cached:
        metadata["input_token_details"] = {"cache_read": cached}
    if reasoning:
        metadata["output_token_details"] = {"reasoning": reasoning}
    return metadata


def to_token_usage(usage: dict[str, Any] | None) -> dict[str, Any] | None:
    """Codex usage → the ``token_usage`` shape ChatOpenAI puts in response_metadata."""
    if not usage:
        return None

    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    return {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "prompt_tokens_details": {"cached_tokens": int(usage.get("cached_input_tokens") or 0)},
        "completion_tokens_details": {
            "reasoning_tokens": int(usage.get("reasoning_output_tokens") or 0)
        },
    }
