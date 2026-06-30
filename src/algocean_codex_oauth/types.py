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
