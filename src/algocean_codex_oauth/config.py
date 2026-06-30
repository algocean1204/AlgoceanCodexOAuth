"""Internal configuration for Codex CLI invocations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

SandboxMode = Literal["read-only", "workspace-write", "danger-full-access"]


@dataclass(frozen=True)
class AlgoceanCodexConfig:
    """Settings passed to the Codex CLI for each invocation."""

    model: str = "gpt-5.5"
    codex_bin: str = "codex"
    timeout_sec: int = 180
    sandbox: SandboxMode = "read-only"
    workdir: str | None = None
    ephemeral: bool = True
    ignore_user_config: bool = True
    ignore_rules: bool = True
    skip_git_repo_check: bool = True
    require_chatgpt_login: bool = True
    allow_codex_access_token_env: bool = False
    extra_env: dict[str, str] = field(default_factory=dict)

    @classmethod
    def chat(cls, *, model: str = "gpt-5.5", timeout_sec: int = 180) -> AlgoceanCodexConfig:
        return cls(
            model=model,
            workdir=None,
            sandbox="read-only",
            ephemeral=True,
            timeout_sec=timeout_sec,
        )

    @classmethod
    def repo_read(
        cls,
        workdir: str,
        *,
        model: str = "gpt-5.5",
        timeout_sec: int = 300,
    ) -> AlgoceanCodexConfig:
        return cls(
            model=model,
            workdir=workdir,
            sandbox="read-only",
            ephemeral=True,
            ignore_user_config=False,
            ignore_rules=False,
            timeout_sec=timeout_sec,
        )

    @classmethod
    def repo_write(
        cls,
        workdir: str,
        *,
        model: str = "gpt-5.5",
        timeout_sec: int = 600,
    ) -> AlgoceanCodexConfig:
        return cls(
            model=model,
            workdir=workdir,
            sandbox="workspace-write",
            ephemeral=False,
            ignore_user_config=False,
            ignore_rules=False,
            timeout_sec=timeout_sec,
        )
