"""ChatGPT OAuth validation and API key environment blocking."""

from __future__ import annotations

import os
import subprocess

from .config import AlgoceanCodexConfig
from .errors import AlgoceanCodexOAuthError

_OAUTH_MARKERS = ("chatgpt", "oauth", "account", "workspace")
_API_KEY_MARKERS = ("api key", "apikey", "api_key")


def build_safe_env(config: AlgoceanCodexConfig) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    env.pop("CODEX_API_KEY", None)

    if not config.allow_codex_access_token_env:
        env.pop("CODEX_ACCESS_TOKEN", None)

    env.update(config.extra_env)
    return env


def assert_chatgpt_oauth_login(config: AlgoceanCodexConfig) -> None:
    process = subprocess.run(
        [config.codex_bin, "login", "status"],
        env=build_safe_env(config),
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    output = f"{process.stdout}\n{process.stderr}".strip()
    normalized = output.lower()

    if process.returncode != 0:
        raise AlgoceanCodexOAuthError(
            "Codex ChatGPT OAuth login is required.\n"
            "Run:\n"
            "  codex logout\n"
            "  unset OPENAI_API_KEY CODEX_API_KEY\n"
            "  codex login\n\n"
            f"Current status:\n{output}"
        )

    if any(marker in normalized for marker in _API_KEY_MARKERS):
        raise AlgoceanCodexOAuthError(
            "Codex appears to be authenticated via API key. "
            "Use ChatGPT OAuth to consume subscription credits instead.\n"
            "Run:\n"
            "  codex logout\n"
            "  unset OPENAI_API_KEY CODEX_API_KEY\n"
            "  codex login\n\n"
            f"Current status:\n{output}"
        )

    if not any(marker in normalized for marker in _OAUTH_MARKERS):
        raise AlgoceanCodexOAuthError(
            "Could not confirm ChatGPT OAuth login from `codex login status`.\n"
            f"Current status:\n{output}"
        )
