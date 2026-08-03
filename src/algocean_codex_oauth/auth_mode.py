"""Authentication mode constants and environment helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

AuthSetting = Literal["oauth", "api_key"]

oauth: AuthSetting = "oauth"
api_key: AuthSetting = "api_key"

API_KEY_ENV = "ALGOCEANCODEXOAUTH_API"
AUTH_ENV = "ALGOCEANCODEXOAUTH_AUTH"

DOTENV_FILE = ".env"


def normalize_auth(value: AuthSetting | str) -> AuthSetting:
    if value not in (oauth, api_key):
        raise ValueError(f"auth must be oauth or api_key, got {value!r}")
    return value  # type: ignore[return-value]


def read_dotenv(path: str | os.PathLike[str] = DOTENV_FILE) -> dict[str, str]:
    """Parse a `KEY=VALUE` .env file. Missing or unreadable file → empty dict.

    Kept dependency-free on purpose: this format needs a few lines, not a package.
    """
    values: dict[str, str] = {}
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return values

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip("\"'")
    return values


def read_setting(name: str) -> str:
    """Real environment wins; a local .env is the personal-project fallback."""
    value = os.getenv(name, "").strip()
    if value:
        return value
    return read_dotenv().get(name, "").strip()


def resolve_auth_from_env(default: AuthSetting = oauth) -> AuthSetting:
    raw = read_setting(AUTH_ENV)
    if raw:
        return normalize_auth(raw.lower())

    # Setting the key but forgetting AUTH is the common slip. Falling back to
    # oauth there looks fine locally and then dies on a machine without codex.
    if read_setting(API_KEY_ENV):
        return api_key

    return default


def load_api_key_from_env() -> str:
    api_key_value = read_setting(API_KEY_ENV)
    if not api_key_value:
        raise ValueError(
            f"auth=api_key requires {API_KEY_ENV} in the environment "
            f"or in a {DOTENV_FILE} file in the working directory."
        )
    return api_key_value
