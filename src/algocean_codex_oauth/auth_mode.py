"""Authentication mode constants and environment helpers."""

from __future__ import annotations

import os
from typing import Literal

AuthSetting = Literal["oauth", "api_key"]

oauth: AuthSetting = "oauth"
api_key: AuthSetting = "api_key"

API_KEY_ENV = "ALGOCEANCODEXOAUTH_API"
AUTH_ENV = "ALGOCEANCODEXOAUTH_AUTH"


def normalize_auth(value: AuthSetting | str) -> AuthSetting:
    if value not in (oauth, api_key):
        raise ValueError(f"auth must be oauth or api_key, got {value!r}")
    return value  # type: ignore[return-value]


def resolve_auth_from_env(default: AuthSetting = oauth) -> AuthSetting:
    raw = os.getenv(AUTH_ENV, default).strip().lower()
    return normalize_auth(raw)


def load_api_key_from_env() -> str:
    api_key_value = os.getenv(API_KEY_ENV, "").strip()
    if not api_key_value:
        raise ValueError(
            f"auth=api_key requires {API_KEY_ENV} in the environment or .env file."
        )
    return api_key_value
