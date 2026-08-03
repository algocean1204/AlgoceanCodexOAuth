"""Codex model catalog — available models and their reasoning efforts.

The catalog is read at runtime from ``codex debug models``.

``ignore_user_config`` must match the flag the run will actually use, because
``codex exec --ignore-user-config`` also ignores ``model_catalog_json``: custom
providers in ``~/.codex/config.toml`` are only reachable when it is False.
Listing them otherwise sends users to models the backend rejects with HTTP 400.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from typing import Literal

from .errors import AlgoceanCodexOAuthError

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"]

#: Full enum the Codex backend accepts. Per-model support is narrower — see :class:`ModelInfo`.
KNOWN_EFFORTS: tuple[str, ...] = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
    "ultra",
)

# The catalog omits "none" from supported_reasoning_levels, but the backend
# reports it as valid for every reasoning model.
_ALWAYS_ALLOWED: tuple[str, ...] = ("none",)

CATALOG_TIMEOUT_SEC = 20


@dataclass(frozen=True)
class ModelInfo:
    """One entry of the Codex model catalog."""

    slug: str
    display_name: str
    efforts: tuple[str, ...]
    default_effort: str | None = None
    context_window: int | None = None
    supported_in_api: bool = True
    visibility: str = "list"

    @property
    def listed(self) -> bool:
        return self.visibility == "list"

    def supports(self, effort: str) -> bool:
        return effort in self.efforts or effort in _ALWAYS_ALLOWED


# codex-cli 0.144.1 bundled catalog — used only when the CLI cannot be reached.
_FALLBACK_MODELS: tuple[ModelInfo, ...] = (
    ModelInfo("gpt-5.6-sol", "GPT-5.6-Sol", ("low", "medium", "high", "xhigh", "max", "ultra"), "low", 372000),
    ModelInfo("gpt-5.6-terra", "GPT-5.6-Terra", ("low", "medium", "high", "xhigh", "max", "ultra"), "medium", 372000),
    ModelInfo("gpt-5.6-luna", "GPT-5.6-Luna", ("low", "medium", "high", "xhigh", "max"), "medium", 372000),
    ModelInfo("gpt-5.5", "GPT-5.5", ("low", "medium", "high", "xhigh"), "medium", 272000),
    ModelInfo("gpt-5.4", "GPT-5.4", ("low", "medium", "high", "xhigh"), "medium", 272000),
    ModelInfo("gpt-5.4-mini", "GPT-5.4-Mini", ("low", "medium", "high", "xhigh"), "medium", 272000),
    ModelInfo("gpt-5.2", "GPT-5.2", ("low", "medium", "high", "xhigh"), "medium", 272000),
)

_CACHE: dict[tuple[str, bool], tuple[ModelInfo, ...]] = {}


def _parse_catalog(raw: str) -> tuple[ModelInfo, ...]:
    payload = json.loads(raw)
    entries = payload.get("models") or []
    models: list[ModelInfo] = []

    for entry in entries:
        slug = entry.get("slug")
        if not slug:
            continue
        efforts = tuple(
            str(level["effort"])
            for level in (entry.get("supported_reasoning_levels") or [])
            if isinstance(level, dict) and level.get("effort")
        )
        models.append(
            ModelInfo(
                slug=str(slug),
                display_name=str(entry.get("display_name") or slug),
                efforts=efforts,
                default_effort=entry.get("default_reasoning_level"),
                context_window=entry.get("context_window"),
                supported_in_api=bool(entry.get("supported_in_api", True)),
                visibility=str(entry.get("visibility") or "list"),
            )
        )

    return tuple(models)


def _query_catalog(codex_bin: str, *, bundled: bool) -> tuple[ModelInfo, ...]:
    cmd = [codex_bin, "debug", "models"]
    if bundled:
        cmd.append("--bundled")

    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=CATALOG_TIMEOUT_SEC,
        check=False,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        raise AlgoceanCodexOAuthError(
            f"`{' '.join(cmd)}` failed (exit {completed.returncode}): {completed.stderr.strip()[:300]}"
        )
    return _parse_catalog(completed.stdout)


def _load_catalog(codex_bin: str, ignore_user_config: bool) -> tuple[ModelInfo, ...]:
    if shutil.which(codex_bin) is None:
        return _FALLBACK_MODELS

    attempts = (True,) if ignore_user_config else (False, True)
    for bundled in attempts:
        try:
            models = _query_catalog(codex_bin, bundled=bundled)
        except (AlgoceanCodexOAuthError, OSError, subprocess.SubprocessError, json.JSONDecodeError, ValueError):
            continue
        if models:
            return models

    return _FALLBACK_MODELS


def clear_catalog_cache() -> None:
    """Drop the in-process catalog cache (use after changing codex config)."""
    _CACHE.clear()


def list_models(
    *,
    codex_bin: str = "codex",
    ignore_user_config: bool = True,
    include_hidden: bool = False,
    refresh: bool = False,
) -> list[ModelInfo]:
    """Return the model catalog, highest-priority first.

    ``ignore_user_config`` mirrors the codex flag the run will use — keep it in
    sync or the list will offer models the run cannot reach.
    """
    key = (codex_bin, ignore_user_config)
    if refresh:
        _CACHE.pop(key, None)

    if key not in _CACHE:
        _CACHE[key] = _load_catalog(codex_bin, ignore_user_config)

    models = _CACHE[key]
    if include_hidden:
        return list(models)
    return [model for model in models if model.listed]


def model_names(
    *,
    codex_bin: str = "codex",
    ignore_user_config: bool = True,
    include_hidden: bool = False,
) -> list[str]:
    """Return just the model slugs."""
    return [
        model.slug
        for model in list_models(
            codex_bin=codex_bin,
            ignore_user_config=ignore_user_config,
            include_hidden=include_hidden,
        )
    ]


def get_model(
    model: str,
    *,
    codex_bin: str = "codex",
    ignore_user_config: bool = True,
) -> ModelInfo | None:
    """Return catalog info for ``model``, or None when it is not in the catalog."""
    for info in list_models(
        codex_bin=codex_bin,
        ignore_user_config=ignore_user_config,
        include_hidden=True,
    ):
        if info.slug == model:
            return info
    return None


def supported_efforts(
    model: str,
    *,
    codex_bin: str = "codex",
    ignore_user_config: bool = True,
) -> tuple[str, ...]:
    """Reasoning efforts this model accepts. Empty tuple when the model is unknown."""
    info = get_model(model, codex_bin=codex_bin, ignore_user_config=ignore_user_config)
    return info.efforts if info else ()


def default_effort(
    model: str,
    *,
    codex_bin: str = "codex",
    ignore_user_config: bool = True,
) -> str | None:
    """The effort Codex uses for this model when none is requested."""
    info = get_model(model, codex_bin=codex_bin, ignore_user_config=ignore_user_config)
    return info.default_effort if info else None


def validate_effort(
    effort: str,
    *,
    model: str | None = None,
    codex_bin: str = "codex",
    ignore_user_config: bool = True,
) -> str:
    """Normalize and check a reasoning effort, raising before any network round-trip."""
    normalized = str(effort).strip().lower()

    if normalized not in KNOWN_EFFORTS:
        raise AlgoceanCodexOAuthError(
            f"Unknown reasoning_effort {effort!r}. Valid values: {', '.join(KNOWN_EFFORTS)}."
        )

    if model is None:
        return normalized

    info = get_model(model, codex_bin=codex_bin, ignore_user_config=ignore_user_config)
    # Unknown model → let the backend decide; the catalog may just be stale.
    if info is None or not info.efforts:
        return normalized

    if not info.supports(normalized):
        raise AlgoceanCodexOAuthError(
            f"reasoning_effort={normalized!r} is not supported by model {model!r}. "
            f"Supported: {', '.join(info.efforts)} (default: {info.default_effort})."
        )

    return normalized


def format_model_table(
    *,
    codex_bin: str = "codex",
    ignore_user_config: bool = True,
    include_hidden: bool = False,
) -> str:
    """Human-readable model/effort table for help output."""
    models = list_models(
        codex_bin=codex_bin,
        ignore_user_config=ignore_user_config,
        include_hidden=include_hidden,
    )
    if not models:
        return "No models found."

    width = max(len(model.slug) for model in models)
    lines = [f"{'MODEL'.ljust(width)}  {'DEFAULT':<8}  EFFORTS"]
    for model in models:
        efforts = ", ".join(model.efforts) or "-"
        lines.append(f"{model.slug.ljust(width)}  {str(model.default_effort or '-'):<8}  {efforts}")
    return "\n".join(lines)
