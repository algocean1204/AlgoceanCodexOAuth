"""Multi-turn helper — thin wrapper over AlgoceanCodexOAuth codex_resume mode."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage

from .auth_mode import api_key, oauth
from .chat_model import AlgoceanCodexOAuth
from .errors import AlgoceanCodexOAuthError


class AlgoceanCodexSession:
    """Codex exec resume session.

    Prefer using ``AlgoceanCodexOAuth(thread_mode='codex_resume', ephemeral=False)``
    directly for ChatOpenAI-like usage. This class remains as a convenience wrapper.
    """

    def __init__(self, llm: AlgoceanCodexOAuth):
        if llm.auth == api_key:
            raise AlgoceanCodexOAuthError("AlgoceanCodexSession requires auth=oauth.")
        if llm.thread_mode != "codex_resume":
            llm = llm.model_copy(update={"auth": oauth, "thread_mode": "codex_resume", "ephemeral": False})
        self.llm = llm

    @property
    def thread_id(self) -> str | None:
        return self.llm.thread_id

    async def ainvoke(
        self,
        messages: list[BaseMessage],
        *,
        output_schema: dict[str, Any] | None = None,
    ) -> AIMessage:
        return await self.llm.ainvoke(messages, output_schema=output_schema)

    def invoke(
        self,
        messages: list[BaseMessage],
        *,
        output_schema: dict[str, Any] | None = None,
    ) -> AIMessage:
        return self.llm.invoke(messages, output_schema=output_schema)

    def reset(self) -> None:
        self.llm.reset_thread()
