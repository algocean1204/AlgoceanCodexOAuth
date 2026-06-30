"""LangChain BaseChatModel implementation for Codex OAuth and OpenAI API key."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
from typing import Any, AsyncIterator, Literal, Optional, Union

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.utils.pydantic import is_basemodel_subclass
from pydantic import ConfigDict, Field, PrivateAttr

from .auth_mode import api_key, load_api_key_from_env, oauth, resolve_auth_from_env
from .auth_mode import AuthSetting
from .client import CodexExecClient
from .config import AlgoceanCodexConfig, SandboxMode
from .errors import AlgoceanCodexOAuthError
from .llm_mode import OAUTH_LLM_ONLY_PREAMBLE
from .messages import messages_to_delta_prompt, messages_to_prompt
from .schema_utils import normalize_codex_json_schema
from .types import CodexExecResult

StructuredSchema = Union[dict[str, Any], type]
ThreadMode = Literal["messages", "codex_resume"]


class AlgoceanCodexOAuth(BaseChatModel):
    """Drop-in ChatOpenAI replacement — local OAuth or deployed API key."""

    model: str = "gpt-5.5"
    auth: AuthSetting = oauth
    timeout: int = 180
    sandbox: SandboxMode = "read-only"
    workdir: Optional[str] = None
    ephemeral: bool = True
    codex_bin: str = "codex"
    require_chatgpt_login: bool = True
    allow_codex_access_token_env: bool = False
    ignore_user_config: bool = True
    ignore_rules: bool = True
    skip_git_repo_check: bool = True
    extra_env: dict[str, str] = Field(default_factory=dict)
    thread_mode: ThreadMode = "messages"

    _client: CodexExecClient | None = PrivateAttr(default=None)
    _api_backend: BaseChatModel | None = PrivateAttr(default=None)
    _thread_id: str | None = PrivateAttr(default=None)
    _last_message_count: int = PrivateAttr(default=0)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        if self._is_api_key_auth():
            self._init_api_key_backend()
            return
        self._init_oauth_backend()

    def _is_api_key_auth(self) -> bool:
        return self.auth == api_key

    def _init_api_key_backend(self) -> None:
        if self.thread_mode == "codex_resume":
            raise AlgoceanCodexOAuthError(
                "thread_mode='codex_resume' is only supported with auth=oauth."
            )
        if self.workdir is not None:
            raise AlgoceanCodexOAuthError(
                "workdir is only supported with auth=oauth. Use auth=oauth for repo modes."
            )

        try:
            api_key_value = load_api_key_from_env()
        except ValueError as exc:
            raise AlgoceanCodexOAuthError(str(exc)) from exc

        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise AlgoceanCodexOAuthError(
                "auth=api_key requires langchain-openai. "
                "Install with: pip install algocean-codex-oauth"
            ) from exc

        self._api_backend = ChatOpenAI(
            model=self.model,
            api_key=api_key_value,
            timeout=self.timeout,
        )

    def _init_oauth_backend(self) -> None:
        if self.thread_mode == "codex_resume" and self.ephemeral:
            raise AlgoceanCodexOAuthError(
                "thread_mode='codex_resume' requires ephemeral=False because Codex must "
                "persist the thread for exec resume. Use thread_mode='messages' (default) "
                "for ChatOpenAI-style multi-turn via message history."
            )
        self._client = CodexExecClient(self._to_config())

    @property
    def _llm_type(self) -> str:
        if self._is_api_key_auth():
            return "algocean-codex-api-key"
        return "algocean-codex-oauth"

    @property
    def thread_id(self) -> str | None:
        if self._is_api_key_auth():
            return None
        return self._thread_id

    def reset_thread(self) -> None:
        """Start a fresh Codex thread (oauth only)."""
        if self._is_api_key_auth():
            return
        self._thread_id = None
        self._last_message_count = 0

    @classmethod
    def from_env(cls, *, model: str | None = None, **kwargs: Any) -> AlgoceanCodexOAuth:
        auth = resolve_auth_from_env()
        return cls(auth=auth, model=model or "gpt-5.5", **kwargs)

    @classmethod
    def help(cls, topic: str | None = None) -> str:
        """Print usage guide. Topics: install, quickstart, langgraph, auth, multiturn, presets, all."""
        from .help import print_help

        return print_help(topic=topic)

    @classmethod
    def chat(cls, *, model: str = "gpt-5.5", auth: AuthSetting = oauth, **kwargs: Any) -> AlgoceanCodexOAuth:
        return cls(
            auth=auth,
            model=model,
            workdir=None,
            sandbox="read-only",
            ephemeral=True,
            thread_mode="messages",
            ignore_user_config=True,
            ignore_rules=True,
            **kwargs,
        )

    @classmethod
    def repo_read(
        cls,
        workdir: str,
        *,
        model: str = "gpt-5.5",
        auth: AuthSetting = oauth,
        **kwargs: Any,
    ) -> AlgoceanCodexOAuth:
        if auth == api_key:
            raise AlgoceanCodexOAuthError("repo_read() requires auth=oauth.")
        return cls(
            auth=auth,
            model=model,
            workdir=workdir,
            sandbox="read-only",
            ephemeral=True,
            thread_mode="messages",
            ignore_user_config=False,
            ignore_rules=False,
            **kwargs,
        )

    @classmethod
    def repo_write(
        cls,
        workdir: str,
        *,
        model: str = "gpt-5.5",
        auth: AuthSetting = oauth,
        thread_mode: ThreadMode = "codex_resume",
        **kwargs: Any,
    ) -> AlgoceanCodexOAuth:
        if auth == api_key:
            raise AlgoceanCodexOAuthError("repo_write() requires auth=oauth.")
        return cls(
            auth=auth,
            model=model,
            workdir=workdir,
            sandbox="workspace-write",
            ephemeral=False,
            thread_mode=thread_mode,
            ignore_user_config=False,
            ignore_rules=False,
            **kwargs,
        )

    def _is_llm_only_oauth(self) -> bool:
        """OAuth chat path with no workdir — forced api_key-like behavior."""
        return not self._is_api_key_auth() and self.workdir is None

    def _oauth_prompt_preamble(self) -> str | None:
        if self._is_llm_only_oauth():
            return OAUTH_LLM_ONLY_PREAMBLE
        return None

    def _to_config(self) -> AlgoceanCodexConfig:
        ignore_user_config = self.ignore_user_config
        ignore_rules = self.ignore_rules
        sandbox = self.sandbox
        ephemeral = self.ephemeral

        if self._is_llm_only_oauth():
            ignore_user_config = True
            ignore_rules = True
            sandbox = "read-only"
            ephemeral = True

        return AlgoceanCodexConfig(
            model=self.model,
            codex_bin=self.codex_bin,
            timeout_sec=self.timeout,
            sandbox=sandbox,
            workdir=self.workdir,
            ephemeral=ephemeral,
            ignore_user_config=ignore_user_config,
            ignore_rules=ignore_rules,
            skip_git_repo_check=self.skip_git_repo_check,
            require_chatgpt_login=self.require_chatgpt_login,
            allow_codex_access_token_env=self.allow_codex_access_token_env,
            extra_env=dict(self.extra_env),
        )

    def _get_client(self) -> CodexExecClient:
        if self._client is None:
            raise AlgoceanCodexOAuthError("Codex client is not available in auth=api_key mode.")
        return self._client

    def _get_api_backend(self) -> BaseChatModel:
        if self._api_backend is None:
            raise AlgoceanCodexOAuthError("OpenAI backend is not available in auth=oauth mode.")
        return self._api_backend

    def _run_sync(self, coro: Any) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self._is_api_key_auth():
            return self._get_api_backend()._generate(
                messages,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )
        return self._run_sync(self._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs))

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self._is_api_key_auth():
            return await self._get_api_backend()._agenerate(
                messages,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )

        del stop, run_manager

        output_schema = kwargs.get("output_schema")
        result = await self._invoke_codex(messages, output_schema=output_schema)
        ai_message = self._to_ai_message(result)
        self._last_message_count = len(messages)
        return ChatResult(generations=[ChatGeneration(message=ai_message)])

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        if self._is_api_key_auth():
            async for chunk in self._get_api_backend()._astream(
                messages,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            ):
                yield chunk
            return

        del stop

        output_schema = kwargs.get("output_schema")
        prompt, resume_thread_id = self._build_prompt_and_resume(messages)

        accumulated = ""
        async for chunk in self._get_client().exec_stream(
            prompt,
            output_schema=output_schema,
            resume_thread_id=resume_thread_id,
        ):
            accumulated += chunk
            if run_manager:
                await run_manager.on_llm_new_token(chunk)
            yield ChatGenerationChunk(message=AIMessageChunk(content=chunk))

        if accumulated:
            self._last_message_count = len(messages)
            if self._get_client().last_thread_id:
                self._thread_id = self._get_client().last_thread_id

    async def _invoke_codex(
        self,
        messages: list[BaseMessage],
        *,
        output_schema: dict[str, Any] | None = None,
    ) -> CodexExecResult:
        prompt, resume_thread_id = self._build_prompt_and_resume(messages)

        if resume_thread_id is None:
            result = await self._get_client().exec(prompt, output_schema=output_schema)
            if result.thread_id:
                self._thread_id = result.thread_id
            return result

        result = await self._get_client().resume(
            prompt,
            thread_id=resume_thread_id,
            output_schema=output_schema,
        )
        if result.thread_id:
            self._thread_id = result.thread_id
        return result

    def _build_prompt_and_resume(self, messages: list[BaseMessage]) -> tuple[str, str | None]:
        preamble = self._oauth_prompt_preamble()

        if self.thread_mode == "messages":
            return messages_to_prompt(messages, preamble=preamble), None

        if self._thread_id is None:
            return messages_to_prompt(messages, preamble=preamble), None

        delta_prompt = messages_to_delta_prompt(messages, self._last_message_count)
        return delta_prompt, self._thread_id

    def _to_ai_message(self, result: CodexExecResult) -> AIMessage:
        return AIMessage(
            content=result.text,
            response_metadata={
                "provider": "algocean-codex-oauth",
                "auth": oauth,
                "model": self.model,
                "thread_id": result.thread_id or self._thread_id,
                "usage": result.usage,
                "thread_mode": self.thread_mode,
            },
        )

    def with_structured_output(
        self,
        schema: StructuredSchema,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[Any, Any]:
        if self._is_api_key_auth():
            return self._get_api_backend().with_structured_output(
                schema,
                include_raw=include_raw,
                **kwargs,
            )

        del kwargs

        json_schema = _to_json_schema(schema)

        async def _invoke_structured(input_value: Any) -> Any:
            ai_message = await self.ainvoke(input_value, output_schema=json_schema)
            parsed = _parse_structured_content(ai_message.content, schema)

            if include_raw:
                return {"raw": ai_message, "parsed": parsed, "parsing_error": None}
            return parsed

        def _invoke_structured_sync(input_value: Any) -> Any:
            return self._run_sync(_invoke_structured(input_value))

        return RunnableLambda(_invoke_structured_sync, afunc=_invoke_structured)

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "auth": self.auth,
            "sandbox": self.sandbox,
            "workdir": self.workdir,
            "provider": self._llm_type,
            "thread_mode": self.thread_mode,
        }


def _to_json_schema(schema: StructuredSchema) -> dict[str, Any]:
    if isinstance(schema, dict):
        return normalize_codex_json_schema(schema)

    if is_basemodel_subclass(schema):
        return normalize_codex_json_schema(schema.model_json_schema())

    raise AlgoceanCodexOAuthError(
        "Structured output schema must be a JSON Schema dict or Pydantic BaseModel subclass."
    )


def _parse_structured_content(content: str | list[Any], schema: StructuredSchema) -> Any:
    if isinstance(content, list):
        text = json.dumps(content, ensure_ascii=False)
    else:
        text = content

    text = text.strip()
    if not text:
        raise AlgoceanCodexOAuthError("Structured output was empty.")

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AlgoceanCodexOAuthError(f"Structured output was not valid JSON: {text}") from exc

    if is_basemodel_subclass(schema):
        return schema.model_validate(data)

    return data
