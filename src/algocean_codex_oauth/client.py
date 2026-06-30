"""Codex CLI subprocess client (exec / exec resume / stream)."""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from .auth import assert_chatgpt_oauth_login, build_safe_env
from .config import AlgoceanCodexConfig
from .errors import AlgoceanCodexOAuthError
from .types import CodexExecResult


class CodexExecClient:
    def __init__(self, config: AlgoceanCodexConfig):
        self.config = config
        self.last_thread_id: str | None = None
        self.last_usage: dict[str, Any] | None = None
        self._assert_codex_installed()
        if config.require_chatgpt_login:
            assert_chatgpt_oauth_login(config)

    def _assert_codex_installed(self) -> None:
        if shutil.which(self.config.codex_bin) is None:
            raise AlgoceanCodexOAuthError(
                "codex CLI not found. Install with "
                "`npm install -g @openai/codex` and run `codex login`."
            )

    async def exec(
        self,
        prompt: str,
        *,
        output_schema: dict[str, Any] | None = None,
    ) -> CodexExecResult:
        return await self._run(prompt, output_schema=output_schema)

    async def resume(
        self,
        prompt: str,
        *,
        thread_id: str | None = None,
        use_last: bool = False,
        output_schema: dict[str, Any] | None = None,
    ) -> CodexExecResult:
        if not thread_id and not use_last:
            raise ValueError("Either thread_id or use_last=True is required for resume.")
        return await self._run(
            prompt,
            output_schema=output_schema,
            resume_thread_id=thread_id,
            resume_last=use_last,
        )

    async def exec_stream(
        self,
        prompt: str,
        *,
        output_schema: dict[str, Any] | None = None,
        resume_thread_id: str | None = None,
    ) -> AsyncIterator[str]:
        """Yield assistant text chunks from codex exec JSONL stdout."""
        async for chunk in self._run_stream(
            prompt,
            output_schema=output_schema,
            resume_thread_id=resume_thread_id,
        ):
            yield chunk

    async def _run(
        self,
        prompt: str,
        *,
        output_schema: dict[str, Any] | None = None,
        resume_thread_id: str | None = None,
        resume_last: bool = False,
    ) -> CodexExecResult:
        final_text = ""
        usage: dict[str, Any] | None = None
        events: list[dict[str, Any]] = []
        thread_id: str | None = None

        async for chunk in self._run_stream(
            prompt,
            output_schema=output_schema,
            resume_thread_id=resume_thread_id,
            resume_last=resume_last,
            collect_events=events,
        ):
            final_text = chunk

        for event in events:
            if event.get("type") == "turn.completed":
                usage = event.get("usage")
            extracted = self._extract_thread_id([event])
            if extracted:
                thread_id = extracted

        return CodexExecResult(
            text=final_text.strip(),
            usage=usage,
            events=events,
            thread_id=thread_id,
        )

    async def _run_stream(
        self,
        prompt: str,
        *,
        output_schema: dict[str, Any] | None = None,
        resume_thread_id: str | None = None,
        resume_last: bool = False,
        collect_events: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        temp_dir_obj: tempfile.TemporaryDirectory[str] | None = None

        try:
            if self.config.workdir is None:
                temp_dir_obj = tempfile.TemporaryDirectory(prefix="algocean-codex-")
                workdir = Path(temp_dir_obj.name)
            else:
                workdir = Path(self.config.workdir).expanduser().resolve()

            workdir.mkdir(parents=True, exist_ok=True)

            final_output_path = workdir / ".codex_final_output.txt"
            schema_path: Path | None = None

            if output_schema is not None:
                schema_path = workdir / ".codex_output_schema.json"
                schema_path.write_text(
                    json.dumps(output_schema, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

            cmd = self._build_command(
                final_output_path=final_output_path,
                output_schema_path=schema_path,
                resume_thread_id=resume_thread_id,
                resume_last=resume_last,
            )

            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(workdir),
                env=build_safe_env(self.config),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            assert process.stdin is not None
            assert process.stdout is not None

            process.stdin.write(prompt.encode("utf-8"))
            await process.stdin.drain()
            process.stdin.close()

            latest_text = ""
            stderr_chunks: list[str] = []
            return_code = 1

            async def _read_stderr() -> None:
                assert process.stderr is not None
                stderr_data = await process.stderr.read()
                if stderr_data:
                    stderr_chunks.append(stderr_data.decode("utf-8", errors="replace"))

            stderr_task = asyncio.create_task(_read_stderr())

            try:
                assert process.stdout is not None
                while True:
                    line_bytes = await asyncio.wait_for(
                        process.stdout.readline(),
                        timeout=self.config.timeout_sec,
                    )
                    if not line_bytes:
                        break

                    line = line_bytes.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue

                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if collect_events is not None:
                        collect_events.append(event)

                    if event.get("type") == "turn.failed":
                        raise AlgoceanCodexOAuthError(f"Codex execution failed: {event}")

                    text = self._extract_agent_text(event)
                    if text and text != latest_text:
                        delta = text[len(latest_text) :] if text.startswith(latest_text) else text
                        latest_text = text
                        if delta:
                            yield delta

                return_code = await asyncio.wait_for(process.wait(), timeout=self.config.timeout_sec)
            finally:
                await stderr_task

            stderr = "".join(stderr_chunks)

            if return_code != 0:
                stdout_tail = json.dumps(collect_events[-3:]) if collect_events else ""
                raise AlgoceanCodexOAuthError(
                    "codex exec failed\n"
                    f"exit_code={return_code}\n\n"
                    f"STDERR:\n{stderr}\n\n"
                    f"EVENTS_TAIL:\n{stdout_tail}"
                )

            if final_output_path.exists():
                file_text = final_output_path.read_text(encoding="utf-8").strip()
                if file_text and file_text != latest_text:
                    delta = file_text[len(latest_text) :] if file_text.startswith(latest_text) else file_text
                    latest_text = file_text
                    if delta:
                        yield delta
                elif file_text and not latest_text:
                    yield file_text
            elif collect_events:
                fallback = self._fallback_final_message(collect_events)
                if fallback and fallback != latest_text:
                    yield fallback if not latest_text else fallback[len(latest_text) :]

            if collect_events:
                for event in collect_events:
                    if event.get("type") == "turn.completed":
                        self.last_usage = event.get("usage")
                self.last_thread_id = self._extract_thread_id(collect_events)

        finally:
            if temp_dir_obj is not None:
                temp_dir_obj.cleanup()

    def _build_command(
        self,
        *,
        final_output_path: Path,
        output_schema_path: Path | None,
        resume_thread_id: str | None,
        resume_last: bool,
    ) -> list[str]:
        is_resume = bool(resume_thread_id or resume_last)

        if is_resume:
            cmd = [self.config.codex_bin, "exec", "resume"]
            if resume_last:
                cmd.append("--last")
            elif resume_thread_id:
                cmd.append(resume_thread_id)
            cmd.extend(
                [
                    "--json",
                    "--model",
                    self.config.model,
                    "--output-last-message",
                    str(final_output_path),
                ]
            )
            if self.config.ephemeral:
                cmd.append("--ephemeral")
            if self.config.ignore_user_config:
                cmd.append("--ignore-user-config")
            if self.config.ignore_rules:
                cmd.append("--ignore-rules")
            if self.config.skip_git_repo_check:
                cmd.append("--skip-git-repo-check")
            if output_schema_path is not None:
                cmd.extend(["--output-schema", str(output_schema_path)])
            cmd.append("-")
            return cmd

        cmd = [self.config.codex_bin, "exec"]
        cmd.extend(
            [
                "--json",
                "--model",
                self.config.model,
                "--sandbox",
                self.config.sandbox,
                "--output-last-message",
                str(final_output_path),
            ]
        )

        if self.config.ephemeral:
            cmd.append("--ephemeral")

        if self.config.ignore_user_config:
            cmd.append("--ignore-user-config")

        if self.config.ignore_rules:
            cmd.append("--ignore-rules")

        if self.config.skip_git_repo_check:
            cmd.append("--skip-git-repo-check")

        if output_schema_path is not None:
            cmd.extend(["--output-schema", str(output_schema_path)])

        cmd.append("-")
        return cmd

    def _extract_agent_text(self, event: dict[str, Any]) -> str:
        if event.get("type") != "item.completed":
            return ""

        item = event.get("item", {})
        if item.get("type") != "agent_message":
            return ""

        return str(item.get("text", "")).strip()

    def _extract_thread_id(self, events: list[dict[str, Any]]) -> str | None:
        for event in events:
            for key in ("thread_id", "threadId", "id"):
                value = event.get(key)
                if isinstance(value, str) and value:
                    return value

            thread = event.get("thread")
            if isinstance(thread, dict):
                value = thread.get("id")
                if isinstance(value, str) and value:
                    return value

        return None

    def _fallback_final_message(self, events: list[dict[str, Any]]) -> str:
        final_text = ""

        for event in events:
            text = self._extract_agent_text(event)
            if text:
                final_text = text

        return final_text.strip()
