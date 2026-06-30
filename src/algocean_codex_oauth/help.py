"""Interactive guide for AlgoceanCodexOAuth."""

from __future__ import annotations

import sys
from typing import TextIO

from . import __version__

_SECTIONS: dict[str, str] = {
    "overview": """
AlgoceanCodexOAuth — LangChain / LangGraph ChatOpenAI drop-in replacement
Version: {version}

  LangGraph → AlgoceanCodexOAuth(auth=oauth | api_key)
    oauth   → codex exec → ChatGPT OAuth (local dev, subscription credits)
    api_key → langchain_openai.ChatOpenAI → OpenAI billing (production)

LangGraph code stays the same — only swap the llm import and constructor.
""".strip(),
    "install": """
Install
-------
  pip install -U algocean-codex-oauth

  oauth and api_key both work with the same install.

Local oauth (one-time):
  npm install -g @openai/codex
  codex login
  codex login status

Production api_key:
  export ALGOCEANCODEXOAUTH_API=<your-openai-api-key>
  export ALGOCEANCODEXOAUTH_AUTH=api_key   # optional; default is oauth
""".strip(),
    "quickstart": """
Quick Start
-----------
  from algocean_codex_oauth import AlgoceanCodexOAuth, oauth, api_key
  from langchain_core.messages import HumanMessage

  # Local (default)
  llm = AlgoceanCodexOAuth(model="gpt-5.5")
  llm = AlgoceanCodexOAuth(auth=oauth, model="gpt-5.5")

  # Production
  llm = AlgoceanCodexOAuth(auth=api_key, model="gpt-4o")

  # From environment (ALGOCEANCODEXOAUTH_AUTH, ALGOCEANCODEXOAUTH_API)
  llm = AlgoceanCodexOAuth.from_env(model="gpt-4o")

  response = llm.invoke([HumanMessage(content="Hello")])
  await llm.ainvoke([HumanMessage(content="Hello")])
""".strip(),
    "langgraph": """
LangGraph
---------
  from langgraph.graph import StateGraph, END
  from algocean_codex_oauth import AlgoceanCodexOAuth

  llm = AlgoceanCodexOAuth(model="gpt-5.5")

  async def node(state):
      ai = await llm.ainvoke(state["messages"])
      return {"messages": state["messages"] + [ai]}

  # ReAct agent
  from langgraph.prebuilt import create_react_agent
  agent = create_react_agent(llm, tools=[])

  # Structured output
  structured = llm.with_structured_output(MyPydanticModel)

Only change llm when deploying:
  llm = AlgoceanCodexOAuth(auth=api_key, model="gpt-4o")
""".strip(),
    "auth": """
Auth Modes
----------
  oauth (default)
    - Use: local development, Codex subscription credits
    - Requires: codex CLI + codex login (ChatGPT OAuth)
    - Backend: codex exec subprocess
    - LLM-only mode (workdir=None): --ignore-rules, internal chat preamble, read-only sandbox
    - Blocks OPENAI_API_KEY / CODEX_API_KEY in subprocess env

  api_key
    - Use: production servers, OpenAI API billing
    - Requires: ALGOCEANCODEXOAUTH_API=<your-openai-api-key>
    - Backend: langchain_openai.ChatOpenAI (official OpenAI SDK path)
    - Does NOT support: repo_read, repo_write, thread_mode=codex_resume
""".strip(),
    "multiturn": """
Multi-turn
----------
  messages (default) — ChatOpenAI / LangGraph standard:
    messages = [HumanMessage(content="Remember ALPHA7")]
    ai1 = await llm.ainvoke(messages)
    messages += [ai1, HumanMessage(content="What was the code?")]
    ai2 = await llm.ainvoke(messages)

  codex_resume (oauth only) — Codex exec resume:
    llm = AlgoceanCodexOAuth(
        workdir="/path/to/repo",
        ephemeral=False,
        thread_mode="codex_resume",
    )
    llm.reset_thread()  # start fresh conversation
""".strip(),
    "presets": """
Presets (oauth only)
--------------------
  AlgoceanCodexOAuth.chat(model="gpt-5.5")
  AlgoceanCodexOAuth.repo_read(workdir="/path/to/repo")
  AlgoceanCodexOAuth.repo_write(workdir="/path/to/repo")

Environment variables:
  ALGOCEANCODEXOAUTH_API   — OpenAI API key (required for auth=api_key)
  ALGOCEANCODEXOAUTH_AUTH  — oauth | api_key (default: oauth)

Links:
  https://github.com/algocean1204/AlgoceanCodexOAuth
  https://pypi.org/project/algocean-codex-oauth/
""".strip(),
}

_TOPIC_ALIASES: dict[str, str] = {
    "": "overview",
    "overview": "overview",
    "intro": "overview",
    "install": "install",
    "setup": "install",
    "quickstart": "quickstart",
    "quick": "quickstart",
    "start": "quickstart",
    "langgraph": "langgraph",
    "graph": "langgraph",
    "auth": "auth",
    "oauth": "auth",
    "api_key": "auth",
    "multiturn": "multiturn",
    "thread": "multiturn",
    "presets": "presets",
    "env": "presets",
    "all": "all",
}


def build_help_text(*, topic: str | None = None) -> str:
    """Return guide text for the given topic."""
    raw = (topic or "").strip().lower()

    if not raw:
        key = "overview"
    elif raw not in _TOPIC_ALIASES:
        available = ", ".join(sorted(set(_TOPIC_ALIASES.keys()) - {"", "all"}))
        return (
            f"Unknown topic: {topic!r}\n"
            f"Available topics: {available}, all\n"
            "Example: AlgoceanCodexOAuth.help('langgraph')"
        )
    else:
        key = _TOPIC_ALIASES[raw]

    if key == "all":
        parts = [_SECTIONS["overview"].format(version=__version__)]
        for name in ("install", "quickstart", "langgraph", "auth", "multiturn", "presets"):
            parts.append(_SECTIONS[name])
        return "\n\n".join(parts)

    text = _SECTIONS[key]
    if key == "overview":
        text = text.format(version=__version__)
    return text


def print_help(
    *,
    topic: str | None = None,
    file: TextIO | None = None,
) -> str:
    """Print guide to stdout (or file) and return the text."""
    text = build_help_text(topic=topic)
    out = file if file is not None else sys.stdout
    print(text, file=out)
    return text
