# AlgoceanCodexOAuth

LangChain / LangGraph에서 **`ChatOpenAI` 자리에 그대로 꽂는** LLM 래퍼입니다.

- **`auth=oauth`** (기본) — 로컬 Codex CLI + ChatGPT OAuth 구독 한도
- **`auth=api_key`** — OpenAI API key 과금 (`langchain_openai.ChatOpenAI` 위임)

> **PyPI:** [algocean-codex-oauth](https://pypi.org/project/algocean-codex-oauth/) · **GitHub:** [algocean1204/AlgoceanCodexOAuth](https://github.com/algocean1204/AlgoceanCodexOAuth)

---

## 1분 시작

```bash
pip install -U algocean-codex-oauth langgraph langchain-core
codex login   # oauth 사용 시 1회 (ChatGPT 로그인)
```

```python
from algocean_codex_oauth import AlgoceanCodexOAuth
from langchain_core.messages import HumanMessage

llm = AlgoceanCodexOAuth.chat(model="gpt-5.5")
print(llm.invoke([HumanMessage(content="Hello")]).content)
```

LangGraph 노드에도 **동일한 `llm` 객체**를 넣으면 됩니다.

---

## 어떤 auth를 쓸까?

| | `auth=oauth` (기본) | `auth=api_key` |
|---|---|---|
| **언제** | 로컬 개발 PC | 배포 서버, CI |
| **인증** | `codex login` | 환경 변수 `ALGOCEANCODEXOAUTH_API` |
| **과금** | Codex / ChatGPT 구독 | OpenAI API |
| **설치** | `pip install algocean-codex-oauth` (동일) | `pip install algocean-codex-oauth` (동일) |
| **repo 에이전트** (`repo_read` / `repo_write`) | ✅ | ❌ |
| **Codex thread resume** | ✅ | ❌ |

**그래프 코드는 그대로** — `llm`을 만드는 줄만 `auth`와 `model`을 바꾸면 됩니다.

---

## 설치

```bash
pip install -U algocean-codex-oauth
```

oauth / api_key **모두 동일한 패키지**입니다. auth는 코드 또는 환경 변수로 선택합니다.

### oauth — 로컬 개발 (추가 1회)

```bash
npm install -g @openai/codex
codex login
codex login status
```

Codex 인증: [Codex Authentication](https://developers.openai.com/codex/auth)

oauth 모드에서는 ChatGPT OAuth만 사용합니다. OpenAI API key 환경 변수가 설정되어 있으면 oauth 호출이 차단될 수 있습니다.

### api_key — 배포 / 서버 (추가 설정)

```bash
export ALGOCEANCODEXOAUTH_API=<your-openai-api-key>
# 선택: export ALGOCEANCODEXOAUTH_AUTH=api_key
```

LangGraph 프로젝트 의존성 예:

```bash
pip install -U algocean-codex-oauth langgraph langchain-core
```

---

## 기본 사용

### 동기 / 비동기

```python
from algocean_codex_oauth import AlgoceanCodexOAuth, oauth, api_key
from langchain_core.messages import HumanMessage

# oauth (기본) — auth 생략 가능
llm = AlgoceanCodexOAuth(model="gpt-5.5")

# api_key
llm = AlgoceanCodexOAuth(auth=api_key, model="gpt-4o")

response = llm.invoke([HumanMessage(content="FastAPI Depends를 짧게 설명해줘.")])
await llm.ainvoke([HumanMessage(content="...")])
```

### 환경 변수로 모드 선택

```python
from algocean_codex_oauth import AlgoceanCodexOAuth

llm = AlgoceanCodexOAuth.from_env(model="gpt-4o")
```

`ALGOCEANCODEXOAUTH_AUTH=api_key`일 때 `ALGOCEANCODEXOAUTH_API`가 필요합니다.

### 터미널 가이드 — `help()`

```python
from algocean_codex_oauth import AlgoceanCodexOAuth

AlgoceanCodexOAuth.help()              # 개요
AlgoceanCodexOAuth.help("langgraph")   # LangGraph 예제
AlgoceanCodexOAuth.help("auth")        # oauth vs api_key
AlgoceanCodexOAuth.help("all")         # 전체
```

토픽: `install`, `quickstart`, `langgraph`, `auth`, `multiturn`, `presets`, `all`

---

## LangGraph

### 단일 노드 그래프

```python
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

from algocean_codex_oauth import AlgoceanCodexOAuth

class State(TypedDict):
    user_input: str
    answer: str

llm = AlgoceanCodexOAuth(model="gpt-5.5")

async def assistant_node(state: State) -> State:
    messages = [
        SystemMessage(content="간결한 개인 비서."),
        HumanMessage(content=state["user_input"]),
    ]
    ai = await llm.ainvoke(messages)
    return {"answer": ai.content}

graph = StateGraph(State)
graph.add_node("assistant", assistant_node)
graph.set_entry_point("assistant")
graph.add_edge("assistant", END)
app = graph.compile()
```

### 배포 시 — api_key로 동일 그래프

```python
from algocean_codex_oauth import AlgoceanCodexOAuth, api_key

llm = AlgoceanCodexOAuth(auth=api_key, model="gpt-4o")
# 그래프·노드 코드는 동일
```

### ReAct Agent

```python
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from algocean_codex_oauth import AlgoceanCodexOAuth

llm = AlgoceanCodexOAuth(model="gpt-5.5")
agent = create_react_agent(llm, tools=[])
result = agent.invoke({"messages": [HumanMessage(content="hello")]})
```

### Structured Output

```python
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from algocean_codex_oauth import AlgoceanCodexOAuth

class Analysis(BaseModel):
    summary: str = Field(description="요약")
    risk_level: str = Field(description="low | medium | high")

llm = AlgoceanCodexOAuth(model="gpt-5.5")
structured = llm.with_structured_output(Analysis)
result = structured.invoke([HumanMessage(content="위험도를 평가해줘.")])
```

---

## ChatOpenAI와 동일한 API

| ChatOpenAI | AlgoceanCodexOAuth |
|---|---|
| `llm.invoke(messages)` | 동일 |
| `await llm.ainvoke(messages)` | 동일 |
| `llm.astream(messages)` | 동일 |
| `llm.with_structured_output(schema)` | 동일 |
| LangGraph `state["messages"]` 멀티턴 | `thread_mode="messages"` (기본) |
| Codex thread resume | `thread_mode="codex_resume"` (oauth 전용) |
| 새 대화 | `llm.reset_thread()` (oauth) |

---

## Preset — 용도별 바로 쓰기

```python
from algocean_codex_oauth import AlgoceanCodexOAuth

# LangGraph Q&A — 순수 LLM (AGENTS.md/rules 무시, 기본)
llm = AlgoceanCodexOAuth.chat(model="gpt-5.5")

# repo 읽기 전용 에이전트 (oauth 전용)
llm = AlgoceanCodexOAuth.repo_read(workdir="/path/to/repo")

# repo 쓰기 에이전트 (oauth 전용)
llm = AlgoceanCodexOAuth.repo_write(workdir="/path/to/repo")
```

| Preset | 용도 | auth |
|---|---|---|
| `chat()` | LangGraph Q&A, 일반 대화 | oauth / api_key |
| `repo_read(path)` | 코드베이스 탐색 | oauth |
| `repo_write(path)` | 코드 수정 에이전트 | oauth |

### LLM-only (`chat()`, oauth)

LangGraph Q&A처럼 **일반 LLM**에 가깝게 동작합니다.

- 프로젝트 `AGENTS.md` / Codex rules **적용 안 함**
- read-only sandbox, 단발성 대화에 적합
- repo 에이전트가 필요하면 `repo_read()` / `repo_write()` 사용

---

## 멀티턴

### messages 모드 (기본, ChatOpenAI와 동일)

```python
messages = [HumanMessage(content="코드네임은 ALPHA7")]
ai1 = await llm.ainvoke(messages)
messages += [ai1, HumanMessage(content="코드네임이 뭐야?")]
ai2 = await llm.ainvoke(messages)
```

LangGraph `state["messages"]` 패턴과 1:1 동일합니다.

### codex_resume (oauth, repo 작업)

```python
llm = AlgoceanCodexOAuth(
    model="gpt-5.5",
    workdir="/path/to/repo",
    sandbox="read-only",
    ephemeral=False,
    thread_mode="codex_resume",
)

await llm.ainvoke([HumanMessage(content="첫 질문")])
await llm.ainvoke([HumanMessage(content="이어서")])
llm.reset_thread()
```

### Streaming

```python
async for chunk in llm.astream([HumanMessage(content="hello")]):
    print(chunk.content, end="", flush=True)
```

---

## 로컬 ↔ 배포 전환

```python
import os
from algocean_codex_oauth import AlgoceanCodexOAuth, oauth, api_key

def get_llm():
    if os.getenv("ALGOCEANCODEXOAUTH_API"):
        return AlgoceanCodexOAuth(auth=api_key, model="gpt-4o")
    return AlgoceanCodexOAuth(auth=oauth, model="gpt-5.5")
```

---

## 생성자 옵션

```python
AlgoceanCodexOAuth(
    model="gpt-5.5",
    auth=oauth,                    # oauth | api_key
    timeout=180,
    sandbox="read-only",           # oauth 전용
    workdir=None,                  # oauth 전용
    ephemeral=True,                # oauth 전용
    thread_mode="messages",        # messages | codex_resume (oauth)
    codex_bin="codex",
    require_chatgpt_login=True,
)
```

## 환경 변수

| 변수 | 설명 |
|---|---|
| `ALGOCEANCODEXOAUTH_API` | api_key 모드 OpenAI API key |
| `ALGOCEANCODEXOAUTH_AUTH` | `oauth` 또는 `api_key` (기본 `oauth`) |

---

## 제한 사항

- **oauth** — 개인 로컬 / 개인 구독 용도. SaaS 서버 배포에는 부적합.
- **api_key** — OpenAI API 과금. repo sandbox / codex resume 미지원.
- **oauth** — Codex CLI(`codex`)가 PATH에 있어야 합니다.

---

## License

MIT

## Links

- [GitHub](https://github.com/algocean1204/AlgoceanCodexOAuth)
- [Codex CLI](https://developers.openai.com/codex/cli/reference)
- [Codex Auth](https://developers.openai.com/codex/auth)
