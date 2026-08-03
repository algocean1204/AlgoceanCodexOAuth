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

llm = AlgoceanCodexOAuth.chat(model="gpt-5.5", reasoning_effort="high")
print(llm.invoke([HumanMessage(content="Hello")]).content)

AlgoceanCodexOAuth.print_models()   # 쓸 수 있는 모델 + 모델별 effort
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
| **tool calling** (`bind_tools`) | ✅ | ✅ |
| **`reasoning_effort` / `verbosity`** | ✅ | ✅ |
| **토큰 사용량** (`usage_metadata`) | ✅ | ✅ |

**그래프 코드는 그대로** — `llm`을 만드는 줄만 `auth`와 `model`을 바꾸면 됩니다.
두 모드의 남은 차이는 [제한 사항](#oauth--api_key-차이-그-외는-동일)에 표로 정리돼 있습니다.

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

### 환경 변수 / `.env`로 모드 선택

```python
from algocean_codex_oauth import AlgoceanCodexOAuth

llm = AlgoceanCodexOAuth.from_env(model="gpt-5.5")
```

`from_env()`가 모드를 정하는 규칙 — 위에서부터 먼저 맞는 것:

| 상황 | 결과 |
|---|---|
| `ALGOCEANCODEXOAUTH_AUTH` 지정 | 그 값 (`oauth` / `api_key`) |
| `ALGOCEANCODEXOAUTH_API`만 있음 | `api_key` |
| 둘 다 없음 | `oauth` (구독) |

키만 넣고 `AUTH`를 잊어도 조용히 구독 모드로 새지 않습니다.
구독을 쓰고 싶은데 키가 환경에 남아 있다면 `ALGOCEANCODEXOAUTH_AUTH=oauth`를 명시하세요.

두 변수 모두 **실제 환경변수 → 작업 디렉터리의 `.env`** 순으로 읽습니다.

```bash
# .env
ALGOCEANCODEXOAUTH_API=sk-...
```

`.env` 파싱은 `KEY=VALUE` 한 줄 형식만 지원하며 추가 의존성은 없습니다.

### 터미널 가이드 — `help()`

```python
from algocean_codex_oauth import AlgoceanCodexOAuth

AlgoceanCodexOAuth.help()              # 개요
AlgoceanCodexOAuth.help("langgraph")   # LangGraph 예제
AlgoceanCodexOAuth.help("auth")        # oauth vs api_key
AlgoceanCodexOAuth.help("all")         # 전체
```

토픽: `install`, `quickstart`, `langgraph`, `auth`, `multiturn`, `models`, `presets`, `all`
(`models`는 `effort` · `reasoning`으로도 호출됩니다.)

---

## 모델 목록 · reasoning effort

```python
from algocean_codex_oauth import AlgoceanCodexOAuth

AlgoceanCodexOAuth.print_models()
# MODEL          DEFAULT   EFFORTS
# gpt-5.6-sol    low       low, medium, high, xhigh, max, ultra
# gpt-5.5        medium    low, medium, high, xhigh
# ...

AlgoceanCodexOAuth.efforts("gpt-5.5")   # ('low', 'medium', 'high', 'xhigh')
AlgoceanCodexOAuth.models()             # list[ModelInfo]
```

목록은 설치된 `codex debug models`에서 런타임에 읽으므로 codex를 업데이트하면 자동 반영됩니다.
codex CLI가 없으면 내장 목록으로 폴백합니다.

### effort 적용

```python
llm = AlgoceanCodexOAuth(model="gpt-5.5", reasoning_effort="high")
llm = AlgoceanCodexOAuth.chat(model="gpt-5.5", reasoning_effort="low")
llm = AlgoceanCodexOAuth.repo_write("/path/repo", reasoning_effort="xhigh")

llm.effective_reasoning_effort   # 지정값, 없으면 모델 기본값
llm.available_efforts            # 이 인스턴스가 실제로 쓸 수 있는 effort
```

값: `none` · `minimal` · `low` · `medium` · `high` · `xhigh` · `max` · `ultra`
— **모델마다 지원 범위가 다르고**, 미지원 값은 API 호출 전에 `AlgoceanCodexOAuthError`로 막힙니다.

```python
AlgoceanCodexOAuth(model="gpt-5.5", reasoning_effort="max")
# AlgoceanCodexOAuthError: reasoning_effort='max' is not supported by model 'gpt-5.5'.
#   Supported: low, medium, high, xhigh (default: medium).
```

### 응답에서 확인

```python
ai = llm.invoke([HumanMessage(content="...")])
ai.response_metadata["reasoning_effort"]                  # 실제 적용된 effort
ai.response_metadata["usage"]["reasoning_output_tokens"]  # 추론 토큰 소모량
```

`auth=api_key`에서는 `reasoning_effort`가 `langchain_openai.ChatOpenAI`로 그대로 전달됩니다.

> `chat()`·`repo_read()`·`repo_write()`는 codex 설정 로딩 여부가 달라 모델 목록도 다릅니다.
> `chat()`은 `--ignore-user-config`로 돌아 codex 내장 목록만, repo preset은 `~/.codex/config.toml`의
> 커스텀 provider까지 봅니다. 목록 API도 같은 구분을 따릅니다:
> `AlgoceanCodexOAuth.models(ignore_user_config=False)`.

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
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from algocean_codex_oauth import AlgoceanCodexOAuth

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

llm = AlgoceanCodexOAuth.chat(model="gpt-5.5", reasoning_effort="low")
agent = create_react_agent(llm, tools=[add])

result = agent.invoke({"messages": [("user", "What is 17 plus 25? Use the tool.")]})
print(result["messages"][-1].content)   # 42
```

### Structured Output

```python
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from algocean_codex_oauth import AlgoceanCodexOAuth

class Item(BaseModel):
    name: str = Field(description="item name")
    qty: int = Field(description="quantity")

class Cart(BaseModel):
    items: list[Item] = Field(description="line items")
    total: int = Field(description="total quantity")
    note: Optional[str] = Field(default=None, description="optional note")

llm = AlgoceanCodexOAuth.chat(model="gpt-5.5")
cart = llm.with_structured_output(Cart).invoke(
    [HumanMessage(content="Cart has 3 apples and 5 pears. Return structured data.")]
)
# Cart(items=[Item(name='apples', qty=3), Item(name='pears', qty=5)], total=8, note=None)
```

중첩 모델·리스트·`Optional` 모두 지원합니다. `include_raw=True`면 `{"raw", "parsed", "parsing_error"}`를 돌려줍니다.

---

## Tool calling

```python
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

llm = AlgoceanCodexOAuth.chat(model="gpt-5.5")

ai = llm.bind_tools([add]).invoke([HumanMessage(content="1+2?")])
ai.tool_calls   # [{'name': 'add', 'args': {'a': 1, 'b': 2}, 'id': 'call_0', ...}]

agent = create_react_agent(llm, tools=[add])   # 그대로 동작
```

`tool_choice`도 ChatOpenAI와 동일하게 `"auto"` / `"required"` / `"none"` / 툴 이름을 받습니다.

> oauth 모드에는 Codex CLI에 tool-call API가 없어 `--output-schema` 기반으로 에뮬레이션합니다.
> 호출부 코드는 ChatOpenAI와 동일하고, 바인딩하지 않은 툴 이름은 프롬프트에서 차단됩니다.
> api_key 모드는 ChatOpenAI 네이티브 tool calling에 그대로 위임합니다.

---

## 토큰 사용량

```python
ai = llm.invoke([HumanMessage(content="...")])

ai.usage_metadata
# {'input_tokens': 100, 'output_tokens': 30, 'total_tokens': 130,
#  'input_token_details': {'cache_read': 40},
#  'output_token_details': {'reasoning': 12}}

ai.response_metadata["model_name"]     # ChatOpenAI 호환 키
ai.response_metadata["token_usage"]
ai.response_metadata["finish_reason"]  # 'stop' | 'tool_calls'
```

LangGraph·LangSmith·콜백의 토큰 집계가 ChatOpenAI와 동일하게 동작합니다.
스트리밍에서도 마지막 청크에 usage가 실립니다.

---

## ChatOpenAI와 동일한 API

| ChatOpenAI | AlgoceanCodexOAuth |
|---|---|
| `llm.invoke(messages)` | 동일 |
| `await llm.ainvoke(messages)` | 동일 |
| `llm.astream(messages)` | 동일 |
| `llm.batch(inputs)` | 동일 |
| `llm.with_structured_output(schema)` | 동일 (중첩·리스트·`Optional` 포함) |
| `llm.bind_tools(tools, tool_choice=...)` | 동일 |
| `ai.tool_calls` | 동일 |
| `ai.usage_metadata` | 동일 |
| `ai.response_metadata["model_name" / "token_usage" / "finish_reason"]` | 동일 |
| `stop=[...]` | 동일 |
| `reasoning_effort` / `verbosity` | 동일 |
| `create_react_agent(llm, tools=[...])` | 수정 없이 동작 |
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

> oauth 모드에서 `codex exec --json`은 토큰 델타 이벤트를 내보내지 않습니다.
> 응답은 완료 시점에 한 청크로 도착합니다 (TTFT ≈ 전체 응답시간).
> api_key 모드는 OpenAI SDK 토큰 스트리밍을 그대로 씁니다.

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
    reasoning_effort=None,         # none|minimal|low|medium|high|xhigh|max|ultra (모델별 상이)
    verbosity=None,                # low | medium | high — 양쪽 모드 지원
    temperature=None,              # ↓ api_key 전용. oauth 는 codex CLI 미지원 →
    max_tokens=None,               #    response_metadata["unsupported_params"] 로 알려줌
    top_p=None,
    seed=None,
    stop=None,                     # 양쪽 모드 지원 (oauth 는 후처리 절단)
    model_kwargs={},               # api_key 로 그대로 전달
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
| `ALGOCEANCODEXOAUTH_API` | api_key 모드 OpenAI API key. 이것만 있어도 `from_env()`는 api_key로 갑니다 |
| `ALGOCEANCODEXOAUTH_AUTH` | `oauth` 또는 `api_key`. 지정하면 항상 이 값이 우선 |

둘 다 실제 환경변수를 먼저 보고, 없으면 작업 디렉터리의 `.env`에서 읽습니다.

---

## 제한 사항

- **oauth** — 개인 로컬 / 개인 구독 용도. SaaS 서버 배포에는 부적합.
- **api_key** — OpenAI API 과금. repo sandbox / codex resume 미지원.
- **oauth** — Codex CLI(`codex`)가 PATH에 있어야 합니다.

### oauth ↔ api_key 차이 (그 외는 동일)

| 기능 | oauth | api_key |
|---|---|---|
| invoke / ainvoke / astream / batch | ✅ | ✅ |
| 멀티턴 (messages) · structured output | ✅ | ✅ |
| `bind_tools` · `tool_calls` | ✅ (output-schema 에뮬레이션) | ✅ (네이티브) |
| `usage_metadata` · `token_usage` · `finish_reason` | ✅ | ✅ |
| `reasoning_effort` · `verbosity` · `stop` | ✅ | ✅ |
| `temperature` · `max_tokens` · `top_p` · `seed` | ❌ codex CLI 미지원<br>`unsupported_params` 로 통지 | ✅ |
| 토큰 단위 스트리밍 | ❌ 완료 시 1청크 | ✅ |
| `repo_read` / `repo_write` / `codex_resume` | ✅ | ❌ |

---

## License

MIT

## Links

- [GitHub](https://github.com/algocean1204/AlgoceanCodexOAuth)
- [Codex CLI](https://developers.openai.com/codex/cli/reference)
- [Codex Auth](https://developers.openai.com/codex/auth)
