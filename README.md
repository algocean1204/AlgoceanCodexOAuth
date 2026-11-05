<h1 align="center">$\bf{\large{\color{#6580DD} AlgoceanCodexOAuth }}$</h1>

**LangChain / LangGraph에서 `ChatOpenAI` 대신 쓰는 LLM 래퍼입니다.**

로컬에서는 Codex CLI의 ChatGPT OAuth 로그인으로, 배포 서버에서는 OpenAI API key로 같은 그래프 코드를 실행합니다. 모델을 만드는 한 줄만 바꾸면 됩니다.

[![PyPI](https://img.shields.io/pypi/v/algocean-codex-oauth?style=flat-square)](https://pypi.org/project/algocean-codex-oauth/)
![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

<hr>

## 개발 환경

### Language
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

### Framework & Library
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?style=for-the-badge&logo=langgraph&logoColor=white)

### AI & Model
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)
![Codex CLI](https://img.shields.io/badge/Codex%20CLI-000000?style=for-the-badge&logo=openai&logoColor=white)

### Build & Deploy
![PyPI](https://img.shields.io/badge/PyPI-3775A9?style=for-the-badge&logo=pypi&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)

<hr>

## 주요 기능

### 1. 두 가지 인증 모드
- `auth=oauth`(기본): 로컬 Codex CLI(`codex exec`)를 호출하고 ChatGPT 구독 한도를 사용합니다.
- `auth=api_key`: `langchain_openai.ChatOpenAI`에 위임하고 OpenAI API로 과금됩니다.
- `from_env()`는 `ALGOCEANCODEXOAUTH_AUTH`, `ALGOCEANCODEXOAUTH_API` 순으로 모드를 정하고, 둘 다 없으면 oauth를 씁니다.

### 2. ChatOpenAI와 같은 호출 방식
- `invoke`, `ainvoke`, `astream`, `batch`, `with_structured_output`, `bind_tools`를 지원합니다.
- `usage_metadata`와 `response_metadata`(`model_name`, `token_usage`, `finish_reason`)를 ChatOpenAI와 같은 키로 채웁니다.
- `create_react_agent(llm, tools=[...])`에 수정 없이 넣을 수 있습니다.

### 3. 모델별 reasoning effort 검증
- `codex debug models`에서 모델 목록과 지원 effort를 런타임에 읽고, CLI가 없으면 내장 목록을 사용합니다.
- 모델이 지원하지 않는 `reasoning_effort`는 API 호출 전에 `AlgoceanCodexOAuthError`로 막습니다.
- `print_models()`, `efforts(model)`로 사용 가능한 값을 확인할 수 있습니다.

### 4. 용도별 Preset
- `chat()`: 프로젝트 `AGENTS.md`와 Codex rules를 적용하지 않는 read-only 대화용 설정입니다.
- `repo_read(workdir)`: 코드베이스를 읽기 전용으로 탐색합니다. (oauth 전용)
- `repo_write(workdir)`: `workspace-write` 샌드박스에서 코드를 수정합니다. (oauth 전용)

<hr>

## 인증 모드 비교

| 기능 | oauth | api_key |
|---|---|---|
| 인증 | `codex login` | `ALGOCEANCODEXOAUTH_API` |
| 과금 | ChatGPT / Codex 구독 | OpenAI API |
| invoke / ainvoke / astream / batch | O | O |
| structured output, 멀티턴(messages) | O | O |
| `bind_tools` / `tool_calls` | O (`--output-schema` 기반 에뮬레이션) | O (네이티브) |
| `reasoning_effort` / `verbosity` / `stop` | O | O |
| `temperature` / `max_tokens` / `top_p` / `seed` | X (`unsupported_params`로 알림) | O |
| 토큰 단위 스트리밍 | X (완료 시 한 청크) | O |
| `repo_read` / `repo_write` / `codex_resume` | O | X |

oauth 모드는 개인 로컬 개발용이며 Codex CLI(`codex`)가 PATH에 있어야 합니다. 서버 배포에는 api_key 모드를 사용합니다.

<hr>

## 설치 및 실행

```bash
pip install -U algocean-codex-oauth
```

oauth 모드는 Codex CLI 로그인이 한 번 필요합니다.

```bash
npm install -g @openai/codex
codex login
```

api_key 모드는 환경 변수나 작업 디렉터리의 `.env`에 키를 넣습니다.

```bash
export ALGOCEANCODEXOAUTH_API=<your-openai-api-key>
```

## 사용 예시

```python
from algocean_codex_oauth import AlgoceanCodexOAuth, api_key
from langchain_core.messages import HumanMessage

llm = AlgoceanCodexOAuth.chat(model="gpt-5.5", reasoning_effort="high")      # oauth
# llm = AlgoceanCodexOAuth(auth=api_key, model="gpt-4o")                     # api_key
# llm = AlgoceanCodexOAuth.from_env(model="gpt-5.5")                         # 환경 변수로 선택

print(llm.invoke([HumanMessage(content="Hello")]).content)
AlgoceanCodexOAuth.print_models()   # 사용 가능한 모델과 모델별 effort
```

LangGraph ReAct 에이전트에도 같은 객체를 넣습니다.

```python
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

agent = create_react_agent(llm, tools=[add])
result = agent.invoke({"messages": [("user", "What is 17 plus 25? Use the tool.")]})
```

터미널에서 `AlgoceanCodexOAuth.help("langgraph")`를 호출하면 토픽별 예제를 볼 수 있습니다. (`install`, `quickstart`, `langgraph`, `auth`, `multiturn`, `models`, `presets`, `all`)

<hr>

## 라이선스

[MIT License](LICENSE)

## 링크

- [PyPI: algocean-codex-oauth](https://pypi.org/project/algocean-codex-oauth/)
- [Codex CLI Reference](https://developers.openai.com/codex/cli/reference)
- [Codex Authentication](https://developers.openai.com/codex/auth)
