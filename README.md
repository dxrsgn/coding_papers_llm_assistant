# Coding Papers LLM Assistant

A LangGraph-powered multi-agent assistant that helps with study and coding workflows. The system answers repository questions, summarizes code, and reviews related papers by delegating queries between specialized  agents. 

Author: Turushev Timur j4312  
Course: Advanced NLP ITMO

## Overview

The assistant targets deep-work sessions where developers bounce between code and research. Assistant is built with supervisor MAS patter, where supervisor can delegate subtasks for following agents:

- **Researcher** — responsible for research related queries and fetching arxiv papers.
- **Dev Lead** — navigates the repository, inspects git history, and summarizes file contents through a code-reader tool (subagent).

Both specialist agents keep feeding structured context back to the supervisor, so later turns can reuse prior findings. 


## Architecture

![Architecture diagram](docs/arch.drawio.png)


| #   | Node                          | File                       | Responsibility                                                                                               | Notes / Tools / State                      |
| --- | ---------------------------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------ | ------------------------------------------ |
| 1   | prepare_user_input | —                          | Normalizes user query and injects a `HumanMessage` into state.                                                           | —                                          |
| 2   | Supervisor                         | `src/agent/supervisor.py`  | Follows the supervisor + specialists MAS pattern; delegates tasks via routing node.                          | Subagents (wrapped as tools)                                         |
| 2.1 | Supervisor routing                 | `src/agent/supervisor.py`  | LangGraph tool node wrapping specialist subgraphs as tools; follows LangGraph official supervisor pattern guidance. | —                                          |
| 3   | Researcher subgraph                | `src/agent/researcher.py`  | Answers research questions; can call a tool node.                                                            | Tool: `search_arxiv`                       |
| 3.1 | Researcher summarizer              | `src/agent/researcher.py`  | Summarizes research history; updates shared state between specialists.                                       | Updates shared memory field in state `research_context`                  |
| 4   | Coder subgraph                     | `src/agent/devlead.py`     | Answers code related questions; can call a tool node and file reader subagent.              | Tools: `call_code_reader`, `list_directory`, `get_git_history`, `get_file_history` |
| 4.1 | Coder summarizer                   | `src/agent/devlead.py`     | Summarizes coder message history; updates shared state between specialists.                                  | Updates shared memory field in state `code_context`                      |
| 4.2 | Code  reader                  | `src/agent/devlead.py` | Dedicated subagent (wrapped as a tool) for reading and summarizing file contents.                           | Tools: `read_file_content`, `recall_file_summary`, `memorize_file_summary`. Saves summaries into long term memory (to DB or local folder)                            |


## Tools
| Tool                    | What it does                                                    | Inputs                                     | Output                                                                  | Notes                                                                                          |
| ----------------------- | --------------------------------------------------------------- | ------------------------------------------ | ----------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `search_arxiv`          | Searches arXiv for papers matching a text query.                | `query`                            | List of papers (titles + short descriptions/abstract snippets).         | Used by the Researcher tool node.                                                              |
| `call_code_reader`      | Invokes the code-reader agent to summarize a file.              | `file_path`                                | Summary of the file’s contents.                                         | Acts as a wrapper around the code-reader subagent/tool.                                        |
| `list_diretory`         | Produces a directory tree view for a given path.                | `path`                                     | Directory tree as a string.                                             | Respects `.gitignore` and `.dockerignore`.                                                     |
| `get_git_history`       | Retrieves recent commit history from the current repository.    | `limit`                               | Last `limit` commits.                                                   | Repo is currently fixed/bound to this repo; dynamic repos are WIP. Essentially runs `git log`. |
| `get_file_history`      | Fetches recent change history for a specific file.              | `file_path`, `limit`                  | Last `limit` diffs/patches for that file.                               |  Basically runs smth like `git log -n5 -p -- main.py`                                                        |
| `read_file_content`     | Reads a file’s raw contents.                                    | `file_path`                                | File contents as a string.                                              |                                                 |
| `recall_file_summary`   | Retrieves a previously stored summary for a given file content. | `file_content`, `use_db`             | Stored summary (if available).                                          | When `use_db=true`, fetches from DB; otherwise reads from local storage.                       |
| `memorize_file_summary` | Stores a summary for a given file content in long-term memory.  | `file_content`, `summary`, `use_db` (bool) | Write confirmation / stored entry reference (implementation-dependent). | Uses a hash of file content as the key; `use_db` switches storage to DB vs local folder.       |




## Memory
Assistant store session information (context) via langgraph's checkpointers. If `DATABASE_URL` is presented in envs, memory is stored in DB, otherwise in RAM  
Previ
Graph's state along with message history has two fields for shared memory betwen researcher and coder agents - `research_context` and `coder_context`  
Summaries of files are also stored in long term memory (by hash of content). If `DATABASE_URL` is presented in envs, summaries are stored in DB, otherwise in local folder defined by `LONG_TERM_MEMORY_DIR`

## Error handling
1.  LLM calls are using native ChatOpenAI's retries  
2.  Structured output calls to LLM are wrapped into custom runnable with fallback to raw json parsing  
3.  Arxiv search is also using retries  
4.  Code related tools wrapped in try excepts with fallback to some placeholder/errmsg output  

Also, its not implemented there now, but for safer excecution its better to add message history trimming in order to avoid errors of exceeding max_token limit in LLM calls
## Repository Layout

```
.
├── backend/               # FastAPI app exposing /chat and /health
├── docs/                  # Architecture diagram (draw.io + PNG export)
├── src/
│   ├── agent/             # LangGraph nodes, subgraphs, supervisor, state
│   ├── database/          # Async SQLAlchemy models + summary storage
│   ├── prompts/           # System/user prompts per agent
│   └── tools/             # LangChain tool implementations (git, arXiv, memory)
├── main.py                # CLI entry point (demo script)
├── docker-compose.yml     # App + Postgres + Phoenix + pgAdmin stack
├── Dockerfile             # FastAPI production image
├── pyproject.toml         # Dependencies (LangChain, LangGraph, etc.)
└── insomnia_collection.json # Ready-made HTTP requests for /chat
```



## Getting Started

### Prerequisites
- Python 3.12+
- Git
- (Optional) Docker / Docker Compose
- (Optional) Running Postgres instance if you want persistent checkpoints/memory in cli app

### Installation
```bash
poetry install
```

## Configuration

Create a `.env` file (used by both CLI and FastAPI) with at least:

| Variable | Required | Description |
| --- | --- | --- |
| `OPENAI_API_KEY` | Yes | Key for your model api. |
| `OPENAI_API_BASE` | Yes | Base URL of the model endpoint, e.g., `http://localhost:8000/v1`. |
| `MODEL` | Yes | Qwen model name registered with vLLM (e.g., `qwen3-32b`). |
| `DATABASE_URL` | No | `postgresql+psycopg://user:pass@host:port/db`. Enables LangGraph checkpointing + summary DB. |
| `LONG_TERM_MEMORY_DIR` | No | Fallback path for cached summaries if no DB is available (`.cache/agent_memory`). |


## Running the Assistant

### Interactive CLI
```bash
# run tracing server
poetry run phoenix server
# run cli itself
poetry run python main.py
```


### Docker Compose stack
```bash
docker compose up --build
```
Services:
- `db` — for checkpointing and summary storage.
- `phoenix` — Arize Phoenix UI (`http://localhost:6006`) for tracing.
- `pgadmin` — optional DB GUI (`http://localhost:5050`, admin@admin.com/admin).
- `app` — FastAPI service on `http://localhost:8000`.

Example of curl to backend:  
```bash
curl --request POST \
  --url http://localhost:8000/chat \
  --header 'Content-Type: application/json' \
  --data '{
  "message": "hi how do u do",
  "thread_id": "default_thread"
}'
```
Also examples of requests could be imported to postman/insomnia via `insomnia_collection.json` file


## Experiments and Informal Evaluation

### One conceptual/theoretical question about MAS or LLM agents
```
Question: What is an ReACT llm agent  
Agent response: An **ReACT LLM agent** is a framework that enhances the problem-solving capabilities of large language models (LLMs) by integrating **reasoning** and **acting** in a dynamic, iterative process .....
```
![Theoretical question](docs/theoretical_question.png)
Agent answered correctly though it didn't decide to search arxiv papers, probably it was already sure in its knowledge  
Memory wasn't also used, becaues there is no need in any


### One design/architecture question
```
Question: how one can design visual slam system  
Agent response: To design a Visual SLAM (Simultaneous Localization and Mapping) system, you need to integrate both theoretical understanding and practical implementation. Here is a structured approach to designing such a system: ...

```
![Design question](docs/design_question.png)
Agent answered kinda correctly. To solve this task it firstly called researcher agent, who later searched arxiv for releavant papers. Then after fetching useful info from papers, it delegated task to code agent for exact implementation examples.  
As for memory, coder agent used `research_context` obtained from researcher's work


### One implementation/coding question
```
Question: how can i make retries for my httpx client  
Agent response: To implement retries for your HTTPX client in Python, you can use either the built-in retry functionality or the `tenacity` library for more advanced control. Here's a breakdown of both approaches and best practices: ....

```
![Coding question](docs/coding_question.png)
The response was usefull. No tool calls, since those are not needed, same for memory


### A couple of queries related to the everyday tasks your assistant is meant to help with
```
Question: what was the last thing i did in this code repo and tell me what does the file i changed does
Agent response: e last action you performed in the code repository was a refactoring of the file `src/agent/devlead.py` in a commit with the message "chore: refactor of devlead" on December 15, 2025. ....

```
![Other question](docs/other_question.png)
Response was useful, firstly supervisor called coder agent to get git history for this repo. Then supervisor using information from that run, called coder again to read content of changed in commit file. If i would've previously run some other queries related to target filed, summary of this file's content would've been fetched from long term memory

## Reflection
What worked well: agent specialization, single responsibility of agents, tools  
Where the system behaved unexpectedly or failed: excessive or on contrary abscence of `search_arxiv` tool calls  
How you would extend or refine the system if you had more time: first of all deeply enhance prompts, add more tools for coder and researcher, make memory dedicated for specific user (now most of it is stored in checkpointer where memory is identified by thread_id), refactor, frontend, vector search over files in repo, make git tools dynamic (now for sake of simplicity those are just bounded to current dir), human in the loop tool accept/reject, add message history pruning (to avoid exceeding max tokens limit)