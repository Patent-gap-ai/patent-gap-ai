# Running the ChromaDB memory layer

All commands run inside the `Backend` folder. Your project's `.env` is read automatically.

## 1. One-time setup
```
pip install chromadb fastmcp mongomock # only these are needed for the standalone self-test
```
MongoDB must be running (the same connection string your project already uses).
For local use, set `CHROMA_MODE=persistent` in `.env` (the default starts a local Chroma server on port 8001).

## 2. Choose how text becomes vectors
| MEMORY_EMBED_PROVIDER | Needs | Use for |
|---|---|---|
| `demo` | nothing | trying the code. NOT meaning-based, only shared words match |
| `local` | `sentence-transformers`, internet once (~90 MB download) | real searches, offline afterwards |
| `gemini` (default) | `GEMINI_API_KEY` | real searches |
| `openai` | `OPENAI_API_KEY` | real searches |

PowerShell: `$env:MEMORY_EMBED_PROVIDER="demo"`   cmd: `set MEMORY_EMBED_PROVIDER=demo`   Mac/Linux: `export MEMORY_EMBED_PROVIDER=demo`
Pick ONE provider per user and keep it. A different one is refused (their vectors are not compatible).

## 3. Run
```
python test_memory.py          # standalone self-test with mocked Mongo adapters + temporary Chroma
python run_memory_demo.py      # the same flow on YOUR MongoDB + ChromaDB, using a throwaway demo case
```
The Flask app also exposes case memory endpoints. They use the logged-in session identity (not `X-User-ID`):
`GET /api/cases/<case_id>/memory`, `POST /api/cases/<case_id>/memory`,
`POST /api/cases/<case_id>/memory/search`, and `POST /api/cases/<case_id>/memory/context`.
For a first local trial, set `MEMORY_EMBED_PROVIDER=demo`; this only matches overlapping words and is not suitable for investigations.
MCP server for one agent (the launcher sets who it is, the model cannot):
```
MEMORY_USER_ID=alice MEMORY_AGENT=claim_isolator_classifier MEMORY_CASE_ID=<case id> python -m mcp_tools.memory_mcp
```

## 4. If something fails
- `No module named database` -> you are not inside `Backend`.
- Mongo connection error -> start MongoDB; check PYTHON_MONGODB_CONNECT_STRING.
- Chroma server error -> set `CHROMA_MODE=persistent`.
- "embedding model is not available" -> set MEMORY_EMBED_PROVIDER (step 2).
- Memory saved but `indexed: False` -> the model was down; later call `reindex_unindexed(<owner id>)`.
