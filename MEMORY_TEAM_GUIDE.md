# Memory layer: how teammates use it (API version 1.0)

Import only from `structures.memory` (the names in `__all__`). Every call is keyword-only and needs:
- `user_id`: the VERIFIED user (from login, never from the model or a request body)
- `agent`: one of the names in `AGENT_ACCESS` (`claim_isolator_classifier`, `patent_infringement_analyzer`, ...)
- `case_id` (and `patent_id` when a case holds several patents)

## Jinsy (orchestrator / reruns)
```python
from structures import memory as m
# before an agent runs: give it earlier findings for its prompt
ctx = m.build_agent_context(user_id=u, agent="patent_infringement_analyzer", case_id=c, patent_id=p, query=claim_text)
# after it finishes: save what it found (always saved as unverified)
m.save_memories(user_id=u, agent="claim_isolator_classifier", memory_type="claim", case_id=c, patent_id=p, contents=claims)
# after the guardrail checks pass:
m.set_memory_status(user_id=u, agent="verification", memory_id=mid, status="verified", verification={"check": "quote_in_source"})
# case deleted:
m.delete_case_memories(user_id=u, agent="host", case_id=c)
```
Re-run after an outage: `m.reindex_unindexed(owner_id)`.

## Neethu (MCP tools)
Launch one server per agent run; the launcher fixes who is calling:
```python
from mcp_tools.memory_mcp import build_memory_server
server = build_memory_server(user_id=u, agent="claim_isolator_classifier", case_id=c)
```
Tools the model sees depend on the agent: save_memory, save_memories, correct_memory, search_memory,
get_agent_context, get_memory, list_case_memory, memory_summary (+ verify_memory for verification/user/host).
Errors return `{"error": "PERMISSION_DENIED" | "INVALID_INPUT" | "UNAVAILABLE", "message": ...}`.

## Ardra (Flask)
Pass the authenticated `session` user as `user_id`. Do NOT read it from the `X-User-ID` header for memory calls.
PermissionError -> HTTP 403, ValueError -> 400, RuntimeError -> 503.

## Rules everyone must know
- Memory is a lead, not proof. Check `verified` / `status` on every result.
- A correction is a new version (`update_memory`); old versions stay as history.
- One embedding provider per user (`MEMORY_EMBED_PROVIDER`); a different one is refused.
- To change who may read/write what, edit `AGENT_ACCESS` in `structures/memory.py` and rerun `python test_memory.py`.
