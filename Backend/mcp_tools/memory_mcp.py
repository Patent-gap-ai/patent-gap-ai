"""Memory tools for ONE agent, served over MCP (FastMCP).

Identity is fixed by whoever LAUNCHES the server (the orchestrator), never by the model:
    MEMORY_USER_ID=alice MEMORY_AGENT=claim_isolator_classifier MEMORY_CASE_ID=case1 python -m mcp_tools.memory_mcp
The model only sends the content or the query. It cannot pick another user, agent or case, and it is
only shown the tools its agent is allowed to use (see AGENT_ACCESS in structures/memory.py).
Errors come back as {"error": CODE, "message": ...} so the model can read them and correct itself.
"""
import functools
import os

try:
    from fastmcp import FastMCP
except ImportError:  # same fallback style as mcp_tools/mcp.py
    from mcp.server.fastmcp import FastMCP

from structures import memory as m


def _safe(fn):
    @functools.wraps(fn)
    def run(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except PermissionError as e:
            return {"error": "PERMISSION_DENIED", "message": str(e)}
        except ValueError as e:
            return {"error": "INVALID_INPUT", "message": str(e)}
        except RuntimeError as e:
            return {"error": "UNAVAILABLE", "message": str(e)}
    return run


def build_memory_server(user_id: str, agent: str, case_id: str = "", embedder=None) -> "FastMCP":
    rules = m.AGENT_ACCESS.get(agent)
    if not user_id or rules is None:
        raise ValueError("a known agent and a user_id are required")
    who = dict(user_id=user_id, agent=agent)
    mcp = FastMCP(f"patent-gap-memory-{agent}")

    if rules.get("write"):
        @mcp.tool
        @_safe
        def save_memory(memory_type: str, content: str, patent_id: str = "", source_ref: str = "", evidence: list[dict] | None = None) -> dict:
            """Save one finding in this case's memory. memory_type is one of the types you may write.
            evidence: [{"url": ..., "quote": ...}]. New memory is always unverified."""
            return m.save_memory(**who, case_id=case_id, memory_type=memory_type, content=content,
                                 patent_id=patent_id, source_ref=source_ref, evidence=evidence, embedder=embedder)

        @mcp.tool
        @_safe
        def save_memories(memory_type: str, contents: list[str], patent_id: str = "", source_ref: str = "") -> dict:
            """Save several findings of one type at once (for example all claims of a patent)."""
            return {"saved": m.save_memories(**who, case_id=case_id, memory_type=memory_type,
                                             contents=contents, patent_id=patent_id, source_ref=source_ref, embedder=embedder)}

        @mcp.tool
        @_safe
        def correct_memory(memory_id: str, content: str) -> dict:
            """Replace a finding with a corrected version. The old version is kept as history."""
            return m.update_memory(**who, memory_id=memory_id, content=content, embedder=embedder)

    if rules.get("read"):
        @mcp.tool
        @_safe
        def search_memory(query: str, k: int = 5, memory_type: str | None = None, patent_id: str = "",
                          verified_only: bool = False, across_cases: bool = False) -> dict:
            """Find saved findings similar to the query. Each result shows its status (verified or not)
            and source. A match is a lead, not proof."""
            return {"results": m.retrieve_memory(**who, case_id=case_id, query=query, patent_id=patent_id, k=min(max(k, 1), 20),
                                                 memory_type=memory_type, verified_only=verified_only,
                                                 across_cases=across_cases, embedder=embedder)}

        @mcp.tool
        @_safe
        def get_agent_context(query: str, patent_id: str = "", k: int = 5, max_chars: int = 3000) -> dict:
            """Ready-to-use prompt text with the best matching findings, labelled and sourced."""
            return {"context": m.build_agent_context(**who, case_id=case_id, query=query, k=min(max(k, 1), 20),
                                                     max_chars=min(max_chars, 8000), embedder=embedder)}

        @mcp.tool
        @_safe
        def get_memory(memory_id: str) -> dict:
            """One saved finding by id."""
            return m.get_memory(**who, memory_id=memory_id) or {"error": "NOT_FOUND", "message": "no such memory"}

        @mcp.tool
        @_safe
        def list_case_memory(memory_type: str | None = None, patent_id: str = "") -> dict:
            """All current findings of this case that you may read."""
            return {"memories": m.list_case_memories(**who, case_id=case_id, memory_type=memory_type, patent_id=patent_id)}

        @mcp.tool
        @_safe
        def memory_summary() -> dict:
            """Counts of saved findings by type and status for this case."""
            return m.memory_summary(**who, case_id=case_id)

    if agent in m.STATUS_SETTERS:
        @mcp.tool
        @_safe
        def verify_memory(memory_id: str, status: str, note: str = "") -> dict:
            """Mark a finding verified, disputed or unverified. 'verified' needs evidence or a source."""
            return m.set_memory_status(**who, memory_id=memory_id, status=status,
                                       verification={"note": note} if note else None)

    return mcp


def main() -> None:
    build_memory_server(os.environ.get("MEMORY_USER_ID", ""), os.environ.get("MEMORY_AGENT", ""),
                        os.environ.get("MEMORY_CASE_ID", "")).run()


if __name__ == "__main__":
    main()
