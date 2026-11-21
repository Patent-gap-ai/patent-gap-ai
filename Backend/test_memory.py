"""Run from the Backend folder: python test_memory.py
Standalone memory-layer test: mocked Mongo helpers + temporary local Chroma and test embeddings.
It does not need the full Patent Gap app, MongoDB, API keys, or internet. Needs mongomock,
chromadb, and fastmcp installed in the active environment."""
import hashlib
import asyncio
import math
import os
import sys
import tempfile
import types
import re

TMP = tempfile.mkdtemp()
os.environ.update({"ENVIRONMENT": "dev", "CASE_DATABASE_NAME_DEV": "cases_t",
                   "CHROMA_MODE": "persistent", "CHROMA_PERSIST_DIR": TMP})
sys.path.insert(0, os.getcwd())

import mongomock
DB = mongomock.MongoClient()["patent_gap_test"]

# Provide only the adapters used by the memory layer. This keeps its self-test runnable
# when the rest of Patent Gap AI (database/, env_controller.py, ai_workflow/) is absent.
import chromadb
from chromadb.config import Settings

chroma_client = chromadb.PersistentClient(path=TMP, settings=Settings(anonymized_telemetry=False))
database_stub = types.ModuleType("database")
database_stub.connect_to_database = lambda: DB
database_stub.addDataById = lambda db, collection, record: (
    db[collection].insert_one(record).inserted_id
)
database_stub.updateDataById = lambda db, collection, values, record_id: (
    db[collection].update_one({"_id": record_id}, {"$set": values}).matched_count > 0
)
database_stub.deleteDataById = lambda db, collection, record_id: (
    db[collection].delete_one({"_id": record_id}).deleted_count > 0
)
database_stub.getDataById = lambda db, collection, record_id: db[collection].find_one({"_id": record_id})
database_stub.paginateDataByQuery = lambda db, collection, query=None, page=1, sort_field="created_at", sort_direction=-1: {
    "items": list(db[collection].find(query or {}).sort(sort_field, sort_direction)),
    "pagination": {"page": page, "page_size": 0, "total": db[collection].count_documents(query or {}),
                   "total_pages": 0, "has_next": False},
}
database_stub.chroma = types.SimpleNamespace(
    get_chroma_client=lambda: chroma_client,
    _sanitize_user_id=lambda value: re.sub(r"[^a-zA-Z0-9_-]+", "_", (value or "anonymous").strip()) or "anonymous",
    _null_embedding_function=lambda: None,
)
sys.modules["database"] = database_stub

env_stub = types.ModuleType("env_controller")
env_stub.getCaseDatabaseName = lambda: "cases_t"
env_stub.getAuditDatabaseName = lambda: "audit_dev"
sys.modules["env_controller"] = env_stub

ai_stub = types.ModuleType("ai_workflow")
ai_config_stub = types.ModuleType("ai_workflow.config")
ai_config_stub.get_gemini_embedding_model = lambda: "text-embedding-004"
ai_embeddings_stub = types.ModuleType("ai_workflow.embeddings")
ai_embeddings_stub.embed_texts_gemini = lambda texts: None
ai_embeddings_stub.embed_texts_openai = lambda texts: None
sys.modules["ai_workflow"] = ai_stub
sys.modules["ai_workflow.config"] = ai_config_stub
sys.modules["ai_workflow.embeddings"] = ai_embeddings_stub

import structures.memory as mem
from structures.memory import Embedder


def words_embedding(texts):
    out = []
    for t in texts:
        v = [0.0] * 64
        for w in t.lower().split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / n for x in v])
    return out


E = Embedder("test:words-64", words_embedding)
DOWN = Embedder("test:words-64", lambda texts: None)      # same model, but unavailable
OTHER = Embedder("test:another-model", words_embedding)


def raises(exc, fn, **kw):
    try:
        fn(**kw)
    except exc:
        return True
    return False


# --- two cases. alice owns case1 and case2; bob is an editor, carol a viewer, dave has no access.
DB["cases_t"].insert_many([
    {"_id": "case1", "created_by": "alice", "assigned_editors": ["bob"], "assigned_to": "carol"},
    {"_id": "case2", "created_by": "alice"},
    {"_id": "case3", "created_by": "alice"},
])
vec = lambda owner: mem._vector_collection(owner)
base = dict(user_id="alice", case_id="case1", embedder=E)

# 1. save: Mongo record + vector, unverified by default
r = mem.save_memory(agent="claim_isolator_classifier", memory_type="claim",
                    content="Claim 1: a vacuum cleaner with a cyclone separator and dust container", **base)
assert r["created"] and r["indexed"]
rec = DB["memory_dev"].find_one({"_id": r["memory_id"]})
assert rec["status"] == "unverified" and rec["embed_model"] == "test:words-64" and vec("alice").count() == 1

# 2. exact duplicate is not saved twice
again = mem.save_memory(agent="claim_isolator_classifier", memory_type="claim",
                        content="claim 1:  a vacuum cleaner with a cyclone separator and dust container", **base)
assert again == {"memory_id": r["memory_id"], "created": False, "indexed": True}

# 3. agent matrix: authorised read ok; wrong type, wrong write, unknown agent, status forgery all denied
got = mem.get_memory(user_id="alice", agent="patent_infringement_analyzer", memory_id=r["memory_id"])
assert got["verified"] is False and got["status"] == "unverified"             # status is always visible
assert raises(PermissionError, mem.get_memory, user_id="alice", agent="patent_data_isolator", memory_id=r["memory_id"])
assert raises(PermissionError, mem.save_memory, agent="product_source_miner", memory_type="claim", content="x y", **base)
assert raises(PermissionError, mem.get_memory, user_id="alice", agent="stranger", memory_id=r["memory_id"])
assert raises(ValueError, mem.save_memory, agent="claim_isolator_classifier", memory_type="claim", content="z", status="verified", **base)

# 4. user and case access: carol views but cannot write; dave sees nothing; bob (editor) can write
assert mem.get_memory(user_id="carol", agent="chart_builder", memory_id=r["memory_id"])["memory_type"] == "claim"
assert raises(PermissionError, mem.save_memory, agent="claim_isolator_classifier", memory_type="claim", content="c", user_id="carol", case_id="case1", embedder=E)
assert raises(PermissionError, mem.get_memory, user_id="dave", agent="user", memory_id=r["memory_id"])
assert raises(PermissionError, mem.save_memory, agent="claim_isolator_classifier", memory_type="claim", content="d", user_id="dave", case_id="case1", embedder=E)
b = mem.save_memory(agent="product_source_miner", memory_type="product", content="Acme vac 3000 cyclone product page", user_id="bob", case_id="case1", embedder=E)
assert b["created"] and DB["memory_dev"].find_one({"_id": b["memory_id"]})["owner_id"] == "alice"   # stored in the case owner's index
assert vec("bob") is None and vec("alice").count() == 2

# 5. semantic search: best match first, scoped to the case, carrying status
mem.save_memory(agent="patent_data_isolator", memory_type="patent", content="Battery charger with pulsed current", **base)
hits = mem.retrieve_memory(user_id="alice", agent="patent_infringement_analyzer", case_id="case1",
                           query="cyclone vacuum cleaner dust", embedder=E)
assert "vacuum" in hits[0]["content"] and hits[0]["status"] == "unverified" and hits[0]["score"] > hits[-1]["score"]
only_claims = mem.retrieve_memory(user_id="alice", agent="patent_infringement_analyzer", case_id="case1",
                                  query="cyclone vacuum", memory_type="claim", embedder=E)
assert [h["memory_type"] for h in only_claims] == ["claim"]
assert raises(PermissionError, mem.retrieve_memory, user_id="alice", agent="chart_builder", case_id="case1", query="x", memory_type="patent", embedder=E)

# 6. memory from another case, found only with across_cases, and only for the owner
mem.save_memory(agent="claim_isolator_classifier", memory_type="claim", content="Claim 9: vacuum nozzle with rotating brush",
                user_id="alice", case_id="case2", embedder=E)
one_case = mem.retrieve_memory(user_id="alice", agent="claim_isolator_classifier", case_id="case1", query="vacuum nozzle brush", embedder=E)
assert all(h["case_id"] == "case1" for h in one_case)
past = mem.retrieve_memory(user_id="alice", agent="claim_isolator_classifier", across_cases=True, query="vacuum nozzle brush", embedder=E)
assert past[0]["case_id"] == "case2"
assert mem.retrieve_memory(user_id="carol", agent="claim_isolator_classifier", across_cases=True, query="vacuum", embedder=E) == []

# 7. a correction is a new version; the old one is hidden and its vector removed; history stays in Mongo
n_before = vec("alice").count()
upd = mem.update_memory(user_id="alice", agent="claim_isolator_classifier", memory_id=r["memory_id"],
                        content="Claim 1: a vacuum cleaner with a cyclone separator, dust container and filter", embedder=E)
new = DB["memory_dev"].find_one({"_id": upd["memory_id"]})
assert new["version"] == 2 and new["supersedes"] == r["memory_id"]
assert DB["memory_dev"].find_one({"_id": r["memory_id"]})["status"] == "superseded" and vec("alice").count() == n_before
assert r["memory_id"] not in [h["memory_id"] for h in mem.retrieve_memory(user_id="alice", agent="claim_isolator_classifier", case_id="case1", query="vacuum cyclone", embedder=E)]
assert raises(ValueError, mem.update_memory, user_id="alice", agent="claim_isolator_classifier", memory_id=r["memory_id"], content="again", embedder=E)

# 8. verification: only allowed roles, only with evidence; verified_only filters
a = mem.save_memory(agent="patent_infringement_analyzer", memory_type="analysis", content="Product Acme vac 3000 reads on claim 1 cyclone", **base)["memory_id"]
assert raises(PermissionError, mem.set_memory_status, user_id="alice", agent="patent_infringement_analyzer", memory_id=a, status="verified")
assert raises(ValueError, mem.set_memory_status, user_id="alice", agent="verification", memory_id=a, status="verified")   # no evidence
ev = mem.save_memory(agent="patent_infringement_analyzer", memory_type="analysis", content="Acme vac 3000 has a cyclone separator",
                     evidence=[{"url": "https://acme.example/vac", "quote": "cyclone separator"}], **base)["memory_id"]
done = mem.set_memory_status(user_id="alice", agent="verification", memory_id=ev, status="verified", verification={"check": "quote_in_source"})
assert done["verified"] and done["verification"]["check"] == "quote_in_source"
vo = mem.retrieve_memory(user_id="alice", agent="report_builder", case_id="case1", query="Acme cyclone", verified_only=True, embedder=E)
assert [h["memory_id"] for h in vo] == [ev]

# 9. embedding down: the record is kept unindexed (no fake vector), then re-indexed later
off = mem.save_memory(agent="claim_isolator_classifier", memory_type="claim", content="Claim 4: a hinge with a spring", user_id="alice", case_id="case1", embedder=DOWN)
assert off["created"] and off["indexed"] is False and DB["memory_dev"].find_one({"_id": off["memory_id"]})["indexed"] is False
assert mem.reindex_unindexed("alice", E) == 1 and "hinge" in mem.retrieve_memory(user_id="alice", agent="claim_isolator_classifier", case_id="case1", query="hinge spring", embedder=E)[0]["content"]

# 10. one embedding model per user: a different one is refused
assert raises(ValueError, mem.save_memory, agent="claim_isolator_classifier", memory_type="claim", content="Claim 5: a latch", user_id="alice", case_id="case1", embedder=OTHER)

# 11. private and user-level scopes
p = mem.save_memory(agent="claim_isolator_classifier", memory_type="claim", content="my private working note", scope="agent", **base)["memory_id"]
assert mem.get_memory(user_id="alice", agent="claim_isolator_classifier", memory_id=p)["scope"] == "agent"
assert raises(PermissionError, mem.get_memory, user_id="alice", agent="patent_infringement_analyzer", memory_id=p)
pref = mem.save_memory(agent="user", memory_type="preference", content="Always quote evidence word for word", scope="user", user_id="alice", embedder=E)["memory_id"]
assert DB["memory_dev"].find_one({"_id": pref})["status"] == "verified"
assert mem.retrieve_memory(user_id="alice", agent="claim_isolator_classifier", query="quote evidence", embedder=E)[0]["memory_id"] == pref
assert raises(PermissionError, mem.get_memory, user_id="dave", agent="user", memory_id=pref)

# 12. delete: needs the delete role on the case AND an agent that may delete
assert raises(PermissionError, mem.delete_memory, user_id="bob", agent="user", memory_id=b["memory_id"])                 # editor, not owner
assert raises(PermissionError, mem.delete_memory, user_id="alice", agent="claim_isolator_classifier", memory_id=b["memory_id"])
n = vec("alice").count()
assert mem.delete_memory(user_id="alice", agent="user", memory_id=b["memory_id"]) and vec("alice").count() == n - 1
assert DB["memory_dev"].find_one({"_id": b["memory_id"]}) is None

# 13. access revoked: bob stops being an editor and loses memory access at once
assert mem.retrieve_memory(user_id="bob", agent="claim_isolator_classifier", case_id="case1", query="vacuum", embedder=E)
DB["cases_t"].update_one({"_id": "case1"}, {"$set": {"assigned_editors": []}})
assert raises(PermissionError, mem.retrieve_memory, user_id="bob", agent="claim_isolator_classifier", case_id="case1", query="vacuum", embedder=E)

# 14. case lifecycle: delete every memory of a case from both stores
assert raises(PermissionError, mem.delete_case_memories, user_id="carol", agent="user", case_id="case1")
count_case1 = DB["memory_dev"].count_documents({"case_id": "case1"})
assert mem.delete_case_memories(user_id="alice", agent="user", case_id="case1") == count_case1 > 0
assert DB["memory_dev"].count_documents({"case_id": "case1"}) == 0
assert vec("alice").get(where={"mem_case_id": "case1"})["ids"] == []
assert DB["memory_dev"].count_documents({"case_id": "case2"}) == 1                     # other cases untouched

# 15. audit trail: actions and denials recorded, never the memory text
kinds = {row["kind"] for row in DB["audit_dev"].find({})}
assert {"memory_saved", "memory_updated", "memory_retrieved", "memory_status_changed", "memory_deleted", "memory_denied"} <= kinds
assert "cyclone" not in str(list(DB["audit_dev"].find({})))

# 16. batch save: ONE embedding call for many memories; re-running creates nothing and embeds nothing
calls = []
def counting(texts):
    calls.append(len(texts))
    return words_embedding(texts)
C = Embedder("test:words-64", counting)
claims = ["Claim 2: the cleaner of claim 1 wherein the filter is washable", "Claim 3: the cleaner of claim 1 with a hose",
          "Claim 4: a battery pack with a latch"]
saved = mem.save_memories(user_id="alice", agent="claim_isolator_classifier", memory_type="claim", contents=claims, case_id="case2", embedder=C)
assert [x["created"] for x in saved] == [True] * 3 and all(x["indexed"] for x in saved) and calls == [3]
again = mem.save_memories(user_id="alice", agent="claim_isolator_classifier", memory_type="claim", contents=claims, case_id="case2", embedder=C)
assert [x["created"] for x in again] == [False] * 3 and calls == [3]                                  # no new embedding call
mixed = mem.save_memories(user_id="alice", agent="claim_isolator_classifier", memory_type="claim",
                          contents=[claims[0], "Claim 5: a bumper"], case_id="case2", embedder=C)
assert [x["created"] for x in mixed] == [False, True] and calls == [3, 1]
assert raises(PermissionError, mem.save_memories, user_id="alice", agent="product_source_miner", memory_type="claim", contents=["x y"], case_id="case2", embedder=C)
assert raises(ValueError, mem.save_memories, user_id="alice", agent="claim_isolator_classifier", memory_type="claim", contents=["ok", "  "], case_id="case2", embedder=C)

# 17. listing and summary respect the agent matrix and the case
listed = mem.list_case_memories(user_id="alice", agent="patent_infringement_analyzer", case_id="case2")
assert len(listed) == 5 and {x["memory_type"] for x in listed} == {"claim"}                         # 1 earlier + 4 new claims
assert mem.list_case_memories(user_id="alice", agent="patent_data_isolator", case_id="case2", memory_type="patent") == []
assert mem.memory_summary(user_id="alice", agent="report_builder", case_id="case2") == {"total": 5, "by_type": {"claim": 5}, "by_status": {"unverified": 5}}
assert raises(PermissionError, mem.list_case_memories, user_id="dave", agent="user", case_id="case2")
assert raises(PermissionError, mem.list_case_memories, user_id="alice", agent="patent_data_isolator", case_id="case2", memory_type="claim")

# 18. agent context: labelled, sourced, size-limited; empty when there is nothing
ev2 = mem.save_memory(user_id="alice", agent="patent_infringement_analyzer", memory_type="analysis", case_id="case2", embedder=E,
                      content="Acme filter is washable like claim 2", evidence=[{"url": "https://acme.example/filter", "quote": "washable filter"}])["memory_id"]
mem.set_memory_status(user_id="alice", agent="verification", memory_id=ev2, status="verified")
ctx = mem.build_agent_context(user_id="alice", agent="report_builder", case_id="case2", query="washable filter", embedder=E)
assert ctx.startswith("Earlier findings") and "[VERIFIED] (analysis" in ctx and "source=https://acme.example/filter" in ctx and "[UNVERIFIED] (claim" in ctx
short = mem.build_agent_context(user_id="alice", agent="report_builder", case_id="case2", query="washable filter", max_chars=120, embedder=E)
assert short.count("\n") <= 1 and len(short) < len(ctx)
assert mem.build_agent_context(user_id="alice", agent="report_builder", case_id="case3", query="anything", embedder=E) == ""

# 19. embedding providers: demo and local (stubbed), pinned names, a failing model gives None not fake vectors
os.environ["MEMORY_EMBED_PROVIDER"] = "demo"
demo = mem.default_embedder()
assert demo.name.startswith("demo:") and len(demo(["a b"])[0]) == 64
class Arr(list):
    def tolist(self): return list(self)
class StubModel:
    def __init__(self, name): self.name = name
    def encode(self, texts, normalize_embeddings=True): return Arr([[0.1, 0.2, 0.3] for _ in texts])
sys.modules["sentence_transformers"] = types.SimpleNamespace(SentenceTransformer=StubModel)
os.environ["MEMORY_EMBED_PROVIDER"] = "local"
local = mem.default_embedder()
assert local.name == "local:sentence-transformers/all-MiniLM-L6-v2" and local(["x", "y"]) == [[0.1, 0.2, 0.3]] * 2
def boom(name): raise OSError("no internet and no cached model")
sys.modules["sentence_transformers"] = types.SimpleNamespace(SentenceTransformer=boom)
mem._local_model = None
assert mem.default_embedder()(["x"]) is None
os.environ["MEMORY_EMBED_PROVIDER"] = "openai"
assert mem.default_embedder().name.startswith("openai:")
del os.environ["MEMORY_EMBED_PROVIDER"]
assert mem.default_embedder().name.startswith("gemini:")

# 20. MCP: each agent sees only its tools; identity is fixed by the launcher; errors are readable
from mcp_tools.memory_mcp import build_memory_server
from fastmcp import Client

async def mcp_checks():
    async with Client(build_memory_server("alice", "claim_isolator_classifier", "case2", embedder=E)) as c:
        tools = {t.name: t for t in await c.list_tools()}
        assert set(tools) == {"save_memory", "save_memories", "correct_memory", "search_memory", "get_agent_context",
                              "get_memory", "list_case_memory", "memory_summary"}
        assert not {"user_id", "agent", "case_id"} & set(tools["save_memory"].input_schema["properties"])
        res = await c.call_tool("save_memory", {"memory_type": "claim", "content": "Claim 6: a wheel with a brake"})
        assert res.data["created"] is True
        found = await c.call_tool("search_memory", {"query": "wheel brake"})
        assert "wheel" in found.data["results"][0]["content"]
        bad = await c.call_tool("save_memory", {"memory_type": "product", "content": "x y"})
        assert bad.data["error"] == "PERMISSION_DENIED"
    async with Client(build_memory_server("alice", "chart_builder", "case2", embedder=E)) as c:   # read-only agent: no save tools at all
        assert {t.name for t in await c.list_tools()} == {"search_memory", "get_agent_context", "get_memory", "list_case_memory", "memory_summary"}
    async with Client(build_memory_server("alice", "verification", "case2", embedder=E)) as c:
        names = {t.name for t in await c.list_tools()}
        assert "verify_memory" in names and "save_memory" in names
        out = await c.call_tool("verify_memory", {"memory_id": "mem_missing", "status": "verified"})
        assert out.data["error"] == "INVALID_INPUT"
    async with Client(build_memory_server("dave", "claim_isolator_classifier", "case2", embedder=E)) as c:   # dave has no access to case2
        out = await c.call_tool("save_memory", {"memory_type": "claim", "content": "Claim 7: a spring"})
        assert out.data["error"] == "PERMISSION_DENIED"
        out = await c.call_tool("search_memory", {"query": "vacuum"})
        assert out.data["error"] == "PERMISSION_DENIED"
asyncio.run(mcp_checks())
assert raises(ValueError, build_memory_server, user_id="alice", agent="stranger")
assert raises(ValueError, build_memory_server, user_id="", agent="user")

# 21. patent id: one case holds memory about several patents; searches and lists can be limited to one
mem.save_memories(user_id="alice", agent="claim_isolator_classifier", memory_type="claim", case_id="case3", patent_id="US100",
                  contents=["Claim 1 of US100: a hinge with a spring", "Claim 2 of US100: the hinge with a damper"], embedder=E)
mem.save_memories(user_id="alice", agent="claim_isolator_classifier", memory_type="claim", case_id="case3", patent_id="US200",
                  contents=["Claim 1 of US200: a hinge with a spring and a latch"], embedder=E)
only100 = mem.retrieve_memory(user_id="alice", agent="patent_infringement_analyzer", case_id="case3", patent_id="US100", query="hinge spring", k=10, embedder=E)
assert len(only100) == 2 and {h["patent_id"] for h in only100} == {"US100"}
both = mem.retrieve_memory(user_id="alice", agent="patent_infringement_analyzer", case_id="case3", query="hinge spring", k=10, embedder=E)
assert {h["patent_id"] for h in both} == {"US100", "US200"}
assert [h["patent_id"] for h in mem.list_case_memories(user_id="alice", agent="user", case_id="case3", patent_id="US200")] == ["US200"]
same = "Claim 1 of US100: a hinge with a spring"
assert mem.save_memory(user_id="alice", agent="claim_isolator_classifier", memory_type="claim", case_id="case3", patent_id="US300", content=same, embedder=E)["created"]
assert mem.save_memory(user_id="alice", agent="claim_isolator_classifier", memory_type="claim", case_id="case3", patent_id="US100", content=same, embedder=E)["created"] is False
ctx3 = mem.build_agent_context(user_id="alice", agent="report_builder", case_id="case3", patent_id="US200", query="hinge", embedder=E)
assert "patent=US200" in ctx3 and "US100" not in ctx3
fixed = mem.update_memory(user_id="alice", agent="claim_isolator_classifier", memory_id=only100[0]["memory_id"], content="Claim 1 of US100: a hinge with a strong spring", embedder=E)
assert mem.get_memory(user_id="alice", agent="user", memory_id=fixed["memory_id"])["patent_id"] == "US100"

async def mcp_patent_checks():
    async with Client(build_memory_server("alice", "claim_isolator_classifier", "case3", embedder=E)) as c:
        await c.call_tool("save_memory", {"memory_type": "claim", "content": "Claim 1 of US400: a gasket", "patent_id": "US400"})
        got = await c.call_tool("search_memory", {"query": "gasket", "patent_id": "US400"})
        assert [h["patent_id"] for h in got.data["results"]] == ["US400"]
asyncio.run(mcp_patent_checks())

# 22. public surface: every name in __all__ exists and public calls are keyword-only
import inspect
assert all(hasattr(mem, n) for n in mem.__all__) and mem.MEMORY_API_VERSION == "1.0"
for fn in (mem.save_memory, mem.save_memories, mem.retrieve_memory, mem.get_memory, mem.update_memory, mem.set_memory_status,
           mem.delete_memory, mem.delete_case_memories, mem.list_case_memories, mem.memory_summary, mem.build_agent_context):
    assert all(q.kind is q.KEYWORD_ONLY for q in inspect.signature(fn).parameters.values()), fn.__name__

print("all memory tests passed")
