"""Investigation memory manager.

One central place for saving, finding and protecting what the investigations learn.
  MongoDB = the truth (full records).   ChromaDB = a searchable index that points back to Mongo.

Rules enforced here in plain code, never by a model:
  1. Every call names the user and the agent. Unknown agent = no access.
  2. AGENT_ACCESS says which memory types each agent may read or write (draft, team to confirm).
  3. The live case record decides what the user may do (view / edit / delete), so revoking
     access takes effect immediately.
  4. Only 'verification', 'user' or 'host' can mark memory verified, and only with evidence.
  5. Memory is never overwritten: a correction is a new version that supersedes the old one.
  6. Memory vectors come from ONE pinned embedding model and live in their own Chroma collection.
     If that model is down we keep the record unindexed. We never use the project's TF-IDF
     fallback (a different vector space with the same 384 numbers as some real models), and never
     store made-up vectors.
  7. Retrieval re-reads Mongo and re-checks permission for every hit; Chroma is only a hint.
  8. Audit rows hold ids and actions, never memory content.
Identity (user_id) must already be authenticated by the caller (the Flask / MCP layer).
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from ai_workflow.config import get_gemini_embedding_model
from ai_workflow.embeddings import embed_texts_gemini, embed_texts_openai
from database import addDataById, chroma, connect_to_database, deleteDataById, updateDataById
from env_controller import getCaseDatabaseName
from structures.audit import add_audit_entry

MEMORY_API_VERSION = "1.0"  # bump the first number when a public function changes in a way that breaks callers

__all__ = [  # the public surface: teammates call these, nothing else
    "MEMORY_API_VERSION", "MEMORY_TYPES", "STATUSES", "SCOPES", "AGENT_ACCESS", "STATUS_SETTERS",
    "Embedder", "default_embedder", "check_memory_permission",
    "save_memory", "save_memories", "get_memory", "update_memory", "set_memory_status",
    "delete_memory", "delete_case_memories", "retrieve_memory", "list_case_memories",
    "memory_summary", "build_agent_context", "reindex_unindexed",
]

MEMORY_TYPES = ("patent", "claim", "product", "analysis", "verification", "clarification", "preference")
STATUSES = ("suggestion", "unverified", "verified", "disputed", "superseded")
SCOPES = ("agent", "case", "user")  # agent: only its author; case: agents allowed by the matrix; user: no case

_CASE_TYPES = {"patent", "claim", "product", "analysis", "verification", "clarification"}
_EVERYTHING = set(MEMORY_TYPES)

# DRAFT matrix. Agent names follow the team lead's plan; confirm with Jinsy / Neethu.
# "user" is the human expert acting through the app; "host" is trusted application code.
AGENT_ACCESS: dict[str, dict[str, set[str]]] = {
    "patent_data_isolator": {"read": {"patent", "preference"}, "write": {"patent"}},
    "claim_isolator_classifier": {"read": {"patent", "claim", "clarification", "preference"}, "write": {"claim"}},
    "patent_source_miner": {"read": {"patent", "claim", "clarification", "preference"}, "write": {"patent"}},
    "product_source_miner": {"read": {"claim", "product", "clarification", "preference"}, "write": {"product"}},
    "patent_infringement_analyzer": {"read": {"patent", "claim", "product", "clarification", "preference"}, "write": {"analysis"}},
    "product_infringement_analyzer": {"read": {"patent", "claim", "product", "clarification", "preference"}, "write": {"analysis"}},
    "verification": {"read": _CASE_TYPES | {"preference"}, "write": {"verification"}},
    "chart_builder": {"read": {"claim", "product", "analysis", "verification"}, "write": set()},
    "report_builder": {"read": {"claim", "product", "analysis", "verification"}, "write": set()},
    "user": {"read": _EVERYTHING, "write": {"clarification", "preference"}, "delete": _EVERYTHING},
    "host": {"read": _EVERYTHING, "write": _EVERYTHING, "delete": _EVERYTHING},
}
STATUS_SETTERS = ("verification", "user", "host")


# ---------------------------------------------------------------- the pinned embedding model
class Embedder:
    """A named embedding model. All memory vectors of a user must come from the same one."""

    def __init__(self, name: str, fn: Callable[[list[str]], Optional[list[list[float]]]]):
        self.name, self._fn = name, fn

    def __call__(self, texts: list[str]) -> Optional[list[list[float]]]:
        return self._fn(texts)


_local_model = None


def _local_model_name() -> str:
    return os.environ.get("MEMORY_LOCAL_MODEL") or "sentence-transformers/all-MiniLM-L6-v2"


def _local_embed(texts: list[str]) -> Optional[list[list[float]]]:
    """Local sentence-transformers model. Needs `pip install sentence-transformers`; the first call
    downloads the model once (about 90 MB), after that it works offline. Long text is cut at the
    model's limit (about 256 word pieces), so keep memories short or split them."""
    global _local_model
    try:
        if _local_model is None:
            from sentence_transformers import SentenceTransformer
            _local_model = SentenceTransformer(_local_model_name())
        return _local_model.encode(texts, normalize_embeddings=True).tolist()
    except Exception as e:
        print(f"LOG: local embedding failed ({type(e).__name__}: {e})")
        return None


def _demo_embed(texts: list[str]) -> list[list[float]]:
    """NOT semantic: hashed word counts, so only shared words match. For trying the code with no
    model and no internet. Never use it for real investigations."""
    out = []
    for text in texts:
        v = [0.0] * 64
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / norm for x in v])
    return out


def default_embedder() -> Embedder:
    """MEMORY_EMBED_PROVIDER = gemini (default) | openai | local | demo.
    One provider per user's memory, no automatic fallback: their vectors differ in size and meaning."""
    provider = (os.environ.get("MEMORY_EMBED_PROVIDER") or "gemini").strip().lower()
    if provider == "openai":
        return Embedder("openai:text-embedding-3-small", embed_texts_openai)
    if provider == "local":
        return Embedder(f"local:{_local_model_name()}", _local_embed)
    if provider == "demo":
        return Embedder("demo:hashed-words-64", _demo_embed)
    return Embedder(f"gemini:{get_gemini_embedding_model()}", embed_texts_gemini)


# ---------------------------------------------------------------- storage helpers
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _collection_name() -> str:
    env = os.environ.get("ENVIRONMENT")
    if env == "prod":
        return os.environ.get("MEMORY_DATABASE_NAME_PROD") or "memory"
    if env == "test":
        return os.environ.get("MEMORY_DATABASE_NAME_TEST") or "memory_test"
    return os.environ.get("MEMORY_DATABASE_NAME_DEV") or "memory_dev"


def _coll():
    # Reads go straight to the collection with an exact filter. getDataById() scans the
    # whole collection when an id is missing, which is wrong for a permission-checked store.
    return connect_to_database()[_collection_name()]


def _vector_collection(owner_id: str, embedder: Optional[Embedder] = None):
    """The owner's memory vectors (separate from the claim index). With an embedder: create if
    needed and refuse a different embedding model. Without: just fetch, None if absent (for deletes)."""
    client = chroma.get_chroma_client()
    name = f"memory_{chroma._sanitize_user_id(owner_id)[:48]}_{hashlib.sha1(owner_id.encode()).hexdigest()[:8]}"
    try:
        col = client.get_collection(name=name)
    except Exception:
        if embedder is None:
            return None
        kwargs: dict[str, Any] = {"name": name, "metadata": {"hnsw:space": "cosine", "embed_model": embedder.name}}
        ef = chroma._null_embedding_function()
        if ef is not None:
            kwargs["embedding_function"] = ef
        col = client.get_or_create_collection(**kwargs)
    pinned = (col.metadata or {}).get("embed_model")
    if embedder is not None and pinned != embedder.name:
        raise ValueError(f"this user's memory vectors use '{pinned}', not '{embedder.name}'; "
                         "re-index with one model instead of mixing")
    return col


def _load_case(case_id: str) -> Optional[dict]:
    if not case_id:
        return None
    return connect_to_database()[getCaseDatabaseName()].find_one({"_id": case_id})


def _case_roles(user_id: str, case: Optional[dict]) -> set[str]:
    """Mirrors structures/cases.has_edit_permission / has_delete_permission, plus a view rule
    (cases.py has none; this is the union of who can see or change a case today). Looks the
    case up directly because cases.get_case_by_id() is defined twice and the live one scans every case."""
    if not case or not user_id:
        return set()
    owner_like = user_id in (case.get("created_by"), case.get("owner"))
    editor = user_id in (case.get("assigned_editors") or [])
    viewer = user_id in (case.get("assigned_to"), case.get("accepted_by"))
    roles: set[str] = set()
    if owner_like or editor or viewer:
        roles.add("view")
    if owner_like or editor:
        roles.add("edit")
    if owner_like:
        roles.add("delete")
    return roles


def _owner_of(case: Optional[dict], user_id: str) -> str:
    """Whose collection holds a case's memory: the case creator's, so every editor shares one place."""
    return ((case or {}).get("created_by") or (case or {}).get("owner") or user_id) if case else user_id


def _audit(user_id: str, case_id: str, kind: str, agent: str, **info: Any) -> None:
    payload = {"agent": agent, **{k: v for k, v in info.items() if v not in (None, "")}}
    add_audit_entry(user_id, case_id or "user_memory", kind, text=f"{kind} by {agent}",
                    payload=payload, status=str(info.get("status", "")), source="memory")


def _public(rec: dict, score: Optional[float] = None) -> dict:
    out = {k: rec.get(k) for k in ("memory_id", "memory_type", "scope", "content", "status", "evidence",
                                   "source_ref", "case_id", "created_by_agent", "version", "supersedes",
                                   "verification", "created_at", "patent_id")}
    out["verified"] = rec.get("status") == "verified"
    if score is not None:
        out["score"] = score
    return out


# ---------------------------------------------------------------- permissions
def check_memory_permission(user_id: str, agent: str, action: str, *, case_id: str = "",
                            memory_type: Optional[str] = None, record: Optional[dict] = None,
                            case: Optional[dict] = None) -> tuple[bool, str]:
    """Can this user, acting through this agent, do `action` (read/write/update/delete)? -> (allowed, reason)"""
    if not user_id:
        return False, "user_id is required"
    rules = AGENT_ACCESS.get(agent)
    if rules is None:
        return False, f"unknown agent '{agent}'"
    if action not in ("read", "write", "update", "delete"):
        return False, f"unknown action '{action}'"
    mtype = memory_type or (record or {}).get("memory_type")
    if mtype not in MEMORY_TYPES:
        return False, "unknown memory_type"
    if mtype not in rules.get("write" if action == "update" else action, set()):
        return False, f"agent '{agent}' may not {action} {mtype} memory"
    if record is not None:
        case_id = record.get("case_id") or ""
        if record.get("scope") == "agent" and agent not in (record.get("created_by_agent"), "host"):
            return False, "this memory is private to the agent that wrote it"
    if not case_id:  # user-level memory belongs to exactly one user
        owner = (record or {}).get("owner_id", user_id)
        return (True, "ok") if owner == user_id else (False, "not your user-level memory")
    roles = _case_roles(user_id, case if case is not None else _load_case(case_id))
    need = {"read": "view", "write": "edit", "update": "edit", "delete": "delete"}[action]
    if need not in roles:
        return False, f"user has no {need} access to this case"
    return True, "ok"


# ---------------------------------------------------------------- indexing
def _index_many(recs: list[dict], embedder: Embedder) -> list[bool]:
    """Embed all records in ONE call and write them to Chroma. Returns indexed True/False per record."""
    if not recs:
        return []
    try:
        vectors = embedder([r["content"] for r in recs])
        if not vectors or len(vectors) != len(recs):
            return [False] * len(recs)  # model unavailable: keep the records, mark unindexed, retry later
        by_owner: dict[str, list[int]] = {}
        for i, rec in enumerate(recs):
            by_owner.setdefault(rec["owner_id"], []).append(i)
        for owner, rows in by_owner.items():
            _vector_collection(owner, embedder).upsert(
                ids=[recs[i]["_id"] for i in rows], documents=[recs[i]["content"] for i in rows],
                embeddings=[vectors[i] for i in rows],
                metadatas=[{"memory_id": recs[i]["_id"], "mem_case_id": recs[i]["case_id"],
                            "patent_id": recs[i].get("patent_id", ""),
                            "memory_type": recs[i]["memory_type"], "scope": recs[i]["scope"]} for i in rows])
        for rec in recs:
            updateDataById(connect_to_database(), _collection_name(), {"indexed": True, "embed_model": embedder.name}, rec["_id"])
        return [True] * len(recs)
    except ValueError:
        raise  # a pinned-model mismatch is a configuration error: say so loudly
    except Exception as e:  # never lose the Mongo records because the index failed
        print(f"LOG: memory index failed ({type(e).__name__}: {e})")
        return [False] * len(recs)


def reindex_unindexed(owner_id: str, embedder: Optional[Embedder] = None) -> int:
    """Retry the index for records saved while embedding was unavailable. Returns how many succeeded."""
    embedder = embedder or default_embedder()
    pending = list(_coll().find({"owner_id": owner_id, "indexed": False, "status": {"$ne": "superseded"}}))
    return sum(_index_many(pending, embedder))


# ---------------------------------------------------------------- operations
def _save_many(*, user_id: str, agent: str, memory_type: str, contents: list[str], case_id: str, patent_id: str,
               scope: str, status: str, evidence: Optional[list], source_ref: str, supersedes: Optional[str],
               embedder: Optional[Embedder]) -> list[dict]:
    if memory_type not in MEMORY_TYPES or scope not in SCOPES:
        raise ValueError("unknown memory_type or scope")
    if (scope == "user") == bool(case_id):
        raise ValueError("user-level memory has no case_id; agent and case memory need one")
    if status not in ("suggestion", "unverified"):
        raise ValueError("new memory is 'suggestion' or 'unverified'; use set_memory_status to verify")
    contents = [(c or "").strip() for c in contents]
    if not contents or not all(contents):
        raise ValueError("content is required")
    if supersedes and len(contents) != 1:
        raise ValueError("a correction replaces exactly one memory")

    case = _load_case(case_id)
    if case_id and case is None:
        raise ValueError("case not found")
    ok, why = check_memory_permission(user_id, agent, "write", case_id=case_id, memory_type=memory_type, case=case)
    if not ok:
        _audit(user_id, case_id, "memory_denied", agent, action="write", memory_type=memory_type, reason=why)
        raise PermissionError(why)

    old = None
    if supersedes:
        old = _coll().find_one({"_id": supersedes})
        ok, why = check_memory_permission(user_id, agent, "update", record=old, memory_type=memory_type, case=case) \
            if old else (False, "memory to supersede not found")
        if ok and (old["case_id"] != case_id or old["memory_type"] != memory_type):
            ok, why = False, "can only supersede memory of the same case and type"
        if not ok:
            _audit(user_id, case_id, "memory_denied", agent, action="update", memory_id=supersedes, reason=why)
            raise PermissionError(why)

    embedder = embedder or default_embedder()
    owner_id = _owner_of(case, user_id)
    if agent == "user" and memory_type in ("clarification", "preference"):
        status = "verified"  # the expert's own statement
    results: list[dict] = []
    fresh: list[dict] = []
    for content in contents:
        key = hashlib.sha1(f"{owner_id}|{case_id}|{patent_id}|{memory_type}|{' '.join(content.lower().split())}".encode()).hexdigest()
        existing = None if supersedes else _coll().find_one({"dedup_key": key, "status": {"$ne": "superseded"}})
        if existing:
            results.append({"memory_id": existing["_id"], "created": False, "indexed": bool(existing.get("indexed"))})
            continue
        rec = {
            "_id": f"mem_{uuid.uuid4().hex[:10]}", "memory_id": None, "owner_id": owner_id, "case_id": case_id,
            "patent_id": patent_id, "memory_type": memory_type, "scope": scope, "content": content, "status": status,
            "evidence": list(evidence or []), "source_ref": source_ref, "verification": {},
            "created_by_agent": agent, "created_by_user": user_id,
            "version": (old["version"] + 1) if old else 1, "supersedes": supersedes or "",
            "dedup_key": key, "indexed": False, "embed_model": embedder.name,
            "created_at": _now(), "updated_at": _now(),
        }
        rec["memory_id"] = rec["_id"]
        added = addDataById(connect_to_database(), _collection_name(), rec)
        if "DocumentCreationError" in str(added):
            raise RuntimeError(f"memory could not be saved: {added}")
        fresh.append(rec)
        results.append({"memory_id": rec["_id"], "created": True, "indexed": False})
    if old and fresh:
        updateDataById(connect_to_database(), _collection_name(), {"status": "superseded", "updated_at": _now()}, supersedes)
        try:
            col = _vector_collection(old["owner_id"])
            if col is not None:
                col.delete(ids=[supersedes])
        except Exception as e:
            print(f"LOG: could not drop superseded vector ({e})")
    flags = _index_many(fresh, embedder)  # ONE embedding call for the whole batch
    indexed = {rec["_id"]: flag for rec, flag in zip(fresh, flags)}
    for result in results:
        if result["memory_id"] in indexed:
            result["indexed"] = indexed[result["memory_id"]]
    if fresh:
        _audit(user_id, case_id, "memory_updated" if old else "memory_saved", agent, memory_type=memory_type,
               status=status, supersedes=supersedes, count=len(fresh), memory_ids=[r["_id"] for r in fresh],
               indexed=sum(flags))
    return results


def save_memory(*, user_id: str, agent: str, memory_type: str, content: str, case_id: str = "",
                patent_id: str = "", scope: str = "case", status: str = "unverified", evidence: Optional[list] = None,
                source_ref: str = "", supersedes: Optional[str] = None,
                embedder: Optional[Embedder] = None) -> dict:
    """Save one memory. Returns {memory_id, created, indexed}. `created` is False for an exact duplicate."""
    return _save_many(user_id=user_id, agent=agent, memory_type=memory_type, contents=[content], case_id=case_id,
                      patent_id=patent_id, scope=scope, status=status, evidence=evidence, source_ref=source_ref,
                      supersedes=supersedes, embedder=embedder)[0]


def save_memories(*, user_id: str, agent: str, memory_type: str, contents: list[str], case_id: str = "",
                  patent_id: str = "", scope: str = "case", status: str = "unverified", evidence: Optional[list] = None,
                  source_ref: str = "", embedder: Optional[Embedder] = None) -> list[dict]:
    """Save several memories of one type (for example all claims of a patent) with ONE embedding call.
    Returns one {memory_id, created, indexed} per content, in order."""
    return _save_many(user_id=user_id, agent=agent, memory_type=memory_type, contents=list(contents), case_id=case_id,
                      patent_id=patent_id, scope=scope, status=status, evidence=evidence, source_ref=source_ref,
                      supersedes=None, embedder=embedder)


def get_memory(*, user_id: str, agent: str, memory_id: str) -> Optional[dict]:
    """One memory by id. None if it does not exist. PermissionError if the caller may not read it."""
    rec = _coll().find_one({"_id": memory_id})
    if rec is None:
        return None
    ok, why = check_memory_permission(user_id, agent, "read", record=rec)
    if not ok:
        _audit(user_id, rec["case_id"], "memory_denied", agent, action="read", memory_id=memory_id, reason=why)
        raise PermissionError(why)
    return _public(rec)


def update_memory(*, user_id: str, agent: str, memory_id: str, content: str, evidence: Optional[list] = None,
                  source_ref: str = "", embedder: Optional[Embedder] = None) -> dict:
    """A correction = a new version that supersedes the old one. History is kept."""
    old = _coll().find_one({"_id": memory_id})
    if old is None:
        raise ValueError("memory not found")
    if old["status"] == "superseded":
        raise ValueError("this version was already superseded; update the newest version")
    return save_memory(user_id=user_id, agent=agent, memory_type=old["memory_type"], content=content,
                       case_id=old["case_id"], patent_id=old.get("patent_id", ""), scope=old["scope"], evidence=evidence or old["evidence"],
                       source_ref=source_ref or old["source_ref"], supersedes=memory_id, embedder=embedder)


def set_memory_status(*, user_id: str, agent: str, memory_id: str, status: str,
                      verification: Optional[dict] = None) -> dict:
    """Mark memory verified / disputed / unverified. 'verified' needs evidence or a source reference."""
    rec = _coll().find_one({"_id": memory_id})
    if rec is None:
        raise ValueError("memory not found")
    ok, why = (False, f"agent '{agent}' may not change memory status") if agent not in STATUS_SETTERS else \
        check_memory_permission(user_id, agent, "read", record=rec)
    if ok:
        roles = _case_roles(user_id, _load_case(rec["case_id"])) if rec["case_id"] else {"edit"}
        ok, why = ("edit" in roles, "user has no edit access to this case")
    if not ok:
        _audit(user_id, rec["case_id"], "memory_denied", agent, action="status", memory_id=memory_id, reason=why)
        raise PermissionError(why)
    if status not in ("verified", "disputed", "unverified") or rec["status"] == "superseded":
        raise ValueError("status must be verified, disputed or unverified, on a current version")
    if status == "verified" and not (rec["evidence"] or rec["source_ref"]):
        raise ValueError("cannot verify memory that has no evidence or source reference")
    proof = {"by": agent, "user": user_id, "at": _now(), **(verification or {})}
    updateDataById(connect_to_database(), _collection_name(), {"status": status, "verification": proof, "updated_at": _now()}, memory_id)
    _audit(user_id, rec["case_id"], "memory_status_changed", agent, memory_id=memory_id, status=status)
    return _public({**rec, "status": status, "verification": proof})


def delete_memory(*, user_id: str, agent: str, memory_id: str) -> bool:
    rec = _coll().find_one({"_id": memory_id})
    if rec is None:
        return False
    ok, why = check_memory_permission(user_id, agent, "delete", record=rec)
    if not ok:
        _audit(user_id, rec["case_id"], "memory_denied", agent, action="delete", memory_id=memory_id, reason=why)
        raise PermissionError(why)
    gone = deleteDataById(connect_to_database(), _collection_name(), memory_id)
    try:
        col = _vector_collection(rec["owner_id"])
        if col is not None:
            col.delete(ids=[memory_id])
    except Exception as e:
        print(f"LOG: vector delete failed for {memory_id} ({e})")
    _audit(user_id, rec["case_id"], "memory_deleted", agent, memory_id=memory_id, memory_type=rec["memory_type"])
    return gone


def delete_case_memories(*, user_id: str, agent: str, case_id: str) -> int:
    """Lifecycle: remove every memory of a case (call before or when the case is deleted).
    Needs delete access to the case; trusted 'host' code may call it after the case is gone."""
    if not all(t in AGENT_ACCESS.get(agent, {}).get("delete", set()) for t in MEMORY_TYPES):
        raise PermissionError(f"agent '{agent}' may not delete memory")
    if agent != "host" and "delete" not in _case_roles(user_id, _load_case(case_id)):
        _audit(user_id, case_id, "memory_denied", agent, action="delete_case", reason="no delete access")
        raise PermissionError("user has no delete access to this case")
    records = list(_coll().find({"case_id": case_id}))
    by_owner: dict[str, list[str]] = {}
    for rec in records:
        by_owner.setdefault(rec["owner_id"], []).append(rec["_id"])
    _coll().delete_many({"case_id": case_id})
    for owner, ids in by_owner.items():
        try:
            col = _vector_collection(owner)
            if col is not None:
                col.delete(ids=ids)
        except Exception as e:
            print(f"LOG: vector delete failed for case {case_id} ({e})")
    _audit(user_id, case_id, "memory_deleted", agent, count=len(records))
    return len(records)


def retrieve_memory(*, user_id: str, agent: str, query: str, case_id: str = "", patent_id: str = "", across_cases: bool = False,
                    memory_type: Optional[str] = None, k: int = 5, verified_only: bool = False,
                    embedder: Optional[Embedder] = None) -> list[dict]:
    """Semantic search. case_id searches one case; across_cases=True searches the user's own earlier
    investigations; case_id='' alone means user-level memory. Every hit is re-read from Mongo and
    re-checked, and each carries its status so a suggestion is never mistaken for a verified fact."""
    query = (query or "").strip()
    readable = AGENT_ACCESS.get(agent, {}).get("read", set())
    if memory_type and memory_type not in MEMORY_TYPES:
        raise ValueError("unknown memory_type")
    if not readable or (memory_type and memory_type not in readable):
        _audit(user_id, case_id, "memory_denied", agent, action="read", reason="agent may not read this type")
        raise PermissionError(f"agent '{agent}' may not read {memory_type or 'any'} memory")
    case = _load_case(case_id)
    if case_id:
        ok, why = check_memory_permission(user_id, agent, "read", case_id=case_id, memory_type=memory_type or sorted(readable)[0], case=case)
        if not ok:
            _audit(user_id, case_id, "memory_denied", agent, action="read", reason=why)
            raise PermissionError(why)
    if not query:
        return []
    embedder = embedder or default_embedder()
    vectors = embedder([query])
    if not vectors:
        raise RuntimeError("embedding model unavailable; cannot search memory")

    col = _vector_collection(user_id if (across_cases or not case_id) else _owner_of(case, user_id), embedder)
    conds: list[dict] = []
    if not across_cases:
        conds.append({"mem_case_id": case_id})
    if patent_id:
        conds.append({"patent_id": patent_id})
    conds.append({"memory_type": memory_type} if memory_type else {"memory_type": {"$in": sorted(readable)}})
    where = conds[0] if len(conds) == 1 else {"$and": conds}
    found = col.query(query_embeddings=vectors, n_results=max(k * 4, 10), where=where)
    ids = (found.get("ids") or [[]])[0]
    distances = (found.get("distances") or [[]])[0]
    if not ids:
        return []

    records = {r["_id"]: r for r in _coll().find({"_id": {"$in": ids}})}
    cases: dict[str, Optional[dict]] = {case_id: case}
    results: list[dict] = []
    for mid, dist in zip(ids, distances):
        rec = records.get(mid)
        if rec is None or rec["status"] == "superseded" or (verified_only and rec["status"] != "verified"):
            continue
        if rec["case_id"] not in cases:
            cases[rec["case_id"]] = _load_case(rec["case_id"])
        ok, _ = check_memory_permission(user_id, agent, "read", record=rec, case=cases[rec["case_id"]])
        if ok:
            results.append(_public(rec, round(1 - dist, 4)))
        if len(results) >= k:
            break
    _audit(user_id, case_id, "memory_retrieved", agent, count=len(results), across_cases=across_cases or None)
    return results


def list_case_memories(*, user_id: str, agent: str, case_id: str, memory_type: Optional[str] = None,
                       patent_id: str = "", include_superseded: bool = False) -> list[dict]:
    """Every memory of a case that this agent may read (no search), oldest first."""
    readable = AGENT_ACCESS.get(agent, {}).get("read", set())
    if memory_type and memory_type not in MEMORY_TYPES:
        raise ValueError("unknown memory_type")
    ok, why = check_memory_permission(user_id, agent, "read", case_id=case_id,
                                      memory_type=memory_type or (sorted(readable)[0] if readable else None))
    if not ok:
        _audit(user_id, case_id, "memory_denied", agent, action="list", reason=why)
        raise PermissionError(why)
    types = [memory_type] if memory_type else sorted(readable)
    query: dict = {"case_id": case_id, "memory_type": {"$in": types}}
    if patent_id:
        query["patent_id"] = patent_id
    if not include_superseded:
        query["status"] = {"$ne": "superseded"}
    case = _load_case(case_id)
    rows = [r for r in _coll().find(query).sort("created_at", 1)
            if check_memory_permission(user_id, agent, "read", record=r, case=case)[0]]
    return [_public(r) for r in rows]


def memory_summary(*, user_id: str, agent: str, case_id: str) -> dict:
    """How much memory a case has: counts by type and by status (current versions only)."""
    rows = list_case_memories(user_id=user_id, agent=agent, case_id=case_id)
    summary: dict[str, Any] = {"total": len(rows), "by_type": {}, "by_status": {}}
    for r in rows:
        summary["by_type"][r["memory_type"]] = summary["by_type"].get(r["memory_type"], 0) + 1
        summary["by_status"][r["status"]] = summary["by_status"].get(r["status"], 0) + 1
    return summary


def build_agent_context(*, user_id: str, agent: str, query: str, case_id: str = "", patent_id: str = "", across_cases: bool = False,
                        memory_type: Optional[str] = None, k: int = 5, max_chars: int = 3000,
                        verified_only: bool = False, embedder: Optional[Embedder] = None) -> str:
    """Text to put in an agent's prompt: the best matching memories, each labelled VERIFIED or not and
    with its source, trimmed to a size budget so small local models are not flooded. '' if nothing matches."""
    hits = retrieve_memory(user_id=user_id, agent=agent, query=query, case_id=case_id, patent_id=patent_id, across_cases=across_cases,
                           memory_type=memory_type, k=k, verified_only=verified_only, embedder=embedder)
    lines: list[str] = []
    used = 0
    for h in hits:
        first = (h.get("evidence") or [{}])[0]
        source = h.get("source_ref") or (first.get("url", "") if isinstance(first, dict) else "")
        line = (f"[{'VERIFIED' if h['verified'] else h['status'].upper()}] ({h['memory_type']}, id={h['memory_id']}"
                f"{', patent=' + h['patent_id'] if h.get('patent_id') else ''}"
                f"{', source=' + source if source else ''}) {h['content']}")
        if used + len(line) > max_chars:
            break
        lines.append(line)
        used += len(line)
    if not lines:
        return ""
    return "Earlier findings. A match is a lead, not proof; only VERIFIED items were checked.\n" + "\n".join(lines)
