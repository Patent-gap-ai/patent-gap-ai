"""Build the patent agents and run the review flow.

Flow (PatentAgent.run):
  1. fetch the reference patent and the candidate patent  (two MCP tools, JSON)
  2. validate both with the MCP validate tool             (all required fields present?)
  3. if both are valid -> put the claims on the board and run the three agents in turn:
       element_parser     reference claims -> elements + keywords
       feature_extractor  candidate text   -> features, each with a verbatim quote as proof
       comparer           elements + features -> one verdict per element (yes / no / unclear)
     Each agent's output is checked; if it is rejected the agent re-runs with a FIX_LIST (board.next_fix).
  4. return everything as one JSON-ready dict

The model comes from ai_workflow.model_loader, the prompts from ai_workflow.static_prompts.
The MCP tools decide validity; the LLM only does the language work (parsing claims).
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from ai_workflow.board import ClaimBoard, Element, Feature, Verdict
from ai_workflow.model_loader import load_model
from ai_workflow.static_prompts import SCHEMAS, SYSTEM_PROMPTS, USER_BUILDERS
from settings import COMMITTEE


class AgentOutputError(RuntimeError):
    """An agent's reply was not usable (not JSON, wrong shape, empty)."""


# ---------------------------------------------------------------------------
# Agents: one role + one instruction sheet + one model call
# ---------------------------------------------------------------------------
@dataclass
class Agent:
    role: str
    system: str
    schema: Optional[dict]
    model: Any                      # a LoadedModel from model_loader (anything with .chat())

    def ask(self, user: str) -> dict:
        """Send this role's instructions + the user block to the model and return the parsed JSON."""
        messages = [{"role": "system", "content": self.system}, {"role": "user", "content": user}]
        # `format` makes Ollama force the reply to match the JSON schema.
        reply = self.model.chat(messages, format=self.schema or "json", options={"temperature": 0})
        return self._parse(reply)

    @staticmethod
    def _parse(reply: str) -> dict:
        text = (reply or "").strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)      # remove code fences if the model added them
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise AgentOutputError(f"reply is not valid JSON ({exc.msg})") from exc
        if not isinstance(data, dict):
            raise AgentOutputError("reply must be one JSON object")
        return data


def create_agent(role: str, model: Any) -> Agent:
    """One worker: its system prompt and JSON schema come from static_prompts."""
    return Agent(role, SYSTEM_PROMPTS[role], SCHEMAS[role], model)


def create_committee(model: Any) -> dict[str, Agent]:
    """The three patent roles, sharing one loaded model (the model is loaded only once)."""
    return {role: create_agent(role, model) for role in SYSTEM_PROMPTS}


# ---------------------------------------------------------------------------
# The workflow
# ---------------------------------------------------------------------------
class PatentAgent:
    """The committee plus an MCP client and a small, explicit tool workflow."""

    def __init__(self, model: Any):
        self.model = model
        self.committee = create_committee(model)

    @staticmethod
    def _tool_text(result: Any) -> str:
        """Extract text returned by an MCP tool call."""
        if getattr(result, "isError", False):
            raise RuntimeError(f"MCP tool returned an error: {result}")

        for block in getattr(result, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                return text

        structured = getattr(result, "structuredContent", None)
        if structured is not None:
            return json.dumps(structured)

        raise RuntimeError("The MCP tool returned no readable content.")

    # -- step 1 + 2: fetch and validate (MCP tools) -------------------------
    async def _fetch_and_validate(self, patent_id: str) -> dict:
        project_root = Path(__file__).resolve().parents[1]
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[str(project_root / "mcp_tools" / "cases.py")],
            env=None,
        )
        out: dict[str, Any] = {}
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                async def call(tool: str, **arguments: Any) -> str:
                    return self._tool_text(await session.call_tool(tool, arguments=arguments))

                ref_json = await call("get_reference_patent", patent_id=patent_id)
                cand_json = await call("get_candidate_patent", patent_id=patent_id)
                out["reference_patent"] = json.loads(ref_json)
                out["candidate_patent"] = json.loads(cand_json)
                # Validate the exact JSON the fetch tools returned.
                out["validation"] = {
                    "reference": json.loads(await call("validate_patent_information", patent_json=ref_json)),
                    "candidate": json.loads(await call("validate_patent_information", patent_json=cand_json)),
                }
        return out

    # -- step 3: element parser agent ----------------------------------------
    @staticmethod
    def _numbered_claims(patent: dict) -> str:
        return "\n".join(f"{i}. {c}" for i, c in enumerate(patent.get("claims", []), 1))

    @staticmethod
    def _candidate_text(patent: dict) -> str:
        return (f"{patent.get('title', '')}\n{patent.get('abstract', '')}\n"
                + PatentAgent._numbered_claims(patent))

    @staticmethod
    def _norm(text: str) -> str:
        """Lower-case and collapse whitespace, so quotes are compared fairly."""
        return " ".join(text.lower().split())

    # -- output checks: accept a reply only if it is usable, else raise AgentOutputError ---
    @staticmethod
    def _check_elements(data: dict, board: ClaimBoard, last_try: bool) -> list[Element]:
        raw = data.get("elements")
        if not isinstance(raw, list) or not raw:
            raise AgentOutputError("'elements' must be a non-empty list")
        elements = []
        for i, item in enumerate(raw, 1):
            text = item.get("text") if isinstance(item, dict) else None
            if not isinstance(text, str) or not text.strip():
                raise AgentOutputError(f"element {i} has no text")
            words = item.get("keywords") if isinstance(item.get("keywords"), list) else []
            # keep only keywords that really appear in the element (drops invented ones)
            keywords = [w.strip() for w in words if isinstance(w, str) and w.strip().lower() in text.lower()]
            elements.append(Element(f"E{i}", text.strip(), keywords))
        return elements

    def _check_features(self, data: dict, board: ClaimBoard, last_try: bool) -> list[Feature]:
        raw = data.get("features")
        if not isinstance(raw, list) or not raw:
            raise AgentOutputError("'features' must be a non-empty list")
        source = self._norm(board.candidate_text)
        good: list[Feature] = []
        bad: list[str] = []
        for item in raw:
            text = item.get("text") if isinstance(item, dict) else None
            quote = item.get("quote") if isinstance(item, dict) else None
            if not (isinstance(text, str) and text.strip() and isinstance(quote, str) and quote.strip()):
                bad.append("a feature without text or quote")
            elif self._norm(quote) not in source:
                bad.append(f"quote not found in the candidate text: '{quote.strip()[:60]}'")
            else:
                good.append(Feature(f"F{len(good) + 1}", text.strip(), quote.strip()))
        if bad and not (last_try and good):      # on the last try, keep the features that are proven
            raise AgentOutputError("; ".join(bad[:3]))
        if not good:
            raise AgentOutputError("no feature has a verbatim quote")
        return good

    def _check_verdicts(self, data: dict, board: ClaimBoard, last_try: bool) -> list[Verdict]:
        raw = data.get("verdicts")
        if not isinstance(raw, list) or not raw:
            raise AgentOutputError("'verdicts' must be a non-empty list")
        element_ids = {e.element_id for e in board.elements}
        feature_ids = {f.feature_id for f in board.features}
        by_element: dict[str, Verdict] = {}
        for item in raw:
            if not isinstance(item, dict):
                raise AgentOutputError("each verdict must be an object")
            eid = str(item.get("element_id", "")).strip()
            fid = str(item.get("feature_id", "") or "").strip()
            answer = str(item.get("infringes", "")).strip().lower()
            if eid not in element_ids:
                raise AgentOutputError(f"unknown element_id '{eid}'")
            if answer not in ("yes", "no", "unclear"):
                raise AgentOutputError(f"{eid}: infringes must be yes, no or unclear")
            if fid and fid not in feature_ids:
                raise AgentOutputError(f"{eid}: unknown feature_id '{fid}'")
            if answer == "yes" and not fid:
                raise AgentOutputError(f"{eid}: a 'yes' must name the matching feature_id")
            by_element[eid] = Verdict(eid, fid, answer, str(item.get("reason", "")).strip())
        missing = [e.element_id for e in board.elements if e.element_id not in by_element]
        if missing:
            raise AgentOutputError("no verdict for " + ", ".join(missing))
        return [by_element[e.element_id] for e in board.elements]      # one verdict per element, in order

    # -- run one agent, with the fix-list retry loop ---------------------------
    def _run_role(self, role: str, board: ClaimBoard, check: Callable[[dict, ClaimBoard, bool], Any]) -> Any:
        """Ask `role`; if `check` rejects the reply, write a FIX_LIST on the board and ask again."""
        agent = self.committee[role]
        tries = 1 + COMMITTEE["max_rounds"]
        last_error = ""
        for attempt in range(1, tries + 1):
            try:
                result = check(agent.ask(USER_BUILDERS[role](board)), board, attempt == tries)
                board.next_fix = [f for f in board.next_fix if f.get("owner") != role]
                return result
            except AgentOutputError as exc:
                last_error = str(exc)
                board.next_fix = [f for f in board.next_fix if f.get("owner") != role] + [
                    {"owner": role, "problem": last_error, "fix": "follow the Format exactly and use only the given text"}]
        raise AgentOutputError(f"{role} failed after {tries} tries: {last_error}")

    @staticmethod
    def _overall_match(verdicts: list[Verdict]) -> str:
        """All claim elements must be met for a match; one 'no' means no match; otherwise unclear."""
        answers = [v.infringes for v in verdicts]
        if all(a == "yes" for a in answers):
            return "yes"
        return "no" if "no" in answers else "unclear"

    # -- the whole flow --------------------------------------------------------
    async def run(self, patent_id: str = "US20240012345A1") -> dict:
        result: dict[str, Any] = {"agent": "Patent Review Agent", "case_id": patent_id}
        result.update(await self._fetch_and_validate(patent_id))

        checks = result["validation"]
        errors = [f"{name}: {e}" for name, v in checks.items() for e in v.get("errors", [])]
        if not all(v.get("valid") for v in checks.values()):
            result.update(status="FAILURE", stage="validation", errors=errors or ["validation failed"])
            return result

        board = ClaimBoard(patent_id=patent_id,
                           ref_claim=self._numbered_claims(result["reference_patent"]),
                           candidate_text=self._candidate_text(result["candidate_patent"]))

        # The three agents run in this order; each fills one part of the board.
        for role, field_name, check in (
            ("element_parser", "elements", self._check_elements),
            ("feature_extractor", "features", self._check_features),
            ("comparer", "verdicts", self._check_verdicts),
        ):
            try:
                setattr(board, field_name, self._run_role(role, board, check))
            except AgentOutputError as exc:
                result.update(status="FAILURE", stage=role, errors=[str(exc)], board=board.to_dict())
                return result

        met = sum(v.infringes == "yes" for v in board.verdicts)
        result.update(
            status="SUCCESS", stage="done",
            elements=[{"element_id": e.element_id, "text": e.text, "keywords": e.keywords} for e in board.elements],
            features=[{"feature_id": f.feature_id, "text": f.text, "quote": f.quote} for f in board.features],
            verdicts=[{"element_id": v.element_id, "feature_id": v.feature_id, "infringes": v.infringes,
                       "reason": v.reason} for v in board.verdicts],
            overall_match=self._overall_match(board.verdicts),
            elements_met=f"{met}/{len(board.elements)}",
            board=board.to_dict())
        return result


def build_agent(model_key: str | None = None) -> PatentAgent:
    """Load a local model with model_loader and create the agents on it."""
    model = load_model(model_key=model_key or COMMITTEE["model_key"])
    return PatentAgent(model)
