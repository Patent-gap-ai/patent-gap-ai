"""Run the model-driven MCP tool loop (fetch_by_id). The 3-agent committee is commented out for now.

(Original docstring kept below; the committee steps B/C no longer run.)

Build the patent agents and run the review flow.

Flow (PatentAgent.run):
  A. TOOL LOOP - the model itself decides which MCP tools to call (steps 1-10 below)
       it fetches the reference patent, fetches the candidate patent, and validates both
  B. CHECK THE TRACE - the program reads the tool trace (not the model's words) to get the patents and
       the validation results; if a tool was skipped or a patent is invalid, the run stops with FAILURE
  C. COMMITTEE - if both patents are valid, three agents run in turn on the claim board:
       element_parser     reference claims -> elements + keywords
       feature_extractor  candidate text   -> features, each with a verbatim quote as proof
       comparer           elements + features -> one verdict per element (yes / no / unclear)
     Each agent's output is checked; if it is rejected the agent re-runs with a FIX_LIST (board.next_fix).
  D. return everything as one JSON-ready dict (including the tool trace)

The tool loop (PatentAgent._run_tool_loop):
   1. open the MCP connection and initialize the session
   2. ask the session for the tool list (name, description, input schema)
   3. start the message list: system prompt + the one user request
   4. send the list + the tool schemas to the model
   5. read the assistant message
   6. no tool call in it -> its text is the final answer, go to step 10
   7. for each tool call, call the named tool on the open session with the model's arguments
   8. append the assistant message, then each tool result, to the message list
   9. go back to step 4, unless the step cap is reached
  10. close the session; return the final text plus the tool trace

The model comes from ai_workflow.model_loader, the prompts from ai_workflow.static_prompts.
The MCP validate tool decides validity; the LLM does the calling and the language work.
"""
from __future__ import annotations

import asyncio
import json
import os
# import re
import sys
# from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# from ai_workflow.board import ClaimBoard, Element, Feature, Verdict
from ai_workflow.model_loader import load_model
from ai_workflow.static_prompts import TOOL_AGENT_SYSTEM_PROMPT, TOOL_AGENT_USER_PROMPT
# from ai_workflow.static_prompts import SCHEMAS, SYSTEM_PROMPTS, USER_BUILDERS   # committee (commented out for now)
from settings import COMMITTEE

# MCP tools that only the code may call (the model never sees them)
CODE_ONLY_TOOLS = {"create_patent"}


# class AgentOutputError(RuntimeError):
#     """An agent's reply was not usable (not JSON, wrong shape, empty)."""
#
#
# # ---------------------------------------------------------------------------
# # Agents: one role + one instruction sheet + one model call
# # ---------------------------------------------------------------------------
# @dataclass
# class Agent:
#     role: str
#     system: str
#     schema: Optional[dict]
#     model: Any                      # a LoadedModel from model_loader (anything with .chat())
#
#     def ask(self, user: str) -> dict:
#         """Send this role's instructions + the user block to the model and return the parsed JSON."""
#         messages = [{"role": "system", "content": self.system}, {"role": "user", "content": user}]
#         # `format` makes Ollama force the reply to match the JSON schema.
#         reply = self.model.chat(messages, format=self.schema or "json", options={"temperature": 0})
#         return self._parse(reply)
#
#     @staticmethod
#     def _parse(reply: str) -> dict:
#         text = (reply or "").strip()
#         text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)      # remove code fences if the model added them
#         try:
#             data = json.loads(text)
#         except json.JSONDecodeError as exc:
#             raise AgentOutputError(f"reply is not valid JSON ({exc.msg})") from exc
#         if not isinstance(data, dict):
#             raise AgentOutputError("reply must be one JSON object")
#         return data
#
#
# def create_agent(role: str, model: Any) -> Agent:
#     """One worker: its system prompt and JSON schema come from static_prompts."""
#     return Agent(role, SYSTEM_PROMPTS[role], SCHEMAS[role], model)
#
#
# def create_committee(model: Any) -> dict[str, Agent]:
#     """The three patent roles, sharing one loaded model (the model is loaded only once)."""
#     return {role: create_agent(role, model) for role in SYSTEM_PROMPTS}


# ---------------------------------------------------------------------------
# The workflow
# ---------------------------------------------------------------------------
class PatentAgent:
    """A tool-calling loop over MCP tools. Only fetch_by_id is used; the committee is commented out."""

    def __init__(self, model: Any, log: Callable[[str], None] = print):
        self.model = model
#         self.committee = create_committee(model)
        self.log = log                      # progress messages; pass `lambda _: None` to silence

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

    # -- the tool loop (steps 1-10) -------------------------------------------
    @staticmethod
    def _ollama_tools(listed: list) -> list[dict]:
        """MCP tool list -> the tool schemas the model receives (name, description, input schema)."""
        return [{"type": "function",
                 "function": {"name": t.name, "description": t.description or "", "parameters": t.inputSchema}}
                for t in listed]

    @staticmethod
    def _clean_arguments(args: Any, schema: dict | None) -> dict:
        """The arguments the model supplied, with one fix: where a tool wants a JSON *string* and the model
        sent a JSON object instead, turn it into text (the data itself is not changed)."""
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if not isinstance(args, dict):
            args = {}
        props = (schema or {}).get("properties", {})
        return {k: (json.dumps(v) if props.get(k, {}).get("type") == "string" and not isinstance(v, str) else v)
                for k, v in args.items()}

    async def _call_tool(self, session: ClientSession, schemas: dict, name: str, args: dict) -> str:
        """Call one tool on the open session. A failure becomes an 'ERROR: ...' text the model can read."""
        if name not in schemas:
            return f"ERROR: unknown tool '{name}'. Available tools: {', '.join(schemas)}"
        try:
            return self._tool_text(await session.call_tool(name, arguments=args))
        except Exception as exc:  # noqa: BLE001 - fed back to the model instead of crashing the run
            return f"ERROR: {type(exc).__name__}: {exc}"

    async def _run_tool_loop(self, patent_id: str, user_id: str) -> dict:
        project_root = Path(__file__).resolve().parents[1]
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "src.mcp_tools.mcp"],         # server lives in mcp_tools/mcp.py
            cwd=str(project_root),                 # so `mcp_tools` and `settings` can be imported
            env={**os.environ, "PATENT_GAP_USER_ID": str(user_id)},   # API keys + the owner of the new patent
        )
        trace: list[dict] = []
        final_text = ""
        steps = 0
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()                                           # 1
                listed = (await session.list_tools()).tools                          # 2
                schemas = {t.name: t.inputSchema for t in listed}
                # create_patent is called by this code, not by the model (a small model may skip it or
                # invent arguments), so it is kept off the model's tool list.
                tools = self._ollama_tools([t for t in listed if t.name not in CODE_ONLY_TOOLS])
                messages: list[dict] = [                                             # 3
                    {"role": "system", "content": TOOL_AGENT_SYSTEM_PROMPT},
                    {"role": "user", "content": TOOL_AGENT_USER_PROMPT.format(patent_id=patent_id)},
                ]
                capped = True
                for steps in range(1, COMMITTEE["max_tool_steps"] + 1):
                    message = await asyncio.to_thread(                               # 4 + 5
                        self.model.chat_message, messages, tools=tools, options={"temperature": 0})
                    calls = message.get("tool_calls") or []
                    if not calls:                                                    # 6
                        final_text = (message.get("content") or "").strip()
                        capped = False
                        break
                    messages.append(message)                                         # 8 (assistant first)
                    for call in calls:                                               # 7
                        fn = call.get("function", {})
                        name = fn.get("name", "")
                        args = self._clean_arguments(fn.get("arguments", {}), schemas.get(name))
                        self.log(f"  step {steps}: model called {name}")
                        text = await self._call_tool(session, schemas, name, args)
                        trace.append({"step": steps, "tool": name, "arguments": args,
                                      "result": text, "error": text.startswith("ERROR:")})
                        messages.append({"role": "tool", "tool_name": name, "content": text})   # 8
                    # The fetched JSON can be large (3 sources). Do not send it back to the model:
                    # once fetch_by_id has succeeded we already have everything we need.
                    if any(x["tool"] == "fetch_by_id" and not x["error"] for x in trace):
                        capped = False
                        break
                    # 9: loop again, unless the for-loop runs out of steps (the cap)

                # 9b: the model has called fetch_by_id. Now THIS CODE calls the create_patent tool on the same
                # open session. The tool itself checks that the fetch was valid, inserts the case into the
                # database, and records an alert for success or failure, so it is called after every fetch.
                fetched = next((x for x in reversed(trace)
                                if x["tool"] == "fetch_by_id" and not x["error"]), None)
                if fetched is not None:
                    data = self._json_object(fetched["result"]) or {}
                    args = {"patent_id": data.get("patent_id") or patent_id}
                    self.log("  inserting the case with create_patent")
                    text = await self._call_tool(session, schemas, "create_patent", args)
                    trace.append({"step": steps, "tool": "create_patent", "arguments": args,
                                  "result": text, "error": text.startswith("ERROR:")})
        # 10: the session is closed here (end of the two `async with` blocks): the MCP server process stops
        return {"final_text": final_text, "trace": trace, "steps": steps, "capped": capped}

    # -- read the trace: what was really fetched and validated ----------------
    @staticmethod
    def _json_object(text: Any) -> dict | None:
        try:
            value = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

#     def _collect(self, trace: list[dict]) -> tuple[dict, dict, list[str]]:
#         """Return (patents, validation results, problems) taken from the tool trace.
#         A patent counts as validated only if the validate tool was called with exactly the JSON that the
#         fetch tool returned, so the model cannot validate data it changed."""
#         patents: dict[str, dict] = {}
#         problems: list[str] = []
#         for key, tool in (("reference", "get_reference_patent"), ("candidate", "get_candidate_patent")):
#             ok = [t for t in trace if t["tool"] == tool and not t["error"]]
#             data = self._json_object(ok[-1]["result"]) if ok else None
#             if data is None:
#                 problems.append(f"the model did not fetch the {key} patent (tool {tool})")
#             else:
#                 patents[key] = data
#         validation: dict[str, dict] = {}
#         for key, data in patents.items():
#             for t in trace:
#                 if (t["tool"] == "validate_patent_information" and not t["error"]
#                         and self._json_object(t["arguments"].get("patent_json")) == data):
#                     validation[key] = self._json_object(t["result"]) or {}
#             if key not in validation:
#                 problems.append(f"the {key} patent was not validated with the exact fetched JSON")
#         return patents, validation, problems
#
#     @staticmethod
#     def _finish(result: dict, loop: dict) -> dict:
#         """Add the tool-loop details at the end of the result, so the main answer is read first."""
#         result.update(tool_steps=loop["steps"], step_cap_reached=loop["capped"], tool_trace=loop["trace"])
#         return result
#
#     # -- step 3: element parser agent ----------------------------------------
#     @staticmethod
#     def _numbered_claims(patent: dict) -> str:
#         return "\n".join(f"{i}. {c}" for i, c in enumerate(patent.get("claims", []), 1))
#
#     @staticmethod
#     def _candidate_text(patent: dict) -> str:
#         return (f"{patent.get('title', '')}\n{patent.get('abstract', '')}\n"
#                 + PatentAgent._numbered_claims(patent))
#
#     @staticmethod
#     def _norm(text: str) -> str:
#         """Lower-case and collapse whitespace, so quotes are compared fairly."""
#         return " ".join(text.lower().split())
#
#     # -- output checks: accept a reply only if it is usable, else raise AgentOutputError ---
#     @staticmethod
#     def _check_elements(data: dict, board: ClaimBoard, last_try: bool) -> list[Element]:
#         raw = data.get("elements")
#         if not isinstance(raw, list) or not raw:
#             raise AgentOutputError("'elements' must be a non-empty list")
#         elements = []
#         for i, item in enumerate(raw, 1):
#             text = item.get("text") if isinstance(item, dict) else None
#             if not isinstance(text, str) or not text.strip():
#                 raise AgentOutputError(f"element {i} has no text")
#             words = item.get("keywords") if isinstance(item.get("keywords"), list) else []
#             # keep only keywords that really appear in the element (drops invented ones)
#             keywords = [w.strip() for w in words if isinstance(w, str) and w.strip().lower() in text.lower()]
#             elements.append(Element(f"E{i}", text.strip(), keywords))
#         return elements
#
#     def _check_features(self, data: dict, board: ClaimBoard, last_try: bool) -> list[Feature]:
#         raw = data.get("features")
#         if not isinstance(raw, list) or not raw:
#             raise AgentOutputError("'features' must be a non-empty list")
#         source = self._norm(board.candidate_text)
#         good: list[Feature] = []
#         bad: list[str] = []
#         for item in raw:
#             text = item.get("text") if isinstance(item, dict) else None
#             quote = item.get("quote") if isinstance(item, dict) else None
#             if not (isinstance(text, str) and text.strip() and isinstance(quote, str) and quote.strip()):
#                 bad.append("a feature without text or quote")
#             elif self._norm(quote) not in source:
#                 bad.append(f"quote not found in the candidate text: '{quote.strip()[:60]}'")
#             else:
#                 good.append(Feature(f"F{len(good) + 1}", text.strip(), quote.strip()))
#         if bad and not (last_try and good):      # on the last try, keep the features that are proven
#             raise AgentOutputError("; ".join(bad[:3]))
#         if not good:
#             raise AgentOutputError("no feature has a verbatim quote")
#         return good
#
#     def _check_verdicts(self, data: dict, board: ClaimBoard, last_try: bool) -> list[Verdict]:
#         raw = data.get("verdicts")
#         if not isinstance(raw, list) or not raw:
#             raise AgentOutputError("'verdicts' must be a non-empty list")
#         element_ids = {e.element_id for e in board.elements}
#         feature_ids = {f.feature_id for f in board.features}
#         by_element: dict[str, Verdict] = {}
#         for item in raw:
#             if not isinstance(item, dict):
#                 raise AgentOutputError("each verdict must be an object")
#             eid = str(item.get("element_id", "")).strip()
#             fid = str(item.get("feature_id", "") or "").strip()
#             answer = str(item.get("infringes", "")).strip().lower()
#             if eid not in element_ids:
#                 raise AgentOutputError(f"unknown element_id '{eid}'")
#             if answer not in ("yes", "no", "unclear"):
#                 raise AgentOutputError(f"{eid}: infringes must be yes, no or unclear")
#             if fid and fid not in feature_ids:
#                 raise AgentOutputError(f"{eid}: unknown feature_id '{fid}'")
#             if answer == "yes" and not fid:
#                 raise AgentOutputError(f"{eid}: a 'yes' must name the matching feature_id")
#             by_element[eid] = Verdict(eid, fid, answer, str(item.get("reason", "")).strip())
#         missing = [e.element_id for e in board.elements if e.element_id not in by_element]
#         if missing:
#             raise AgentOutputError("no verdict for " + ", ".join(missing))
#         return [by_element[e.element_id] for e in board.elements]      # one verdict per element, in order
#
#     # -- run one agent, with the fix-list retry loop ---------------------------
#     def _run_role(self, role: str, board: ClaimBoard, check: Callable[[dict, ClaimBoard, bool], Any]) -> Any:
#         """Ask `role`; if `check` rejects the reply, write a FIX_LIST on the board and ask again."""
#         agent = self.committee[role]
#         tries = 1 + COMMITTEE["max_rounds"]
#         last_error = ""
#         for attempt in range(1, tries + 1):
#             try:
#                 result = check(agent.ask(USER_BUILDERS[role](board)), board, attempt == tries)
#                 board.next_fix = [f for f in board.next_fix if f.get("owner") != role]
#                 return result
#             except AgentOutputError as exc:
#                 last_error = str(exc)
#                 board.next_fix = [f for f in board.next_fix if f.get("owner") != role] + [
#                     {"owner": role, "problem": last_error, "fix": "follow the Format exactly and use only the given text"}]
#         raise AgentOutputError(f"{role} failed after {tries} tries: {last_error}")
#
#     @staticmethod
#     def _overall_match(verdicts: list[Verdict]) -> str:
#         """All claim elements must be met for a match; one 'no' means no match; otherwise unclear."""
#         answers = [v.infringes for v in verdicts]
#         if all(a == "yes" for a in answers):
#             return "yes"
#         return "no" if "no" in answers else "unclear"
#
#     # -- the whole flow --------------------------------------------------------
#     async def run(self, patent_id: str = "US20240012345A1") -> dict:
#         loop = await self._run_tool_loop(patent_id)
#         patents, checks, problems = self._collect(loop["trace"])
#         result: dict[str, Any] = {"agent": "Patent Review Agent", "case_id": patent_id,
#                                   "agent_summary": loop["final_text"],
#                                   "reference_patent": patents.get("reference"),
#                                   "candidate_patent": patents.get("candidate"),
#                                   "validation": checks}
#         if loop["capped"]:
#             problems.append(f"the step cap ({COMMITTEE['max_tool_steps']}) was reached before the model finished")
#         if problems:
#             result.update(status="FAILURE", stage="tool_loop", errors=problems)
#             return self._finish(result, loop)
#
#         errors = [f"{name}: {e}" for name, v in checks.items() for e in v.get("errors", [])]
#         if not all(v.get("valid") for v in checks.values()):
#             result.update(status="FAILURE", stage="validation", errors=errors or ["validation failed"])
#             return self._finish(result, loop)
#
#         board = ClaimBoard(patent_id=patent_id,
#                            ref_claim=self._numbered_claims(result["reference_patent"]),
#                            candidate_text=self._candidate_text(result["candidate_patent"]))
#
#         # The three agents run in this order; each fills one part of the board.
#         for role, field_name, check in (
#             ("element_parser", "elements", self._check_elements),
#             ("feature_extractor", "features", self._check_features),
#             ("comparer", "verdicts", self._check_verdicts),
#         ):
#             try:
#                 setattr(board, field_name, self._run_role(role, board, check))
#             except AgentOutputError as exc:
#                 result.update(status="FAILURE", stage=role, errors=[str(exc)], board=board.to_dict())
#                 return self._finish(result, loop)
#
#         met = sum(v.infringes == "yes" for v in board.verdicts)
#         result.update(
#             status="SUCCESS", stage="done",
#             elements=[{"element_id": e.element_id, "text": e.text, "keywords": e.keywords} for e in board.elements],
#             features=[{"feature_id": f.feature_id, "text": f.text, "quote": f.quote} for f in board.features],
#             verdicts=[{"element_id": v.element_id, "feature_id": v.feature_id, "infringes": v.infringes,
#                        "reason": v.reason} for v in board.verdicts],
#             overall_match=self._overall_match(board.verdicts),
#             elements_met=f"{met}/{len(board.elements)}",
#             board=board.to_dict())
#         return self._finish(result, loop)
#

    # -- the whole flow --------------------------------------------------------
    async def run(self, patent_id: str, user_id: str) -> dict:
        """Let the model call fetch_by_id, save the result with create_patent, and return the fetched JSON
        plus a "db_insert" entry with the outcome of the save. The output is taken from the tool trace,
        not from the model's words."""
        loop = await self._run_tool_loop(patent_id, user_id)
        calls = [t for t in loop["trace"] if t["tool"] == "fetch_by_id" and not t["error"]]
        if not calls:
            raise RuntimeError("the model did not call fetch_by_id (step cap reached: "
                               f"{loop['capped']}); trace: {loop['trace']}")
        data = self._json_object(calls[-1]["result"])
        if data is None:
            raise RuntimeError("fetch_by_id did not return a JSON object")

        saves = [t for t in loop["trace"] if t["tool"] == "create_patent"]
        if not saves:      # should not happen: the code calls create_patent after every fetch
            data["db_insert"] = {"success": False, "message": "Case was not created: create_patent was not called."}
        else:
            data["db_insert"] = self._json_object(saves[-1]["result"]) or {
                "success": False, "message": saves[-1]["result"][:300]}
        return data


def build_agent(model_key: str | None = None) -> PatentAgent:
    """Load a local model with model_loader and create the agents on it."""
    model = load_model(model_key=model_key or COMMITTEE["model_key"])
    return PatentAgent(model)