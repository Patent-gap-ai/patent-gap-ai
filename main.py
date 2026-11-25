"""Fetch one patent by id through the model-driven MCP tool loop and print ONLY the JSON.

    python main.py [patent_id]

The JSON is printed to the console and also saved to output/<patent_id>.json.
(The three-agent committee is commented out in ai_workflow/agent_builder.py for now.)
"""
import asyncio
import contextlib
import json
import sys
from pathlib import Path

from ai_workflow.agent_builder import build_agent

CASE_ID = "US20240012345A1"      # default; pass another id on the command line to override


async def parent_patent_info_fetching(patent_id: str = CASE_ID) -> dict:
    with contextlib.redirect_stdout(sys.stderr):        # model-loading messages must not pollute the JSON
        agent = build_agent()
    agent.log = lambda _: None                          # silence "step 1: model called ..."
    result = await agent.run(patent_id)

    out = Path("output") / f"{patent_id}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))                 # the only console output
    return result


if __name__ == "__main__":
    asyncio.run(parent_patent_info_fetching(sys.argv[1] if len(sys.argv) > 1 else CASE_ID))
