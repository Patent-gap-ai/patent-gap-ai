import asyncio
import contextlib
import json
import sys
from pathlib import Path

from ai_workflow.agent_builder import build_agent


def parent_patent_info_fetching(app, patent_id: str, user_id: str) -> dict:
    patent_id = patent_id.strip().upper()

    with contextlib.redirect_stdout(sys.stderr):
        agent = build_agent()
    agent.log = lambda _: None
    result = asyncio.run(agent.run(patent_id, user_id))   # fetch + create_patent (see agent_builder)

    out = Path("output") / f"{patent_id}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result