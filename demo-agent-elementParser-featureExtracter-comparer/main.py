"""Run the patent review flow for one case id: python main.py [patent_id]

  1. the model is loaded (model_loader) and the agents are created
  2. the reference and candidate patents are fetched through the MCP tools (hardcoded data for now)
  3. both are validated by the MCP validate tool
  4. if valid, the three agents run in turn:
       element_parser -> feature_extractor -> comparer
The result is printed as JSON.
"""
import asyncio
import json
import sys

from ai_workflow.agent_builder import build_agent

CASE_ID = "US20240012345A1"      # hardcoded for now; pass another id on the command line to override


async def main(patent_id: str = CASE_ID) -> dict:
    print("Loading the model...")
    agent = build_agent()

    print(f"\nRunning the patent review for {patent_id}...\n")
    result = await agent.run(patent_id)

    print("=== AGENT RESULT ===")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else CASE_ID))
