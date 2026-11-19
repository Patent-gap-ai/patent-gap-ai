"""Run the example patent MCP agent: python main.py"""
import asyncio
import json

from ai_workflow.agent_builder import build_agent


async def main() -> None:
    print("Loading the model...")
    agent = build_agent()  # Automatically selects a compatible local model.

    print("\nRunning patent information workflow...\n")
    result = await agent.run("US20240012345A1")

    print("=== AGENT RESULT ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
