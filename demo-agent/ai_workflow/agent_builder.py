"""Build and run a simple agent that calls two MCP patent tools."""
import json
import sys
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from ai_workflow.model_loader import load_model
from ai_workflow.static_prompts import (
    PATENT_AGENT_SYSTEM_PROMPT,
    PATENT_AGENT_USER_PROMPT,
)


class PatentAgent:
    """An LLM plus an MCP client and a small, explicit tool workflow."""

    def __init__(self, model: Any):
        self.model = model

    @staticmethod
    def _tool_text(result: Any) -> str:
        """Extract text returned by an MCP tool call."""
        if getattr(result, "isError", False):
            raise RuntimeError(f"MCP tool returned an error: {result}")

        # Current MCP SDK returns text blocks in result.content.
        for block in getattr(result, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                return text

        # Accommodate SDK versions that expose structured content.
        structured = getattr(result, "structuredContent", None)
        if structured is not None:
            return json.dumps(structured)

        raise RuntimeError("The MCP tool returned no readable content.")

    async def run(self, patent_id: str = "US20240012345A1") -> dict:
        """Fetch patent JSON, validate it with a second MCP tool, return JSON."""
        project_root = Path(__file__).resolve().parents[1]
        server_file = project_root / "mcp_tools" / "cases.py"

        server_params = StdioServerParameters(
            command=sys.executable,
            args=[str(server_file)],
            env=None,
        )

        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                # TOOL 1: fetch patent information.
                patent_result = await session.call_tool(
                    "get_patent_information",
                    arguments={"patent_id": patent_id},
                )
                patent_json = self._tool_text(patent_result)
                patent_data = json.loads(patent_json)

                # TOOL 2: validate the exact JSON returned by tool 1.
                validation_result = await session.call_tool(
                    "validate_patent_information",
                    arguments={"patent_json": patent_json},
                )
                validation_json = self._tool_text(validation_result)
                validation_data = json.loads(validation_json)

        # Ask the loaded LLM to explain the tool results. The validation
        # tool, not the LLM, remains authoritative for SUCCESS or FAILURE.
        explanation = self.model.chat([
            {"role": "system", "content": PATENT_AGENT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": PATENT_AGENT_USER_PROMPT.format(
                    patent_json=json.dumps(patent_data, indent=2),
                    validation_json=json.dumps(validation_data, indent=2),
                ),
            },
        ])

        return {
            "agent": "Patent Information Validation Agent",
            "patent_id": patent_data.get("patent_id"),
            "status": validation_data.get("status", "FAILURE"),
            "valid": validation_data.get("valid", False),
            "patent_information": patent_data,
            "validation": validation_data,
            "agent_summary": explanation.strip(),
        }


def build_agent(model_key: str | None = None) -> PatentAgent:
    """Load a suitable local model using the existing model_loader."""
    model = load_model(model_key="qwen2.5-1.5b")
    return PatentAgent(model)
