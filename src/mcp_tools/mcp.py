"""Runs the MCP server defined in cases.py (the tools are registered there with @mcp.tool()).

Run from the project root:  python -m mcp_tools.mcp
(Run it as a module. `python mcp_tools/mcp.py` would make this file shadow the `mcp` SDK package.)
"""
from src.mcp_tools.cases import mcp


if __name__ == "__main__":
    # MCP uses stdio; do not print logs to stdout in this process.
    mcp.run(transport="stdio")
