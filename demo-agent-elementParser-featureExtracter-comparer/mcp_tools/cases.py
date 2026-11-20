"""Small MCP server exposing example patent-case tools.

Tools
  get_reference_patent(patent_id)   -> JSON of the reference patent (its claims = ref_claim)
  get_candidate_patent(patent_id)   -> JSON of the candidate patent (its text = candidate_text)
  validate_patent_information(json) -> checks that all required fields are present

The data is hardcoded for now. Both patents use the same sample format.
Run directly with: python -m mcp_tools.cases
"""
import json
from typing import Any

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Patent Case Tools")


@mcp.tool()
def get_reference_patent(patent_id: str = "US20240012345A1") -> str:
    """Return the reference patent (the patent being protected) as a JSON string."""
    patent = {
        "patent_id": patent_id,
        "title": "Wireless Power Transfer System",
        "inventors": ["Alex Morgan", "Jamie Lee"],
        "filing_date": "2023-06-15",
        "publication_date": "2024-01-18",
        "abstract": (
            "A system for transferring electrical power wirelessly between "
            "a transmitter and a receiver using resonant coupling."
        ),
        "claims": [
            "A wireless power transfer system comprising: a power transmitter; and a power "
            "receiver configured to receive power from the power transmitter.",
            "The system of claim 1, wherein the receiver uses resonant coupling.",
        ],
    }
    return json.dumps(patent)


@mcp.tool()
def get_candidate_patent(patent_id: str = "US20240012345A1") -> str:
    """Return the candidate patent/product (the one being compared) as a JSON string.
    `patent_id` is the case id; the candidate has its own patent number."""
    patent = {
        "patent_id": "US20230099876A1",
        "title": "Resonant Wireless Charging Pad",
        "inventors": ["Sam Rivera"],
        "filing_date": "2023-03-02",
        "publication_date": "2023-09-28",
        "abstract": (
            "A charging pad with a transmitter coil that sends power to a receiver coil "
            "in a phone through resonant coupling."
        ),
        "claims": [
            "A charging pad comprising a transmitter coil that sends power wirelessly to a receiver coil.",
            "The charging pad of claim 1, wherein the coils are tuned to the same resonant frequency.",
        ],
    }
    return json.dumps(patent)


@mcp.tool()
def validate_patent_information(patent_json: str) -> str:
    """Validate that a patent JSON object contains the required fields."""
    required_fields = ["patent_id", "title", "inventors", "filing_date", "abstract", "claims"]

    try:
        patent: Any = json.loads(patent_json)
    except json.JSONDecodeError as exc:
        return json.dumps({
            "status": "FAILURE",
            "valid": False,
            "errors": [f"Invalid JSON: {exc.msg}"]
        })

    if not isinstance(patent, dict):
        return json.dumps({
            "status": "FAILURE",
            "valid": False,
            "errors": ["Patent information must be a JSON object."]
        })

    missing_fields = [
        field for field in required_fields
        if field not in patent or patent[field] in (None, "", [])
    ]

    errors = [f"Missing or empty required field: {field}" for field in missing_fields]
    return json.dumps({
        "status": "SUCCESS" if not errors else "FAILURE",
        "valid": not errors,
        "checked_fields": required_fields,
        "errors": errors
    })


if __name__ == "__main__":
    # MCP uses stdio; do not print logs to stdout in this process.
    mcp.run(transport="stdio")
