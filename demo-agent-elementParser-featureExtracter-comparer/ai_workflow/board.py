"""The claim board: the one shared notepad for a single patent review.

The pipeline writes to it, the agents' prompts read from it. Agents never talk to each
other directly; they only see what the prompt builders copy from the board for them.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Element:
    element_id: str                                      # "E1", "E2", ...
    text: str                                            # one claim limitation, in the claim's own words
    keywords: list[str] = field(default_factory=list)    # key technical words of that limitation


@dataclass
class Feature:
    feature_id: str                                      # "F1", "F2", ...
    text: str
    quote: str = ""                                      # verbatim proof copied from the candidate text


@dataclass
class Verdict:
    element_id: str
    feature_id: str
    infringes: str                                       # "yes" | "no" | "unclear"
    reason: str = ""


@dataclass
class ClaimBoard:
    patent_id: str = ""
    ref_claim: str = ""                                  # reference patent claims (element parser reads this)
    candidate_text: str = ""                             # candidate patent text (feature extractor reads this)
    rules: str = ""                                      # ANALYSIS_RULES set by the team (optional)
    notes: str = ""                                      # CONFIRMED_FACTS confirmed by the user (optional)
    elements: list[Element] = field(default_factory=list)
    features: list[Feature] = field(default_factory=list)
    verdicts: list[Verdict] = field(default_factory=list)
    next_fix: list[dict[str, Any]] = field(default_factory=list)   # [{"owner", "problem", "fix"}]

    def to_dict(self) -> dict[str, Any]:
        """Plain dict, safe to json.dumps."""
        return asdict(self)
