"""Extract a verdict (A / B / tie) from free-form judge output."""

from __future__ import annotations

import re

_TAGS = {"A": "A", "B": "B", "TIE": "tie", "C": "tie", "TIED": "tie"}

# [[A]], [[ B ]], [[Tie]], [[C]] (MT-Bench uses C for tie)
_BRACKET = re.compile(r"\[\[\s*(A|B|C|TIE|TIED)\s*\]\]", re.I)
# "Verdict: A", "Final verdict - Assistant B", "**Verdict:** Tie"
_VERDICT_LINE = re.compile(
    r"verdict\W{0,6}(?:is\s+)?(?:a\s+(?=tie))?(?:assistant\s+)?(A|B|TIE|TIED)\b",
    re.I,
)
# Whole reply is just a label, possibly wrapped: "A", "[A]", "**B**", "Tie."
_BARE = re.compile(r"^\W*(?:assistant\s+)?(A|B|TIE|TIED)\W*$", re.I)


def parse_verdict(text: str | None) -> str | None:
    """Return "A", "B", "tie", or None if no verdict can be found.

    The last bracketed tag wins, since reasoning often mentions both answers
    before the final verdict line.
    """
    if not text:
        return None
    for pattern in (_BRACKET, _VERDICT_LINE):
        matches = pattern.findall(text)
        if matches:
            return _TAGS[matches[-1].upper()]
    m = _BARE.match(text.strip())
    if m:
        return _TAGS[m.group(1).upper()]
    return None
