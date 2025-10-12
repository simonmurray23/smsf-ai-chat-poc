from typing import List, Tuple

Section = str  # "setup" | "fees" | "trustees" | "compliance" | "other"

# Seed keyword lists (edit in PRs for audit)
_SECTION_KEYWORDS = {
    "setup": ["set up", "establish", "create", "open", "rollover"],
    "fees": ["fee", "levy", "cost", "charge", "audit fee", "admin fee", "supervisory"],
    "trustees": ["trustee", "director", "member", "responsibilities", "individual", "corporate"],
    "compliance": ["return", "audit", "contravention", "ATO", "reporting", "residency", "sisa", "sisor"],
}

def route_sections(query: str, k: int = 2) -> List[Section]:
    """
    Heuristic router: pick up to k sections that match keywords in the query.
    Always includes 'other' if confidence is low.
    """
    q = query.lower()
    scores = {sec: 0 for sec in _SECTION_KEYWORDS.keys()}
    for sec, kws in _SECTION_KEYWORDS.items():
        for kw in kws:
            if kw in q:
                scores[sec] += 1
    ranked = [s for s,_ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True) if scores[s] > 0]
    chosen = (ranked[:k] if ranked else []) or ["other"]
    return chosen

def explain_routing(query: str) -> Tuple[List[Section], str]:
    """
    Returns (sections, rationale_text) for audit.
    """
    secs = route_sections(query)
    rationale = f"Router selected {secs} based on keyword matches; query='{query[:120]}...'"
    return secs, rationale
