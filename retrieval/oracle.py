from typing import List, Tuple

def assess_coverage(snippets: List[str], min_chars: int = 600, min_sources: int = 2) -> Tuple[bool, str]:
    """
    Require enough total text and at least N distinct source docs (proxied by snippet count).
    TODO: wire distinct sources via snippet meta in app pipeline.
    """
    total = sum(len(s or "") for s in snippets)
    distinct = len(snippets)
    ok = (total >= min_chars) and (distinct >= min_sources)
    why = f"total_chars={total}, distinct_snippets={distinct}, min_chars={min_chars}, min_sources={min_sources}"
    return ok, why

def compose_prompt(query: str, snippets: List[dict]) -> str:
    """
    Build strict prompt: ONLY use provided snippets. Include inline citation info.
    TODO: shape to your Titan system/user message style.
    """
    return f"ANSWER STRICTLY FROM SNIPPETS. Q: {query}\nSNIPPETS:\n" + "\n---\n".join(
        [s.get('text','') for s in snippets]
    )

def render_refusal(query: str, suggestions: List[str]) -> dict:
    """
    Structured refusal payload for your UI.
    """
    return {
        "mode": "refusal",
        "query": query,
        "message": "I can’t answer confidently from the approved snippets provided.",
        "suggestions": suggestions,
        "disclaimer": "General information only – not financial advice."
    }
