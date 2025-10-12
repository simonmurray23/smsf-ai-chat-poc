# Makes this directory a package and exposes public API.
from .router import route_sections, explain_routing
from .search_bm25 import load_bm25_index, bm25_search, debug_bm25_breakdown
from .re_rank import re_rank_with_embeddings, embed_text_bedrock
from .oracle import assess_coverage, compose_prompt, render_refusal

__all__ = [
    "route_sections", "explain_routing",
    "load_bm25_index", "bm25_search", "debug_bm25_breakdown",
    "re_rank_with_embeddings", "embed_text_bedrock",
    "assess_coverage", "compose_prompt", "render_refusal"
]
