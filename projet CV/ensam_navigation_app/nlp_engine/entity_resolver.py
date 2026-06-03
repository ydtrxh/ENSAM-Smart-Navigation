"""
entity_resolver.py — Stage 2 of the NLP pipeline.

Deterministic, LLM-free entity resolution using RapidFuzz.

Takes the raw destination string extracted by the LLM (Stage 1) and
resolves it to a canonical campus node_id by fuzzy-matching against the
pre-normalized alias table in CampusKnowledge.

Key design constraints:
  - No LLM calls. Pure Python + RapidFuzz only.
  - The alias table is already normalized at startup (see campus_knowledge.py).
  - Only the QUERY string is normalized here at call time.
  - token_set_ratio scorer handles word-order variation and partial matches
    (e.g. "labo de génie civil" → "laboratoires de genie civil").
"""

import logging
from rapidfuzz import process, fuzz

from .campus_knowledge import CampusKnowledge, normalize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Query normalization
# ---------------------------------------------------------------------------

def resolve_entity(
    raw_destination: str,
    knowledge: CampusKnowledge,
    threshold: int = 65,
) -> dict:
    """
    Stage 2: Resolve a raw destination string to a campus node_id.

    Uses RapidFuzz token_set_ratio against the pre-built alias table.
    token_set_ratio is chosen because it is robust to:
      - Word-order variation: "civil labo" ↔ "labo civil"
      - Partial containment: "biblio" ↔ "bibliothèque"
      - Extra filler words: "emmène-moi au département MICS"

    Args:
        raw_destination: The destination string as extracted by the LLM.
        knowledge:       Loaded CampusKnowledge instance.
        threshold:       Minimum fuzzy score (0–100) to accept a match.
                         Configurable via config.yaml (resolution_threshold).
                         Default: 65.

    Returns:
        {
            "node_id":     str | None   — matched node ID or None
            "label":       str | None   — canonical label or None
            "fuzzy_score": int          — best score (0–100)
            "resolved":    bool         — True if score >= threshold
        }
    """
    if not raw_destination or not raw_destination.strip():
        logger.warning("resolve_entity received an empty destination string.")
        return _no_match()

    query = normalize(raw_destination)
    logger.debug(f"Stage 2 — resolving query: {query!r}  (original: {raw_destination!r})")

    if not knowledge._normalized_alias_table:
        logger.error("Alias table is empty — campus GeoJSON may not have loaded correctly.")
        return _no_match()

    # Extract the alias strings (first element of each tuple) for RapidFuzz
    alias_strings = [alias for alias, _ in knowledge._normalized_alias_table]

    for alias, node_id in knowledge._normalized_alias_table:
        if alias == query:
            label = knowledge.node_registry[node_id]["label"]
            logger.info(
                f"Resolved '{raw_destination}' by exact alias match -> "
                f"node_id={node_id!r} label={label!r}"
            )
            return {
                "node_id": node_id,
                "label": label,
                "fuzzy_score": 100,
                "resolved": True,
            }

    for alias, node_id in sorted(knowledge._normalized_alias_table, key=lambda item: len(item[0]), reverse=True):
        if len(alias) >= 3 and alias in query:
            label = knowledge.node_registry[node_id]["label"]
            logger.info(
                f"Resolved '{raw_destination}' by contained alias match -> "
                f"node_id={node_id!r} label={label!r}"
            )
            return {
                "node_id": node_id,
                "label": label,
                "fuzzy_score": 100,
                "resolved": True,
            }

    # Run fuzzy matching: returns (best_match_string, score, index)
    result = process.extractOne(
        query,
        alias_strings,
        scorer=fuzz.token_set_ratio,
    )

    if result is None:
        logger.debug("RapidFuzz returned no result.")
        return _no_match()

    best_alias, best_score, best_idx = result

    # Log top 3 candidates for every query (DEBUG level)
    top3 = process.extract(
        query,
        alias_strings,
        scorer=fuzz.token_set_ratio,
        limit=3,
    )
    logger.debug(
        f"Top-3 candidates for {query!r}: "
        + ", ".join(f"{a!r}={s}" for a, s, _ in top3)
    )

    # Retrieve the node_id associated with the best alias
    best_node_id = knowledge._normalized_alias_table[best_idx][1]
    best_label   = knowledge.node_registry[best_node_id]["label"]

    resolved = int(best_score) >= threshold

    if resolved:
        logger.info(
            f"Resolved '{raw_destination}' → node_id={best_node_id!r} "
            f"label={best_label!r}  score={best_score}"
        )
    else:
        logger.info(
            f"Resolution FAILED for '{raw_destination}': "
            f"best match was {best_node_id!r} ({best_score}) < threshold={threshold}"
        )

    return {
        "node_id":     best_node_id if resolved else None,
        "label":       best_label   if resolved else None,
        "fuzzy_score": int(best_score),
        "resolved":    resolved,
    }


def _no_match() -> dict:
    """Safe fallback for unresolvable inputs."""
    return {
        "node_id":     None,
        "label":       None,
        "fuzzy_score": 0,
        "resolved":    False,
    }
