"""
pipeline.py — Full NLP pipeline orchestrator.

Composes Stage 1 (LLM intent extraction) and Stage 2 (deterministic entity
resolution) into a single NavigationIntent output.

Pipeline flow:
  Whisper → NLPPipeline.process() → NavigationIntent.node_id → A* routing → map

The two stages are strictly separated:
  - Stage 1 (LLM):         extract raw destination string from utterance
  - Stage 2 (RapidFuzz):   resolve raw string to a campus node_id

The LLM is never trusted for building names or node IDs.
All entity resolution is grounded in the campus GeoJSON alias table.
"""

import logging
import yaml
from dataclasses import dataclass
from pathlib import Path

from .llm_backend import get_backend, LLMBackend
from .campus_knowledge import CampusKnowledge
from .intent_extractor import extract_intent
from .entity_resolver import resolve_entity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class NavigationIntent:
    """
    The unified output of the NLP pipeline.

    Attributes:
        raw_utterance:  Original text as received from Whisper.
        intent:         Extracted intent type: navigate_to | locate | unknown.
        node_id:        Campus node ID for routing (e.g. 'bibliotheque'),
                        or None if unresolved. Connect directly to the
                        navigation graph imported from data/campus_graph.json.
        label:          Human-readable canonical building name, or None.
        confidence:     Normalized fuzzy score (fuzzy_score / 100).
                        Range: 0.0–1.0.
        language:       Detected input language ('fr', 'ar', 'en', 'mixed').
                        INFORMATIONAL ONLY — never used in routing decisions.
        resolved:       True if a campus node was successfully matched.

    Note on confidence:
        Confidence is the normalized fuzzy match score only.
        LLM self-reported confidence is excluded — it is unreliable and
        not present in the extraction schema.
    """
    raw_utterance: str
    intent:        str         # navigate_to | locate | unknown
    node_id:       str | None
    label:         str | None
    confidence:    float       # fuzzy_score / 100
    language:      str         # informational only, never used in routing
    resolved:      bool


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------

class NLPPipeline:
    """
    Two-stage NLP pipeline for campus navigation intent resolution.

    Instantiate once at application startup (cold-start loads the LLM and
    the campus alias table). Then call process() or process_batch() per request.

    Args:
        config: Parsed YAML config dict. Expected keys:
                  backend, geojson_path, resolution_threshold, temperature,
                  ollama_model, llamacpp_model_path, llamacpp_n_threads
    """

    def __init__(self, config: dict):
        logger.info("Initializing NLPPipeline...")

        self._config    = config
        self._threshold = config.get("resolution_threshold", 65)

        # Try to initialise the LLM backend; degrade gracefully if unavailable
        try:
            self._backend = get_backend(config)
            self._llm_available = True
        except Exception as e:
            logger.warning(
                f"LLM backend unavailable ({e}). "
                "NLPPipeline will run in RapidFuzz-only mode (no LLM)."
            )
            self._backend = None
            self._llm_available = False

        geojson_path = config.get("geojson_path", "data/campus.geojson")
        self._knowledge = CampusKnowledge(geojson_path)

        logger.info(
            f"NLPPipeline ready. backend={config.get('backend')} | "
            f"llm_available={self._llm_available} | "
            f"threshold={self._threshold} | "
            f"nodes={len(self._knowledge.node_registry)}"
        )

    @property
    def llm_available(self) -> bool:
        """True if the LLM backend is reachable; False in RapidFuzz-only mode."""
        return self._llm_available

    # ── Single utterance ──────────────────────────────────────────────────

    def process(self, utterance: str) -> NavigationIntent:
        """
        Run the full two-stage pipeline on a single utterance.

        Stage 1 — Intent extraction (LLM):
            Receives raw utterance, returns intent + raw destination string.
            The LLM never maps to node IDs.

        Stage 2 — Entity resolution (deterministic):
            Fuzzy-matches the raw destination string against the alias table.
            Returns node_id and score.
            Skipped entirely if Stage 1 returned intent=unknown.

        Args:
            utterance: Raw Whisper-transcribed text (any language).

        Returns:
            NavigationIntent dataclass with all fields populated.
        """
        logger.info(f"Processing utterance: {utterance!r}")

        # ── Stage 1: LLM intent extraction (skipped if backend unavailable) ─
        if self._backend is not None:
            try:
                extraction = extract_intent(utterance, self._backend)
            except Exception as exc:
                logger.warning(
                    "LLM extraction failed (%s). Falling back to RapidFuzz-only mode.",
                    exc,
                )
                extraction = {
                    "intent": "navigate_to",
                    "raw_destination": utterance.strip(),
                    "language_detected": "mixed",
                }
        else:
            # Fallback: treat the entire utterance as the raw destination
            extraction = {
                "intent": "navigate_to",
                "raw_destination": utterance.strip(),
                "language_detected": "mixed",
            }

        intent   = extraction["intent"]
        raw_dest = extraction["raw_destination"]
        language = extraction["language_detected"]

        # ── Stage 2: Entity resolution (skip if intent is unknown) ────────
        if intent == "unknown" or raw_dest is None:
            logger.info("Intent is 'unknown' — skipping entity resolution.")
            return NavigationIntent(
                raw_utterance=utterance,
                intent="unknown",
                node_id=None,
                label=None,
                confidence=0.0,
                language=language,
                resolved=False,
            )

        resolution = resolve_entity(raw_dest, self._knowledge, self._threshold)

        # Confidence is the normalized fuzzy score only. LLM output is not used
        # in confidence calculation because LLM self-reported confidence is
        # unreliable.
        confidence = resolution["fuzzy_score"] / 100.0

        return NavigationIntent(
            raw_utterance=utterance,
            intent=intent,
            node_id=resolution["node_id"],
            label=resolution["label"],
            confidence=confidence,
            language=language,       # informational only, never used in routing
            resolved=resolution["resolved"],
        )

    # ── Batch processing ──────────────────────────────────────────────────

    def process_batch(self, utterances: list[str]) -> list[NavigationIntent]:
        """
        Run the pipeline sequentially over a list of utterances.

        Useful for offline evaluation, test suites, and benchmarking.
        Each utterance is processed independently — no shared state.

        Args:
            utterances: List of raw transcribed strings.

        Returns:
            List of NavigationIntent, one per input utterance.
        """
        results = []
        for i, utt in enumerate(utterances):
            logger.debug(f"Batch item {i+1}/{len(utterances)}: {utt!r}")
            result = self.process(utt)
            results.append(result)
        return results


# ---------------------------------------------------------------------------
# Config loader helper
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load the YAML configuration file.

    Args:
        config_path: Path to config.yaml (relative or absolute).

    Returns:
        Dict of configuration values.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at '{config_path}'. "
            "Copy config.yaml to the working directory and set correct paths."
        )
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)
