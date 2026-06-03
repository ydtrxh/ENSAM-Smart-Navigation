"""
Stage 1: LLM intent extraction.

The LLM extracts only the raw destination mention from an utterance. It never
maps to node IDs and never invents campus building names.

The extraction schema is intentionally minimal. Future intent types such as
find_nearest, route_between, or nearest_service can be added by updating only
the system prompt and schema; the LLMBackend abstraction requires no changes.
"""

from __future__ import annotations

import json
import logging

from .llm_backend import LLMBackend

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a JSON extraction system. You output only raw JSON. No explanation. No preamble. No markdown. No code blocks.

Your task: identify the navigation destination in the user's message.

Rules:
- Extract the destination exactly as the user said it. Preserve their words.
- Do NOT map to building IDs. Do NOT use knowledge of any campus.
- Do NOT invent or guess destination names.
- If no destination is explicitly mentioned, return intent "unknown" and raw_destination null.
- Your entire response must be a single JSON object. Nothing else.

Output schema:
{"intent": "navigate_to|locate|unknown", "raw_destination": "string or null", "language_detected": "fr|ar|en|mixed"}"""

VALID_INTENTS = {"navigate_to", "locate", "unknown"}
VALID_LANGUAGES = {"fr", "ar", "en", "mixed"}


def extract_intent(utterance: str, backend: LLMBackend) -> dict:
    """
    Extract the destination mention from a transcribed utterance.

    Temperature 0.0 is required for deterministic JSON extraction. Any value
    above 0 risks malformed output.
    """
    if not utterance or not utterance.strip():
        logger.warning("extract_intent received an empty utterance.")
        return _unknown_result()

    raw_response = backend.generate(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=utterance,
        temperature=0.0,
    )
    logger.debug("LLM raw response: %r", raw_response)
    return _parse_response(raw_response)


def _parse_response(raw: str) -> dict:
    cleaned = raw.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        logger.debug("No JSON object found in LLM response: %r", raw)
        return _unknown_result()

    try:
        data = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        logger.debug("Malformed JSON from LLM: %r", raw)
        return _unknown_result()

    intent = data.get("intent", "unknown")
    if intent not in VALID_INTENTS:
        intent = "unknown"

    raw_destination = data.get("raw_destination")
    if raw_destination is not None and not isinstance(raw_destination, str):
        raw_destination = str(raw_destination)
    if raw_destination is not None:
        raw_destination = raw_destination.strip() or None
    if intent == "unknown":
        raw_destination = None

    # Language detection from the LLM is informational only and unreliable for
    # mixed-language input. It must not influence routing decisions.
    language = data.get("language_detected", "mixed")
    if language not in VALID_LANGUAGES:
        language = "mixed"

    return {
        "intent": intent,
        "raw_destination": raw_destination,
        "language_detected": language,
    }


def _unknown_result() -> dict:
    return {
        "intent": "unknown",
        "raw_destination": None,
        "language_detected": "mixed",
    }
