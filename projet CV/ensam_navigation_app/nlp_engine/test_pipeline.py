"""
End-to-end test runner for the ENSAM NLP pipeline.

Run from the project root:
    python -m nlp_engine.test_pipeline

It can also be run from inside nlp_engine/:
    python test_pipeline.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from nlp_engine.pipeline import NLPPipeline, NavigationIntent, load_config
else:
    from .pipeline import NLPPipeline, NavigationIntent, load_config


logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)

TEST_CASES = [
    ("french", "emmène-moi à la bibliothèque"),
    ("french", "je veux aller au département MICS"),
    ("french", "je cherche la salle de conférence"),
    ("french", "où est l'administration"),
    ("french", "comment aller à l'amphi 450"),
    ("english", "take me to amphi 450"),
    ("english", "where is the energy department"),
    ("english", "I need to go to TD1"),
    ("english", "where is admin 2"),
    ("english", "take me to the research center"),
    ("arabic", "وين المكتبة"),
    ("arabic", "كيفاش نوصل للإدارة"),
    ("arabic", "فين قاعة المؤتمرات"),
    ("mixed", "فين TD1"),
    ("mixed", "emmène-moi au GMS"),
    ("mixed", "électrique département"),
    ("short", "buvette"),
    ("short", "GMS"),
    ("short", "TD2"),
    ("short", "génie civil"),
    ("adversarial", "take me to the moon"),
    ("adversarial", "I forgot the building name"),
    ("adversarial", "where is my classroom"),
]


def _config_path() -> Path:
    return Path(__file__).resolve().parent / "config.yaml"


def print_result(index: int, category: str, utterance: str, result: NavigationIntent) -> None:
    print(f"{index:02d}. [{category}] {utterance}")
    print(f"    intent     : {result.intent}")
    print(f"    node_id    : {result.node_id}")
    print(f"    label      : {result.label}")
    print(f"    confidence : {result.confidence:.2f}")
    print(f"    language   : {result.language}")
    print(f"    resolved   : {result.resolved}")
    print()


def main() -> None:
    config = load_config(str(_config_path()))
    config["geojson_path"] = str(Path(__file__).resolve().parent / config.get("geojson_path", "data/campus.geojson"))

    print("=" * 72)
    print("ENSAM NLP PIPELINE - END-TO-END TEST")
    print("=" * 72)
    print(f"Backend   : {config.get('backend')}")
    print(f"Model     : {config.get('ollama_model')}")
    print(f"GeoJSON   : {config.get('geojson_path')}")
    print(f"Threshold : {config.get('resolution_threshold')}")
    print()

    pipeline = NLPPipeline(config)

    results: list[tuple[str, str, NavigationIntent | None, Exception | None]] = []
    for category, utterance in TEST_CASES:
        try:
            results.append((category, utterance, pipeline.process(utterance), None))
        except Exception as exc:
            results.append((category, utterance, None, exc))

    for index, (category, utterance, result, exc) in enumerate(results, start=1):
        if exc is not None:
            print(f"{index:02d}. [{category}] {utterance}")
            print(f"    exception: {exc}")
            print()
        elif result is not None:
            print_result(index, category, utterance, result)

    valid_results = [r for _, _, r, exc in results if r is not None and exc is None]
    resolved = sum(1 for r in valid_results if r.resolved)
    unknown = sum(1 for r in valid_results if not r.resolved)
    failed = sum(1 for *_, exc in results if exc is not None)
    avg_conf = sum(r.confidence for r in valid_results if r.resolved) / resolved if resolved else 0.0

    arabic_cases = [(r, exc) for category, _, r, exc in results if category == "arabic"]
    mixed_cases = [(r, exc) for category, _, r, exc in results if category == "mixed"]
    arabic_resolved = sum(1 for r, exc in arabic_cases if exc is None and r is not None and r.resolved)
    mixed_resolved = sum(1 for r, exc in mixed_cases if exc is None and r is not None and r.resolved)

    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Total utterances  : {len(TEST_CASES)}")
    print(f"Resolved          : {resolved}")
    print(f"Unknown           : {unknown}")
    print(f"Average confidence: {avg_conf:.2f}")
    print(f"Arabic resolved   : {arabic_resolved}/3")
    print(f"Mixed resolved    : {mixed_resolved}/3")
    print(f"Failed (exception): {failed}")


if __name__ == "__main__":
    main()
