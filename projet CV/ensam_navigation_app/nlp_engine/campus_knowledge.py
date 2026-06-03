"""
campus_knowledge.py — Campus GeoJSON knowledge base.

Loads the campus GeoJSON once at startup, builds a node registry and
a pre-normalized alias table for fast fuzzy matching at query time.

Design notes:
  - GeoJSON is parsed with Python's built-in json module only.
    No GIS library (geojson, geopandas, shapely) is used or permitted.
  - The alias table is built and normalized ONCE at startup.
    It is NEVER re-normalized at query time. This is a deliberate design
    decision to keep the hot query path (resolve_entity) as fast as possible.
  - Aliases are the primary performance lever of this module.
    Improving aliases improves resolution more than upgrading models.
    Add language variants (Arabic, French, English), abbreviations,
    common misspellings, and transliterated forms to each feature's
    "aliases" list in the GeoJSON.
"""

import json
import logging
import unicodedata
import string
from pathlib import Path

logger = logging.getLogger(__name__)


class CampusGeoJSONError(ValueError):
    """Raised when campus GeoJSON is missing or malformed."""


def normalize(text: str) -> str:
    """
    Normalize a string for alias matching:
      1. Lowercase
      2. Strip diacritics via NFD decomposition + remove combining chars
      3. Remove punctuation

    This function is applied to aliases at startup (once) and to the
    query string at call time. Both sides use the same normalization so
    comparison is always apples-to-apples.

    Note: Arabic text does NOT use diacritics normalization in the same way
    as French. Arabic letters are preserved; only harakat (short vowel marks,
    Unicode category Mn) are stripped, which is the desired behavior.
    """
    # Step 1: lowercase
    text = text.lower().strip()
    # Step 2: NFD decomposition → strip combining characters (diacritics)
    nfd = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
    # Step 3: remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Step 4: collapse whitespace
    text = " ".join(text.split())
    return text


class CampusKnowledge:
    """
    In-memory knowledge base built from the campus GeoJSON.

    Attributes:
        node_registry (dict[str, dict]):
            Maps node_id → full properties dict for that feature.

        _normalized_alias_table (list[tuple[str, str]]):
            List of (normalized_alias, node_id) pairs, built ONCE at startup.
            # Aliases are normalized once at startup and cached.
            # Do not normalize the alias table inside the query path.
    """

    def __init__(self, geojson_path: str):
        """
        Load and parse the campus GeoJSON, then build the alias table.

        Args:
            geojson_path: Absolute or relative path to the campus .geojson file.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError:        If the file is not valid JSON or is missing
                               required fields (id, label, aliases).
        """
        path = Path(geojson_path)
        if not path.exists():
            raise CampusGeoJSONError(
                f"Campus GeoJSON not found at '{geojson_path}'. "
                "Check the geojson_path setting in config.yaml."
            )

        logger.info(f"Loading campus knowledge from: {path}")

        try:
            with path.open(encoding="utf-8") as f:
                raw = json.load(f)
        except json.JSONDecodeError as e:
            raise CampusGeoJSONError(
                f"Failed to parse campus GeoJSON '{geojson_path}': {e}"
            ) from e

        if "features" not in raw or not isinstance(raw["features"], list):
            raise CampusGeoJSONError(
                f"GeoJSON '{geojson_path}' is missing a 'features' array."
            )

        # ── Build node registry and alias table ───────────────────────────
        self.node_registry: dict[str, dict] = {}

        # Aliases are the primary performance lever of this module. Improving
        # aliases improves resolution more than upgrading models.
        # Aliases are normalized once at startup and cached.
        # Do not normalize the alias table inside the query path.
        self._normalized_alias_table: list[tuple[str, str]] = []

        for feature in raw["features"]:
            props = feature.get("properties", {})

            # Validate required fields — fail loudly, never silently
            node_id = props.get("id")
            label   = props.get("label")
            aliases = props.get("aliases")

            if not node_id:
                raise CampusGeoJSONError(
                    f"A GeoJSON feature is missing 'id' in its properties: {props}"
                )
            if not label:
                raise CampusGeoJSONError(
                    f"Feature '{node_id}' is missing 'label' in its properties."
                )
            if not isinstance(aliases, list):
                raise CampusGeoJSONError(
                    f"Feature '{node_id}' has no 'aliases' list. "
                    "Every node must declare at least one alias."
                )
            if len(aliases) == 0:
                raise CampusGeoJSONError(
                    f"Feature '{node_id}' has an empty 'aliases' list. "
                    "Add at least one alias string."
                )

            # Store full metadata
            self.node_registry[node_id] = props

            # Add the canonical label itself as a normalised alias
            all_aliases = [label] + aliases
            for alias in all_aliases:
                if not isinstance(alias, str) or not alias.strip():
                    continue
                normalized = normalize(alias)
                if normalized:
                    self._normalized_alias_table.append((normalized, node_id))

        self._validate_against_campus_graph(path)

        node_count  = len(self.node_registry)
        alias_count = len(self._normalized_alias_table)
        logger.info(
            f"Campus knowledge loaded: {node_count} nodes, "
            f"{alias_count} normalized aliases."
        )

    def _validate_against_campus_graph(self, geojson_path: Path) -> None:
        """
        Validate NLP node IDs against data/campus_graph.json when available.

        The navigation app routes through Neo4j populated from campus_graph.json,
        so NLP node IDs must exist in that graph.
        """
        graph_path = self._resolve_campus_graph_path(geojson_path)
        if not graph_path.exists():
            logger.warning("campus_graph.json not found at %s; skipping node_id validation.", graph_path)
            return

        try:
            with graph_path.open(encoding="utf-8") as f:
                graph = json.load(f)
        except json.JSONDecodeError as exc:
            raise CampusGeoJSONError(f"Failed to parse campus graph '{graph_path}': {exc}") from exc

        graph_ids = {node.get("id") for node in graph.get("nodes", []) if node.get("id")}
        missing = sorted(set(self.node_registry) - graph_ids)
        if missing:
            raise CampusGeoJSONError(
                "NLP GeoJSON contains node_id values missing from data/campus_graph.json: "
                + ", ".join(missing)
            )

    def _resolve_campus_graph_path(self, geojson_path: Path) -> Path:
        """
        Locate the campus graph in either the current or target repository layout.

        During the structure migration, the graph may live at:
          - data/campus_graph.json
          - graph/data/campus_graph.json

        When the full app is available, app.config is the source of truth. When
        the NLP package is tested independently, walk upward from the GeoJSON
        file and use the first known graph path that exists.
        """
        try:
            from app.config import settings

            configured = Path(settings.get_path("campus_graph_json"))
            if configured.exists():
                return configured
        except Exception:
            pass

        for parent in [geojson_path.resolve().parent, *geojson_path.resolve().parents]:
            candidates = [
                parent / "data" / "campus_graph.json",
                parent / "graph" / "data" / "campus_graph.json",
            ]
            for candidate in candidates:
                if candidate.exists():
                    return candidate

        return geojson_path.resolve().parent / "campus_graph.json"

    # ── Public API ─────────────────────────────────────────────────────────

    def get_all_labels(self) -> list[str]:
        """Return canonical labels for all registered nodes."""
        return [props["label"] for props in self.node_registry.values()]

    def get_node(self, node_id: str) -> dict:
        """
        Return the full metadata dict for a given node_id.

        Args:
            node_id: The node identifier (e.g. 'bibliotheque').

        Returns:
            Properties dict from the GeoJSON feature.

        Raises:
            KeyError: If node_id is not in the registry.
        """
        if node_id not in self.node_registry:
            raise KeyError(
                f"Node '{node_id}' not found in campus registry. "
                f"Known nodes: {sorted(self.node_registry.keys())}"
            )
        return self.node_registry[node_id]
