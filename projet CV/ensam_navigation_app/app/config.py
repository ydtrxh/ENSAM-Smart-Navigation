from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent


DEFAULT_CONFIG: dict[str, Any] = {
    "paths": {
        "buildings_json": [
            "graph/data/buildings.json",
            "data/buildings.json",
        ],
        "campus_graph_json": [
            "graph/data/campus_graph.json",
            "data/campus_graph.json",
        ],
        "campus_geojson": [
            "graph/data/campus.geojson",
            "data/campus.geojson",
            "nlp_engine/data/campus.geojson",
        ],
        "model_metrics": [
            "outputs/evaluation/model_metrics.json",
            "outputs/eval_report.json",
            "model_metrics.json",
        ],
        "nlp_config": [
            "configs/nlp.yaml",
            "nlp_engine/config.yaml",
        ],
        "gallery_pkl": [
            "models/checkpoints/gallery.pkl",
            "checkpoints/gallery.pkl",
            "cv_engine/outputs/gallery.pkl",
        ],
        "best_model_pth": [
            "models/checkpoints/best_model.pth",
            "checkpoints/best_model.pth",
            "cv_engine/checkpoints/best_model.pth",
        ],
        "campus_svg": [
            "app/static/campus_map_2d1.svg",
        ],
    }
}


class AppConfig:
    """Runtime path configuration with migration-safe fallbacks."""

    def __init__(self, config_path: str | os.PathLike[str] | None = None):
        self.project_root = PROJECT_ROOT
        self.config_path = Path(config_path) if config_path else PROJECT_ROOT / "configs" / "app.yaml"
        self.config_data = copy.deepcopy(DEFAULT_CONFIG)
        self._load_yaml()

    def _load_yaml(self) -> None:
        if not self.config_path.exists():
            return

        try:
            with self.config_path.open("r", encoding="utf-8") as f:
                user_config = yaml.safe_load(f) or {}
        except Exception as exc:
            print(f"[Config] Error loading {self.config_path}: {exc}")
            return

        paths = user_config.get("paths")
        if isinstance(paths, dict):
            self.config_data["paths"].update(paths)

    def _to_abs_path(self, path_value: str | os.PathLike[str]) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path
        return self.project_root / path

    def get_path(self, key: str, must_exist: bool = False) -> str:
        """
        Return an absolute path for a configured key.

        If a key maps to a list, the first existing path is returned. If no
        candidate exists, the first candidate is returned so callers can still
        create files at the preferred future location.
        """
        path_value = self.config_data.get("paths", {}).get(key)
        if not path_value:
            if must_exist:
                raise KeyError(f"Unknown configured path key: {key}")
            return ""

        candidates = path_value if isinstance(path_value, list) else [path_value]
        abs_candidates = [self._to_abs_path(candidate) for candidate in candidates]

        for candidate in abs_candidates:
            if candidate.exists():
                return str(candidate)

        if must_exist:
            formatted = ", ".join(str(path) for path in abs_candidates)
            raise FileNotFoundError(f"No configured path exists for key '{key}': {formatted}")

        return str(abs_candidates[0])

    def get_path_candidates(self, key: str) -> list[str]:
        path_value = self.config_data.get("paths", {}).get(key)
        if not path_value:
            return []
        candidates = path_value if isinstance(path_value, list) else [path_value]
        return [str(self._to_abs_path(candidate)) for candidate in candidates]


settings = AppConfig()
