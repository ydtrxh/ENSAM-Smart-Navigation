from pathlib import Path

from app.config import PROJECT_ROOT, settings


def test_runtime_config_resolves_current_source_files():
    required_keys = [
        "buildings_json",
        "campus_graph_json",
        "campus_geojson",
        "nlp_config",
        "campus_svg",
    ]

    missing = []
    for key in required_keys:
        path = Path(settings.get_path(key))
        if not path.exists():
            missing.append((key, str(path)))

    assert not missing, f"Configured source paths do not exist: {missing}"


def test_runtime_config_resolves_active_cv_artifacts():
    required_keys = ["best_model_pth", "gallery_pkl"]

    missing = []
    for key in required_keys:
        path = Path(settings.get_path(key))
        if not path.exists():
            missing.append((key, str(path)))

    assert not missing, f"Configured CV artifact paths do not exist: {missing}"


def test_runtime_path_candidates_are_absolute():
    for key in ["campus_graph_json", "best_model_pth", "gallery_pkl"]:
        candidates = settings.get_path_candidates(key)
        assert candidates, f"No candidates configured for {key}"
        assert all(Path(candidate).is_absolute() for candidate in candidates)


def test_project_root_points_to_repository_root():
    assert (PROJECT_ROOT / "README.md").exists()
    assert (PROJECT_ROOT / "app").is_dir()
    assert (PROJECT_ROOT / "data").is_dir()
