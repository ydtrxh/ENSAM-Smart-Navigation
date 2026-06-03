import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_all_building_node_ids_exist_in_campus_graph():
    buildings = load_json(DATA_DIR / "buildings.json")
    graph = load_json(DATA_DIR / "campus_graph.json")

    graph_node_ids = {node["id"] for node in graph.get("nodes", [])}
    missing = [
        entry["node_id"]
        for entry in buildings
        if entry.get("node_id") not in graph_node_ids
    ]

    assert not missing, f"Missing graph nodes for buildings.json node_id values: {missing}"


def test_campus_graph_edges_reference_existing_nodes():
    graph = load_json(DATA_DIR / "campus_graph.json")
    graph_node_ids = {node["id"] for node in graph.get("nodes", [])}

    missing_edges = []
    for edge in graph.get("edges", []):
        source = edge.get("from", edge.get("source"))
        target = edge.get("to", edge.get("target"))
        if source not in graph_node_ids or target not in graph_node_ids:
            missing_edges.append({"source": source, "target": target})

    assert not missing_edges, f"Edges reference missing nodes: {missing_edges}"


def test_campus_graph_edges_have_positive_distance():
    graph = load_json(DATA_DIR / "campus_graph.json")

    invalid_edges = []
    for edge in graph.get("edges", []):
        source = edge.get("from", edge.get("source"))
        target = edge.get("to", edge.get("target"))
        distance = edge.get("distance", edge.get("weight", edge.get("distance_meters")))
        if distance is None or float(distance) <= 0:
            invalid_edges.append({"source": source, "target": target, "distance": distance})

    assert not invalid_edges, f"Edges with missing or non-positive distance: {invalid_edges}"
