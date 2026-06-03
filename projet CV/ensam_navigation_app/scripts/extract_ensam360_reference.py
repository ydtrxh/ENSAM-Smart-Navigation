import argparse
import json
import re
from pathlib import Path


NODE_PATTERN = re.compile(
    r"""\{\s*id:\s*"(.*?)"\s*,\s*label:\s*\[(.*?)\]\s*,\s*name:\s*"(.*?)".*?"""
    r"""description:\s*"(.*?)".*?map_coords:\s*\[(.*?)\].*?"""
    r"""pano_url:\s*"(.*?)".*?flat_url:\s*"(.*?)".*?floor:\s*(\d+)""",
    re.DOTALL,
)

CONNECTION_PATTERN = re.compile(
    r"MATCH\s*\(a:Location\s*\{id:\s*'(.*?)'\}\)\s*"
    r"MATCH\s*\(b:Location\s*\{id:\s*'(.*?)'\}\)",
    re.DOTALL,
)


def parse_labels(raw_labels):
    labels = []
    for item in raw_labels.split(","):
        cleaned = item.strip().strip("\"'")
        if cleaned:
            labels.append(cleaned)
    return labels


def parse_coords(raw_coords):
    return [float(part.strip()) for part in raw_coords.split(",")]


def extract_reference(seed_path):
    seed_text = Path(seed_path).read_text(encoding="utf-8", errors="replace")
    stripped = re.sub(r"^\s*//\s*", "", seed_text, flags=re.MULTILINE)

    nodes_by_id = {}
    for match in NODE_PATTERN.finditer(seed_text):
        node = {
            "id": match.group(1),
            "labels": parse_labels(match.group(2)),
            "name": match.group(3),
            "description": match.group(4),
            "coords": parse_coords(match.group(5)),
            "pano_url": match.group(6),
            "flat_url": match.group(7),
            "floor": int(match.group(8)),
        }
        nodes_by_id[node["id"]] = node

    connections = []
    seen = set()
    for match in CONNECTION_PATTERN.finditer(stripped):
        src, dst = match.group(1), match.group(2)
        if (src, dst) not in seen:
            seen.add((src, dst))
            connections.append({"from": src, "to": dst})

    missing_nodes = []
    for edge in connections:
        if edge["from"] not in nodes_by_id or edge["to"] not in nodes_by_id:
            missing_nodes.append(edge)

    return {
        "source": str(seed_path),
        "notes": [
            "Reference model only. Coordinates are from the old React/Leaflet map and must be remapped before use.",
            "Use node roles and connection patterns as architecture guidance, not as a direct replacement.",
        ],
        "nodes": list(nodes_by_id.values()),
        "edges": connections,
        "missing_node_edges": missing_nodes,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract old ENSAM360 routing model as a reference JSON file.")
    parser.add_argument(
        "--seed",
        default=r"C:\Users\Hp\OneDrive\Desktop\Nouveau dossier\PythonProjects\ENSAM360-main\ENSAM360-main\backend\scripts\seed.js",
    )
    parser.add_argument("--out", default="data/ensam360_old_routing_reference.json")
    args = parser.parse_args()

    reference = extract_reference(args.seed)
    out_path = Path(args.out)
    out_path.write_text(json.dumps(reference, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Old routing reference written to {out_path}")
    print(f"Nodes: {len(reference['nodes'])}")
    print(f"Edges: {len(reference['edges'])}")
    print(f"Edges with missing nodes: {len(reference['missing_node_edges'])}")


if __name__ == "__main__":
    main()
