import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


BUILDING_HINTS = {
    "administration",
    "academic",
    "libraries",
    "laboratories",
    "amphitheaters",
    "student services",
    "sports facilities",
    "residential",
}

WAYPOINT_HINTS = {
    "couloir",
    "intersection",
    "waypoint",
    "path",
    "road",
    "entrance",
    "door",
    "stairs",
}


def edge_endpoints(edge):
    src = edge.get("from", edge.get("source"))
    dst = edge.get("to", edge.get("target"))
    return src, dst


def edge_distance(edge, src_node=None, dst_node=None):
    value = edge.get("distance", edge.get("weight", edge.get("distance_meters")))
    if value is not None:
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
    if src_node and dst_node:
        ax, ay = src_node.get("coords", [0, 0])
        bx, by = dst_node.get("coords", [0, 0])
        return math.hypot(ax - bx, ay - by)
    return 0.0


def node_role(node):
    explicit = node.get("node_role") or node.get("role")
    if explicit:
        return str(explicit).lower()

    ntype = str(node.get("type", "")).lower()
    name = str(node.get("name", "")).lower()
    nid = str(node.get("id", "")).lower()
    text = " ".join([ntype, name, nid])

    if any(hint in text for hint in WAYPOINT_HINTS):
        if "entrance" in text or "entree" in text or "entrée" in text:
            return "entrance"
        if "door" in text or "porte" in text:
            return "door"
        if "stairs" in text or "escalier" in text:
            return "stairs"
        return "path"

    if any(hint in text for hint in BUILDING_HINTS):
        return "building"

    if nid.startswith("n_"):
        return "path"

    return "unknown"


def audit_graph(graph_path, buildings_path=None, long_edge_threshold=120.0):
    graph_path = Path(graph_path)
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in graph.get("nodes", [])}
    edges = graph.get("edges", [])

    missing_node_edges = []
    duplicate_edges = []
    self_loops = []
    long_edges = []
    building_junction_edges = []
    seen = set()
    degree = Counter()
    roles = {}

    for node_id, node in nodes.items():
        roles[node_id] = node_role(node)

    for edge in edges:
        src, dst = edge_endpoints(edge)
        if not src or not dst or src not in nodes or dst not in nodes:
            missing_node_edges.append(edge)
            continue

        if src == dst:
            self_loops.append(edge)

        key = tuple(sorted([src, dst]))
        if key in seen:
            duplicate_edges.append(edge)
        seen.add(key)

        degree[src] += 1
        degree[dst] += 1

        dist = edge_distance(edge, nodes[src], nodes[dst])
        src_role = roles[src]
        dst_role = roles[dst]
        if dist >= long_edge_threshold:
            long_edges.append({
                "from": src,
                "to": dst,
                "distance": round(dist, 2),
                "from_role": src_role,
                "to_role": dst_role,
            })

        if src_role == "building" and dst_role == "building":
            building_junction_edges.append({
                "from": src,
                "to": dst,
                "reason": "building_to_building",
            })
        elif src_role == "building" and dst_role not in {"entrance", "door", "path"}:
            building_junction_edges.append({
                "from": src,
                "to": dst,
                "reason": f"building_to_{dst_role}",
            })
        elif dst_role == "building" and src_role not in {"entrance", "door", "path"}:
            building_junction_edges.append({
                "from": src,
                "to": dst,
                "reason": f"{src_role}_to_building",
            })

    isolated_nodes = sorted(node_id for node_id in nodes if degree[node_id] == 0)
    role_counts = Counter(roles.values())

    building_node_id_errors = []
    if buildings_path:
        buildings_path = Path(buildings_path)
        if buildings_path.exists():
            buildings = json.loads(buildings_path.read_text(encoding="utf-8"))
            for item in buildings:
                node_id = item.get("node_id")
                if node_id not in nodes:
                    building_node_id_errors.append(item)

    adjacency = defaultdict(list)
    for edge in edges:
        src, dst = edge_endpoints(edge)
        if src in nodes and dst in nodes:
            adjacency[src].append(dst)
            adjacency[dst].append(src)

    building_access = []
    for node_id, role in roles.items():
        if role != "building":
            continue
        neighbors = adjacency[node_id]
        neighbor_roles = sorted({roles.get(n, "unknown") for n in neighbors})
        if "entrance" not in neighbor_roles and "door" not in neighbor_roles:
            building_access.append({
                "id": node_id,
                "name": nodes[node_id].get("name", node_id),
                "neighbor_roles": neighbor_roles,
                "recommendation": "Add an entrance/door node between this building and the path graph.",
            })

    return {
        "graph": str(graph_path),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "role_counts": dict(role_counts),
        "missing_node_edges": missing_node_edges,
        "duplicate_edges": duplicate_edges,
        "self_loops": self_loops,
        "isolated_nodes": isolated_nodes,
        "long_edge_threshold": long_edge_threshold,
        "long_edges": sorted(long_edges, key=lambda item: item["distance"], reverse=True),
        "building_junction_edges": building_junction_edges,
        "building_node_id_errors": building_node_id_errors,
        "buildings_without_entrance_or_door_neighbor": building_access,
    }


def main():
    parser = argparse.ArgumentParser(description="Audit campus routing graph topology.")
    parser.add_argument("--graph", default="data/campus_graph.json")
    parser.add_argument("--buildings", default="data/buildings.json")
    parser.add_argument("--long-edge-threshold", type=float, default=120.0)
    parser.add_argument("--out", default="data/routing_audit_report.json")
    args = parser.parse_args()

    report = audit_graph(args.graph, args.buildings, args.long_edge_threshold)
    out_path = Path(args.out)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Routing audit written to {out_path}")
    print(f"Nodes: {report['node_count']} | Edges: {report['edge_count']}")
    print(f"Long edges: {len(report['long_edges'])}")
    print(f"Buildings without entrance/door neighbor: {len(report['buildings_without_entrance_or_door_neighbor'])}")
    print(f"Missing-node edges: {len(report['missing_node_edges'])}")


if __name__ == "__main__":
    main()
