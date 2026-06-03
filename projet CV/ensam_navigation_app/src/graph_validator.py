import json
import networkx as nx

def validate_graph(graph_dict, buildings_meta):
    """
    Validates the generated graph dict against the spec requirements.
    Returns (passed: bool, report: dict, errors: list)
    """
    report = {
        "total_nodes": len(graph_dict.get("nodes", [])),
        "total_edges": len(graph_dict.get("edges", [])),
        "total_buildings": len(buildings_meta),
        "buildings_without_node": [],
        "isolated_nodes": [],
        "disconnected_components": 0,
        "self_loops": [],
        "duplicate_edges": [],
        "passed": True,
        "warnings": []
    }
    errors = []
    
    G = nx.Graph()
    
    nodes_set = set()
    for n in graph_dict.get("nodes", []):
        nodes_set.add(n["id"])
        G.add_node(n["id"])
        
    for b in buildings_meta:
        if b["node_id"] not in nodes_set:
            report["buildings_without_node"].append(b["id"])
            errors.append(f"Building {b['id']} has node_id {b['node_id']} which is not in the graph.")
            
    seen_edges = set()
    for e in graph_dict.get("edges", []):
        u = e.get("source", e.get("from"))
        v = e.get("target", e.get("to"))
        dist = e.get("distance_meters", e.get("distance", e.get("weight", 0)))

        if not u or not v:
            errors.append(f"Edge has missing endpoints: {e}")
            continue
        
        if dist <= 0:
            errors.append(f"Edge {u}-{v} has non-positive distance {dist}.")
            
        if u == v:
            report["self_loops"].append(u)
            errors.append(f"Self-loop detected on node {u}.")
            
        edge_id = tuple(sorted([u, v]))
        if edge_id in seen_edges:
            report["duplicate_edges"].append(f"{u}-{v}")
            errors.append(f"Duplicate edge detected between {u} and {v}.")
        else:
            seen_edges.add(edge_id)
            
        G.add_edge(u, v, weight=dist)
        
    isolated = list(nx.isolates(G))
    if isolated:
        report["isolated_nodes"] = isolated
        errors.append(f"Graph contains {len(isolated)} isolated nodes.")
        
    components = list(nx.connected_components(G))
    report["disconnected_components"] = len(components)
    if len(components) != 1:
        errors.append(f"Graph is not fully connected. Found {len(components)} components.")
        
    if errors:
        report["passed"] = False
        
    return report["passed"], report, errors
