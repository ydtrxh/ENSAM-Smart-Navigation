import json
import networkx as nx
from scipy.spatial import KDTree
import math

class NavigationGraph:
    def __init__(self, graph_path, buildings_path):
        with open(graph_path, 'r', encoding='utf-8') as f:
            self.graph_data = json.load(f)
            
        with open(buildings_path, 'r', encoding='utf-8') as f:
            self.buildings_meta = json.load(f)
            
        self.G = nx.Graph()
        self.nodes = {}
        
        # Build networkx graph
        for node in self.graph_data.get("nodes", []):
            self.nodes[node["id"]] = node
            self.G.add_node(node["id"], **node)
            
        for edge in self.graph_data.get("edges", []):
            self.G.add_edge(edge["source"], edge["target"], weight=edge["distance_meters"])
            
        # Build KDTree for spatial lookups (using SVG coordinates)
        self.node_ids = list(self.nodes.keys())
        coords = [[self.nodes[nid]["x_svg"], self.nodes[nid]["y_svg"]] for nid in self.node_ids]
        if coords:
            self.kdtree = KDTree(coords)
        else:
            self.kdtree = None
            
        self.scale_factor = self.graph_data.get("meta", {}).get("scale_factor", 1.0)
            
    def find_nearest_node(self, x_svg: float, y_svg: float) -> str:
        """
        Returns the ID of the nearest graph node to the given SVG coordinates.
        """
        if not self.kdtree:
            return None
        distance, index = self.kdtree.query([x_svg, y_svg])
        return self.node_ids[index]

    def astar(self, start_node_id: str, end_node_id: str) -> dict:
        """
        Returns shortest path and metrics between two nodes.
        Heuristic: Euclidean distance in SVG coordinates * scale_factor.
        """
        if start_node_id not in self.G or end_node_id not in self.G:
            return None
            
        def heuristic(u, v):
            u_node = self.nodes[u]
            v_node = self.nodes[v]
            dist = math.sqrt((u_node["x_svg"] - v_node["x_svg"])**2 + (u_node["y_svg"] - v_node["y_svg"])**2)
            return dist * self.scale_factor

        try:
            path = nx.astar_path(self.G, start_node_id, end_node_id, heuristic=heuristic, weight='weight')
            distance = nx.astar_path_length(self.G, start_node_id, end_node_id, heuristic=heuristic, weight='weight')
            
            coords_svg = [[self.nodes[nid]["x_svg"], self.nodes[nid]["y_svg"]] for nid in path]
            coords_norm = [[self.nodes[nid]["x_norm"], self.nodes[nid]["y_norm"]] for nid in path]
            
            return {
                "path": path,
                "distance_meters": round(distance, 2),
                "coordinates_svg": coords_svg,
                "coordinates_norm": coords_norm
            }
        except nx.NetworkXNoPath:
            return None

    def route_to_building(self, building_id: str, start_node_id: str) -> dict:
        """
        Looks up building_id in buildings list to retrieve node_id,
        then calls astar(start_node_id, node_id).
        """
        target_node = None
        for b in self.buildings_meta:
            if b["id"] == building_id:
                target_node = b["node_id"]
                break
                
        if not target_node:
            raise ValueError(f"Building {building_id} not found in buildings_meta.")
            
        return self.astar(start_node_id, target_node)

# Expose module-level functions matching the spec
def find_nearest_node(x_svg: float, y_svg: float, graph: dict) -> str:
    # Build tree on the fly (for standalone use as requested in spec)
    nodes = {n["id"]: n for n in graph.get("nodes", [])}
    node_ids = list(nodes.keys())
    coords = [[nodes[nid]["x_svg"], nodes[nid]["y_svg"]] for nid in node_ids]
    if not coords: return None
    kdtree = KDTree(coords)
    _, index = kdtree.query([x_svg, y_svg])
    return node_ids[index]

def astar(start_node_id: str, end_node_id: str, graph: dict) -> dict:
    # Standalone function as requested in spec
    G = nx.Graph()
    nodes = {n["id"]: n for n in graph.get("nodes", [])}
    for n in graph.get("nodes", []): G.add_node(n["id"], **n)
    for e in graph.get("edges", []): G.add_edge(e["source"], e["target"], weight=e["distance_meters"])
    scale_factor = graph.get("meta", {}).get("scale_factor", 1.0)
    
    def heuristic(u, v):
        u_node, v_node = nodes[u], nodes[v]
        dist = math.sqrt((u_node["x_svg"] - v_node["x_svg"])**2 + (u_node["y_svg"] - v_node["y_svg"])**2)
        return dist * scale_factor

    try:
        path = nx.astar_path(G, start_node_id, end_node_id, heuristic=heuristic, weight='weight')
        distance = nx.astar_path_length(G, start_node_id, end_node_id, heuristic=heuristic, weight='weight')
        coords_svg = [[nodes[nid]["x_svg"], nodes[nid]["y_svg"]] for nid in path]
        coords_norm = [[nodes[nid]["x_norm"], nodes[nid]["y_norm"]] for nid in path]
        return {"path": path, "distance_meters": round(distance, 2), "coordinates_svg": coords_svg, "coordinates_norm": coords_norm}
    except nx.NetworkXNoPath:
        return None

def route_to_building(building_id: str, start_node_id: str, buildings: list, graph: dict) -> dict:
    target_node = None
    for b in buildings:
        if b["id"] == building_id:
            target_node = b["node_id"]
            break
    if not target_node: return None
    return astar(start_node_id, target_node, graph)
