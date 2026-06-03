"""
Navigation Engine Module — ENSAM Navigation
Calculates the shortest path using Neo4j Graph Database.
Generates turn-by-turn instructions based on geometry.
"""

import math
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class NavigationEngine:
    def __init__(self, graph_store):
        """
        Initialize with a reference to the Neo4j GraphStore instance.
        """
        self.store = graph_store

    def get_shortest_path(self, start_id: str, end_id: str) -> Dict:
        """
        Compute the shortest path using Neo4j APOC Dijkstra algorithm.
        """
        if not self.store.driver:
            return {"error": "Database connection not available."}

        # Query using APOC Dijkstra for optimal path finding based on distance
        query = """
        MATCH (start:Node {id: $start_id})
        MATCH (end:Node {id: $end_id})
        CALL apoc.algo.dijkstra(start, end, 'CONNECTED_TO>', 'distance') YIELD path, weight
        RETURN
            nodes(path) AS path_nodes,
            [r IN relationships(path) | {
                from: startNode(r).id,
                to: endNode(r).id,
                distance: r.distance,
                yaw: r.yaw,
                pitch: r.pitch
            }] AS path_edges,
            weight AS total_distance
        """
        
        try:
            with self.store.driver.session() as session:
                result = session.run(query, start_id=start_id, end_id=end_id)
                record = result.single()
                
                if not record:
                    return {"error": "No path exists between these two locations."}
                
                nodes_data = record["path_nodes"]
                edges_data = record["path_edges"]
                total_dist = record["total_distance"]
                
                # Format node path and metadata
                path_ids = []
                nodes_meta = []
                for n in nodes_data:
                    n_id = n["id"]
                    path_ids.append(n_id)
                    nodes_meta.append({
                        "id": n_id,
                        "name": n.get("name", n_id),
                        "type": n.get("type", "unknown"),
                        "node_role": n.get("node_role", ""),
                        "visible_destination": n.get("visible_destination", True),
                        "floor": n.get("floor", 0),
                        "coords": [n.get("x_svg", 0), n.get("y_svg", 0)]
                    })
                
                # Generate turn-by-turn directions
                instructions = self._generate_instructions(path_ids, nodes_meta)
                
                return {
                    "path": path_ids,
                    "nodes": nodes_meta,
                    "edges": edges_data,
                    "distance": round(total_dist, 1),
                    "instructions": instructions
                }
                
        except Exception as e:
            logger.error(f"Error computing path in Neo4j: {e}")
            return {"error": f"Pathfinding failed: {str(e)}"}

    def _calculate_angle(self, p1: List[float], p2: List[float], p3: List[float]) -> float:
        """Calculate the turn angle between three points (p1 -> p2 -> p3)."""
        dx1 = p2[0] - p1[0]
        dy1 = p2[1] - p1[1]
        dx2 = p3[0] - p2[0]
        dy2 = p3[1] - p2[1]
        
        heading1 = math.atan2(dy1, dx1)
        heading2 = math.atan2(dy2, dx2)
        angle = math.degrees(heading2 - heading1)
        
        if angle > 180: angle -= 360
        elif angle < -180: angle += 360
        return angle

    def _determine_turn(self, angle: float) -> str:
        if -20 <= angle <= 20: return "straight"
        elif 20 < angle <= 160: return "right"
        elif -160 <= angle < -20: return "left"
        else: return "around"

    def _get_edge_distance(self, u: str, v: str) -> float:
        query = """
        MATCH (a:Node {id: $u})-[r:CONNECTED_TO]-(b:Node {id: $v})
        RETURN r.distance as dist LIMIT 1
        """
        with self.store.driver.session() as session:
            res = session.run(query, u=u, v=v).single()
            if res: return res["dist"]
        return 0.0

    def _generate_instructions(self, path: List[str], nodes_meta: List[Dict]) -> List[Dict]:
        instructions = []
        if len(path) < 2:
            instructions.append({"type": "arrive", "text": "You are already at your destination.", "distance": 0})
            return instructions

        next_node = nodes_meta[1]
        current_dist = self._get_edge_distance(path[0], path[1])
        
        instructions.append({
            "type": "start",
            "text": f"Head towards {next_node.get('name', 'the next point')}.",
            "distance": round(current_dist, 1)
        })

        for i in range(1, len(path) - 1):
            prev = nodes_meta[i-1]
            curr = nodes_meta[i]
            nxt = nodes_meta[i+1]
            
            if curr.get('floor', 0) != nxt.get('floor', 0):
                direction = "up" if nxt.get('floor', 0) > curr.get('floor', 0) else "down"
                instructions.append({"type": "stairs", "text": f"Take the stairs {direction}.", "distance": 0})
                continue
            
            prev_coords = prev.get('coords')
            curr_coords = curr.get('coords')
            nxt_coords = nxt.get('coords')
            edge_weight = self._get_edge_distance(path[i], path[i+1])
            
            if prev_coords and curr_coords and nxt_coords:
                angle = self._calculate_angle(prev_coords, curr_coords, nxt_coords)
                turn = self._determine_turn(angle)
                
                if turn == "straight":
                    instructions.append({"type": "straight", "text": f"Continue straight.", "distance": round(edge_weight, 1)})
                else:
                    instructions.append({"type": "turn", "text": f"Turn {turn} at {curr.get('name', 'intersection')}.", "distance": round(edge_weight, 1)})
            else:
                instructions.append({"type": "proceed", "text": f"Proceed to {nxt.get('name')}.", "distance": round(edge_weight, 1)})

        final_node = nodes_meta[-1]
        instructions.append({"type": "arrive", "text": f"Arrived at {final_node.get('name')}.", "distance": 0})
        return instructions
