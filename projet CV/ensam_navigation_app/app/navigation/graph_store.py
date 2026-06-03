"""
Graph Store Module — ENSAM Navigation
Connects to Neo4j to fetch graph metadata and nodes.
"""
from neo4j import GraphDatabase
import logging
import os

logger = logging.getLogger(__name__)

class GraphStore:
    def __init__(self, uri, user, password):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        self._connect()

    def _connect(self):
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.driver.verify_connectivity()
            logger.info("Connected to Neo4j successfully.")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j at {self.uri}: {e}")

    def get_node_metadata(self, node_id: str) -> dict:
        """Fetch all metadata properties for a given node ID."""
        if not self.driver:
            return {}
            
        query = "MATCH (n:Node {id: $node_id}) RETURN n"
        with self.driver.session() as session:
            result = session.run(query, node_id=node_id)
            record = result.single()
            if record:
                node = record["n"]
                return {
                    "id": node["id"],
                    "name": node.get("name", node["id"]),
                    "type": node.get("type", "unknown"),
                    "node_role": node.get("node_role", ""),
                    "visible_destination": node.get("visible_destination", True),
                    "floor": node.get("floor", 0),
                    "coords": [node.get("x_svg", 0), node.get("y_svg", 0)],
                    "x_norm": node.get("x_norm", 0),
                    "y_norm": node.get("y_norm", 0)
                }
        return {}

    def get_all_nodes(self) -> list:
        """Returns a list of all nodes with their metadata."""
        if not self.driver:
            return []
            
        nodes = []
        query = "MATCH (n:Node) RETURN n"
        with self.driver.session() as session:
            result = session.run(query)
            for record in result:
                node = record["n"]
                nodes.append({
                    "id": node["id"],
                    "name": node.get("name", node["id"]),
                    "type": node.get("type", "unknown"),
                    "node_role": node.get("node_role", ""),
                    "visible_destination": node.get("visible_destination", True),
                    "floor": node.get("floor", 0),
                    "coords": [node.get("x_svg", 0), node.get("y_svg", 0)],
                    "x_norm": node.get("x_norm", 0),
                    "y_norm": node.get("y_norm", 0)
                })
        return nodes

    def get_all_edges(self) -> list:
        """Returns all edges for map rendering."""
        if not self.driver:
            return []
            
        edges = []
        query = """
        MATCH (a:Node)-[r:CONNECTED_TO]->(b:Node)
        RETURN a.id as src, b.id as dst, r.distance as weight, r.yaw as yaw, r.pitch as pitch
        """
        with self.driver.session() as session:
            result = session.run(query)
            for record in result:
                edges.append({
                    "from": record["src"],
                    "to": record["dst"],
                    "weight": record["weight"],
                    "yaw": record["yaw"],
                    "pitch": record["pitch"]
                })
        return edges

    def close(self):
        if self.driver:
            self.driver.close()
