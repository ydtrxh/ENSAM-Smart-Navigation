# Neo4j Integration

The project uses Neo4j AuraDB to store the campus navigation graph and compute shortest paths.

## AuraDB Configuration

Create a `.env` file:

```env
NEO4J_URI=neo4j+s://your-aura-instance.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-aura-generated-password
```

## Synchronization

The synchronization script imports `data/campus_graph.json` into Neo4j:

```bash
python scripts\sync_campus_graph.py
```

It creates:

- `Node` nodes,
- `CONNECTED_TO` relationships,
- bidirectional edges,
- distance attributes used for routing.

## Shortest Path Query

The runtime navigation engine uses APOC Dijkstra:

```cypher
MATCH (start:Node {id: $start_id})
MATCH (end:Node {id: $end_id})
CALL apoc.algo.dijkstra(start, end, 'CONNECTED_TO>', 'distance')
YIELD path, weight
RETURN nodes(path), relationships(path), weight
```

## Data Contract

Every edge must reference existing nodes and must have a positive distance.

Every `node_id` in `data/buildings.json` must exist in `data/campus_graph.json`.

These rules are checked by contract tests in `tests/test_graph_contract.py`.

