# Streamlit Deployment

The application runs as a Streamlit web app.

## Local Run

```bash
python -m streamlit run app\main.py
```

Open:

```text
http://localhost:8501
```

## Required Environment Variables

```env
NEO4J_URI=neo4j+s://your-aura-instance.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-aura-generated-password
```

## Deployment Checklist

- Confirm the app can connect to AuraDB.
- Confirm `data/campus_graph.json` matches the Neo4j graph.
- Confirm `checkpoints/best_model.pth` and `checkpoints/gallery.pkl` exist.
- Confirm `data/buildings.json` node IDs exist in the graph.
- Confirm Streamlit static serving can access the SVG map.
- Do not publish `.env` or model checkpoints unless intentionally using Git LFS or private storage.

## Static Assets

The campus map SVG is served from:

```text
app/static/campus_map_2d1.svg
```

The map is rendered through Leaflet using simple SVG coordinate bounds.

