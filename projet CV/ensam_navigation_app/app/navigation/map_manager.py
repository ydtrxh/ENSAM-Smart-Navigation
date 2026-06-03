"""
Map rendering module using Leaflet embedded inside Streamlit.
"""
import streamlit.components.v1 as components
import json

def generate_map_html(nodes: dict, edges: list, current_path=None, current_location: str = None) -> str:
    """
    Generates the HTML/JS for a Leaflet map.
    """
    # Filter edges to a JS-friendly format
    js_edges = []
    for edge in edges:
        u = nodes.get(edge["from"])
        v = nodes.get(edge["to"])
        if u and v:
            js_edges.append([ [u["coords"][1], u["coords"][0]], [v["coords"][1], v["coords"][0]] ])

    # Active path coords
    path_coords = []
    if current_path and len(current_path) > 1:
        for nid in current_path:
            nd = nodes.get(nid)
            if nd:
                path_coords.append([nd["coords"][1], nd["coords"][0]])
                
    # Current location
    curr_coord = None
    if current_location and current_location in nodes:
        nd = nodes[current_location]
        curr_coord = [nd["coords"][1], nd["coords"][0]]
        
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <style>
            #map {{ height: 500px; width: 100%; }}
            body {{ margin: 0; padding: 0; }}
        </style>
    </head>
    <body>
        <div id="map"></div>
        <script>
            // Setup Leaflet map with simple CRS (for pixel/SVG coordinates)
            var map = L.map('map', {{
                crs: L.CRS.Simple,
                minZoom: -2,
                maxZoom: 3
            }});

            var bounds = [[0, 0], [1480, 2092]];
            
            // SVG served via Streamlit static file server (enableStaticServing = true in .streamlit/config.toml)
            L.imageOverlay('app/static/campus_map_2d1.svg', bounds).addTo(map);
            map.fitBounds(bounds);

            // Draw walkable edges (Grey dashed)
            var edges = {json.dumps(js_edges)};
            edges.forEach(function(edge) {{
                L.polyline(edge, {{color: '#94A3B8', weight: 2, opacity: 0.4, dashArray: '6, 6'}}).addTo(map);
            }});

            // Draw active path
            var pathCoords = {json.dumps(path_coords)};
            if (pathCoords.length > 0) {{
                L.polyline(pathCoords, {{color: '#DC2626', weight: 6, opacity: 0.85}}).addTo(map);
                
                // Destination marker
                var dest = pathCoords[pathCoords.length - 1];
                L.circleMarker(dest, {{radius: 8, fillColor: '#DC2626', color: '#fff', weight: 2, fillOpacity: 1}}).addTo(map);
            }}

            // Draw current location
            var currCoord = {json.dumps(curr_coord)};
            if (currCoord) {{
                L.circleMarker(currCoord, {{
                    radius: 10,
                    fillColor: '#1D4ED8',
                    color: '#fff',
                    weight: 3,
                    fillOpacity: 1
                }}).bindTooltip("Vous êtes ici", {{permanent: true, direction: 'right'}}).addTo(map);
            }}
        </script>
    </body>
    </html>
    """
    return html

def render_map(nodes: dict, edges: list, current_path=None, current_location: str = None):
    html = generate_map_html(nodes, edges, current_path, current_location)
    components.html(html, height=520, width=1000)
