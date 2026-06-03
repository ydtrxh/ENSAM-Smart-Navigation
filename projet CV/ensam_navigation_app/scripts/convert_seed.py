import re
import os
import json

js_file = r"C:\Users\Hp\OneDrive\Desktop\Nouveau dossier\PythonProjects\ENSAM360-main\ENSAM360-main\backend\scripts\seed.js"
out_file = r"c:\Users\Hp\OneDrive\Desktop\Nouveau dossier\PythonProjects\projet CV\projet CV\ensam_navigation_app\scripts\seed_neo4j.py"

with open(js_file, "r", encoding="utf-8") as f:
    content = f.read()

# ──────────────────────────────────────────────
# 1. Extract NODES
# ──────────────────────────────────────────────
nodes = []
pattern = re.compile(
    r"""\{\s*id:\s*"(.*?)"\s*,\s*label:\s*\[.*?\]\s*,\s*name:\s*"(.*?)".*?"""
    r"""description:\s*"(.*?)".*?map_coords:\s*\[(.*?)\].*?"""
    r"""pano_url:\s*"(.*?)".*?flat_url:\s*"(.*?)".*?floor:\s*(\d+)""",
    re.DOTALL
)
for m in pattern.finditer(content):
    try:
        coords_raw = m.group(4).strip()
        coords = [int(x.strip()) for x in coords_raw.split(",")]
        nodes.append({
            "id": m.group(1),
            "name": m.group(2),
            "description": m.group(3),
            "coords": coords,
            "pano_url": m.group(5),
            "flat_url": m.group(6),
            "floor": int(m.group(7))
        })
    except Exception as e:
        print(f"Node parse error: {e}")

unique_nodes = {n["id"]: n for n in nodes}
nodes = list(unique_nodes.values())
print(f"Noeuds extraits: {len(nodes)}")

# ──────────────────────────────────────────────
# 2. Extract CONNECTIONS (including commented-out ones)
# ──────────────────────────────────────────────
# Strip all comment markers so we can search freely
# We remove // at the start of lines (within the multi-line comment blocks too)
content_stripped = re.sub(r"^\s*//\s*", "", content, flags=re.MULTILINE)

conn_pattern = re.compile(
    r"MATCH\s*\(a:Location\s*\{id:\s*'(.*?)'\}\)\s*"
    r"MATCH\s*\(b:Location\s*\{id:\s*'(.*?)'\}\)"
)
connections = []
seen = set()
for m in conn_pattern.finditer(content_stripped):
    src, dst = m.group(1), m.group(2)
    key = (src, dst)
    if key not in seen:
        seen.add(key)
        connections.append([src, dst])

print(f"Connexions extraites: {len(connections)}")
if connections:
    print("Exemples:", connections[:3])

# ──────────────────────────────────────────────
# 3. Generate seed_neo4j.py
# ──────────────────────────────────────────────
script = f'''import os
import math
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
USER     = os.getenv("NEO4J_USER",     "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "ensam360password")

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

NODES = {json.dumps(nodes, indent=4, ensure_ascii=False)}

CONNECTIONS = {json.dumps(connections, indent=4)}

def dist(c1, c2):
    return int(math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2))

node_map = {{n["id"]: n for n in NODES}}

def seed_db():
    with driver.session() as session:
        print("Nettoyage de la base de donnees...")
        session.run("MATCH (n) DETACH DELETE n")

        print(f"Creation de {{len(NODES)}} noeuds...")
        for node in NODES:
            session.run("""
                MERGE (n:Node {{id: $id}})
                SET n.name        = $name,
                    n.description = $description,
                    n.x_svg       = $x,
                    n.y_svg       = $y,
                    n.pano_url    = $pano_url,
                    n.flat_url    = $flat_url,
                    n.floor       = $floor
            """,
                id=node["id"], name=node["name"],
                description=node["description"],
                x=node["coords"][0], y=node["coords"][1],
                pano_url=node["pano_url"], flat_url=node["flat_url"],
                floor=node["floor"]
            )

        print(f"Creation de {{len(CONNECTIONS)*2}} connexions...")
        skipped = 0
        for src, dst in CONNECTIONS:
            sn = node_map.get(src)
            dn = node_map.get(dst)
            if not sn or not dn:
                print(f"  Warning: noeud manquant  {{src}} -> {{dst}}")
                skipped += 1
                continue
            d = dist(sn["coords"], dn["coords"])
            session.run(
                "MATCH (a:Node {{id: $src}}), (b:Node {{id: $dst}}) "
                "MERGE (a)-[r:CONNECTED_TO]->(b) SET r.distance = $d",
                src=src, dst=dst, d=d
            )
            session.run(
                "MATCH (a:Node {{id: $dst}}), (b:Node {{id: $src}}) "
                "MERGE (a)-[r:CONNECTED_TO]->(b) SET r.distance = $d",
                src=src, dst=dst, d=d
            )
        if skipped:
            print(f"  {{skipped}} connexions ignorees (noeud introuvable).")
        print("Base de donnees initialisee avec succes !")

if __name__ == "__main__":
    seed_db()
    driver.close()
    print("Connexion fermee.")
'''

with open(out_file, "w", encoding="utf-8") as f:
    f.write(script)

print(f"seed_neo4j.py genere avec {len(nodes)} noeuds et {len(connections)} connexions.")
