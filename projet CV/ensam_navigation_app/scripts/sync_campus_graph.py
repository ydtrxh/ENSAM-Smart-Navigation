import os
import json
import sys
import argparse
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Add project root to sys.path to allow importing app.config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config import settings
from dotenv import load_dotenv

# Enforce UTF-8 output just in case
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# Charger les variables d'environnement depuis le fichier .env
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(env_path)

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "ensam360password")

print("--------------------------------------------------")
print("SYNCHRONISATION CAMPUS_GRAPH.JSON -> NEO4J AURADB")
print("--------------------------------------------------")
print(f"Connexion a : {URI}")
print(f"Utilisateur : {USER}")

# Initialiser le pilote Neo4j
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

def sync_data():
    # 1. Charger les données du fichier campus_graph.json
    json_path = settings.get_path("campus_graph_json")
    if not json_path:
        json_path = os.path.join(os.path.dirname(__file__), "..", "data", "campus_graph.json")
    
    if not os.path.exists(json_path):
        print(f"[ERREUR] Le fichier {json_path} est introuvable.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        graph_data = json.load(f)

    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    print(f"[INFO] Fichier charge avec succes :")
    print(f"   - {len(nodes)} noeuds trouves.")
    print(f"   - {len(edges)} liaisons (aretes) trouvees.")

    with driver.session() as session:
        # 2. Nettoyage de la base de données existante (nœuds de type Node uniquement)
        print("\n[INFO] Nettoyage des anciens noeuds et relations de type 'Node'...")
        session.run("MATCH (n:Node) DETACH DELETE n")
        print("   -> Nettoyage termine.")

        # 3. Création des nœuds
        print(f"\n[INFO] Importation de {len(nodes)} noeuds dans Neo4j...")
        for node in nodes:
            nid = node["id"]
            name = node.get("name", nid)
            ntype = node.get("type", "Couloir/Intersection")
            node_role = node.get("node_role", node.get("role", ""))
            visible_destination = node.get("visible_destination", True)
            coords = node.get("coords", [0, 0])
            floor = node.get("floor", 0)
            description = node.get("description", f"Lieu du campus : {name} ({ntype})")

            # Insertion Cypher
            session.run("""
                MERGE (n:Node {id: $id})
                SET n.name        = $name,
                    n.type        = $type,
                    n.node_role   = $node_role,
                    n.visible_destination = $visible_destination,
                    n.x_svg       = $x,
                    n.y_svg       = $y,
                    n.floor       = $floor,
                    n.description = $description
            """,
                id=nid,
                name=name,
                type=ntype,
                node_role=node_role,
                visible_destination=visible_destination,
                x=coords[0],
                y=coords[1],
                floor=floor,
                description=description
            )
        print("   -> Tous les noeuds ont ete crees.")

        # 4. Création des relations (bidirectionnelles)
        print(f"\n[INFO] Importation de {len(edges)} relations bidirectionnelles...")
        relations_crees = 0
        for edge in edges:
            src = edge.get("from", edge.get("source"))
            dst = edge.get("to", edge.get("target"))
            if not src or not dst:
                print(f"   [WARN] Relation ignoree, endpoints manquants : {edge}")
                continue

            dist = edge.get("distance", edge.get("weight", edge.get("distance_meters", 1.0)))
            yaw = edge.get("yaw")
            pitch = edge.get("pitch")

            # Création bidirectionnelle (Aller)
            session.run("""
                MATCH (a:Node {id: $src}), (b:Node {id: $dst})
                MERGE (a)-[r:CONNECTED_TO]->(b)
                SET r.distance = $dist,
                    r.yaw = $yaw,
                    r.pitch = $pitch
            """, src=src, dst=dst, dist=dist, yaw=yaw, pitch=pitch)

            # Création bidirectionnelle (Retour)
            session.run("""
                MATCH (a:Node {id: $dst}), (b:Node {id: $src})
                MERGE (a)-[r:CONNECTED_TO]->(b)
                SET r.distance = $dist,
                    r.yaw = $yaw,
                    r.pitch = $pitch
            """, src=src, dst=dst, dist=dist, yaw=yaw, pitch=pitch)
            
            relations_crees += 1

        print(f"   -> {relations_crees} connexions bidirectionnelles importees (total {relations_crees * 2} relations).")

    print("\n[SUCCES] Synchronisation avec Neo4j AuraDB terminee avec succes !")

if __name__ == "__main__":
    try:
        sync_data()
    except Exception as e:
        print(f"\n[ERREUR] Une erreur est survenue lors de la synchronisation : {e}")
    finally:
        driver.close()
        print("[INFO] Connexion Neo4j fermee.")
