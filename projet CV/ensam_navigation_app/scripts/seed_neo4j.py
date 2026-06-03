import os
import math
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
USER     = os.getenv("NEO4J_USER",     "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "ensam360password")

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

NODES = [
    {
        "id": "amphi_parking",
        "name": "Amphi Parking",
        "description": "Intersection entre l'amphi et parking",
        "coords": [
            380,
            317
        ],
        "pano_url": "src/assets/image/amphi_parking.jpg",
        "flat_url": "src/assets/image/amphi_parking_flat.jpg",
        "floor": 0
    },
    {
        "id": "amphis_door",
        "name": "Porte Amphi",
        "description": "La porte d'entree de l'amphi",
        "coords": [
            350,
            317
        ],
        "pano_url": "src/assets/image/amphis_door.jpg",
        "flat_url": "src/assets/image/amphis_door_flat.jpg",
        "floor": 0
    },
    {
        "id": "AUF",
        "name": "Batiment AUF",
        "description": "Le batiment AUF des lagnues",
        "coords": [
            270,
            401
        ],
        "pano_url": "src/assets/image/AUF.jpg",
        "flat_url": "src/assets/image/AUF_flat.jpg",
        "floor": 0
    },
    {
        "id": "buvette",
        "name": "Buvette",
        "description": "La buvette des etudiants",
        "coords": [
            639,
            317
        ],
        "pano_url": "src/assets/image/buvette.jpg",
        "flat_url": "src/assets/image/buvette_flat.jpg",
        "floor": 0
    },
    {
        "id": "centre_de_recherche",
        "name": "Centre de Recherche",
        "description": "Centre de recherche avec salles de dessin",
        "coords": [
            440,
            400
        ],
        "pano_url": "src/assets/image/centre_de_recherche.jpg",
        "flat_url": "src/assets/image/centre_de_recherche_flat.jpg",
        "floor": 0
    },
    {
        "id": "centre_de_recherche_door",
        "name": "La porte de Centre de Recherche",
        "description": "La porte du Centre de recherche",
        "coords": [
            441,
            350
        ],
        "pano_url": "src/assets/image/centre_de_recherche_door.jpg",
        "flat_url": "src/assets/image/centre_de_recherche_door_flat.jpg",
        "floor": 0
    },
    {
        "id": "centre_de_recherche_entry",
        "name": "Entree de Centre de Recherche",
        "description": "L'entree de Centre de recherche",
        "coords": [
            441,
            316
        ],
        "pano_url": "src/assets/image/centre_de_recherche_entry.jpg",
        "flat_url": "src/assets/image/centre_de_recherche_entry_flat.jpg",
        "floor": 0
    },
    {
        "id": "civil",
        "name": "Departement civil",
        "description": "Le department genie civil",
        "coords": [
            542,
            400
        ],
        "pano_url": "src/assets/image/civil.jpg",
        "flat_url": "src/assets/image/civil_flat.jpg",
        "floor": 0
    },
    {
        "id": "civil_entry",
        "name": "Entree civil",
        "description": "Entree du Departement Genie Civil",
        "coords": [
            542,
            350
        ],
        "pano_url": "src/assets/image/civil_entry.jpg",
        "flat_url": "src/assets/image/civil_entry_flat.jpg",
        "floor": 0
    },
    {
        "id": "civil_gms_entry",
        "name": "Intersection GMS Civil",
        "description": "Intersection entre le Atelier Tp GMS et le Dept Civil",
        "coords": [
            542,
            316
        ],
        "pano_url": "src/assets/image/civil_gms_entry.jpg",
        "flat_url": "src/assets/image/civil_gms_entry_flat.jpg",
        "floor": 0
    },
    {
        "id": "couloir_intersection",
        "name": "Couloir Intersection",
        "description": "Couloir de l'intersection entre chemin vers Buvette et chemin vers Bibliothèque",
        "coords": [
            379,
            399
        ],
        "pano_url": "src/assets/image/couloir_intersection.jpg",
        "flat_url": "src/assets/image/couloir_intersection_flat.jpg",
        "floor": 0
    },
    {
        "id": "couloir_right_td1",
        "name": "Couloir droit de TD1",
        "description": "Couloir droit du Td1",
        "coords": [
            320,
            482
        ],
        "pano_url": "src/assets/image/couloir_right_td1.jpg",
        "flat_url": "src/assets/image/couloir_right_td1_flat.jpg",
        "floor": 1
    },
    {
        "id": "energitique_door",
        "name": "Porte Energitique",
        "description": "La prote du Departement Energitique",
        "coords": [
            670,
            405
        ],
        "pano_url": "src/assets/image/energitique_door.jpg",
        "flat_url": "src/assets/image/energitique_door_flat.jpg",
        "floor": 0
    },
    {
        "id": "energitique_entry",
        "name": "Entree Energitique",
        "description": "Entree du Departement Energitique",
        "coords": [
            670,
            317
        ],
        "pano_url": "src/assets/image/energitique_entry.jpg",
        "flat_url": "src/assets/image/energitique_entry_flat.jpg",
        "floor": 0
    },
    {
        "id": "ensam_entry",
        "name": "Entree Ensam",
        "description": "Entree de L'Ecole Nationale Superieure d'Arts et Metiers de Meknes",
        "coords": [
            82,
            330
        ],
        "pano_url": "src/assets/image/ensam_entry.jpg",
        "flat_url": "src/assets/image/ensam_entry_flat.jpg",
        "floor": 0
    },
    {
        "id": "fabrication_mecanique_entry",
        "name": "Entree Tp Fabrication mecanique",
        "description": "Entree du TP en fabrication mecanique",
        "coords": [
            493,
            316
        ],
        "pano_url": "src/assets/image/fabrication_mecanique_entry.jpg",
        "flat_url": "src/assets/image/fabrication_mecanique_entry_flat.jpg",
        "floor": 0
    },
    {
        "id": "parking",
        "name": "Parking",
        "description": "Parking de l'ENSAM",
        "coords": [
            378,
            240
        ],
        "pano_url": "src/assets/image/parking.jpg",
        "flat_url": "src/assets/image/parking_flat.jpg",
        "floor": 0
    },
    {
        "id": "pointeuse_entry",
        "name": "Espace de pointeuses",
        "description": "Espace de pointeuses a cote du TD2",
        "coords": [
            187,
            401
        ],
        "pano_url": "src/assets/image/pointeuse_entry.jpg",
        "flat_url": "src/assets/image/pointeuse_entry_flat.jpg",
        "floor": 0
    },
    {
        "id": "salle_info_7",
        "name": "Salle 7 Info",
        "description": "Salle 7 de la filiere Genie Informatique,Ingenierie Logicielle et Systemes Intelligents",
        "coords": [
            324,
            435
        ],
        "pano_url": "src/assets/image/salle_info_7.jpg",
        "flat_url": "src/assets/image/salle_info_7_flat.jpg",
        "floor": 1
    },
    {
        "id": "stairs_right_td1",
        "name": "Escaliers Droit de TD1",
        "description": "Escaliers droit du TD1",
        "coords": [
            348,
            482
        ],
        "pano_url": "src/assets/image/stairs_right_td1.jpg",
        "flat_url": "src/assets/image/stairs_right_td1_flat.jpg",
        "floor": 0
    },
    {
        "id": "td1",
        "name": "TD1",
        "description": "le Td TD1",
        "coords": [
            320,
            452
        ],
        "pano_url": "src/assets/image/td1.jpg",
        "flat_url": "src/assets/image/td1_flat.jpg",
        "floor": 0
    },
    {
        "id": "td1_door",
        "name": "Porte de TD1",
        "description": "La porte du TD1",
        "coords": [
            348,
            452
        ],
        "pano_url": "src/assets/image/td1_door.jpg",
        "flat_url": "src/assets/image/td1_door_flat.jpg",
        "floor": 0
    },
    {
        "id": "td1_entry",
        "name": "Entrée de TD1",
        "description": "Entrée du Td TD1",
        "coords": [
            379,
            452
        ],
        "pano_url": "src/assets/image/td1_entry.jpg",
        "flat_url": "src/assets/image/td1_entry_flat.jpg",
        "floor": 0
    },
    {
        "id": "td2",
        "name": "TD2",
        "description": "Le td TD2",
        "coords": [
            215,
            470
        ],
        "pano_url": "src/assets/image/td2.jpg",
        "flat_url": "src/assets/image/td2_flat.jpg",
        "floor": 0
    },
    {
        "id": "td2_entry",
        "name": "Entree TD2",
        "description": "Entree du TD2",
        "coords": [
            211,
            401
        ],
        "pano_url": "src/assets/image/td2_entry.jpg",
        "flat_url": "src/assets/image/td2_entry_flat.jpg",
        "floor": 0
    },
    {
        "id": "terrain_fonderie",
        "name": "Intersection Terrain Fonderie",
        "description": "Intersection entre le chemin menant au Terrain de Foot et La buvette des Profs avec le chemin menant au TP (labaux) de Fonderie",
        "coords": [
            720,
            317
        ],
        "pano_url": "src/assets/image/terrain_fonderie.jpg",
        "flat_url": "src/assets/image/terrain_fonderie_flat.jpg",
        "floor": 0
    },
    {
        "id": "administration",
        "name": "Administration",
        "description": "Bâtiment de l'administration.",
        "coords": [
            80,
            540
        ],
        "pano_url": "src/assets/image/administration.jpg",
        "flat_url": "src/assets/image/administration_flat.jpg",
        "floor": 0
    },
    {
        "id": "Admnetud",
        "name": "Route vers l'Administration des Étudiants et la Bibliothèque",
        "description": "Jonction vers l'administration des étudiants et la bibliothèque.",
        "coords": [
            95,
            610
        ],
        "pano_url": "src/assets/image/routebib.jpg",
        "flat_url": "src/assets/image/routebib_flat.jpg",
        "floor": 0
    },
    {
        "id": "Administration_etud",
        "name": "Administration des Étudiants",
        "description": "Bureau de l'administration des étudiants.",
        "coords": [
            180,
            623
        ],
        "pano_url": "src/assets/image/adminetud.jpg",
        "flat_url": "src/assets/image/adminetud_flat.jpg",
        "floor": 0
    },
    {
        "id": "couloirForum",
        "name": "Couloir Administration des etudiants",
        "description": "Couloir menant au forum.",
        "coords": [
            180,
            647
        ],
        "pano_url": "src/assets/image/couloirforum.jpg",
        "flat_url": "src/assets/image/couloirforum_flat.jpg",
        "floor": 0
    },
    {
        "id": "couloir1",
        "name": "Couloir Bibliothèque",
        "description": "Premier couloir principal.",
        "coords": [
            215,
            647
        ],
        "pano_url": "src/assets/image/image5.jpg",
        "flat_url": "src/assets/image/image5_flat.jpg",
        "floor": 0
    },
    {
        "id": "Bibliotheque_et_centre_de_langue",
        "name": "Bibliothèque et Centre de Langue",
        "description": "Bibliothèque principale et centre de langue.",
        "coords": [
            220,
            730
        ],
        "pano_url": "src/assets/image/bib.jpg",
        "flat_url": "src/assets/image/bib_flat.jpg",
        "floor": 0
    },
    {
        "id": "escalier1",
        "name": "Escalier Bibliothèque",
        "description": "Premier escalier d'accès a la bibliotheque.",
        "coords": [
            200,
            700
        ],
        "pano_url": "src/assets/image/escalier1.jpg",
        "flat_url": "src/assets/image/escalier1_flat.jpg",
        "floor": 0
    },
    {
        "id": "escalier2",
        "name": "Entrée Bibliothèque",
        "description": "Deuxième escalier d'accès.",
        "coords": [
            200,
            715
        ],
        "pano_url": "src/assets/image/image2.jpg",
        "flat_url": "src/assets/image/image2_flat.jpg",
        "floor": 1
    },
    {
        "id": "couloir2",
        "name": "Deuxième couloir principal",
        "description": "Deuxième couloir principal.",
        "coords": [
            377,
            647
        ],
        "pano_url": "src/assets/image/image6.jpg",
        "flat_url": "src/assets/image/image6_flat.jpg",
        "floor": 1
    },
    {
        "id": "couloircctd1",
        "name": "Couloir Centre de Calcul",
        "description": "Couloir du centre de calcul.",
        "coords": [
            377,
            610
        ],
        "pano_url": "src/assets/image/couloircctd1.jpg",
        "flat_url": "src/assets/image/couloircctd1_flat.jpg",
        "floor": 1
    },
    {
        "id": "CouloirAEEE",
        "name": "Couloir de l'Entrée de AEEE",
        "description": "Couloir d'entrée du département AEEE.",
        "coords": [
            379,
            476
        ],
        "pano_url": "src/assets/image/couloira3e.jpg",
        "flat_url": "src/assets/image/couloira3e_flat.jpg",
        "floor": 1
    },
    {
        "id": "EntreeA3e",
        "name": "Entrée de AEEE",
        "description": "Entrée du département AEEE.",
        "coords": [
            415,
            476
        ],
        "pano_url": "src/assets/image/EntreeA3E.jpg",
        "flat_url": "src/assets/image/EntreeA3E_flat.jpg",
        "floor": 1
    },
    {
        "id": "AEEE",
        "name": "Département AEEE",
        "description": "Département d'Automatique, Électronique, Énergie et Environnement.",
        "coords": [
            415,
            487
        ],
        "pano_url": "src/assets/image/a3e.jpg",
        "flat_url": "src/assets/image/a3e_flat.jpg",
        "floor": 1
    },
    {
        "id": "a3einside",
        "name": "Département AEEE (Intérieur)",
        "description": "Intérieur du département AEEE.",
        "coords": [
            460,
            520
        ],
        "pano_url": "src/assets/image/a3e2.jpg",
        "flat_url": "src/assets/image/a3e2_flat.jpg",
        "floor": 1
    },
    {
        "id": "mathinfo",
        "name": "Entrée Département Mathématiques-Informatique",
        "description": "Département de Mathématiques et Informatique.",
        "coords": [
            420,
            610
        ],
        "pano_url": "src/assets/image/cc_outside.jpg",
        "flat_url": "src/assets/image/cc_outside_flat.jpg",
        "floor": 1
    },
    {
        "id": "mathinfo_inside",
        "name": "Département Mathématiques-Informatique (Intérieur)",
        "description": "Intérieur du département Mathématiques-Informatique.",
        "coords": [
            475,
            653
        ],
        "pano_url": "src/assets/image/mathinfo.jpg",
        "flat_url": "src/assets/image/mathinfo_flat.jpg",
        "floor": 1
    },
    {
        "id": "amphie_et_salle_de_conference",
        "name": "Amphithéâtre 3 et Salle de Conférence",
        "description": "Jonction entre l'amphithéâtre 3 et la salle de conférence.",
        "coords": [
            377,
            755
        ],
        "pano_url": "src/assets/image/image7.jpg",
        "flat_url": "src/assets/image/image7_flat.jpg",
        "floor": 1
    },
    {
        "id": "entree_salle_conference",
        "name": "Entrée de la Salle de Conférence",
        "description": "Entrée de la salle de conférence.",
        "coords": [
            355,
            755
        ],
        "pano_url": "src/assets/image/image8.jpg",
        "flat_url": "src/assets/image/image8_flat.jpg",
        "floor": 1
    },
    {
        "id": "salle_conference",
        "name": "Salle de Conférence",
        "description": "Salle de conférence principale.",
        "coords": [
            355,
            800
        ],
        "pano_url": "src/assets/image/image9.jpg",
        "flat_url": "src/assets/image/image9_flat.jpg",
        "floor": 1
    },
    {
        "id": "entree_emphi3",
        "name": "Entrée de l'Amphithéâtre 3",
        "description": "Entrée de l'amphithéâtre 3.",
        "coords": [
            410,
            755
        ],
        "pano_url": "src/assets/image/amphi3.jpg",
        "flat_url": "src/assets/image/amphi3_flat.jpg",
        "floor": 1
    },
    {
        "id": "Amphi3",
        "name": "Amphithéâtre 3",
        "description": "Amphithéâtre 3 pour les conférences et cours.",
        "coords": [
            300,
            300
        ],
        "pano_url": "src/assets/image/image11.jpg",
        "flat_url": "src/assets/image/image11_flat.jpg",
        "floor": 1
    }
]

CONNECTIONS = [
    [
        "ensam_entry",
        "administration"
    ],
    [
        "administration",
        "Admnetud"
    ],
    [
        "Admnetud",
        "Administration_etud"
    ],
    [
        "Administration_etud",
        "couloirForum"
    ],
    [
        "couloirForum",
        "couloir1"
    ],
    [
        "couloir1",
        "Bibliotheque_et_centre_de_langue"
    ],
    [
        "Bibliotheque_et_centre_de_langue",
        "escalier1"
    ],
    [
        "escalier1",
        "escalier2"
    ],
    [
        "couloir1",
        "couloir2"
    ],
    [
        "couloir2",
        "couloircctd1"
    ],
    [
        "couloircctd1",
        "mathinfo"
    ],
    [
        "mathinfo",
        "mathinfo_inside"
    ],
    [
        "couloircctd1",
        "CouloirAEEE"
    ],
    [
        "CouloirAEEE",
        "EntreeA3e"
    ],
    [
        "EntreeA3e",
        "AEEE"
    ],
    [
        "AEEE",
        "a3einside"
    ],
    [
        "CouloirAEEE",
        "td1_entry"
    ],
    [
        "couloirtd1",
        "CouloirTD1TD2"
    ],
    [
        "couloir2",
        "amphie_et_salle_de_conference"
    ],
    [
        "amphie_et_salle_de_conference",
        "entree_salle_conference"
    ],
    [
        "entree_salle_conference",
        "salle_conference"
    ],
    [
        "amphie_et_salle_de_conference",
        "entree_emphi3"
    ],
    [
        "entree_emphi3",
        "Amphi3"
    ],
    [
        "ensam_entry",
        "pointeuse_entry"
    ],
    [
        "pointeuse_entry",
        "td2_entry"
    ],
    [
        "td2_entry",
        "td2"
    ],
    [
        "td2_entry",
        "AUF"
    ],
    [
        "AUF",
        "couloir_intersection"
    ],
    [
        "couloir_intersection",
        "amphi_parking"
    ],
    [
        "couloir_intersection",
        "td1_entry"
    ],
    [
        "td1_entry",
        "td1_door"
    ],
    [
        "td1_door",
        "td1"
    ],
    [
        "td1_door",
        "stairs_right_td1"
    ],
    [
        "td1",
        "stairs_right_td1"
    ],
    [
        "stairs_right_td1",
        "couloir_right_td1"
    ],
    [
        "couloir_right_td1",
        "salle_info_7"
    ],
    [
        "amphi_parking",
        "amphis_door"
    ],
    [
        "amphi_parking",
        "parking"
    ],
    [
        "amphi_parking",
        "centre_de_recherche_entry"
    ],
    [
        "centre_de_recherche_entry",
        "centre_de_recherche_door"
    ],
    [
        "centre_de_recherche_entry",
        "fabrication_mecanique_entry"
    ],
    [
        "centre_de_recherche_door",
        "centre_de_recherche"
    ],
    [
        "fabrication_mecanique_entry",
        "civil_gms_entry"
    ],
    [
        "civil_gms_entry",
        "civil_entry"
    ],
    [
        "civil_gms_entry",
        "buvette"
    ],
    [
        "civil_entry",
        "civil"
    ],
    [
        "buvette",
        "energitique_entry"
    ],
    [
        "energitique_entry",
        "energitique_door"
    ],
    [
        "energitique_entry",
        "terrain_fonderie"
    ],
    [
        "mathinfo_inside",
        "labo_indus"
    ]
]

def dist(c1, c2):
    return int(math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2))

node_map = {n["id"]: n for n in NODES}

def seed_db():
    with driver.session() as session:
        print("Nettoyage de la base de donnees...")
        session.run("MATCH (n) DETACH DELETE n")

        print(f"Creation de {len(NODES)} noeuds...")
        for node in NODES:
            session.run("""
                MERGE (n:Node {id: $id})
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

        print(f"Creation de {len(CONNECTIONS)*2} connexions...")
        skipped = 0
        for src, dst in CONNECTIONS:
            sn = node_map.get(src)
            dn = node_map.get(dst)
            if not sn or not dn:
                print(f"  Warning: noeud manquant  {src} -> {dst}")
                skipped += 1
                continue
            d = dist(sn["coords"], dn["coords"])
            session.run(
                "MATCH (a:Node {id: $src}), (b:Node {id: $dst}) "
                "MERGE (a)-[r:CONNECTED_TO]->(b) SET r.distance = $d",
                src=src, dst=dst, d=d
            )
            session.run(
                "MATCH (a:Node {id: $dst}), (b:Node {id: $src}) "
                "MERGE (a)-[r:CONNECTED_TO]->(b) SET r.distance = $d",
                src=src, dst=dst, d=d
            )
        if skipped:
            print(f"  {skipped} connexions ignorees (noeud introuvable).")
        print("Base de donnees initialisee avec succes !")

if __name__ == "__main__":
    seed_db()
    driver.close()
    print("Connexion fermee.")
