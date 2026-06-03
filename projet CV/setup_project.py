# setup_project.py
import os

# Définition de l'arborescence et du contenu des fichiers
project_structure = {
    "ensam_navigation_app/requirements.txt": """streamlit>=1.30.0
streamlit-folium>=0.15.0
folium>=0.15.0
networkx>=3.0
tensorflow>=2.15.0
numpy>=1.24.0
langchain>=0.1.0
langchain-groq>=0.1.0
python-dotenv>=1.0.0
pillow>=10.0.0
""",

    "ensam_navigation_app/.env.example": """# Copie ce fichier et renomme le en ".env" à la racine du projet
GROQ_API_KEY=votre_cle_api_groq_ici
""",

    "ensam_navigation_app/README.md": """# Application de Navigation Indoor ENSAM

## Installation
1. Installez les dépendances : `pip install -r requirements.txt`
2. Lancez l'application : `streamlit run app/main.py`

## Utilisation
- Utilisez l'onglet **ADMIN** pour insérer vos premiers points ou importer un fichier JSON.
- Utilisez l'onglet **UTILISATEUR** pour tester la localisation par photo, le chatbot et l'itinéraire visuel.
""",

    "ensam_navigation_app/app/cv_module/__init__.py": "",
    "ensam_navigation_app/app/nlp_module/__init__.py": "",
    "ensam_navigation_app/app/navigation/__init__.py": "",

    "ensam_navigation_app/app/data_manager.py": """import json
import os
import shutil

class DataManager:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.img_dir = os.path.join(data_dir, "departments")
        os.makedirs(self.img_dir, exist_ok=True)
        
        self.db = {
            "departments": {}, 
            "nodes": {},       
            "edges": []        
        }
        
    def add_department(self, name, uploaded_files):
        dept_dir = os.path.join(self.img_dir, name)
        os.makedirs(dept_dir, exist_ok=True)
        
        saved_paths = []
        for file in uploaded_files:
            file_path = os.path.join(dept_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            saved_paths.append(file_path)
            
        self.db["departments"][name] = list(set(self.db["departments"].get(name, []) + saved_paths))
        return True

    def remove_department(self, name):
        if name in self.db["departments"]:
            del self.db["departments"][name]
            dept_dir = os.path.join(self.img_dir, name)
            if os.path.exists(dept_dir):
                shutil.rmtree(dept_dir)
            return True
        return False

    def add_node(self, node_id, name, lat, lon, node_type):
        self.db["nodes"][str(node_id)] = {
            "name": name,
            "lat": float(lat),
            "lon": float(lon),
            "type": node_type
        }

    def remove_node(self, node_id):
        node_id = str(node_id)
        if node_id in self.db["nodes"]:
            del self.db["nodes"][node_id]
            self.db["edges"] = [e for e in self.db["edges"] if e["from"] != node_id and e["to"] != node_id]
            return True
        return False

    def add_edge(self, from_node, to_node, distance):
        from_node, to_node = str(from_node), str(to_node)
        for edge in self.db["edges"]:
            if (edge["from"] == from_node and edge["to"] == to_node) or (edge["from"] == to_node and edge["to"] == from_node):
                return False
        self.db["edges"].append({"from": from_node, "to": to_node, "distance": float(distance)})
        return True

    def remove_edge(self, from_node, to_node):
        from_node, to_node = str(from_node), str(to_node)
        initial_len = len(self.db["edges"])
        self.db["edges"] = [
            e for e in self.db["edges"] 
            if not ((e["from"] == from_node and e["to"] == to_node) or (e["from"] == to_node and e["to"] == from_node))
        ]
        return len(self.db["edges"]) < initial_len

    def get_all_departments(self): return self.db["departments"]
    def get_all_nodes(self): return self.db["nodes"]
    def get_all_edges(self): return self.db["edges"]

    def export_to_json(self, filename="navigation_db.json"):
        path = os.path.join(self.data_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.db, f, indent=4, ensure_ascii=False)
        return path

    def import_from_json(self, filename="navigation_db.json"):
        path = os.path.join(self.data_dir, filename)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self.db = json.load(f)
            return True
        return False
""",

    "ensam_navigation_app/app/cv_module/train_model.py": """import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import json

def train_cv_model(data_dir="data"):
    img_dir = os.path.join(data_dir, "departments")
    model_path = os.path.join(data_dir, "model.h5")
    classes_path = os.path.join(data_dir, "classes.json")
    
    if not os.path.exists(img_dir):
        return False, "Aucune donnée d'image trouvée."
        
    classes = [d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))]
    if len(classes) < 2:
        return False, "Il faut au moins 2 départements avec des photos pour entraîner le modèle."

    with open(classes_path, "w") as f:
        json.dump(classes, f)

    try:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            img_dir, image_size=(224, 224), batch_size=4, validation_split=0.2, subset="training", seed=123
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            img_dir, image_size=(224, 224), batch_size=4, validation_split=0.2, subset="validation", seed=123
        )
    except Exception as e:
        return False, f"Erreur lors du chargement des images : {str(e)}"

    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(classes), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_ds, validation_data=val_ds, epochs=3)
    model.save(model_path)
    
    return True, f"Modèle entraîné avec succès sur {len(classes)} classes !"
""",

    "ensam_navigation_app/app/cv_module/predict.py": """import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image

def predict_department(image_file, data_dir="data", threshold=0.6):
    model_path = os.path.join(data_dir, "model.h5")
    classes_path = os.path.join(data_dir, "classes.json")
    
    if not os.path.exists(model_path) or not os.path.exists(classes_path):
        return "Inconnu (Modèle non entraîné)", 0.0

    with open(classes_path, "r") as f:
        classes = json.load(f)

    model = tf.keras.models.load_model(model_path)
    
    img = Image.open(image_file).convert('RGB').resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]
    max_idx = np.argmax(predictions)
    confidence = float(predictions[max_idx])

    if confidence >= threshold:
        return classes[max_idx], confidence
    return "Inconnu (Confiance trop faible)", confidence
""",

    "ensam_navigation_app/app/nlp_module/chatbot.py": """import os
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

class CampusChatbot:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY", "")
        self.llm = None
        if self.api_key:
            try:
                self.llm = ChatGroq(groq_api_key=self.api_key, model_name="llama-3.3-70b-specdec")
            except:
                self.llm = None

    def ask(self, user_question, current_location, history, available_destinations):
        dest_list_str = ", ".join(available_destinations)
        
        if not self.llm:
            q = user_question.lower()
            matched = [d for d in available_destinations if d.lower() in q]
            if matched:
                return f"D'accord, je vois que tu veux aller à : **{matched[0]}**. J'ai configuré ton itinéraire sur la carte ci-dessous.", matched[0]
            return "Je suis en mode hors-ligne. Pour trouver ton chemin, mentionne explicitement le nom d'un bâtiment ou d'une salle valide dans ton message.", None

        system_prompt = f\"\"\"Tu es le guide d'orientation indoor de l'ENSAM.
L'utilisateur est actuellement localisé à : {current_location}.
Les destinations valides et configurées sur le campus sont STRICTEMENT limitées à cette liste : [{dest_list_str}].

Instructions :
1. Aide poliment l'utilisateur.
2. Si sa question implique qu'il souhaite se rendre à un endroit de la liste, termine obligatoirement ta réponse par le tag exact sous la forme [DESTINATION:Nom_Exact].
3. Si la destination demandée n'est pas dans la liste, explique gentiment que ce lieu n'est pas cartographié.
\"\"\"
        messages = [SystemMessage(content=system_prompt)]
        for h in history:
            messages.append(HumanMessage(content=h.get("user", "")))
        
        messages.append(HumanMessage(content=user_question))
        
        try:
            response = self.llm.invoke(messages).content
            target_destination = None
            if "[DESTINATION:" in response:
                parts = response.split("[DESTINATION:")
                target_destination = parts[1].split("]")[0].strip()
                response = parts[0] + " (Itinéraire en cours de traçage...)"
                
            return response, target_destination
        except Exception as e:
            return f"Erreur de communication avec l'IA. Mode autonome activé.", None
""",

    "ensam_navigation_app/app/navigation/graph_manager.py": """import networkx as nx

class NavigationEngine:
    def __init__(self):
        self.G = nx.Graph()

    def build_graph(self, nodes, edges):
        self.G.clear()
        for node_id, data in nodes.items():
            self.G.add_node(str(node_id), **data)
        for edge in edges:
            self.G.add_edge(str(edge["from"]), str(edge["to"]), weight=float(edge["distance"]))

    def get_shortest_path(self, start_node, end_node):
        start_node, end_node = str(start_node), str(end_node)
        if not self.G.has_node(start_node) or not self.G.has_node(end_node):
            return None, 0, ["Point de départ ou d'arrivée manquant dans le graphe."]
            
        try:
            path = nx.shortest_path(self.G, source=start_node, target=end_node, weight='weight')
            distance = nx.shortest_path_length(self.G, source=start_node, target=end_node, weight='weight')
            
            instructions = []
            for i in range(len(path) - 1):
                curr_name = self.G.nodes[path[i]]['name']
                next_name = self.G.nodes[path[i+1]]['name']
                dist = self.G.edges[path[i], path[i+1]]['weight']
                instructions.append(f"Marcher de **{curr_name}** vers **{next_name}** sur {dist} mètres.")
                
            return path, distance, instructions
        except nx.NetworkXNoPath:
            return None, 0, ["Aucun chemin continu n'existe entre ces deux points. Veuillez connecter le graphe côté Admin."]
""",

    "ensam_navigation_app/app/navigation/map_manager.py": """import folium

def generate_map(nodes, edges, current_path=None):
    if nodes:
        lats = [n['lat'] for n in nodes.values()]
        lons = [n['lon'] for n in nodes.values()]
        center = [sum(lats)/len(lats), sum(lons)/len(lons)]
    else:
        center = [48.8316, 2.3650] 
        
    m = folium.Map(location=center, zoom_start=19, max_zoom=22, tiles="OpenStreetMap")

    for edge in edges:
        f_node = nodes.get(edge["from"])
        t_node = nodes.get(edge["to"])
        if f_node and t_node:
            folium.PolyLine(
                locations=[[f_node['lat'], f_node['lon']], [t_node['lat'], t_node['lon']]],
                color="gray", weight=2, opacity=0.5
            ).add_to(m)

    for node_id, data in nodes.items():
        color = "blue"
        if data['type'] == "Entrée": color = "green"
        elif data['type'] == "Amphi/Salle": color = "purple"
        
        if current_path and str(node_id) in current_path:
            if str(node_id) == current_path[0]: color = "orange" 
            elif str(node_id) == current_path[-1]: color = "red" 
        
        folium.Marker(
            location=[data['lat'], data['lon']],
            popup=f"<b>{data['name']}</b> ({data['type']})<br>ID: {node_id}",
            icon=folium.Icon(color=color, icon="info-sign")
        ).add_to(m)

    if current_path and len(current_path) > 1:
        path_coords = []
        for n_id in current_path:
            node = nodes.get(n_id)
            if node:
                path_coords.append([node['lat'], node['lon']])
        
        folium.PolyLine(locations=path_coords, color="red", weight=6, opacity=0.9).add_to(m)
        
    return m
""",

    "ensam_navigation_app/app/main.py": """import streamlit as st
from streamlit_folium import st_folium
import os
from dotenv import load_dotenv

load_dotenv()

from data_manager import DataManager
from cv_module.train_model import train_cv_model
from cv_module.predict import predict_department
from nlp_module.chatbot import CampusChatbot
from navigation.graph_manager import NavigationEngine
from navigation.map_manager import generate_map

st.set_page_config(page_title="ENSAM Indoor Nav", layout="wide")
st.title("🏫 Système de Navigation Indoor - ENSAM")

if "dm" not in st.session_state:
    st.session_state.dm = DataManager()
    st.session_state.dm.import_from_json() 

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_location" not in st.session_state:
    st.session_state.current_location = None

if "target_node_id" not in st.session_state:
    st.session_state.target_node_id = None

dm = st.session_state.dm
nodes = dm.get_all_nodes()
edges = dm.get_all_edges()
depts = dm.get_all_departments()

nav_engine = NavigationEngine()
nav_engine.build_graph(nodes, edges)

onglets = st.tabs(["⚙️ ADMINISTRATION & SETUP", "📱 INTERFACE UTILISATEUR"])

with onglets[0]:
    st.header("Gestion de la Base de Données du Campus")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Ajout des zones (Départements)")
        with st.form("dept_form", clear_on_submit=True):
            dept_name = st.text_input("Nom du Département")
            uploaded_imgs = st.file_uploader("Photos pour entraînement CV", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])
            if st.form_submit_button("Enregistrer le département"):
                if dept_name and uploaded_imgs:
                    dm.add_department(dept_name, uploaded_imgs)
                    st.success(f"Département {dept_name} enregistré.")
                    st.rerun()

        st.subheader("2. Cartographie (Noeuds / Points clés)")
        with st.form("node_form", clear_on_submit=True):
            n_id = st.text_input("ID Unique (ex: 'salle_102')")
            n_name = st.text_input("Nom d'affichage")
            n_lat = st.number_input("Latitude", format="%.6f", value=48.8316)
            n_lon = st.number_input("Longitude", format="%.6f", value=2.3650)
            n_type = st.selectbox("Type de point", ["Couloir/Intersection", "Amphi/Salle", "Entrée", "Bureau"])
            if st.form_submit_button("Ajouter le point"):
                if n_id and n_name:
                    dm.add_node(n_id, n_name, n_lat, n_lon, n_type)
                    st.success(f"Point {n_id} ajouté.")
                    st.rerun()

    with col2:
        st.subheader("3. Maillage (Edges / Chemins)")
        with st.form("edge_form", clear_on_submit=True):
            all_node_ids = list(nodes.keys())
            f_node = st.selectbox("Du Point (A)", all_node_ids if all_node_ids else ["Aucun point"])
            t_node = st.selectbox("Au Point (B)", all_node_ids if all_node_ids else ["Aucun point"])
            dist = st.number_input("Distance en mètres", min_value=1.0, value=5.0)
            if st.form_submit_button("Créer le chemin"):
                if f_node != "Aucun point" and t_node != "Aucun point" and f_node != t_node:
                    if dm.add_edge(f_node, t_node, dist):
                        st.success("Liaison établie.")
                        st.rerun()

        st.subheader("🧠 Intelligence Artificielle (Vision)")
        if st.button("🔄 Lancer le Ré-entraînement du modèle Vision"):
            with st.spinner("Entraînement de MobileNetV2..."):
                success, msg = train_cv_model()
                if success: st.success(msg)
                else: st.error(msg)

    st.divider()
    b_col1, b_col2 = st.columns(2)
    if b_col1.button("💾 Exporter la base en JSON"):
        path = dm.export_to_json()
        st.success(f"Sauvegardé : {path}")
    if b_col2.button("📥 Importer depuis le JSON"):
        if dm.import_from_json(): st.rerun()

with onglets[1]:
    st.header("📱 Navigation Indoor en Temps Réel")
    if not nodes:
        st.warning("Le graphe est vide. Ajoutez des données dans l'onglet ADMIN.")
    else:
        u_col1, u_col2 = st.columns([1, 2])
        with u_col1:
            uploaded_photo = st.file_uploader("Prends une photo pour te localiser", type=['jpg','jpeg','png'])
            if uploaded_photo:
                detected_dept, conf = predict_department(uploaded_photo)
                st.session_state.current_location = detected_dept
                st.info(f"📍 Emplacement : **{detected_dept}** ({int(conf*100)}%)")
            else:
                st.session_state.current_location = st.selectbox("Ma position", list(nodes.keys()), format_func=lambda x: nodes[x]['name'])
            
            st.divider()
            for msg in st.session_state.chat_history:
                with st.chat_message("user"): st.write(msg["user"])
                with st.chat_message("assistant"): st.write(msg["assistant"])
            
            if user_query := st.chat_input("Où veux-tu aller ?"):
                destinations_labels = {data['name']: n_id for n_id, data in nodes.items()}
                chatbot = CampusChatbot()
                reply, target_name = chatbot.ask(user_query, st.session_state.current_location, st.session_state.chat_history, list(destinations_labels.keys()))
                st.session_state.chat_history.append({"user": user_query, "assistant": reply})
                if target_name and target_name in destinations_labels:
                    st.session_state.target_node_id = destinations_labels[target_name]
                st.rerun()
                
        with u_col2:
            active_path = None
            if st.session_state.target_node_id:
                start_node = st.session_state.current_location
                if start_node not in nodes:
                    possible_starts = [nid for nid, ndat in nodes.items() if ndat['name'] == start_node or start_node in ndat['name']]
                    start_node = possible_starts[0] if possible_starts else list(nodes.keys())[0]

                active_path, total_dist, steps = nav_engine.get_shortest_path(start_node, st.session_state.target_node_id)
                if active_path:
                    st.success(f"📏 Distance totale : **{total_dist}m**")
                    for s in steps: st.write(s)
            
            map_obj = generate_map(nodes, edges, current_path=active_path)
            map_data = st_folium(map_obj, width="100%", height=500)
            if map_data and map_data.get("last_object_clicked"):
                clicked_lat = map_data["last_object_clicked"]["lat"]
                clicked_lon = map_data["last_object_clicked"]["lng"]
                for nid, ndat in nodes.items():
                    if abs(ndat['lat'] - clicked_lat) < 1e-5 and abs(ndat['lon'] - clicked_lon) < 1e-5:
                        st.session_state.target_node_id = nid
                        st.rerun()
"""
}

# Création des répertoires et fichiers
print("🚀 Génération de l'application de navigation indoor ENSAM...")
for filepath, content in project_structure.items():
    dir_name = os.path.dirname(filepath)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Créé : {filepath}")

# Création du dossier data vide obligatoire
os.makedirs("ensam_navigation_app/data", exist_ok=True)
print("\n✅ Terminé ! Le projet complet est prêt dans le dossier 'ensam_navigation_app/'.")