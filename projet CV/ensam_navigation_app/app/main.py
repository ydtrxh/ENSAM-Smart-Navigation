"""
ENSAM Meknès Indoor Navigation - Streamlit Application
Architecture: CNN (Location) + RapidFuzz NLP (Destination) + A* (Pathfinding)
"""
import streamlit as st
import os
import json
import sys
import html

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

_PROJECT_ROOT = os.path.dirname(_APP_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()

from nlp_engine.pipeline import NLPPipeline, load_config
from cv_engine.inference import predict
from navigation.graph_store import GraphStore
from navigation.engine import NavigationEngine
from navigation.map_manager import render_map
from app.config import settings


def render_metric_card(label: str, value: str):
    if label == "Accuracy Globale" and st.session_state.get("_cv_metrics_stale"):
        measured = st.session_state.get("_cv_metrics_measured_count", 0)
        expected = st.session_state.get("_cv_metrics_expected_count", 0)
        st.warning(
            "Métriques CV masquées : `model_metrics.json` ne correspond pas "
            f"aux classes actuelles ({measured} classes mesurées, "
            f"{expected} classes attendues). Relancez l'évaluation du modèle "
            "pour afficher des performances fiables."
        )
        return

    st.markdown(
        f"""
        <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;
                    padding:14px 16px;margin-bottom:10px;">
            <div style="color:#64748B;font-size:0.78rem;font-weight:600;margin-bottom:6px;">
                {html.escape(label)}
            </div>
            <div style="color:#0F172A;font-size:1.25rem;font-weight:700;line-height:1.25;">
                {html.escape(str(value))}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_simple_table(rows: list[dict]):
    if not rows:
        return
    if st.session_state.get("_cv_metrics_stale"):
        return
    headers = list(rows[0].keys())
    header_html = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    rows_html = ""
    for row in rows:
        rows_html += "<tr>" + "".join(
            f"<td>{html.escape(str(row.get(h, '')))}</td>" for h in headers
        ) + "</tr>"
    st.markdown(
        f"""
        <table style="width:100%;border-collapse:collapse;font-size:0.86rem;">
            <thead>
                <tr style="background:#F1F5F9;color:#334155;">{header_html}</tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
        <style>
            table th, table td {{
                border:1px solid #E2E8F0;
                padding:8px 10px;
                text-align:left;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_current_cv_labels() -> set[str]:
    buildings_path = settings.get_path("buildings_json")
    try:
        with open(buildings_path, "r", encoding="utf-8") as f:
            return {entry["label"] for entry in json.load(f)}
    except Exception:
        return set()


def load_cv_class_to_node_id() -> dict[str, str]:
    buildings_path = settings.get_path("buildings_json")
    try:
        with open(buildings_path, "r", encoding="utf-8") as f:
            buildings = json.load(f)
        return {
            entry["label"]: entry["node_id"]
            for entry in buildings
            if entry.get("label") and entry.get("node_id")
        }
    except Exception:
        return {}


def resolve_cv_artifact_paths() -> tuple[str, str]:
    gallery_path = settings.get_path("gallery_pkl")
    checkpoint_path = settings.get_path("best_model_pth")
    return gallery_path, checkpoint_path


def metrics_match_current_classes(saved_metrics: dict) -> tuple[bool, set[str], set[str]]:
    current_labels = get_current_cv_labels()
    metric_labels = set(saved_metrics.get("class_names") or [])
    if not metric_labels:
        report = saved_metrics.get("report", {})
        metric_labels = {
            label for label in report
            if label not in ("accuracy", "macro avg", "weighted avg")
        }
    return metric_labels == current_labels, current_labels, metric_labels

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="ENSAM Nav — Navigation Intelligente",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==============================================================================
# CSS STYLING
# ==============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stSidebar"] {display: none;}

    .block-container {
        max-width: 1150px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    /* ---- Header ---- */
    .app-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #1D4ED8 50%, #2563EB 100%);
        border-radius: 16px;
        padding: 22px 28px;
        margin-bottom: 24px;
        color: white;
        display: flex;
        align-items: center;
        gap: 16px;
        box-shadow: 0 10px 25px rgba(30, 58, 138, 0.3);
    }
    .app-header h1 { margin: 0; font-size: 1.6rem; font-weight: 700; }
    .app-header p  { margin: 4px 0 0; opacity: 0.85; font-size: 0.9rem; }

    /* ---- Step Cards ---- */
    .step-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 22px 24px;
        margin-bottom: 18px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        border-left: 5px solid #1E3A8A;
    }
    .step-card.green { border-left-color: #10B981; }
    .step-card.red   { border-left-color: #EF4444; }
    .step-card h3 { margin-top: 0; color: #1E293B; font-size: 1.05rem; }
    .step-card p  { color: #64748B; font-size: 0.88rem; margin-bottom: 0; }

    /* ---- Step Badges ---- */
    .step-badge {
        display: inline-block;
        background: #1E3A8A;
        color: white;
        border-radius: 50%;
        width: 28px; height: 28px;
        line-height: 28px;
        text-align: center;
        font-weight: 700;
        font-size: 0.85rem;
        margin-right: 10px;
    }

    /* ---- Instruction steps ---- */
    .route-step {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 0.87rem;
        color: #334155;
    }

    /* ---- Buttons ---- */
    .stButton > button {
        background: linear-gradient(135deg, #1E3A8A, #2563EB) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 22px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        transition: all 0.2s !important;
        box-shadow: 0 3px 10px rgba(30, 58, 138, 0.25) !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 16px rgba(30, 58, 138, 0.35) !important;
    }

    /* ---- Chat messages ---- */
    [data-testid="stChatMessage"] {
        border-radius: 12px !important;
        margin-bottom: 8px !important;
    }

    /* ---- Fix truncated metrics (e.g. Position Actuelle) ---- */
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] > div {
        white-space: normal !important;
        word-wrap: break-word !important;
        overflow: visible !important;
        text-overflow: clip !important;
        font-size: 1.6rem !important;
        line-height: 1.2 !important;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================
@st.cache_resource
def get_graph_store():
    URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    USER = os.getenv("NEO4J_USER", "neo4j")
    PASSWORD = os.getenv("NEO4J_PASSWORD", "ensam360password")
    return GraphStore(URI, USER, PASSWORD)

@st.cache_resource
def get_nav_engine(_store):
    return NavigationEngine(_store)

@st.cache_resource
def get_nlp_pipeline():
    try:
        config_path = settings.get_path("nlp_config")
        # Fallback to old path if the new configs/nlp.yaml doesn't exist yet
        if not os.path.exists(config_path):
            config_path = os.path.join(_PROJECT_ROOT, "nlp_engine", "config.yaml")
        
        config = load_config(config_path)
        
        geojson_path = settings.get_path("campus_geojson")
        config["geojson_path"] = geojson_path
        
        if config.get("llamacpp_model_path"):
            model_path = config["llamacpp_model_path"]
            if not os.path.isabs(model_path):
                preferred_path = os.path.join(_PROJECT_ROOT, model_path)
                legacy_path = os.path.join(_PROJECT_ROOT, "nlp_engine", model_path)
                config["llamacpp_model_path"] = (
                    preferred_path if os.path.exists(preferred_path) else legacy_path
                )
        return NLPPipeline(config)
    except Exception as e:
        import traceback
        print(f"[NLP INIT] Warning: {e}\n{traceback.format_exc()}")
        return None

# Load the resources using the cached functions
graph_store = get_graph_store()
nav_engine = get_nav_engine(graph_store)
nlp_pipeline = get_nlp_pipeline()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "detected_location" not in st.session_state:
    st.session_state.detected_location = None  # Current node_id from CNN

if "target_node_id" not in st.session_state:
    st.session_state.target_node_id = None     # Destination node_id from NLP

if "active_path" not in st.session_state:
    st.session_state.active_path = None


# ==============================================================================
# LOAD DATA & BUILD GRAPH
# ==============================================================================
# Load Data & Build Graph
nav_store = graph_store

# Preload nodes and edges from Neo4j for the frontend map renderer
nodes_list = nav_store.get_all_nodes()
nodes = {n["id"]: n for n in nodes_list}
edges = nav_store.get_all_edges()


# ==============================================================================
# HEADER
# ==============================================================================
st.markdown("""
<div class="app-header">
    <div style="font-size: 2.5rem;">🧭</div>
    <div>
        <h1>ENSAM Meknès — Navigation Intelligente</h1>
        <p>Système de guidage indoor par IA · Vision par Ordinateur + NLP + A*</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# STEP 1: LOCALIZATION VIA CNN
# ==============================================================================
if not st.session_state.detected_location:

    st.markdown("""
    <div class="step-card">
        <h3><span class="step-badge">1</span>📸 Localisation par Vision Artificielle (CNN)</h3>
        <p>Prenez ou importez une photo de la porte ou du couloir où vous vous trouvez.
        Le modèle CNN analysera l'image et détectera automatiquement votre position sur le plan.</p>
    </div>
    """, unsafe_allow_html=True)

    # Single source of truth: CV class labels map to navigation node IDs through data/buildings.json.
    CV_CLASS_TO_NODE_ID = load_cv_class_to_node_id()

    CNN_CLASS_TO_NODE = {}
    for cv_cls, node_id in CV_CLASS_TO_NODE_ID.items():
        if node_id in nodes:
            CNN_CLASS_TO_NODE[cv_cls] = (node_id, nodes[node_id]["name"])

    uploaded_photo = st.file_uploader(
        "📁 Choisissez une image de votre environnement actuel...",
        type=['jpg', 'jpeg', 'png'],
        help="Photo d'une porte de département ou d'un couloir."
    )

    if uploaded_photo is not None:
        col_img, col_result = st.columns([1, 1])

        with col_img:
            st.image(uploaded_photo, caption="Image reçue")

        with col_result:
            with st.spinner("🔍 Analyse en cours par le réseau CNN..."):
                try:
                    gallery_path, checkpoint_path = resolve_cv_artifact_paths()
                    predicted_class, confidence = predict(uploaded_photo, gallery_path, checkpoint_path)
                except Exception as e:
                    st.warning(f"Modèle CNN non disponible ({e}). Mode simulation activé.")
                    predicted_class = "TD1"
                    confidence = 0.50

            if predicted_class == "unknown":
                node_id, display_name = None, "Position inconnue"
            else:
                node_id, display_name = CNN_CLASS_TO_NODE.get(
                    predicted_class, (None, f"Classe non reli?e: {predicted_class}")
                )
            # Store in session state so the confirm button (outside columns) can access them
            st.session_state["_cnn_predicted_class"] = predicted_class
            st.session_state["_cnn_node_id"] = node_id
            st.session_state["_cnn_display_name"] = display_name
            st.session_state["_cnn_confidence"] = confidence

            if node_id:
                st.success("? Position d?tect?e avec succ?s !")
            else:
                st.warning("Position non confirm?e : le mod?le n'a pas reli? cette image ? un noeud du graphe.")
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                render_metric_card("?? Position Actuelle", display_name)
            with col_m2:
                render_metric_card("?? Confiance CNN", f"{confidence * 100:.1f}%")
            render_metric_card("Classe CNN", predicted_class)

            # Display model performance metrics if available
            metrics_path = settings.get_path("model_metrics")

            if os.path.exists(metrics_path):
                with open(metrics_path, "r", encoding="utf-8") as f:
                    saved_metrics = json.load(f)
                metrics_are_current, current_labels, metric_labels = metrics_match_current_classes(saved_metrics)
                st.session_state["_cv_metrics_stale"] = not metrics_are_current
                st.session_state["_cv_metrics_measured_count"] = len(metric_labels)
                st.session_state["_cv_metrics_expected_count"] = len(current_labels)

                st.markdown("---")
                st.markdown("**📊 Performance du Modèle (Test Set)**")
                render_metric_card("Accuracy Globale", f"{saved_metrics['accuracy'] * 100:.2f}%")

                report = saved_metrics.get("report", {})
                table_data = []
                for cls, vals in report.items():
                    if cls in ("accuracy", "macro avg", "weighted avg"):
                        continue
                    table_data.append({
                        "Classe": cls,
                        "Précision": f"{vals['precision']*100:.1f}%",
                        "Recall": f"{vals['recall']*100:.1f}%",
                        "F1-Score": f"{vals['f1-score']*100:.1f}%",
                        "Support": int(vals['support'])
                    })
                if table_data:
                    render_simple_table(table_data)
            else:
                st.info("💡 Exécutez `train_model.py` pour afficher les métriques réelles du modèle.")

    # Confirmation button — placed OUTSIDE columns so it's always fully visible
    if uploaded_photo is not None:
        _detected_name = st.session_state.get("_cnn_display_name", "")
        _detected_conf = st.session_state.get("_cnn_confidence", 0)
        st.markdown("---")
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            _can_confirm = bool(st.session_state.get("_cnn_node_id"))
            if st.button(
                f"? Confirmer : {_detected_name} ({_detected_conf*100:.0f}%) ? Continuer ?",
                type="primary",
                disabled=not _can_confirm,
            ):
                st.session_state.detected_location = st.session_state["_cnn_node_id"]
                st.session_state.chat_history = []
                st.rerun()


# ==============================================================================
# STEPS 2 & 3: NAVIGATION DASHBOARD
# ==============================================================================
else:
    current_node_id = st.session_state.detected_location
    current_node_name = nodes.get(current_node_id, {}).get('name', current_node_id)

    col_left, col_right = st.columns([1, 1], gap="large")

    # ------------------------------------------------------------------
    # LEFT COLUMN: NLP Query + Turn-by-Turn Instructions
    # ------------------------------------------------------------------
    with col_left:
        # Determine NLP mode for display
        _nlp = nlp_pipeline
        if _nlp is not None and hasattr(_nlp, 'llm_available') and _nlp.llm_available:
            _nlp_badge = '<span style="background:#10B981;color:#fff;border-radius:4px;padding:2px 7px;font-size:10px;font-weight:600;margin-left:8px;">LLM ✓</span>'
        else:
            _nlp_badge = '<span style="background:#F59E0B;color:#fff;border-radius:4px;padding:2px 7px;font-size:10px;font-weight:600;margin-left:8px;">RapidFuzz seul</span>'

        st.markdown(f"""
        <div class="step-card green">
            <h3><span class="step-badge" style="background:#10B981">2</span>💬 Saisir votre Destination {_nlp_badge}</h3>
            <p>📍 Position actuelle : <strong>{current_node_name}</strong></p>
        </div>
        """, unsafe_allow_html=True)

        # Chat history display
        for msg in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(msg["user"])
            with st.chat_message("assistant"):
                st.write(msg["assistant"])

        # Text input for destination
        _hint = "Ex: amphi 450, bibliothèque, td 1, buvette..." if (_nlp and hasattr(_nlp,'llm_available') and _nlp.llm_available) else "Tapez le nom exact du lieu (ex: Amphi 450, Buvette, Bibliothèque)..."
        if user_query := st.chat_input(_hint):
            with st.chat_message("user"):
                st.write(user_query)

            # NLP: Extract destination from text
            if nlp_pipeline:
                intent = nlp_pipeline.process(user_query)
                target_node_id = intent.node_id
                nlp_confidence = intent.confidence
                if intent.resolved:
                    reply = f"Navigation vers {intent.label}."
                else:
                    reply = "Je n'ai pas pu identifier votre destination. Pourriez-vous reformuler ?"
            else:
                target_node_id = None
                nlp_confidence = 0.0
                reply = "NLP Pipeline non disponible."

            if target_node_id and target_node_id in nodes:
                st.session_state.target_node_id = target_node_id
                # Compute route immediately
                route_result = nav_engine.get_shortest_path(current_node_id, target_node_id)
                if "error" not in route_result:
                    st.session_state.active_path = route_result["path"]
                    st.session_state.active_dist = route_result["distance"]
                    st.session_state.active_instructions = route_result["instructions"]
                else:
                    st.error(route_result["error"])

            with st.chat_message("assistant"):
                st.write(reply)

            st.session_state.chat_history.append({"user": user_query, "assistant": reply})
            st.rerun()

        # Display turn-by-turn instructions if path exists
        if st.session_state.active_path:
            dest_name = nodes.get(st.session_state.target_node_id, {}).get('name', '')
            instructions = st.session_state.get('active_instructions', [])
            dist = st.session_state.get('active_dist', 0)

            st.markdown("---")
            st.markdown(f"""
            <div class="step-card green">
                <h3>🗺️ Itinéraire vers {dest_name}</h3>
                <p>Distance totale estimée : <strong>{dist:.0f} mètres</strong>
                   · Durée estimée : <strong>{dist/70:.1f} min</strong> à pied</p>
            </div>
            """, unsafe_allow_html=True)

            for step in instructions:
                # Add conditional rendering based on step type
                icon = "🚶"
                if step.get("type") == "turn":
                    icon = "↪️"
                elif step.get("type") == "straight":
                    icon = "⬆️"
                elif step.get("type") == "arrive":
                    icon = "🎯"
                elif step.get("type") == "stairs":
                    icon = "🪜"
                st.markdown(f'<div class="route-step">{icon} {step["text"]}</div>', unsafe_allow_html=True)

        st.markdown("---")
        if st.button("🔄 Nouvelle Localisation (Retour à l'étape 1)"):
            st.session_state.detected_location = None
            st.session_state.target_node_id = None
            st.session_state.active_path = None
            st.session_state.chat_history = []
            st.rerun()

    # ------------------------------------------------------------------
    # RIGHT COLUMN: Interactive Map
    # ------------------------------------------------------------------
    with col_right:
        st.markdown("""
        <div class="step-card red">
            <h3><span class="step-badge" style="background:#EF4444">3</span>🗺️ Carte Interactive du Campus</h3>
            <p>Visualisation du plan de masse avec tracé de l'itinéraire optimal (algorithme A*).</p>
        </div>
        """, unsafe_allow_html=True)

        if not st.session_state.active_path:
            st.info("💡 Entrez votre destination dans le chat à gauche pour générer le tracé.")

        render_map(
            nodes,
            edges,
            current_path=st.session_state.active_path,
            current_location=current_node_id
        )
