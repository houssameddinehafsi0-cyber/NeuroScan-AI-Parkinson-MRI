import streamlit as st
from pathlib import Path
from PIL import Image

APP_DIR = Path(__file__).parent
MODELS_DIR = APP_DIR / "models"

st.set_page_config(
    page_title="NeuroScan AI | Parkinson MRI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: #f5f8fb;
}
.block-container {
    max-width: 1280px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}
[data-testid="stSidebar"] {
    background: #102a43;
}
[data-testid="stSidebar"] * {
    color: #eef5fb !important;
}
.brand {
    padding: 0.5rem 0 1.5rem 0;
}
.brand-title {
    font-size: 1.35rem;
    font-weight: 700;
    letter-spacing: -0.02em;
}
.brand-sub {
    font-size: .75rem;
    opacity: .72;
    margin-top: .2rem;
}
.topbar {
    display:flex;
    justify-content:space-between;
    align-items:center;
    padding: .8rem 1.2rem;
    background:white;
    border:1px solid #e3eaf1;
    border-radius:14px;
    margin-bottom:1.3rem;
}
.status {
    font-size:.78rem;
    color:#1d7a55;
    background:#e9f8f1;
    padding:.45rem .7rem;
    border-radius:20px;
    font-weight:600;
}
.hero {
    padding:2.5rem;
    border-radius:22px;
    background:linear-gradient(135deg,#0b3954 0%,#087e8b 100%);
    color:white;
    margin-bottom:1.5rem;
    box-shadow:0 12px 30px rgba(16,42,67,.12);
}
.hero h1 {
    font-size:2.35rem;
    margin:0 0 .6rem 0;
    letter-spacing:-.04em;
}
.hero p {
    font-size:1rem;
    opacity:.9;
    max-width:720px;
    line-height:1.6;
}
.metric {
    background:white;
    border:1px solid #e3eaf1;
    border-radius:16px;
    padding:1.2rem;
    box-shadow:0 4px 16px rgba(16,42,67,.04);
}
.metric-label {font-size:.78rem;color:#627d98;text-transform:uppercase;letter-spacing:.04em}
.metric-value {font-size:1.55rem;font-weight:700;color:#102a43;margin-top:.35rem}
.panel {
    background:white;
    border:1px solid #e3eaf1;
    border-radius:18px;
    padding:1.35rem;
    margin-bottom:1rem;
}
.panel h3 {color:#102a43;margin-top:0}
.model-card {
    background:#f8fafc;
    border:1px solid #e1e8ef;
    border-radius:14px;
    padding:1rem;
    min-height:135px;
}
.model-name {font-weight:700;color:#102a43}
.model-desc {font-size:.84rem;color:#627d98;line-height:1.5;margin-top:.4rem}
.result-normal {
    padding:1.5rem;
    border-radius:16px;
    background:#eaf8f1;
    border:1px solid #b9e5cf;
}
.result-note {
    padding:1rem;
    border-radius:12px;
    background:#fff8e8;
    border:1px solid #f0d99c;
}
.footer {
    text-align:center;
    color:#829ab1;
    font-size:.75rem;
    padding-top:2rem;
}
div.stButton > button {
    border-radius:10px;
    font-weight:600;
}
</style>
""", unsafe_allow_html=True)

MODELS = {
    "SVM + HOG": MODELS_DIR / "parkinson_svm_hog_model.joblib",
    "Random Forest + HOG": MODELS_DIR / "random_forest_parkinson.joblib",
    "ResNet50": MODELS_DIR / "ResNet50_Parkinson_finetuned.keras",
}
def model_ready(path):
    return path.exists() and path.stat().st_size > 100

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("""
    <div class="brand">
        <div style="font-size:2rem">🧠</div>
        <div class="brand-title">NeuroScan AI</div>
        <div class="brand-sub">Parkinson MRI Classification</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "MENU",
        ["Dashboard", "Analyse IRM", "Modèles & résultats", "À propos"],
        label_visibility="visible"
    )

    st.divider()
    st.caption("PROTOTYPE ACADÉMIQUE")
    st.caption("Classification expérimentale d'images IRM")

# ---------------- Top bar ----------------
st.markdown("""
<div class="topbar">
    <div><b>NeuroScan AI</b> &nbsp; / &nbsp; Analyse IRM cérébrale</div>
    <div class="status">● Système prêt</div>
</div>
""", unsafe_allow_html=True)

# ---------------- Dashboard ----------------
if page == "Dashboard":
    st.markdown("""
    <div class="hero">
        <h1>Analyse intelligente des IRM cérébrales</h1>
        <p>
        Plateforme expérimentale de classification automatique de la maladie de Parkinson
        à partir d’images IRM, basée sur SVM, Random Forest et ResNet50.
        </p>
    </div>
    """, unsafe_allow_html=True)

    a,b,c,d = st.columns(4)
    with a:
        st.markdown('<div class="metric"><div class="metric-label">Modèles</div><div class="metric-value">3</div></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="metric"><div class="metric-label">Meilleure Accuracy</div><div class="metric-value">94.93%</div></div>', unsafe_allow_html=True)
    with c:
        st.markdown('<div class="metric"><div class="metric-label">ROC-AUC</div><div class="metric-value">98.52%</div></div>', unsafe_allow_html=True)
    with d:
        st.markdown('<div class="metric"><div class="metric-label">Résolution</div><div class="metric-value">128×128</div></div>', unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="panel"><h3>Parcours d’analyse</h3>', unsafe_allow_html=True)
    x,y,z = st.columns(3)
    with x:
        st.markdown('<div class="model-card"><div class="model-name">01 · Charger</div><div class="model-desc">Importer une image IRM cérébrale depuis votre ordinateur.</div></div>', unsafe_allow_html=True)
    with y:
        st.markdown('<div class="model-card"><div class="model-name">02 · Analyser</div><div class="model-desc">Sélectionner SVM, Random Forest ou ResNet50.</div></div>', unsafe_allow_html=True)
    with z:
        st.markdown('<div class="model-card"><div class="model-name">03 · Résultat</div><div class="model-desc">Afficher la classe prédite et les informations du modèle.</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.info("Utilisez « Analyse IRM » dans le menu pour commencer.")

# ---------------- Analysis ----------------
elif page == "Analyse IRM":
    st.title("Analyse IRM")
    st.caption("Importez une image puis sélectionnez le modèle à utiliser.")

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Importer une image IRM",
        type=["jpg","jpeg","png"],
        help="Formats actuellement pris en charge par le prototype."
    )

    if not uploaded:
        st.markdown("""
        <div style="text-align:center;padding:2.2rem;color:#627d98">
            <div style="font-size:3rem">🩻</div>
            <b>Déposez votre image IRM ici</b>
            <p style="font-size:.85rem">JPG, JPEG ou PNG</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        left, right = st.columns([1.15, .85])

        with left:
            st.markdown('<div class="panel"><h3>Image examinée</h3>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.caption(f"Fichier : {uploaded.name} · Taille originale : {image.size[0]} × {image.size[1]}")
            st.markdown('</div>', unsafe_allow_html=True)

        with right:
            st.markdown('<div class="panel"><h3>Configuration de l’analyse</h3>', unsafe_allow_html=True)
            model_name = st.selectbox("Modèle", list(MODELS.keys()))
            st.markdown("**Prétraitement prévu**")
            st.write("✓ Conversion de l’image")
            st.write("✓ Redimensionnement 128 × 128")
            if "SVM" in model_name or "Forest" in model_name:
                st.write("✓ Extraction HOG")
            else:
                st.write("✓ Prétraitement compatible ResNet50")
            st.write("")
            ready = model_ready(MODELS[model_name])
            if ready:
                st.success("Modèle détecté")
            else:
                st.warning("Modèle à intégrer")
            analyze = st.button("🔍 Lancer l’analyse", type="primary", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        if analyze:
            if not ready:
                st.markdown("""
                <div class="result-note">
                <b>Analyse non exécutée.</b><br>
                Le fichier du modèle sélectionné est encore un placeholder.
                Remplacez-le par votre modèle entraîné pour activer la prédiction.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Le point d’intégration du modèle est prêt. La fonction de prédiction sera reliée à votre fichier réel.")

        st.markdown("""
        <div class="panel">
        <h3>Interprétation</h3>
        <p style="color:#627d98">
        Le résultat affiché par cette interface doit être interprété comme une classification
        expérimentale issue du modèle sélectionné et non comme un diagnostic médical.
        </p>
        </div>
        """, unsafe_allow_html=True)

# ---------------- Models ----------------
elif page == "Modèles & résultats":
    st.title("Modèles & résultats")
    st.caption("Performances expérimentales rapportées dans le mémoire.")

    rows = [
        ["SVM + HOG","86.34 %","87.59 %","82.45 %","90.65 %","92.78 %"],
        ["Random Forest + HOG","88.81 %","99.56 %","55.38 %","93.08 %","97.12 %"],
        ["ResNet50","94.93 %","98.10 %","85.07 %","96.69 %","98.52 %"],
    ]
    st.dataframe(
        rows,
        column_config={
            0:"Modèle",1:"Accuracy",2:"Sensibilité",3:"Spécificité",4:"F1-score",5:"ROC-AUC"
        },
        hide_index=True,
        use_container_width=True
    )

    st.markdown('<div class="panel"><h3>Architecture des modèles</h3>', unsafe_allow_html=True)
    a,b,c = st.columns(3)
    with a:
        st.markdown('<div class="model-card"><div class="model-name">SVM</div><div class="model-desc">HOG → SVM → Parkinson / Non-Parkinson</div></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="model-card"><div class="model-name">Random Forest</div><div class="model-desc">HOG → Random Forest → Parkinson / Non-Parkinson</div></div>', unsafe_allow_html=True)
    with c:
        st.markdown('<div class="model-card"><div class="model-name">ResNet50</div><div class="model-desc">Image → ResNet50 → Parkinson / Non-Parkinson</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- About ----------------
else:
    st.title("À propos de NeuroScan AI")
    st.markdown("""
    <div class="panel">
    <h3>Objectif</h3>
    <p>
    Développer une interface expérimentale permettant de charger une image IRM cérébrale
    et de préparer sa classification à l’aide de modèles d’apprentissage automatique.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="panel">
    <h3>Technologies prévues</h3>
    <p>Python · Streamlit · SVM · Random Forest · HOG · ResNet50</p>
    </div>
    """, unsafe_allow_html=True)

    st.warning(
        "Cette application est un prototype académique. Elle ne constitue pas un dispositif "
        "médical validé et ne remplace pas l’évaluation d’un professionnel de santé."
    )

st.markdown('<div class="footer">NeuroScan AI · Projet académique · Parkinson MRI Classification</div>', unsafe_allow_html=True)
