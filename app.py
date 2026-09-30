import os
import io
import itertools
import warnings

import joblib
from huggingface_hub import hf_hub_download
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from skimage.feature import hog

warnings.filterwarnings("ignore")

# ============================================================
# NeuroScan AI - Parkinson MRI Classification
# Real model integration
# ============================================================

st.set_page_config(
    page_title="NeuroScan AI | Parkinson MRI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
HF_REPO_ID = "hou88hou/NeuroScan-AI-Parkinson-MRI"

SVM_FILENAME = "parkinson_svm_hog_model.joblib"
RF_FILENAME = "random_forest_parkinson.joblib"
RESNET_FILENAME = "ResNet50_Parkinson_finetuned.keras"

IMG_SIZE = (128, 128)
RESNET_SIZE = (224, 224)

# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: #f5f7fb;
    }
    section[data-testid="stSidebar"] {
        background: #0b1f33;
    }
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    .hero {
        background: linear-gradient(135deg, #0b1f33 0%, #123d5a 100%);
        padding: 30px;
        border-radius: 18px;
        color: white;
        margin-bottom: 24px;
    }
    .hero h1 {
        margin: 0;
        font-size: 34px;
    }
    .hero p {
        margin-top: 8px;
        opacity: 0.88;
        font-size: 16px;
    }
    .card {
        background: white;
        border-radius: 16px;
        padding: 22px;
        border: 1px solid #e5e9f0;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05);
        margin-bottom: 18px;
    }
    .metric-card {
        background: white;
        border-radius: 14px;
        padding: 18px;
        border: 1px solid #e5e9f0;
        text-align: center;
    }
    .metric-value {
        font-size: 30px;
        font-weight: 700;
        color: #0b1f33;
    }
    .metric-label {
        color: #64748b;
        font-size: 14px;
    }
    .success-box {
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        color: #065f46;
        padding: 16px;
        border-radius: 12px;
    }
    .warning-box {
        background: #fffbeb;
        border: 1px solid #fde68a;
        color: #92400e;
        padding: 16px;
        border-radius: 12px;
    }
    .danger-box {
        background: #fef2f2;
        border: 1px solid #fecaca;
        color: #991b1b;
        padding: 16px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Model loading
# -----------------------------
@st.cache_resource
def download_model(filename):
    """Download a public model from Hugging Face and cache it locally."""
    return hf_hub_download(repo_id=HF_REPO_ID, filename=filename)


@st.cache_resource
def load_classical_models():
    svm_path = download_model(SVM_FILENAME)
    rf_path = download_model(RF_FILENAME)
    svm_bundle = joblib.load(svm_path)
    rf_bundle = joblib.load(rf_path)
    return svm_bundle, rf_bundle


@st.cache_resource
def load_resnet():
    import tensorflow as tf
    resnet_path = download_model(RESNET_FILENAME)
    return tf.keras.models.load_model(resnet_path)


def model_files_status():
    # The repository is public; availability is checked when the model is loaded.
    return {
        "SVM + HOG": True,
        "Random Forest + HOG": True,
        "ResNet50": True,
    }


# -----------------------------
# Image / HOG preprocessing
# -----------------------------
def prepare_gray(image):
    image = ImageOps.exif_transpose(image).convert("L")
    image = image.resize(IMG_SIZE)
    return np.asarray(image, dtype=np.float32) / 255.0


def prepare_rgb(image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize(IMG_SIZE)
    return np.asarray(image, dtype=np.float32)


def build_hog_candidates(gray):
    """Exact HOG configuration used during training."""
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return [{
        "features": features.astype(np.float32),
        "orientations": 9,
        "pixels_per_cell": (8, 8),
        "cells_per_block": (2, 2),
    }]


def get_expected_features(model):
    value = getattr(model, "n_features_in_", None)
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def extract_hog_for_model(gray, model):
    candidates = build_hog_candidates(gray)
    expected = get_expected_features(model)

    if expected is None:
        # Fall back to a common configuration.
        return candidates[0]["features"], candidates[0]

    for candidate in candidates:
        if len(candidate["features"]) == expected:
            return candidate["features"], candidate

    raise ValueError(
        f"Le modèle attend {expected} caractéristiques HOG, "
        f"mais aucune configuration HOG courante testée ne produit ce nombre. "
        f"Il faut utiliser exactement les paramètres HOG du notebook d'entraînement."
    )


# -----------------------------
# Prediction helpers
# -----------------------------
def label_from_prediction(pred):
    """
    Converts common binary outputs to a readable label.
    Assumes the trained classifier uses 0/1 or class labels containing
    Parkinson / Non-PD.
    """
    value = pred

    if isinstance(value, np.ndarray):
        value = value.reshape(-1)[0]

    text = str(value).strip().lower()

    if text in {"1", "1.0", "true", "pd", "parkinson", "parkinson's", "positive"}:
        return "Parkinson"
    if text in {"0", "0.0", "false", "non-pd", "non pd", "non_parkinson", "negative"}:
        return "Non-Parkinson"

    return str(value)


def _find_predictor(obj):
    """Find the real sklearn estimator when joblib contains a dictionary/bundle."""
    if hasattr(obj, "predict"):
        return obj

    if isinstance(obj, dict):
        preferred = (
            "model", "classifier", "estimator", "pipeline",
            "svm", "svm_model", "rf", "rf_model",
            "random_forest", "random_forest_model"
        )
        for key in preferred:
            if key in obj:
                found = _find_predictor(obj[key])
                if found is not None:
                    return found
        for value in obj.values():
            found = _find_predictor(value)
            if found is not None:
                return found

    if isinstance(obj, (list, tuple)):
        for value in obj:
            found = _find_predictor(value)
            if found is not None:
                return found

    return None


def _find_transformers(obj):
    """Recover optional preprocessing objects stored beside a classifier."""
    if not isinstance(obj, dict):
        return []

    transformers = []
    for key in ("scaler", "standard_scaler", "pca", "transformer", "preprocessor"):
        value = obj.get(key)
        if value is not None and hasattr(value, "transform"):
            transformers.append(value)
    return transformers


def _unwrap_classical_bundle(bundle):
    model = _find_predictor(bundle)
    if model is None:
        raise TypeError(
            "Le fichier du modèle classique a bien été chargé, mais aucun estimateur "
            "scikit-learn avec predict() n'a été trouvé dans son contenu."
        )
    transformers = _find_transformers(bundle)
    return model, transformers


def classical_prediction(bundle, image):
    model, transformers = _unwrap_classical_bundle(bundle)

    gray = prepare_gray(image)
    features, hog_info = extract_hog_for_model(gray, model)
    X = features.reshape(1, -1)

    for transformer in transformers:
        X = transformer.transform(X)

    prediction = model.predict(X)[0]
    label = label_from_prediction(prediction)

    probability = None
    if hasattr(model, "predict_proba"):
        try:
            probability = float(np.max(model.predict_proba(X)[0]))
        except Exception:
            probability = None

    return label, probability, hog_info


def _resnet_input_shape(model):
    """Read the saved Keras model input shape instead of assuming 224x224x3."""
    shape = getattr(model, "input_shape", None)
    if isinstance(shape, list):
        shape = shape[0]
    if shape is None and getattr(model, "inputs", None):
        shape = tuple(model.inputs[0].shape)
    if shape is None or len(shape) != 4:
        raise ValueError(f"Forme d'entrée Keras non prise en charge : {shape}")

    _, h, w, c = [int(x) if x is not None else None for x in shape]
    if h is None or w is None or c is None:
        raise ValueError(f"Le modèle possède une forme d'entrée dynamique non exploitable : {shape}")
    return h, w, c


def resnet_prediction(model, image):
    import tensorflow as tf

    h, w, channels = _resnet_input_shape(model)

    gray = ImageOps.exif_transpose(image).convert("L")
    gray = gray.resize((w, h))
    arr = np.asarray(gray, dtype=np.float32)

    if channels == 1:
        x = arr[..., None]
    elif channels == 3:
        # Match the documented ResNet50 preprocessing used in the thesis.
        rgb = np.stack([arr, arr, arr], axis=-1)
        x = tf.keras.applications.resnet50.preprocess_input(rgb)
    else:
        raise ValueError(
            f"Le modèle attend {channels} canaux. L'application prend en charge 1 ou 3 canaux."
        )

    x = np.expand_dims(x, axis=0).astype(np.float32)
    output = model.predict(x, verbose=0)
    values = np.asarray(output).reshape(-1)

    if values.size == 1:
        score = float(values[0])
        probability_pd = score if 0 <= score <= 1 else 1 / (1 + np.exp(-score))
    elif values.size == 2:
        # Training convention in the project: class 0 = PD Patients, class 1 = Non PD Patients.
        probs = tf.nn.softmax(values).numpy()
        probability_pd = float(probs[0])
    else:
        raise ValueError(f"Sortie du modèle inattendue : {values.size} valeurs.")

    label = "Parkinson" if probability_pd >= 0.5 else "Non-Parkinson"
    confidence = probability_pd if label == "Parkinson" else 1 - probability_pd
    return label, float(confidence), float(probability_pd), (h, w, channels)


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("## 🧠 NeuroScan AI")
    st.caption("Parkinson MRI Classification")
    st.divider()

    page = st.radio(
        "Navigation",
        ["Dashboard", "Analyse IRM", "Modèles & résultats", "À propos"],
    )

    st.divider()
    st.caption("Prototype académique — non destiné au diagnostic clinique.")


# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <div class="hero">
        <h1>NeuroScan AI</h1>
        <p>Système expérimental d'aide à la classification de la maladie de Parkinson à partir d'images IRM.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Dashboard
# -----------------------------
if page == "Dashboard":
    st.markdown("## Tableau de bord")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">3</div><div class="metric-label">Modèles</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">94.93%</div><div class="metric-label">Accuracy ResNet50</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">98.52%</div><div class="metric-label">ROC-AUC ResNet50</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">128×128</div><div class="metric-label">Résolution</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("### État des modèles")
    status = model_files_status()

    for name, exists in status.items():
        if exists:
            st.markdown(
                f'<div class="success-box">✓ <strong>{name}</strong> — fichier détecté</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="danger-box">✗ <strong>{name}</strong> — fichier manquant</div>',
                unsafe_allow_html=True,
            )

# -----------------------------
# MRI analysis
# -----------------------------
elif page == "Analyse IRM":
    st.markdown("## Analyse d'une image IRM")

    st.markdown(
        """
        <div class="card">
        <strong>Étape 1 — Importer une image</strong><br>
        Chargez une image IRM au format JPG, JPEG ou PNG.
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Sélectionner une image IRM",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded is not None:
        image = Image.open(io.BytesIO(uploaded.read()))

        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(image, caption="Image IRM importée", use_container_width=True)

        with col2:
            st.markdown("### Paramètres")
            model_choice = st.selectbox(
                "Modèle",
                ["ResNet50", "SVM + HOG", "Random Forest + HOG"],
            )

            st.info(
                "SVM/RF : 128×128 + HOG (9 orientations, 8×8, 2×2). ResNet50 : 224×224 + 3 canaux + preprocess_input."
            )

            files = model_files_status()
            selected_file_ok = {
                "ResNet50": files["ResNet50"],
                "SVM + HOG": files["SVM + HOG"],
                "Random Forest + HOG": files["Random Forest + HOG"],
            }[model_choice]

            if selected_file_ok:
                st.success("Modèle détecté et prêt à être chargé.")
            else:
                st.error("Le fichier du modèle sélectionné est introuvable.")

            run = st.button(
                "🔎 Lancer l'analyse",
                type="primary",
                use_container_width=True,
            )

        if run:
            if not selected_file_ok:
                st.error("Impossible de lancer l'analyse : modèle manquant.")
            else:
                try:
                    with st.spinner("Analyse de l'image en cours..."):
                        if model_choice == "SVM + HOG":
                            svm, _ = load_classical_models()
                            label, confidence, hog_info = classical_prediction(svm, image)

                            st.markdown("### Résultat")
                            st.success(f"Classification : **{label}**")
                            if confidence is not None:
                                st.metric("Confiance du modèle", f"{confidence * 100:.2f}%")

                            st.caption(
                                "HOG utilisé : "
                                f"{hog_info['orientations']} orientations, "
                                f"pixels/cellule={hog_info['pixels_per_cell']}, "
                                f"cellules/bloc={hog_info['cells_per_block']}."
                            )

                        elif model_choice == "Random Forest + HOG":
                            _, rf = load_classical_models()
                            label, confidence, hog_info = classical_prediction(rf, image)

                            st.markdown("### Résultat")
                            st.success(f"Classification : **{label}**")
                            if confidence is not None:
                                st.metric("Confiance du modèle", f"{confidence * 100:.2f}%")

                            st.caption(
                                "HOG utilisé : "
                                f"{hog_info['orientations']} orientations, "
                                f"pixels/cellule={hog_info['pixels_per_cell']}, "
                                f"cellules/bloc={hog_info['cells_per_block']}."
                            )

                        else:
                            resnet = load_resnet()
                            label, confidence, _, input_shape = resnet_prediction(resnet, image)

                            st.markdown("### Résultat")
                            st.success(f"Classification : **{label}**")
                            st.metric("Confiance du modèle", f"{confidence * 100:.2f}%")

                            st.caption(
                                f"Entrée du modèle : {input_shape[0]}×{input_shape[1]}×{input_shape[2]} ; prétraitement adapté à la forme réelle du modèle."
                            )

                except Exception as e:
                    st.error("L'analyse n'a pas pu être exécutée.")
                    st.code(str(e))

                    if model_choice in ["SVM + HOG", "Random Forest + HOG"]:
                        st.warning(
                            "Pour SVM/Random Forest, les paramètres HOG doivent être "
                            "identiques à ceux utilisés pendant l'entraînement."
                        )
                    else:
                        st.warning(
                            "Pour ResNet50, le prétraitement doit être identique à celui "
                            "utilisé pendant l'entraînement du modèle."
                        )

    else:
        st.markdown(
            '<div class="warning-box">Veuillez importer une image IRM pour commencer.</div>',
            unsafe_allow_html=True,
        )

# -----------------------------
# Results
# -----------------------------
elif page == "Modèles & résultats":
    st.markdown("## Modèles et résultats expérimentaux")

    st.markdown(
        """
        <div class="card">
        <strong>SVM + HOG</strong><br>
        Accuracy : 86.34% &nbsp; | &nbsp; ROC-AUC : 92.78%
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="card">
        <strong>Random Forest + HOG</strong><br>
        Accuracy : 88.81% &nbsp; | &nbsp; ROC-AUC : 97.12%
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="card">
        <strong>ResNet50 fine-tuned</strong><br>
        Accuracy : 94.93% &nbsp; | &nbsp; Precision : 95.33% &nbsp; | &nbsp;
        Sensitivity : 98.10% &nbsp; | &nbsp; Specificity : 85.07% &nbsp; | &nbsp;
        F1 : 96.69% &nbsp; | &nbsp; ROC-AUC : 98.52%
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------
# About
# -----------------------------
else:
    st.markdown("## À propos")

    st.markdown(
        """
        <div class="card">
        <h3>NeuroScan AI</h3>
        <p>
        Prototype académique de classification d'images IRM pour l'étude de la
        maladie de Parkinson. L'application intègre trois approches :
        SVM avec HOG, Random Forest avec HOG et ResNet50 fine-tuned.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.warning(
        "Important : cette application est un prototype expérimental académique. "
        "Elle n'est pas un dispositif médical et ne doit pas être utilisée seule "
        "pour établir un diagnostic."
    )
