import streamlit as st
from PIL import Image
import torch
import gc

# --------------------------
# Imports dos módulos
# --------------------------
from boudingBox import run_lion_yolo
from clustering import run_clustering_lion
from imageCaption import run_caption
from SpatialGraph import run_spatial_graph
from SemanticGraph import run_semantic_graph  # versão com grafo unificado

# ==========================
# Utils
# ==========================
def clear_memory():
    """Limpeza leve (segura com cache de modelos)."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def show_result_image(img, max_width=500):
    """Mostra imagem de resultado menor mantendo aspect ratio."""
    w, h = img.size
    scale = max_width / w
    new_size = (int(w * scale), int(h * scale))
    st.image(img.resize(new_size), width=max_width)

# ==========================
# Configuração da página
# ==========================
st.set_page_config(
    page_title="LION App",
    layout="wide"
)
st.title("LION - Empowering Multimodal Large Language Model with Dual-Level Visual Knowledge")

# ==========================
# Upload de imagem
# ==========================
uploaded_file = st.file_uploader(
    "Upload uma imagem",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_key = hash(img.tobytes())

    if "img_key" not in st.session_state or st.session_state.img_key != img_key:
        clear_memory()
        st.session_state.img = img
        st.session_state.img_key = img_key
        for key in [
            "bbox_result",
            "cluster_result",
            "caption_result",
            "spatial_result",
            "semantic_result",
            "prolog_result",
        ]:
            st.session_state[key] = None
else:
    st.info("Faça upload de uma imagem para continuar.")
    st.stop()

# ==========================
# Estado inicial
# ==========================
if "active_mode" not in st.session_state:
    st.session_state.active_mode = None

# ==========================
# Selector de modo
# ==========================
mode = st.radio(
    "Escolha o modo",
    [
        "Bounding Box + LION Tags",
        "Clustering",
        "Captioning",
        "Spatial Scene Graph",
        "Semantic Scene Graph",
        "Prolog Representation"
    ],
    horizontal=True
)

# ==========================
# Limpeza ao trocar de aba
# ==========================
if st.session_state.active_mode != mode:
    clear_memory()
    st.session_state.active_mode = mode

# ======================================================
# MODOS
# ======================================================

# 1️⃣ Bounding Box
if mode == "Bounding Box + LION Tags":
    st.header("Bounding Box + LION Tags")
    if st.button("Gerar Bounding Boxes"):
        with st.spinner("Processando YOLO + LION..."):
            st.session_state.bbox_result = run_lion_yolo(st.session_state.img)
    if st.session_state.bbox_result:
        img_res, tags = st.session_state.bbox_result
        show_result_image(img_res, max_width=500)
        st.subheader("Tags")
        st.json(tags)

# 2️⃣ Clustering
elif mode == "Clustering":
    st.header("Clustering de Objetos")
    if st.button("Gerar Clusters"):
        with st.spinner("Processando clustering..."):
            st.session_state.cluster_result = run_clustering_lion(st.session_state.img)
    if st.session_state.cluster_result:
        img_res, clusters = st.session_state.cluster_result
        show_result_image(img_res, max_width=500)
        st.subheader("Clusters")
        st.json(clusters)

# 3️⃣ Captioning
elif mode == "Captioning":
    st.header("Captioning (LION)")
    if st.button("Gerar Descrição"):
        with st.spinner("Gerando caption..."):
            st.session_state.caption_result = run_caption(st.session_state.img)
    if st.session_state.caption_result:
        r = st.session_state.caption_result
        st.write("**Pergunta:**", r["question"])
        st.write("**Tags:**")
        st.json(r["tags"])
        st.write("**Resposta:**")
        st.write(r["answer"])

# 4️⃣ Spatial Graph
elif mode == "Spatial Scene Graph":
    st.header("Spatial Scene Graph")
    if st.button("Gerar Grafo Espacial"):
        with st.spinner("Processando grafo espacial..."):
            img_small = st.session_state.img.resize((640, 480))
            st.session_state.spatial_result = run_spatial_graph(img_small)
    if st.session_state.spatial_result:
        fig, rel = st.session_state.spatial_result
        st.pyplot(fig)
        show_result_image(st.session_state.img, max_width=400)
        st.subheader("Relações")
        st.json(rel)

# 5️⃣ Semantic Graph (Físico + Semântico)
elif mode == "Semantic Scene Graph":
    st.header("Semantic Scene Graph (Physical + Caption-Aligned)")
    if st.button("Gerar Grafo Semântico"):
        with st.spinner("Gerando legenda + grafo..."):
            if not st.session_state.caption_result:
                st.session_state.caption_result = run_caption(st.session_state.img)
            caption = st.session_state.caption_result["answer"]

            fig, rel, objs, semantic_interactions = run_semantic_graph(st.session_state.img, caption)
            st.session_state.semantic_result = (fig, rel, objs, semantic_interactions)

    if st.session_state.semantic_result:
        fig, rel, _, semantic_interactions = st.session_state.semantic_result
        st.pyplot(fig)

        st.subheader("Legenda usada")
        st.write(st.session_state.caption_result["answer"])

        st.subheader("Relações físicas (next_to / holding / etc.)")
        st.json(rel)

        st.subheader("Interações semânticas extraídas da legenda")
        st.json(semantic_interactions)

# 6️⃣ Prolog Representation
elif mode == "Prolog Representation":
    st.header("🧩 Prolog Representation")
    if st.button("Gerar Prolog"):
        if not st.session_state.semantic_result:
            st.warning("Primeiro gera o Semantic Scene Graph.")
        else:
            from prolog_representation import to_prolog
            fig, relations, objects, _ = st.session_state.semantic_result
            prolog_code = to_prolog(objects=objects, relations=relations)
            st.session_state.prolog_result = prolog_code

    if st.session_state.prolog_result:
        st.subheader("Factos Prolog")
        st.code(st.session_state.prolog_result, language="prolog")
