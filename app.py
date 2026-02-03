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
from SemanticGraph import run_semantic_graph


# ==========================
# Utils
# ==========================
def clear_memory():
    """Limpeza leve (segura com cache de modelos)."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def show_result_image(img, max_width=500):
    """
    Mostra imagem de resultado mais pequena,
    mantendo aspect ratio.
    """
    w, h = img.size
    scale = max_width / w
    new_size = (int(w * scale), int(h * scale))
    st.image(img.resize(new_size), width=max_width)


# ==========================
# Configuração da página
# ==========================
st.set_page_config(
    page_title="Visual Reasoning App",
    layout="wide"
)
st.title("🧠 Visual Reasoning Demo")


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

    # Nova imagem → reset de resultados
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

# ==========================
# 1️⃣ Bounding Box
# ==========================
if mode == "Bounding Box + LION Tags":
    st.header("Bounding Box + LION Tags")

    if st.button("Gerar Bounding Boxes"):
        with st.spinner("Processando YOLO + LION..."):
            st.session_state.bbox_result = run_lion_yolo(
                st.session_state.img
            )

    if st.session_state.bbox_result:
        img_res, tags = st.session_state.bbox_result
        show_result_image(img_res, max_width=500)
        st.subheader("Tags")
        st.json(tags)


# ==========================
# 2️⃣ Clustering
# ==========================
elif mode == "Clustering":
    st.header("Clustering de Objetos")

    if st.button("Gerar Clusters"):
        with st.spinner("Processando clustering..."):
            st.session_state.cluster_result = run_clustering_lion(
                st.session_state.img
            )

    if st.session_state.cluster_result:
        img_res, clusters = st.session_state.cluster_result
        show_result_image(img_res, max_width=500)
        st.subheader("Clusters")
        st.json(clusters)


# ==========================
# 3️⃣ Captioning
# ==========================
elif mode == "Captioning":
    st.header("Captioning (LION)")

    if st.button("Gerar Descrição"):
        with st.spinner("Gerando caption..."):
            st.session_state.caption_result = run_caption(
                st.session_state.img
            )

    if st.session_state.caption_result:
        r = st.session_state.caption_result
        st.write("**Pergunta:**", r["question"])
        st.write("**Tags:**")
        st.json(r["tags"])
        st.write("**Resposta:**")
        st.write(r["answer"])


# ==========================
# 4️⃣ Spatial Graph
# ==========================
elif mode == "Spatial Scene Graph":
    st.header("Spatial Scene Graph")

    if st.button("Gerar Grafo Espacial"):
        with st.spinner("Processando grafo espacial..."):
            img_small = st.session_state.img.resize((640, 480))
            st.session_state.spatial_result = run_spatial_graph(img_small)

    if st.session_state.spatial_result:
        fig, rel = st.session_state.spatial_result
        # Mostrar figura do grafo
        st.pyplot(fig)
        # Opcional: criar miniatura da imagem para visualização
        show_result_image(st.session_state.img, max_width=400)
        st.subheader("Relações")
        st.json(rel)


# ==========================
# 5️⃣ Semantic Graph
# ==========================
elif mode == "Semantic Scene Graph":
    st.header("Semantic Scene Graph")

    if st.button("Gerar Grafo Semântico"):
        with st.spinner("Processando grafo semântico..."):
            img_small = st.session_state.img.resize((640, 480))
            st.session_state.semantic_result = run_semantic_graph(img_small)

    if st.session_state.semantic_result:
        fig, rel = st.session_state.semantic_result
        # Mostrar figura do grafo
        st.pyplot(fig)
        # Mostrar miniatura da imagem original (como no clustering)
        show_result_image(st.session_state.img, max_width=400)
        st.subheader("Relações")
        st.json(rel)
