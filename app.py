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
from objectDetection import run_lion_qa

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
    "Upload image",
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
            "qa_result",
            "spatial_result",
            "semantic_result",
            "prolog_result",
        ]:
            st.session_state[key] = None
else:
    st.info("Upload an image to continue")
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
    "Choose mode",
    [
        "Caption",
        "Bounding Box",
        "Clustering",
        "Interactive LION QA",
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

# Captioning
if mode == "Caption":
    st.header("Caption")
    if st.button("Generate Description"):
        with st.spinner("Generating caption..."):
            st.session_state.caption_result = run_caption(st.session_state.img)
    if st.session_state.caption_result:
        r = st.session_state.caption_result
        st.write("**Question:**", r["question"])
        st.write("**Tags:**")
        st.json(r["tags"])
        st.write("**Answer:**")
        st.write(r["answer"])

# Bounding Box
elif mode == "Bounding Box":
    st.header("Bounding Box")

    if st.button("Generate Bounding Boxes"):
        with st.spinner("Processing YOLO + LION..."):
            img_out, detections, tags = run_lion_yolo(st.session_state.img)

            st.session_state.bbox_result = {
                "img": img_out,
                "detections": detections,
                "tags": tags
            }

    if st.session_state.bbox_result:
        r = st.session_state.bbox_result

        st.subheader("Image with Bounding Boxes")
        show_result_image(r["img"], max_width=500)

        st.subheader("Detected Objects")
        st.json(r["detections"])

        st.subheader("Tags LION (usage semantics)")
        st.json(r["tags"])


# Clustering
elif mode == "Clustering":
    st.header("Clustering de Objetos")
    if st.button("Generate Clusters"):
        with st.spinner("Processing clustering..."):
            st.session_state.cluster_result = run_clustering_lion(st.session_state.img)
    if st.session_state.cluster_result:
        img_res, clusters = st.session_state.cluster_result
        show_result_image(img_res, max_width=500)
        st.subheader("Clusters")
        st.json(clusters)

#Object Detection
elif mode == "Interactive LION QA":
    st.header("Interactive LION Question Answering")

    question = st.text_input(
        "Type your question about the image.",
        placeholder="Ex: What are the people doing?"
    )

    if st.button("Ask LION") and question:
        with st.spinner("Processing question..."):
            qa_result = run_lion_qa(
                st.session_state.img,
                question
            )

            # 🔐 guardar no estado
            st.session_state.qa_result = {
                "question": question,
                "img": qa_result[0],
                "answer": qa_result[1],
                "bboxes": qa_result[2],
            }

    # 🔁 mostrar resultado guardado
    if st.session_state.qa_result:
        r = st.session_state.qa_result

        st.subheader("Question")
        st.write(r["question"])

        st.subheader("Image with Bounding Boxes")
        show_result_image(r["img"], max_width=500)

        st.subheader("Answer")
        st.write(r["answer"])

        st.subheader("Bounding Boxes (pixels)")
        st.json(r["bboxes"])



# Spatial Graph
elif mode == "Spatial Scene Graph":
    st.header("Spatial Scene Graph")
    if st.button("Generate Spatial Graph"):
        with st.spinner("Processing spatial graph..."):
            img_small = st.session_state.img.resize((640, 480))
            st.session_state.spatial_result = run_spatial_graph(img_small)
    if st.session_state.spatial_result:
        fig, rel = st.session_state.spatial_result
        st.pyplot(fig)
        show_result_image(st.session_state.img, max_width=400)
        st.subheader("Relações")
        st.json(rel)

# Semantic Graph (Físico + Semântico)
elif mode == "Semantic Scene Graph":
    st.header("Semantic Scene Graph (Physical + Caption-Aligned)")
    if st.button("Generate Semantic Graph"):
        with st.spinner("Generating caption + graph..."):
            if not st.session_state.caption_result:
                st.session_state.caption_result = run_caption(st.session_state.img)
            caption = st.session_state.caption_result["answer"]

            fig, rel, objs, semantic_interactions = run_semantic_graph(st.session_state.img, caption)
            st.session_state.semantic_result = (fig, rel, objs, semantic_interactions)

    if st.session_state.semantic_result:
        fig, rel, _, semantic_interactions = st.session_state.semantic_result
        st.pyplot(fig)

        st.subheader("Subtitle used")
        st.write(st.session_state.caption_result["answer"])

        st.subheader("Physical Relations (next_to / holding / etc.)")
        st.json(rel)

        st.subheader("Semantic Interactions extracted from the caption")
        st.json(semantic_interactions)

# Prolog Representation
elif mode == "Prolog Representation":
    st.header("🧩 Prolog Representation")

    if st.button("Generate Prolog"):
        if not st.session_state.caption_result:
            st.warning("First generate the Captioning.")
        else:
            from prolog_representation import to_prolog
            caption = st.session_state.caption_result["answer"]

            # Gera o código Prolog e relações
            prolog_code, _ = to_prolog(caption=caption)
            st.session_state.prolog_result = prolog_code

            # --- Mostra primeiro os fatos Prolog ---
            st.subheader("Prolog Facts")
            st.code(st.session_state.prolog_result, language="prolog")

            # --- Depois executa consultas Prolog usando pyswip ---
            try:
                from pyswip import Prolog
                prolog = Prolog()

                # Salva o código Prolog em arquivo temporário
                with open("scene.pl", "w") as f:
                    f.write(prolog_code)

                # Carrega o arquivo no Prolog
                prolog.consult("scene.pl")

                st.subheader("Results of Generic Rules")

                # talking(X,Y)
                talking_results = list(prolog.query("talking(X,Y)"))
                if talking_results:
                    st.markdown("**talking(X,Y):**")
                    for sol in talking_results:
                        st.write(f"X = {sol['X']}, Y = {sol['Y']}")
                else:
                    st.write("**talking(X,Y):** No results")

                # near(X,Y)
                near_results = list(prolog.query("near(X,Y)"))
                if near_results:
                    st.markdown("**near(X,Y):**")
                    for sol in near_results:
                        st.write(f"X = {sol['X']}, Y = {sol['Y']}")
                else:
                    st.write("**near(X,Y):** No results")

                # located(X,Place)
                located_results = list(prolog.query("located(X,Place)"))
                if located_results:
                    st.markdown("**located(X,Place):**")
                    for sol in located_results:
                        st.write(f"X = {sol['X']}, Place = {sol['Place']}")
                else:
                    st.write("**located(X,Place):** No results")

            except Exception as e:
                st.error(f"Error running Prolog: {e}")
