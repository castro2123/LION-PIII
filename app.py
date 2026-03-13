from SpatialGraphVideo import generate_spatial_graph_frames
from objectDetection_Video import run_lion_qa_video
from SemanticGraphVideo import generate_semantic_graph_frames
import streamlit as st
from PIL import Image
import torch
import gc
import matplotlib.pyplot as plt
import pandas as pd

# --------------------------
# Imports dos módulos
# --------------------------
from boudingBox import run_lion_yolo
from clustering import run_clustering_lion
from imageCaption import run_caption
from SpatialGraph import run_spatial_graph
from SemanticGraph import run_semantic_graph  # versão com grafo unificado
from objectDetection import run_lion_qa
from videoCaption import run_video_caption
from boundingBox_video import run_yolo_video_fast
from clustering_video import run_clustering_video_streamlit
from utils import show_result_image
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
    "Upload image or video",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)

if uploaded_file:

    file_type = uploaded_file.type

    if "video" in file_type:

        video_bytes = uploaded_file.getvalue()
        video_key = hash(video_bytes)

        if "video_key" not in st.session_state or st.session_state.video_key != video_key:

            clear_memory()

            st.session_state.video_file = uploaded_file
            st.session_state.video_key = video_key
            st.session_state.img = None

            # 🔥 limpar resultados apenas quando o vídeo muda
            for key in [
                "video_results",
                "output_video",
                "video_bbox_results",
                "output_video_bbox",
                "video_clustering_results",
                "output_video_clustering",
                "video_qa_results",
                "video_qa_appearance",
                "output_video_qa",
                "output_graph_video"
            ]:
                st.session_state[key] = None

        # Modos disponíveis para vídeo
        available_modes = [
            "Video Caption",
            "Bounding Box Video",
            "Clustering Video",
            "Interactive Video QA",
            "Spatial Graph Video",
            "Semantic Graph Video"

        ]

    else:
        # Guardar imagem
        img = Image.open(uploaded_file).convert("RGB")
        img_key = hash(img.tobytes())

        if "img_key" not in st.session_state or st.session_state.img_key != img_key:
            clear_memory()
            st.session_state.img = img
            st.session_state.img_key = img_key

            # Limpar resultados de imagem anteriores
            for key in [
                "bbox_result",
                "cluster_result",
                "caption_result",
                "qa_result",
                "spatial_result",
                "semantic_result",
                "prolog_result",
                "video_results",
                "video_file",
                "output_video",
                "active_mode",
                "output_video_qa"
            ]:
                st.session_state[key] = None

        # Modos disponíveis para imagem
        available_modes = [
            "Caption",
            "Bounding Box",
            "Clustering",
            "Interactive LION QA",
            "Spatial Scene Graph",
            "Semantic Scene Graph",
            "Prolog Representation"
        ]

else:
    st.info("Upload an image or video to continue")
    st.stop()

# ==========================
# Selector de modo
# ==========================
mode = st.radio(
    "Choose mode",
    available_modes,
    horizontal=True
)

# ==========================
# Estado inicial
# ==========================
if "active_mode" not in st.session_state:
    st.session_state.active_mode = None


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
# ==========================================
# Semantic Scene Graph (Caption Only)
# ==========================================
elif mode == "Semantic Scene Graph":
    st.header("Semantic Scene Graph (Caption Only)")

    # Garantir que a chave existe no session_state
    if "semantic_result" not in st.session_state:
        st.session_state.semantic_result = None

    if st.button("Generate Semantic Graph"):

        if "img" not in st.session_state or st.session_state.img is None:
            st.warning("Please upload an image first.")
        else:
            with st.spinner("Generating caption + semantic graph..."):
                fig, relations, caption = run_semantic_graph(st.session_state.img)

                st.session_state.semantic_result = {
                    "fig": fig,
                    "relations": relations,
                    "caption": caption
                }

    # Mostrar resultados apenas se existirem
    if st.session_state.semantic_result is not None:

        r = st.session_state.semantic_result

        st.pyplot(r["fig"])
        plt.close(r["fig"])  # evita memory leak

        st.subheader("Caption Used (Auto Generated)")
        st.write(r["caption"])

        st.subheader("Semantic Relations extracted from caption")
        st.json([
            {"subject": s, "predicate": p, "object": o}
            for (s, p, o) in r["relations"]
        ])


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


# ===============================
# Streamlit - Video Caption
# ===============================
elif mode == "Video Caption":
    st.header("🎬 Video Caption (Ultra Fast Optimized)")

    # Inicializar session_state
    for key in ["video_results", "output_video", "video_file"]:
        if key not in st.session_state:
            st.session_state[key] = None

    # Upload do vídeo
    uploaded_video = st.file_uploader(
        "Upload a video file",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:
        st.session_state.video_file = uploaded_video
        st.session_state.video_results = None
        st.session_state.output_video = None

    if st.session_state.video_file is None:
        st.warning("Please upload a video file.")
        st.stop()

    # Configurações
    frame_interval = st.slider(
        "Process 1 frame every N frames",
        min_value=30,
        max_value=300,
        value=60,
        step=10
    )

    # Botão gerar legendas
    if st.button("Generate Video Captions"):
        with st.spinner("Processing video..."):
            temp_path = "temp_video.mp4"
            with open(temp_path, "wb") as f:
                f.write(st.session_state.video_file.getbuffer())

            results, output_video = run_video_caption(
                temp_path,
                frame_interval=frame_interval
            )

            st.session_state.video_results = results
            st.session_state.output_video = output_video

    # Mostrar vídeo centralizado e menor
    if st.session_state.output_video is not None:
        st.subheader("🎥 Captioned Video")
        col1, col2, col3 = st.columns([1, 1.5, 1])
        with col2:
            st.video(st.session_state.output_video, start_time=0)

        # Mostrar captions
        st.subheader("Generated Captions")
        if st.session_state.video_results:
            for r in st.session_state.video_results:
                st.markdown(f"**Time {r['timestamp']}s**")
                st.write(r["caption"])
                st.divider()

elif mode == "Bounding Box Video":
    st.header("🎬 Video Bounding Boxes")

    # Inicializar estado
    for key in ["video_bbox_results", "output_video_bbox"]:
        if key not in st.session_state:
            st.session_state[key] = None

    # Upload
    uploaded_video = st.file_uploader(
        "Upload a video file",
        type=["mp4", "avi", "mov"],
        key="bbox_video_upload"
    )

    if uploaded_video is not None:
        st.session_state.video_file = uploaded_video
        st.session_state.video_bbox_results = None
        st.session_state.output_video_bbox = None

    if "video_file" not in st.session_state or st.session_state.video_file is None:
        st.warning("Please upload a video file.")
        st.stop()

    # Slider para acelerar
    frame_interval = st.slider(
        "Process 1 frame every N frames",
        min_value=1,
        max_value=60,
        value=1
    )

    if st.button("Generate Bounding Boxes for Video"):

        with st.spinner("Processing video..."):

            temp_path = "temp_video_bbox.mp4"
            with open(temp_path, "wb") as f:
                f.write(st.session_state.video_file.getbuffer())

            output_path, frame_results = run_yolo_video_fast(
                temp_path,
                conf_threshold=0.25,
                frame_interval=frame_interval
            )

            st.session_state.video_bbox_results = frame_results
            st.session_state.output_video_bbox = output_path

    # Mostrar vídeo corretamente
    if st.session_state.output_video_bbox is not None:

        st.subheader("🎥 Video with Bounding Boxes")

        with open(st.session_state.output_video_bbox, "rb") as f:
            video_bytes = f.read()

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.video(video_bytes)

        # Mostrar tabela leve
        st.subheader("Sample Frame Detections")

        if st.session_state.video_bbox_results:

            for r in st.session_state.video_bbox_results[:5]:

                st.markdown(f"**Frame {r['frame_number']}**")

                if r["detections"]:
                    df = pd.DataFrame(r["detections"])
                    st.dataframe(df, use_container_width=True)
                else:
                    st.write("No detections")

                st.divider()

if mode == "Clustering Video":
    st.header("🎬 Video Clustering")

    uploaded_video = st.file_uploader(
        "Upload video for clustering",
        type=["mp4","avi","mov"]
    )

    if uploaded_video is not None:
        st.session_state.video_file = uploaded_video

    if st.session_state.video_file is None:
        st.warning("Please upload a video.")
        st.stop()

    frame_interval = st.slider("Process 1 frame every N frames", 1, 60, 5)

    if st.button("Generate Clustering Video"):
        temp_path = "temp_video_clustering.mp4"
        with open(temp_path, "wb") as f:
            f.write(st.session_state.video_file.getbuffer())

        output_path, frame_results = run_clustering_video_streamlit(
            temp_path,
            frame_interval=frame_interval,
            conf_thresh=0.5
        )

        st.session_state.output_video_clustering = output_path
        st.session_state.video_clustering_results = frame_results

    # Mostrar vídeo
    if st.session_state.get("output_video_clustering") is not None:
        with open(st.session_state.output_video_clustering, "rb") as f:
            st.video(f.read())
            st.subheader("Sample Frame Clusters")

        for r in st.session_state.video_clustering_results[:5]:
            st.markdown(f"**Frame {r['frame_number']}**")
            st.json(r["clusters"])

elif mode == "Interactive Video QA":
    st.header("🎬 Interactive Video QA (Object Detection + LION QA)")

    # Upload
    uploaded_video = st.file_uploader(
        "Upload a video file",
        type=["mp4", "avi", "mov"],
        key="qa_video_upload"
    )

    if uploaded_video is not None:
        st.session_state.video_file = uploaded_video
        st.session_state.video_qa_results = None
        st.session_state.output_video_qa = None

    if "video_file" not in st.session_state or st.session_state.video_file is None:
        st.warning("Please upload a video file.")
        st.stop()

    # Pergunta do usuário
    question = st.text_input(
        "Type your question about the video.",
        placeholder="Ex: Is there a knife in the video?"
    )

    frame_interval = st.slider(
        "Process 1 frame every N frames",
        min_value=1,
        max_value=60,
        value=5
    )

    if st.button("Run LION QA on Video") and question:
        with st.spinner("Processing video..."):
            temp_path = "temp_video_qa.mp4"
            with open(temp_path, "wb") as f:
                f.write(st.session_state.video_file.getbuffer())

            # ======================
            # Chamada da função QA
            # ======================
            output_path, frame_results, object_appearance = run_lion_qa_video(
                temp_path,
                question,
                frame_interval=frame_interval,
                display_delay=0  # pode colocar >0 se quiser ver frames no Streamlit
            )

            st.session_state.output_video_qa = output_path
            st.session_state.video_qa_results = frame_results
            st.session_state.video_qa_appearance = object_appearance

    # ======================
    # Mostrar vídeo e resultados
    # ======================
    if st.session_state.output_video_qa is not None:
        st.subheader("🎥 Video with Bounding Boxes (LION QA)")
        with open(st.session_state.output_video_qa, "rb") as f:
            st.video(f.read())

        st.subheader("Frames where object appears")
        if st.session_state.video_qa_appearance:
            for r in st.session_state.video_qa_appearance[:10]:  # mostra os primeiros 10
                st.markdown(f"**Frame {r['frame_number']} ({r['timestamp']:.2f}s)**")
                st.write("Answer:", r["answer"])
                st.json(r["bboxes"])
                st.divider()
        else:
            st.write("No objects detected matching your question.")

elif mode == "Spatial Graph Video":

    st.header("🎬 Interactive Spatial Graph Video Viewer")

    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"], key="graph_video_upload")

    if uploaded_video:
        temp_path = "temp_graph_input.mp4"
        with open(temp_path, "wb") as f:
            f.write(uploaded_video.getbuffer())

        frame_interval = st.slider("Process 1 frame every N frames", 1, 60, 10)

        # Botão para processar o vídeo
        if st.button("Generate Spatial Graphs for Video"):
            with st.spinner("Processing video frames..."):
                frames_data = generate_spatial_graph_frames(temp_path, frame_interval)
                st.session_state.frames_data = frames_data
                st.success(f"Processed {len(frames_data)} frames!")

        # Mostrar grafo interativo
        if "frames_data" in st.session_state:
            frames_data = st.session_state.frames_data

            # Slider para selecionar frame
            frame_nums = [f["frame_number"] for f in frames_data]
            selected_frame = st.select_slider("Select frame", options=frame_nums)

            frame_info = next(f for f in frames_data if f["frame_number"] == selected_frame)

            st.subheader(f"Frame {selected_frame}")
            st.image(frame_info["image"], caption="Original Frame", use_column_width=True)

            st.subheader("Spatial Graph")
            st.pyplot(frame_info["graph_fig"])

            st.subheader("Relations Extracted")
            st.json(frame_info["relations"])

elif mode == "Semantic Graph Video":

    st.header("🎬 Interactive Semantic Graph Video Viewer")

    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"], key="semantic_graph_video_upload")

    if uploaded_video:
        temp_path = "temp_video_semantic.mp4"
        with open(temp_path, "wb") as f:
            f.write(uploaded_video.getbuffer())

        frame_interval = st.slider("Process 1 frame every N frames", 1, 60, 10)

        if st.button("Generate Semantic Graphs for Video"):
            with st.spinner("Processing video frames..."):
                frames_data = generate_semantic_graph_frames(temp_path, frame_interval)
                st.session_state.frames_data = frames_data
                st.success(f"Processed {len(frames_data)} frames!")

        # Interatividade frame a frame
        if "frames_data" in st.session_state:
            frames_data = st.session_state.frames_data
            frame_nums = [f["frame_number"] for f in frames_data]
            selected_frame = st.select_slider("Select frame", options=frame_nums)

            frame_info = next(f for f in frames_data if f["frame_number"] == selected_frame)

            st.subheader(f"Frame {selected_frame}")
            st.image(frame_info["image"], caption="Original Frame", use_column_width=True)

            st.subheader("Semantic Graph")
            st.pyplot(frame_info["graph_fig"])

            st.subheader("Caption")
            st.write(frame_info["caption"])

            st.subheader("Relations Extracted")
            st.json(frame_info["relations"])

