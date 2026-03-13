import cv2
import gc
import torch
import subprocess
import os
import numpy as np
from PIL import Image
import streamlit as st
from clustering import run_clustering_lion
import time


def run_clustering_video_streamlit(
    video_path: str,
    frame_interval: int = 1,
    conf_thresh: float = 0.5,
    display_delay: float = 0.03
):
    """
    Clustering de vídeo com suavização temporal (anti-flicker).
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video file")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Vídeo temporário AVI
    temp_output = "temp_clustering.avi"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

    frame_results = []
    frame_number = 0

    stframe = st.empty()

    # 🔥 Variáveis anti-flicker
    last_processed_frame = None
    prev_frame_out = None
    alpha = 0.7  # peso do frame atual (0.6–0.8 ideal)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % frame_interval == 0:

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Clustering normal
            result_img, clusters = run_clustering_lion(
                img,
                conf_thresh=conf_thresh
            )

            frame_out = cv2.cvtColor(
                np.array(result_img),
                cv2.COLOR_RGB2BGR
            )

            # 🔥 Blur leve para reduzir micro variações
            frame_out = cv2.GaussianBlur(frame_out, (3, 3), 0)

            # 🔥 Suavização temporal real
            if prev_frame_out is not None:
                frame_out = cv2.addWeighted(
                    frame_out, alpha,
                    prev_frame_out, 1 - alpha,
                    0
                )

            prev_frame_out = frame_out.copy()
            last_processed_frame = frame_out.copy()

            # Mostrar no Streamlit
            stframe.image(frame_out, channels="BGR")

            if display_delay > 0:
                time.sleep(display_delay)

            frame_results.append({
                "frame_number": frame_number,
                "clusters": clusters
            })

        else:
            # 🔥 Reutiliza último frame processado
            if last_processed_frame is not None:
                frame_out = last_processed_frame.copy()
            else:
                frame_out = frame

        out.write(frame_out)
        frame_number += 1

    cap.release()
    out.release()

    # Converter AVI → MP4 H264
    final_output = "clustering_video.mp4"
    try:
        subprocess.run([
            "ffmpeg",
            "-y",
            "-i", temp_output,
            "-vcodec", "libx264",
            "-acodec", "aac",
            "-pix_fmt", "yuv420p",
            final_output
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if os.path.exists(temp_output):
            os.remove(temp_output)

    except Exception as e:
        print("Erro FFmpeg:", e)
        return temp_output, frame_results

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return final_output, frame_results
