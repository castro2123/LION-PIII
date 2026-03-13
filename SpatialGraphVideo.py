import cv2
from PIL import Image
import numpy as np
import torch
import gc
from SpatialGraph import run_spatial_graph
from matplotlib.backends.backend_agg import FigureCanvasAgg

def fig_to_frame(fig):
    """Converte matplotlib figure → numpy array RGB"""
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3]  # RGBA -> RGB
    return img

def generate_spatial_graph_frames(video_path, frame_interval=10):
    """
    Processa vídeo e retorna lista de dicionários:
    [{"frame_number": int, "image": PIL.Image, "graph_fig": matplotlib.figure, "relations": list}]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    frames_data = []
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % frame_interval == 0:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            fig, relations = run_spatial_graph(img)
            frames_data.append({
                "frame_number": frame_number,
                "image": img,
                "graph_fig": fig,
                "relations": relations
            })

        frame_number += 1

    cap.release()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return frames_data