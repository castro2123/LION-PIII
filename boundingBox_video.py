import cv2
import gc
import torch
import subprocess
import os
from PIL import Image, ImageDraw, ImageFont
from model_registry import get_models
import numpy as np
from utils import show_result_image


def run_yolo_video_fast(
    video_path: str,
    conf_threshold: float = 0.25,
    frame_interval: int = 1
):
    """
    Processa vídeo frame a frame usando YOLO.
    Gera vídeo final H264 compatível com Streamlit.
    """

    models = get_models()
    yolo = models["yolo_det"]
    device = models["device"]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video file")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ===============================
    # Criar AVI temporário
    # ===============================
    temp_output = "temp_bbox.avi"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

    frame_results = []
    frame_number = 0
    font = ImageFont.load_default()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % frame_interval == 0:

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            with torch.no_grad():
                results = yolo.predict(
                    img,
                    conf=conf_threshold,
                    device=0 if device.startswith("cuda") else "cpu",
                    verbose=False
                )

            boxes = results[0].boxes
            names = results[0].names

            img_out = img.copy()
            draw = ImageDraw.Draw(img_out)
            detections = []

            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
                    score = float(boxes.conf[i])
                    cls_id = int(boxes.cls[i])
                    label = names[cls_id].lower()

                    detections.append({
                        "label": label,
                        "score": round(score, 3),
                        "bbox": [x1, y1, x2, y2]
                    })

                    text = f"{label} ({score:.2f})"
                    tw, th = draw.textbbox((0, 0), text, font=font)[2:]

                    draw.rectangle([x1, y1 - th, x1 + tw, y1], fill="red")
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                    draw.text((x1, y1 - th), text, fill="white", font=font)

            frame = cv2.cvtColor(np.array(img_out), cv2.COLOR_RGB2BGR)

            frame_results.append({
                "frame_number": frame_number,
                "detections": detections
            })

        out.write(frame)
        frame_number += 1

    cap.release()
    out.release()

    # ===============================
    # Converter para MP4 H264
    # ===============================
    final_output = "bbox_video.mp4"

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
        return temp_output, frame_results  # fallback

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return final_output, frame_results
