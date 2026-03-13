import cv2
import gc
import torch
import subprocess
import os
import numpy as np
from PIL import Image, ImageDraw
import streamlit as st
from model_registry import get_models
import time
import re

# ======================================================
# Função de QA para cada frame
# ======================================================
def run_lion_qa_frame(img, question):
    """
    Processa um frame de vídeo com LION QA, retornando:
        - imagem com bounding boxes
        - resposta textual
        - bboxes em pixels
    """
    models = get_models()
    lion = models["lion"]
    processor = models["processor"]
    device = models["device"]

    # preprocess
    processed_img = processor(img)

    # tags globais
    tags = lion.generate_tags(img)

    # pergunta de regiões
    region_question = (
        "Find all objects in the image related to the question and "
        f"return bounding boxes for each instance: {question}"
    )

    with torch.no_grad():
        output_bbox = lion.generate({
            "image": processed_img.unsqueeze(0).to(device),
            "question": [region_question],
            "tags": [tags],
            "category": "region_level",
        })

    matches = re.findall(r"\[([0-9., ]+)\]", output_bbox[0])

    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    bboxes_pixels = []

    for b in matches:
        bbox = eval("[" + b + "]")
        x1 = int(bbox[0] * img.width)
        y1 = int(bbox[1] * img.height)
        x2 = int(bbox[2] * img.width)
        y2 = int(bbox[3] * img.height)
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
        bboxes_pixels.append([x1, y1, x2, y2])

    # resposta textual baseada nas boxes
    if bboxes_pixels:
        boxes_str = "; ".join([f"[{x1},{y1},{x2},{y2}]" for x1, y1, x2, y2 in bboxes_pixels])
        prompt_text = (
            f"Describe in a full detailed sentence all objects in the image "
            f"corresponding to the bounding boxes: {boxes_str}. "
            f"Answer the question: {question}"
        )
    else:
        prompt_text = (
            f"Describe in a full detailed sentence the scene of the image, "
            f"answering the question: {question}"
        )

    with torch.no_grad():
        output_text = lion.generate({
            "image": processed_img.unsqueeze(0).to(device),
            "question": [prompt_text],
            "tags": [tags],
            "category": "image_level",
        })

    answer = output_text[0]
    return img_copy, answer, bboxes_pixels

# ======================================================
# Função de vídeo QA
# ======================================================
def run_lion_qa_video(
    video_path: str,
    question: str,
    frame_interval: int = 5,
    display_delay: float = 0.05
):
    """
    Processa vídeo frame a frame:
        - Executa run_lion_qa_frame
        - Retorna vídeo final com bounding boxes
        - Retorna lista de frames em que o objeto aparece
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video file")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_output = "temp_lionqa.avi"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

    stframe = st.empty()  # placeholder Streamlit
    frame_results = []
    object_appearance = []  # lista de (frame_number, timestamp, bboxes)
    frame_number = 0
    last_frame_out = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % frame_interval == 0:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_out, answer, bboxes = run_lion_qa_frame(img, question)
            frame_out = cv2.cvtColor(np.array(img_out), cv2.COLOR_RGB2BGR)

            # Suavização temporal para evitar flicker
            if last_frame_out is not None:
                frame_out = cv2.addWeighted(frame_out, 0.7, last_frame_out, 0.3, 0)
            last_frame_out = frame_out.copy()

            stframe.image(frame_out, channels="BGR")
            if display_delay > 0:
                time.sleep(display_delay)

            if bboxes:  # objeto encontrado
                timestamp = frame_number / fps
                object_appearance.append({
                    "frame_number": frame_number,
                    "timestamp": timestamp,
                    "bboxes": bboxes,
                    "answer": answer
                })

        else:
            # reutiliza último frame processado
            if last_frame_out is not None:
                frame_out = last_frame_out.copy()
            else:
                frame_out = frame

        out.write(frame_out)
        frame_results.append({
            "frame_number": frame_number,
            "bboxes": bboxes if frame_number % frame_interval == 0 else [],
            "answer": answer if frame_number % frame_interval == 0 else ""
        })

        frame_number += 1

    cap.release()
    out.release()

    # Converter AVI → MP4 H264
    final_output = "lionqa_video.mp4"
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", temp_output,
            "-vcodec", "libx264", "-acodec", "aac",
            "-pix_fmt", "yuv420p", final_output
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists(temp_output):
            os.remove(temp_output)
    except Exception as e:
        print("Erro FFmpeg:", e)
        final_output = temp_output

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return final_output, frame_results, object_appearance
