# ===============================
# Função de Video Caption
# ===============================
import gc
import cv2
import torch
import subprocess
import os
from PIL import Image
from textwrap import wrap
from model_registry import get_models

def run_video_caption(
        video_path: str,
        frame_interval: int = 60,   # processa 1 frame a cada 60 frames
        resize_size: int = 224,
        max_new_tokens: int = 20,
        use_tags: bool = False
):
    """
    Gera legendas para vídeo usando LION + OpenCV
    Compatível com Windows + Streamlit
    Gera MP4 H264 reproduzível no browser
    """

    # ===============================
    # Carregar modelos
    # ===============================
    models = get_models()
    lion = models["lion"]
    processor = models["processor"]
    device = models["device"]

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # ===============================
    # Abrir vídeo
    # ===============================
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Erro ao abrir vídeo.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ===============================
    # Criar vídeo temporário AVI
    # ===============================
    temp_output = "temp_captioned.avi"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

    frame_count = 0
    results = []
    current_caption = ""
    question = "Please describe the image using a single short sentence."

    # ===============================
    # Processamento frame a frame
    # ===============================
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb).resize((resize_size, resize_size))

            with torch.no_grad():
                processed = processor(pil_img).unsqueeze(0).to(device)

                if use_tags:
                    tags = lion.generate_tags(pil_img)
                else:
                    tags = [None]

                output = lion.generate({
                    "image": processed,
                    "question": [question],
                    "tags": tags,
                    "category": "image_level",
                    "num_beams": 1,
                    "max_new_tokens": max_new_tokens,
                    "do_sample": False
                })

                caption = output if isinstance(output, str) else output[0]
                current_caption = caption

                results.append({
                    "frame": frame_count,
                    "timestamp": round(frame_count / fps, 2),
                    "caption": caption
                })

                del processed, output

        # ===============================
        # Desenhar legenda no frame
        # ===============================
        if current_caption:
            # quebra o texto em várias linhas
            wrapped_text = wrap(current_caption, width=40)  # 40 caracteres por linha
            y0 = height - 60
            dy = 25  # espaçamento entre linhas
            for i, line in enumerate(wrapped_text):
                y = y0 + i*dy
                cv2.putText(
                    frame,
                    line,
                    (40, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,          # fonte menor
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    # ===============================
    # Converter para MP4 H264
    # ===============================
    final_output = "captioned_video.mp4"

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

        # remover temporário
        if os.path.exists(temp_output):
            os.remove(temp_output)

    except Exception as e:
        print("Erro ao converter com FFmpeg:", e)
        return results, temp_output  # fallback

    # ===============================
    # Limpeza memória
    # ===============================
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results, final_output
