# ===============================
# VIDEO CAPTIONING MELHORADO
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
        frame_interval: int = 15,     # analisa mais frames
        resize_size: int = 224,
        max_new_tokens: int = 40,    # legenda mais detalhada
        use_tags: bool = True
):
    """
    Gera legendas automáticas para vídeo usando LION.

    Melhorias:
    - legendas mais precisas
    - atualização mais frequente
    - melhor qualidade textual
    - compatível com Streamlit
    - exporta MP4 H264
    """

    # ===============================
    # CARREGAR MODELOS
    # ===============================
    models = get_models()

    lion = models["lion"]
    processor = models["processor"]
    device = models["device"]

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # ===============================
    # ABRIR VÍDEO
    # ===============================
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Erro ao abrir vídeo.")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        fps = 30

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ===============================
    # OUTPUT TEMPORÁRIO
    # ===============================
    temp_output = "temp_captioned.avi"

    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    out = cv2.VideoWriter(
        temp_output,
        fourcc,
        fps,
        (width, height)
    )

    # ===============================
    # VARIÁVEIS
    # ===============================
    frame_count = 0

    current_caption = ""

    results = []

    # PROMPT MELHORADO
    question = (
        "Describe the main action happening in this frame "
        "using one short precise sentence. "
        "Only mention actions and objects clearly visible."
    )

    # ===============================
    # LOOP PRINCIPAL
    # ===============================
    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # ===============================
        # ANALISAR FRAME
        # ===============================
        if frame_count % frame_interval == 0:

            try:

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                pil_img = Image.fromarray(frame_rgb)

                pil_img = pil_img.resize(
                    (resize_size, resize_size)
                )

                with torch.no_grad():

                    processed = processor(pil_img)

                    processed = processed.unsqueeze(0).to(device)

                    # ===============================
                    # TAGS OPCIONAIS
                    # ===============================
                    if use_tags:
                        tags = lion.generate_tags(pil_img)
                    else:
                        tags = [None]

                    # ===============================
                    # GERAR LEGENDA
                    # ===============================
                    output = lion.generate({
                        "image": processed,
                        "question": [question],
                        "tags": tags,
                        "category": "image_level",

                        # melhor qualidade
                        "num_beams": 3,
                        "max_new_tokens": max_new_tokens,
                        "do_sample": False
                    })

                    # normalizar output
                    if isinstance(output, list):
                        caption = output[0]
                    else:
                        caption = output

                    current_caption = caption

                    # guardar resultados
                    results.append({
                        "frame": frame_count,
                        "timestamp": round(frame_count / fps, 2),
                        "caption": caption
                    })

                    # limpeza
                    del processed
                    del output

            except Exception as e:
                print(f"Erro no frame {frame_count}: {e}")

        # ===============================
        # DESENHAR LEGENDA
        # ===============================
        if current_caption:

            # quebrar texto
            wrapped_text = wrap(current_caption, width=45)

            # altura inicial
            y0 = height - (30 * len(wrapped_text)) - 30

            for i, line in enumerate(wrapped_text):

                y = y0 + (i * 30)

                # sombra preta
                cv2.putText(
                    frame,
                    line,
                    (42, y + 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 0),
                    4,
                    cv2.LINE_AA
                )

                # texto principal
                cv2.putText(
                    frame,
                    line,
                    (40, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

        # ===============================
        # ESCREVER FRAME
        # ===============================
        out.write(frame)

        frame_count += 1

    # ===============================
    # FECHAR
    # ===============================
    cap.release()
    out.release()

    # ===============================
    # CONVERTER PARA MP4 H264
    # ===============================
    final_output = "captioned_video.mp4"

    try:

        subprocess.run([
            "ffmpeg",
            "-y",
            "-i", temp_output,

            # vídeo
            "-vcodec", "libx264",
            "-preset", "medium",
            "-crf", "23",

            # browser compatibility
            "-pix_fmt", "yuv420p",

            # áudio
            "-acodec", "aac",

            final_output

        ], stdout=subprocess.DEVNULL,
           stderr=subprocess.DEVNULL)

        # remover AVI temporário
        if os.path.exists(temp_output):
            os.remove(temp_output)

    except Exception as e:

        print("Erro FFmpeg:", e)

        return results, temp_output

    # ===============================
    # LIMPEZA MEMÓRIA
    # ===============================
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ===============================
    # RETORNO
    # ===============================
    return results, final_output