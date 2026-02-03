# caption.py
import gc
import torch
from PIL import Image

# modelos CENTRALIZADOS
from model_registry import get_models


# ------------------------------------------------
# Função principal
# ------------------------------------------------
def run_caption(img: Image.Image) -> dict:
    """
    Recebe uma imagem PIL e retorna:
    - question
    - tags (LION)
    - answer (caption)
    """

    # 🔥 MODELOS ÚNICOS DA APP
    models = get_models()
    lion = models["lion"]
    processor = models["processor"]
    device = models["device"]

    question = "Please describe the image."

    img = img.convert("RGB")

    with torch.no_grad():

        # -------- TAGS LION --------
        tags = lion.generate_tags(img)

        # -------- PREPROCESS --------
        processed_img = processor(img).unsqueeze(0).to(device)

        # -------- CAPTION --------
        output = lion.generate({
            "image": processed_img,
            "question": [question],
            "tags": [tags],
            "category": "image_level",
        })

        answer = output if isinstance(output, str) else output[0]

    # 🧹 LIMPEZA EXPLÍCITA (CRÍTICA)
    del processed_img, output
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "question": question,
        "tags": tags,
        "answer": answer
    }
