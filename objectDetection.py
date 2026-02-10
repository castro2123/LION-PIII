import re
from PIL import ImageDraw
import torch

from model_registry import get_models


# =========================
# Função principal
# =========================
def run_lion_qa(img, question):

    # 🔹 obter modelos partilhados
    models = get_models()
    lion = models["lion"]
    processor = models["processor"]
    device = models["device"]

    # 🔹 preprocess
    processed_img = processor(img)

    # 🔹 gerar tags globais
    tags = lion.generate_tags(img)

    # ==================================================
    # 1️⃣ Pergunta de regiões (bounding boxes)
    # ==================================================
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

    # ==================================================
    # 2️⃣ Resposta textual baseada nas boxes
    # ==================================================
    if bboxes_pixels:
        boxes_str = "; ".join(
            [f"[{x1},{y1},{x2},{y2}]" for x1, y1, x2, y2 in bboxes_pixels]
        )
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
