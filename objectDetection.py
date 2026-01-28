import re
import os
import torch
from PIL import Image, ImageDraw
from models import load_model
from preprocessors.lion_preprocessors import ImageEvalProcessor

# 🔧 Configurar dispositivo
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 1️⃣ Carregar modelo LION
print("Carregando modelo LION...")
lion_model = load_model(
    name="lion_t5",
    model_type="flant5xl",
    is_eval=True,
    device=device
)
lion_preprocessor = ImageEvalProcessor()
print("Modelo carregado!\n")

# 2️⃣ Criar pasta de resultados
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# 3️⃣ Abrir imagem
img_path = input("Caminho da imagem: ").strip()
img = Image.open(img_path).convert("RGB")
processed_img = lion_preprocessor(img)

# 4️⃣ Gerar tags automaticamente
tags = lion_model.generate_tags(img)

# 5️⃣ Loop interativo
counter = 1
while True:
    question = input("\nDigite sua pergunta sobre a imagem (ou 'sair' para encerrar): ").strip()
    if question.lower() in ["sair", "exit", "quit"]:
        print("Encerrando...")
        break

    # 🔹 6️⃣ Pedir todas as instâncias do objeto mencionado
    region_question = f"Find all objects in the image related to the question and return bounding boxes for each instance: {question}"
    output_bbox = lion_model.generate({
        "image": processed_img.unsqueeze(0).to(device),
        "question": [region_question],
        "tags": [tags],
        "category": "region_level",
    })

    matches = re.findall(r"\[([0-9., ]+)\]", output_bbox[0])
    bboxes_pixels = []

    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)

    if matches:
        for b in matches:
            bbox = eval("[" + b + "]")
            x1 = int(bbox[0] * img.width)
            y1 = int(bbox[1] * img.height)
            x2 = int(bbox[2] * img.width)
            y2 = int(bbox[3] * img.height)
            draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 0, 0), width=3)
            bboxes_pixels.append([x1, y1, x2, y2])
        bbox_text = str(bboxes_pixels)
    else:
        bbox_text = "Nenhuma bounding box encontrada."

    # 🔹 7️⃣ Resposta textual detalhada baseada em todas as boxes
    if bboxes_pixels:
        boxes_str = "; ".join([f"[{x1},{y1},{x2},{y2}]" for x1, y1, x2, y2 in bboxes_pixels])
        prompt_text = (
            f"Describe in a full detailed sentence all objects in the image corresponding to the bounding boxes: {boxes_str}. "
            f"Answer the question: {question}"
        )

        output_text = lion_model.generate({
            "image": processed_img.unsqueeze(0).to(device),
            "question": [prompt_text],
            "tags": [tags],
            "category": "image_level",
        })
        resposta_textual = output_text[0]
    else:
        prompt_text = (
            f"Describe in a full detailed sentence the scene of the image, answering the question: {question}"
        )
        output_text = lion_model.generate({
            "image": processed_img.unsqueeze(0).to(device),
            "question": [prompt_text],
            "tags": [tags],
            "category": "image_level",
        })
        resposta_textual = output_text[0]

    print("\nResposta textual detalhada:", resposta_textual)
    print("Bounding boxes (pixels):", bbox_text)

    # 🔹 8️⃣ Salvar resultados
    base_name = f"result_{counter}"
    img_copy.save(os.path.join(results_dir, f"{base_name}.png"))

    with open(os.path.join(results_dir, f"{base_name}.txt"), "w") as f:
        f.write(f"Pergunta: {question}\n")
        f.write(f"Tags: {tags}\n")
        f.write(f"Resposta textual detalhada: {resposta_textual}\n")
        f.write(f"Bounding boxes (pixels): {bbox_text}\n")

    print(f"Resultado salvo em {results_dir}/{base_name}.png e .txt")

    # Mostrar imagem com bounding boxes
    img_copy.show()

    counter += 1
