# boundingBox.py
import sys
import gc
import torch
from PIL import Image, ImageDraw, ImageFont

# modelos CENTRALIZADOS
from model_registry import get_models


# ------------------------------------------------
# Heurística simples de matching YOLO <-> LION tags
# ------------------------------------------------
def match_tag_to_class(yolo_label, tags):
    yolo_label = yolo_label.lower()
    for tag in tags:
        if yolo_label in tag.lower() or tag.lower() in yolo_label:
            return tag
    return yolo_label


# ------------------------------------------------
# Função principal
# ------------------------------------------------
def run_lion_yolo(
    img: Image.Image,
    conf_threshold: float = 0.25
):
    """
    Recebe uma imagem PIL e retorna:
    - img_out: imagem com bounding boxes
    - tags: tags geradas pelo LION
    """

    # 🔥 MODELOS ÚNICOS DA APP
    models = get_models()
    yolo = models["yolo_det"]
    lion = models["lion"]
    device = models["device"]

    img = img.convert("RGB")

    with torch.no_grad():

        # -------- LION TAGS --------
        tags = lion.generate_tags(img)

        # -------- YOLO DETECTION --------
        results = yolo.predict(
            img,
            conf=conf_threshold,
            device=0 if device.startswith("cuda") else "cpu",
            verbose=False
        )

        boxes = results[0].boxes
        class_names = results[0].names

        img_out = img.copy()
        draw = ImageDraw.Draw(img_out)
        font = ImageFont.load_default()

        if boxes is not None:
            for i in range(len(boxes)):
                x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
                score = float(boxes.conf[i])
                class_id = int(boxes.cls[i])

                yolo_label = class_names[class_id]
                final_label = match_tag_to_class(yolo_label, tags)

                # Bounding Box
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

                # Texto
                text = f"{final_label} ({score:.2f})"
                bbox = draw.textbbox((x1, y1), text, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

                draw.rectangle(
                    [x1, y1 - th, x1 + tw, y1],
                    fill="red"
                )

                draw.text(
                    (x1, y1 - th),
                    text,
                    fill="white",
                    font=font
                )

    # 🧹 LIMPEZA EXPLÍCITA (CRÍTICA)
    torch.cuda.empty_cache()
    gc.collect()

    return img_out, tags


# ------------------------------------------------
# PONTO DE ENTRADA (CLI)
# ------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso correto:")
        print("  python boundingBox.py caminho/para/imagem.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    img = Image.open(image_path).convert("RGB")

    img_out, tags = run_lion_yolo(img)

    print("\n🧠 Tags LION detectadas:")
    print(tags)

    img_out.show()
