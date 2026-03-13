# boundingBox.py
import gc
import torch
from PIL import Image, ImageDraw, ImageFont
from model_registry import get_models


def run_lion_yolo(
    img: Image.Image,
    conf_threshold: float = 0.25
):
    """
    Retorna:
    - img_out: imagem com bounding boxes desenhadas
    - detections: lista de dicts com {label, score, bbox}
    - tags: tags LION (uso semântico)
    """

    models = get_models()
    yolo = models["yolo_det"]   # 🔥 YOLO-WORLD
    lion = models["lion"]
    device = models["device"]

    img = img.convert("RGB")

    with torch.no_grad():
        # 🔹 Tags LION
        tags = lion.generate_tags(img)

        # 🔹 YOLO-World
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
        font = ImageFont.load_default()

        detections = []

        if boxes is not None:
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

                # desenhar box
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

                # texto
                tw, th = draw.textbbox((0, 0), text, font=font)[2:]
                draw.rectangle([x1, y1 - th, x1 + tw, y1], fill="red")
                draw.text((x1, y1 - th), text, fill="white", font=font)

    torch.cuda.empty_cache()
    gc.collect()

    return img_out, detections, tags
