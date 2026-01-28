import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from sklearn.cluster import KMeans

from models import load_model
from preprocessors.lion_preprocessors import ImageEvalProcessor


# ------------------------------------------------
# Configuração visual
# ------------------------------------------------
ALPHA = 100
CLUSTER_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255)
]


# ------------------------------------------------
# Lazy loading dos modelos
# ------------------------------------------------
@st.cache_resource
def load_models(yolo_weights="yolov8s-seg.pt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    yolo_model = YOLO(yolo_weights)

    lion_model = load_model(
        "lion_t5",
        "flant5xl",
        is_eval=True,
        device=device
    )
    lion_preprocessor = ImageEvalProcessor()

    return yolo_model, lion_model, device


# ------------------------------------------------
# Função principal
# ------------------------------------------------
def run_clustering_lion(
    image_path: str,
    num_clusters: int = 3,
    conf_threshold: float = 0.25,
    yolo_weights: str = "yolov8s-seg.pt"
):
    yolo_model, lion_model, device = load_models(yolo_weights)

    # Abrir imagem
    img = Image.open(image_path).convert("RGBA")
    width, height = img.size

    # ---------------- YOLO SEG ----------------
    results = yolo_model(
        image_path,
        imgsz=1280,
        conf=conf_threshold,
        iou=0.5,
        verbose=False
    )[0]

    masks = []

    for i in range(len(results.boxes)):
        mask = results.masks.data[i].cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8)

        mask_pil = Image.fromarray(mask * 255).resize(
            (width, height),
            resample=Image.NEAREST
        )
        masks.append(np.array(mask_pil) > 128)

    if len(masks) == 0:
        return img.convert("RGB"), []

    # ---------------- CLUSTERING ----------------
    if len(masks) > 1:
        centers = [
            [xs.mean(), ys.mean()]
            for m in masks
            for ys, xs in [np.where(m)]
        ]
        K = min(num_clusters, len(centers))
        cluster_ids = KMeans(
            n_clusters=K,
            random_state=0,
            n_init=10
        ).fit_predict(np.array(centers))
    else:
        cluster_ids = np.array([0])

    # ---------------- LION TAGS ----------------
    tags = lion_model.generate_tags(img)

    cluster_to_tags = {cid: set() for cid in set(cluster_ids)}
    for cid in cluster_to_tags:
        cluster_to_tags[cid].update(tags)

    # ---------------- OVERLAY ----------------
    overlay = np.zeros((height, width, 4), dtype=np.uint8)

    for mask, cid in zip(masks, cluster_ids):
        color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
        overlay[mask] = [*color, ALPHA]

    img_result = Image.alpha_composite(img, Image.fromarray(overlay))

    # ---------------- LEGENDA ----------------
    draw = ImageDraw.Draw(img_result)
    font = ImageFont.load_default()

    x, y = 10, 10
    for cid, tags in cluster_to_tags.items():
        color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
        text = f"Cluster {cid}: {', '.join(tags)}"

        draw.rectangle(
            [x - 2, y - 2, x + len(text) * 6 + 6, y + 14],
            fill=(*color, 200)
        )
        draw.text((x, y), text, fill=(255, 255, 255), font=font)
        y += 18

    return img_result.convert("RGB"), cluster_to_tags


if __name__ == "__main__":
    image_path = "images/COCO_train2014_000000533220.jpg"

    img_result, clusters = run_clustering_lion(
        image_path=image_path,
        num_clusters=3
    )

    img_result.save("cluster_result.png")
    print("Clusters:", clusters)
    print("Imagem salva como cluster_result.png")
