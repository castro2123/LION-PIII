import gc
import torch
import numpy as np
from PIL import Image
from model_registry import get_models

# ------------------------------------------------
# Heurística simples: filtra tags globais por classe
# ------------------------------------------------
def filter_tags_by_class(tags, cls_name):
    """
    Mantém apenas tags que provavelmente se relacionam com a classe YOLO
    """
    cls_name_lower = cls_name.lower()
    filtered = []
    for t in tags:
        t_lower = t.lower()
        if cls_name_lower in t_lower or cls_name_lower.split()[0] in t_lower:
            filtered.append(t)
    return filtered

# ------------------------------------------------
# Pipeline completo: tags globais → clusters
# ------------------------------------------------
def run_clustering_lion(img: Image.Image, conf_thresh=0.5, crop_size=(224, 224)):
    """
    1. Gera tags globais com LION
    2. Detecta objetos com YOLO-SEG
    3. Atribui a cada cluster apenas tags globais relevantes
    4. Cria clusters separados para tags globais não atribuídas
    """
    ALPHA = 120
    COLORS = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (255, 128, 0), (128, 0, 255)
    ]
    MIN_SIZE = 40
    BG_COLOR = 128

    models = get_models()
    yolo = models["yolo_seg"]
    lion = models["lion"]

    img = img.convert("RGB")
    img_rgba = img.convert("RGBA")
    w, h = img.size

    with torch.no_grad():
        global_tags = lion.generate_tags(img)
        tags_assigned = set()
        res = yolo(img, conf=conf_thresh)[0]
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        clusters = {}

        if res.masks is not None:
            for i in range(len(res.masks.data)):
                m = res.masks.data[i].detach().cpu().numpy()
                m = (m > 0.5).astype(np.uint8)
                m = Image.fromarray(m * 255).resize((w, h), Image.NEAREST)
                m = np.array(m) > 0

                if not np.any(m):
                    continue

                # bbox exata
                ys, xs = np.where(m)
                x1, x2 = xs.min(), xs.max()
                y1, y2 = ys.min(), ys.max()
                if (x2 - x1) < MIN_SIZE or (y2 - y1) < MIN_SIZE:
                    continue

                # crop do objeto com fundo neutro
                crop_rgb = img.crop((x1, y1, x2, y2))
                crop_mask = m[y1:y2, x1:x2]
                crop_np = np.array(crop_rgb)
                crop_np[~crop_mask] = BG_COLOR
                crop = Image.fromarray(crop_np).resize(crop_size)

                # classe YOLO
                cls_id = int(res.boxes.cls[i])
                cls_name = yolo.names[cls_id]

                # ---------- Atribui tags globais relevantes ----------
                cluster_tags = []
                for t in global_tags:
                    if cls_name.lower() in t.lower():
                        cluster_tags.append(t)
                        tags_assigned.add(t)

                # overlay
                color = COLORS[i % len(COLORS)]
                overlay[m] = (*color, ALPHA)

                clusters[i] = {
                    "object_id": i,
                    "class": cls_name,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "tags": cluster_tags
                }

        # ---------- Cria clusters para tags globais não atribuídas ----------
        unassigned_tags = [t for t in global_tags if t not in tags_assigned]
        for t in unassigned_tags:
            clusters[len(clusters)] = {
                "object_id": len(clusters),
                "class": "tag_only",
                "bbox": None,
                "tags": [t]
            }

        # Compoe overlay final
        result = Image.alpha_composite(img_rgba, Image.fromarray(overlay))

    # Limpeza de memória
    del overlay, res
    torch.cuda.empty_cache()
    gc.collect()

    return result.convert("RGB"), clusters
