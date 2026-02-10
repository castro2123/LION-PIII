# SpatialGraph.py
import math
import itertools
import json
import re
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

from model_registry import get_models


# =========================
# CONFIG
# =========================
USE_LION = True        # 👈 muda para False se quiseres instantâneo
DIST_THRESHOLD = 220  # pixels


def run_spatial_graph(
    img: Image.Image,
    max_objects: int = 12,
    max_candidates: int = 40,
    batch_size: int = 2
):
    """
    Retorna:
    - fig: matplotlib figure
    - valid_relations: relações espaciais corretas
    """

    # ------------------------------
    # Modelos
    # ------------------------------
    models = get_models()
    yolo = models["yolo_det"]
    lion = models["lion"]
    processor = models["processor"]
    device = models["device"]

    # ------------------------------
    # Preprocess
    # ------------------------------
    ram_transform = T.Compose([
        T.Resize((384, 384)),
        T.ToTensor(),
        T.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    img_llm = processor(img).unsqueeze(0).to(device)
    img_ram = ram_transform(img).unsqueeze(0).to(device)

    # ------------------------------
    # YOLO
    # ------------------------------
    res = yolo(img, conf=0.3)[0]

    objects = []
    for i, box in enumerate(res.boxes[:max_objects]):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls = yolo.names[int(box.cls[0])]
        objects.append({
            "id": i,
            "name": cls.lower(),
            "box": (x1, y1, x2, y2)
        })

    # Renomear repetidos
    counts = {}
    for o in objects:
        counts[o["name"]] = counts.get(o["name"], 0) + 1
        o["name"] = f"{o['name']}_{counts[o['name']]}"

    # ------------------------------
    # Geometria
    # ------------------------------
    def center(b):
        return ((b[0]+b[2])/2, (b[1]+b[3])/2)

    def width(b): return b[2]-b[0]
    def height(b): return b[3]-b[1]

    def h_overlap(b1, b2):
        return max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))

    def v_overlap(b1, b2):
        return max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))

    def dist(c1, c2):
        return math.dist(c1, c2)

    # ------------------------------
    # Candidatos espaciais (CORRETOS)
    # ------------------------------
    candidates = []

    for o1, o2 in itertools.combinations(objects, 2):
        b1, b2 = o1["box"], o2["box"]
        c1, c2 = center(b1), center(b2)

        if dist(c1, c2) > DIST_THRESHOLD:
            continue

        ho = h_overlap(b1, b2)
        vo = v_overlap(b1, b2)

        min_w = min(width(b1), width(b2))
        min_h = min(height(b1), height(b2))

        # ---------- ABOVE / UNDER ----------
        if ho > 0.4 * min_w:
            if c1[1] < c2[1]:
                candidates.append((o1["name"], "above", o2["name"]))
                candidates.append((o2["name"], "under", o1["name"]))
            else:
                candidates.append((o2["name"], "above", o1["name"]))
                candidates.append((o1["name"], "under", o2["name"]))

        # ---------- NEXT TO ----------
        if vo > 0.4 * min_h:
            candidates.append((o1["name"], "next_to", o2["name"]))
            candidates.append((o2["name"], "next_to", o1["name"]))

        # ---------- ON ----------
        if ho > 0.5 * min_w:
            if abs(b1[3] - b2[1]) < 0.2 * min_h:
                candidates.append((o1["name"], "on", o2["name"]))
            elif abs(b2[3] - b1[1]) < 0.2 * min_h:
                candidates.append((o2["name"], "on", o1["name"]))

        # ---------- IN FRONT / BEHIND ----------
        depth = height(b1) - height(b2)
        if abs(depth) > 0.3 * min_h:
            if depth > 0:
                candidates.append((o1["name"], "in_front_of", o2["name"]))
                candidates.append((o2["name"], "behind", o1["name"]))
            else:
                candidates.append((o2["name"], "in_front_of", o1["name"]))
                candidates.append((o1["name"], "behind", o2["name"]))

    candidates = list(set(candidates))[:max_candidates]

    # ------------------------------
    # Validação com LION (opcional)
    # ------------------------------
    def validate_batch(batch):
        if not USE_LION:
            return [{"subject": s, "predicate": p, "object": o} for s,p,o in batch]

        question = f"""
Select ONLY the spatial relations that are visually correct.

Return ONLY valid JSON:
[
  {{"subject":"obj","predicate":"rel","object":"obj"}}
]

Candidates:
{json.dumps(batch)}
"""

        answer = lion.generate({
            "image": img_llm,
            "ram_image": img_ram,
            "question": [question],
            "category": "complex_reasoning"
        })[0]

        try:
            parsed = json.loads(answer)
            if isinstance(parsed, list):
                return parsed
        except:
            pass

        return [{"subject": s, "predicate": p, "object": o} for s,p,o in batch]

    valid_relations = []
    for i in range(0, len(candidates), batch_size):
        valid_relations.extend(validate_batch(candidates[i:i+batch_size]))

    valid_relations = list({
        (r["subject"], r["predicate"], r["object"])
        for r in valid_relations
    })

    valid_relations = [
        {"subject": s, "predicate": p, "object": o}
        for s,p,o in valid_relations
    ]

    # ------------------------------
    # Grafo
    # ------------------------------
    G = nx.DiGraph()
    for o in objects:
        G.add_node(o["name"])
    for r in valid_relations:
        G.add_edge(r["subject"], r["object"], relation=r["predicate"])

    fig = plt.figure(figsize=(14, 12))
    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=2600, node_color="#b3d9ff")
    nx.draw_networkx_edge_labels(G, pos, nx.get_edge_attributes(G, "relation"))
    plt.axis("off")

    return fig, valid_relations
