# semantic_graph.py
import gc
import re
import json
import math
import itertools
import torch
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

# modelos CENTRALIZADOS
from model_registry import get_models


def run_semantic_graph(
    img: Image.Image,
    dist_threshold: int = 160,
    max_objects: int = 12
):
    """
    Recebe uma imagem PIL e retorna:
    - fig: matplotlib figure com o grafo
    - relations: lista de relações validadas
    """

    # 🔥 MODELOS ÚNICOS DA APP
    models = get_models()
    yolo = models["yolo_det"]
    lion = models["lion"]
    processor = models["processor"]
    device = models["device"]

    img = img.convert("RGB")

    # ------------------------------
    # Pré-processamento
    # ------------------------------
    ram_transform = T.Compose([
        T.Resize((384, 384)),
        T.ToTensor(),
        T.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])

    with torch.no_grad():

        img_llm = processor(img).unsqueeze(0).to(device)
        img_ram = ram_transform(img).unsqueeze(0).to(device)

        # ------------------------------
        # YOLO detecção
        # ------------------------------
        res = yolo(img, conf=0.3)[0]

        objects = []
        for i, box in enumerate(res.boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls = yolo.names[int(box.cls[0])]
            objects.append({
                "id": i,
                "name": cls.lower(),
                "box": (x1, y1, x2, y2)
            })

        objects = objects[:max_objects]

        # ------------------------------
        # Renomear objetos repetidos
        # ------------------------------
        name_count = {}
        for o in objects:
            name = o["name"]
            name_count[name] = name_count.get(name, 0) + 1
            o["name"] = f"{name}_{name_count[name]}"

        # ------------------------------
        # Funções de geometria
        # ------------------------------
        def center(b): return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        def area(b): return (b[2] - b[0]) * (b[3] - b[1])
        def dist(b1, b2): return math.dist(center(b1), center(b2))
        def overlap(b1, b2):
            return max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))

        # ------------------------------
        # Gerar candidatos
        # ------------------------------
        candidates = []
        for o1, o2 in itertools.permutations(objects, 2):
            if o1["id"] == o2["id"]:
                continue

            b1, b2 = o1["box"], o2["box"]

            if overlap(b1, b2) > 0 and area(b1) < area(b2):
                candidates.append((o1["name"], "on", o2["name"]))

            if dist(b1, b2) < dist_threshold:
                candidates.append((o1["name"], "next to", o2["name"]))

            if "person" in o1["name"] and dist(b1, b2) < dist_threshold:
                candidates.append(
                    (o1["name"], "interacting with", o2["name"])
                )

        candidates = list(set(candidates))

        # ------------------------------
        # Validação com LION
        # ------------------------------
        question = f"""
You are a visual reasoning system.

From the list of candidate relations, select ONLY those that are
visually true in the image.

Output ONLY a JSON array:
[
  {{"subject": "object", "predicate": "relation", "object": "object"}}
]

Candidates:
{json.dumps(candidates, indent=2)}
"""

        answer = lion.generate({
            "image": img_llm,
            "ram_image": img_ram,
            "question": [question],
            "category": "complex_reasoning"
        })[0]

    # ------------------------------
    # Parse JSON robusto
    # ------------------------------
    def robust_parse(text, fallback):
        try:
            return json.loads(text)
        except Exception:
            matches = re.findall(r"\[[\s\S]*?\]", text)
            for m in matches:
                try:
                    return json.loads(m)
                except Exception:
                    pass
        return [
            {"subject": s, "predicate": p, "object": o}
            for s, p, o in fallback
        ]

    relations = robust_parse(answer, candidates)

    relations = list({
        (r["subject"], r["predicate"], r["object"])
        for r in relations
    })

    relations = [
        {"subject": s, "predicate": p, "object": o}
        for s, p, o in relations
    ]

    # ------------------------------
    # Construir grafo
    # ------------------------------
    G = nx.DiGraph()
    for o in objects:
        G.add_node(o["name"])
    for r in relations:
        G.add_edge(
            r["subject"],
            r["object"],
            relation=r["predicate"]
        )

    fig = plt.figure(figsize=(14, 12))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=2600,
        node_color="#b3d9ff"
    )
    nx.draw_networkx_edge_labels(
        G,
        pos,
        nx.get_edge_attributes(G, "relation")
    )
    plt.title("Semantic Scene Graph — YOLO + Geometry + LION")
    plt.axis("off")

    # ------------------------------
    # LIMPEZA CRÍTICA
    # ------------------------------
    del img_llm, img_ram, answer, res
    torch.cuda.empty_cache()
    gc.collect()

    return fig, relations
