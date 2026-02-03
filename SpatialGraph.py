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

def run_spatial_graph(
    img: Image.Image,
    dist_threshold: int = 160,
    max_objects: int = 12,
    max_candidates: int = 50,
    batch_size: int = 5
):
    """
    Recebe uma imagem PIL e retorna:
    - fig: matplotlib figure com o grafo
    - valid_relations: lista de relações validadas
    """

    # ------------------------------
    # Modelos (cacheados – dict)
    # ------------------------------
    models = get_models()
    yolo = models["yolo_det"]
    lion = models["lion"]
    processor = models["processor"]
    device = models["device"]

    # ------------------------------
    # Pré-processamento
    # ------------------------------
    ram_transform = T.Compose([
        T.Resize((384, 384)),
        T.ToTensor(),
        T.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])

    img_llm = processor(img).unsqueeze(0).to(device)
    img_ram = ram_transform(img).unsqueeze(0).to(device)

    # ------------------------------
    # YOLO deteção
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
    def overlap(b1, b2): return max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))

    # ------------------------------
    # Gerar candidatos de relações
    # ------------------------------
    candidates = []
    for o1, o2 in itertools.permutations(objects, 2):
        if o1["id"] == o2["id"]:
            continue
        b1, b2 = o1["box"], o2["box"]

        if overlap(b1, b2) > 0 and area(b1) < area(b2):
            candidates.append((o1["name"], "on", o2["name"]))
        if dist(b1, b2) < dist_threshold:
            candidates.append((o1["name"], "next_to", o2["name"]))
        if center(b1)[1] < center(b2)[1]:
            candidates.append((o1["name"], "above", o2["name"]))
        elif center(b1)[1] > center(b2)[1]:
            candidates.append((o1["name"], "under", o2["name"]))

    candidates = list(set(candidates))
    candidates = candidates[:max_candidates]  # 👈 limita para evitar prompts gigantes

    # ------------------------------
    # Função para validar relações em batch
    # ------------------------------
    def validate_batch(batch):
        if not batch:
            return []

        question = f"""
From the list of candidate relations, select ONLY those that are
visually true in the image.

Output ONLY a JSON array:
[
  {{"subject": "object", "predicate": "relation", "object": "object"}}
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

        # Tenta parsear JSON
        try:
            parsed = json.loads(answer)
            if isinstance(parsed, list):
                return parsed
        except:
            matches = re.findall(r"\[[\s\S]*?\]", answer)
            for m in matches:
                try:
                    parsed = json.loads(m)
                    if isinstance(parsed, list):
                        return parsed
                except:
                    continue

        # fallback conservador: retorna todas as relações como estão
        return [{"subject": s, "predicate": p, "object": o} for s, p, o in batch]

    # ------------------------------
    # Validar todas as relações em batches pequenos
    # ------------------------------
    valid_relations = []
    for i in range(0, len(candidates), batch_size):
        valid_relations.extend(validate_batch(candidates[i:i + batch_size]))

    # Remove duplicatas
    valid_relations = list({
        (r["subject"], r["predicate"], r["object"])
        for r in valid_relations
    })
    valid_relations = [
        {"subject": s, "predicate": p, "object": o}
        for s, p, o in valid_relations
    ]

    # ------------------------------
    # Construir grafo
    # ------------------------------
    G = nx.DiGraph()
    for o in objects:
        G.add_node(o["name"])
    for r in valid_relations:
        G.add_edge(r["subject"], r["object"], relation=r["predicate"])

    fig = plt.figure(figsize=(14, 12))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=2600, node_color="#b3d9ff")
    nx.draw_networkx_edge_labels(G, pos, nx.get_edge_attributes(G, "relation"))
    plt.axis("off")

    return fig, valid_relations
