import gc
import math
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import torch
from PIL import Image
import spacy

from model_registry import get_models

# =========================
# NLP
# =========================
nlp = spacy.load("en_core_web_sm")

# =========================
# Geometria
# =========================
def center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)

def dist(b1, b2):
    return math.dist(center(b1), center(b2))

# =========================
# Constantes linguísticas
# =========================
META_SUBJECTS = {"image", "photo", "picture", "scene"}
META_VERBS = {"depict", "show", "illustrate", "feature"}

WEAK_VERBS = {"seem", "appear"}  # ⚠️ NÃO inclui "be"

BAD_OBJECTS = {
    "one", "some", "several", "many", "thing",
    "area", "place", "setting", "side", "variety"
}

PRONOUNS = {"it", "them", "this", "that", "there"}
BAD_SUBJECTS = {"other", "others"}

LOCATION_PREPS = {
    "in", "on", "at", "near", "around", "behind", "under", "over"
}

BAD_PREPS = {"throughout"}

# =========================
# Utils NLP
# =========================
def norm(tok):
    return tok.lemma_.lower()

def clean_entity(s):
    for bad in ["several ", "many ", "group of ", "a group of "]:
        s = s.replace(bad, "")
    return s.strip()

def noun_phrase(tok):
    parts = []
    for t in tok.lefts:
        if t.pos_ in {"ADJ", "NOUN", "PROPN"}:
            parts.append(t.lemma_.lower())
    parts.append(tok.lemma_.lower())
    return " ".join(parts)

# =========================
# 1️⃣ Extração semântica ROBUSTA
# =========================
def extract_semantic_interactions(caption: str):
    doc = nlp(caption)
    relations = []
    last_subjects = []

    for sent in doc.sents:

        # ignorar frases meta
        if any(tok.lemma_ in META_VERBS for tok in sent) and \
           any(tok.text.lower() in META_SUBJECTS for tok in sent):
            continue

        # ignorar "there is / there are"
        if sent.root.lemma_ == "be" and any(tok.text.lower() == "there" for tok in sent):
            continue

        subjects = []

        # --- sujeitos explícitos ---
        for tok in sent:
            if tok.dep_ in ("nsubj", "nsubjpass") and tok.pos_ in {"NOUN", "PROPN"}:
                ent = clean_entity(noun_phrase(tok))
                if ent in BAD_OBJECTS or ent in PRONOUNS or ent in BAD_SUBJECTS:
                    continue
                subjects.append(ent)

        # herdar sujeitos da frase anterior (ex: "with one of the machines...")
        if not subjects and last_subjects:
            subjects = last_subjects.copy()

        if not subjects:
            continue

        last_subjects = subjects.copy()

        # --- verbos ---
        for verb in sent:
            if verb.pos_ != "VERB":
                continue

            v = norm(verb)
            if v in META_VERBS or v in WEAK_VERBS:
                continue

            # 🔑 CASO ESPECIAL: "X is Y" / "X being Y"
            if verb.lemma_ == "be":
                for child in verb.children:
                    if child.dep_ == "attr" and child.pos_ in {"NOUN", "PROPN"}:
                        obj = clean_entity(noun_phrase(child))
                        if obj not in BAD_OBJECTS:
                            for s in subjects:
                                relations.append({
                                    "subject": s,
                                    "predicate": "is",
                                    "object": obj,
                                    "type": "semantic"
                                })
                continue

            # --- objetos e preposições ---
            for child in verb.children:

                # objeto direto
                if child.dep_ in ("dobj", "obj", "attr") and child.pos_ in {"NOUN", "PROPN"}:
                    obj = clean_entity(noun_phrase(child))
                    if obj not in BAD_OBJECTS and obj not in PRONOUNS:
                        for s in subjects:
                            relations.append({
                                "subject": s,
                                "predicate": v,
                                "object": obj,
                                "type": "semantic"
                            })

                # preposições
                if child.dep_ == "prep" and child.lemma_ not in BAD_PREPS:
                    prep = child.lemma_
                    for pobj in child.children:
                        if pobj.pos_ not in {"NOUN", "PROPN"}:
                            continue
                        obj = clean_entity(noun_phrase(pobj))
                        if obj in BAD_OBJECTS or obj in PRONOUNS:
                            continue

                        predicate = prep if prep in LOCATION_PREPS else f"{v}_{prep}"
                        for s in subjects:
                            relations.append({
                                "subject": s,
                                "predicate": predicate,
                                "object": obj,
                                "type": "semantic"
                            })

    # remover duplicados
    uniq, seen = [], set()
    for r in relations:
        key = (r["subject"], r["predicate"], r["object"])
        if key not in seen:
            seen.add(key)
            uniq.append(r)

    return uniq

# =========================
# 2️⃣ Alinhamento com YOLO
# =========================
def align_with_visual_objects(relations, objects):
    by_class = {}
    for o in objects:
        by_class.setdefault(o["class"], []).append(o["name"])

    aligned = []
    for r in relations:
        r = r.copy()
        if r["subject"] in by_class:
            r["subject"] = by_class[r["subject"]][0]
        if r["object"] in by_class:
            r["object"] = by_class[r["object"]][0]
        aligned.append(r)

    return aligned

# =========================
# 3️⃣ Função principal
# =========================
def run_semantic_graph(img: Image.Image, caption: str):
    models = get_models()
    yolo = models["yolo_det"]

    img = img.convert("RGB")

    # -------- YOLO --------
    with torch.no_grad():
        res = yolo(img, conf=0.35)[0]

    objects = []
    counts = {}
    for i, box in enumerate(res.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls = yolo.names[int(box.cls[0])]
        counts[cls] = counts.get(cls, 0) + 1
        objects.append({
            "id": i,
            "name": f"{cls}_{counts[cls]}",
            "class": cls,
            "box": (x1, y1, x2, y2),
            "type": "physical"
        })

    # -------- Relações físicas --------
    physical = []
    for o1, o2 in itertools.combinations(objects, 2):
        if dist(o1["box"], o2["box"]) < 160:
            physical.append({
                "subject": o1["name"],
                "predicate": "next_to",
                "object": o2["name"],
                "type": "physical"
            })

    # -------- Relações semânticas --------
    semantic = extract_semantic_interactions(caption)
    semantic = align_with_visual_objects(semantic, objects)

    # -------- Grafo --------
    G = nx.DiGraph()

    for o in objects:
        G.add_node(o["name"], type="physical")

    for r in semantic:
        G.add_node(r["subject"], type="semantic")
        G.add_node(r["object"], type="semantic")
        G.add_edge(
            r["subject"],
            r["object"],
            label=r["predicate"],
            edge_type="semantic"
        )

    for r in physical:
        G.add_edge(
            r["subject"],
            r["object"],
            label=r["predicate"],
            edge_type="physical"
        )

    # -------- Visualização --------
    fig = plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42, k=1.6)

    colors = [
        "#cce5ff" if G.nodes[n].get("type") == "physical" else "#ffe0e0"
        for n in G.nodes
    ]

    nx.draw_networkx_nodes(G, pos, node_size=1800,
                           node_color=colors, edgecolors="black")
    nx.draw_networkx_labels(G, pos, font_size=9)

    phys = [(u, v) for u, v, d in G.edges(data=True)
            if d["edge_type"] == "physical"]
    sem = [(u, v) for u, v, d in G.edges(data=True)
           if d["edge_type"] == "semantic"]

    nx.draw_networkx_edges(G, pos, edgelist=phys,
                           width=2, edge_color="#1f77b4")
    nx.draw_networkx_edges(G, pos, edgelist=sem,
                           width=2, edge_color="#d62728", style="dashed")

    labels = {(u, v): d["label"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=8)

    plt.axis("off")
    gc.collect()
    torch.cuda.empty_cache()

    return fig, physical, objects, semantic
