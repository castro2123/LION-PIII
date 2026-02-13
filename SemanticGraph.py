import networkx as nx
import matplotlib.pyplot as plt
import spacy
from PIL import Image

nlp = spacy.load("en_core_web_sm")


# =========================
# Helpers
# =========================
def noun_phrase(tok):
    parts = []
    for t in tok.lefts:
        if t.dep_ in {"amod", "compound"}:
            parts.append(t.lemma_.lower())
    parts.append(tok.lemma_.lower())
    return " ".join(parts)


def clean_entity(text):
    text = text.lower()
    replacements = [
        "a couple of ",
        "couple of ",
        "several ",
        "two ",
        "one of the ",
        "one of ",
    ]
    for r in replacements:
        text = text.replace(r, "")
    return text.strip()


# =========================
# Extract Scene Graph
# =========================
def extract_semantic_relations(caption: str):
    doc = nlp(caption)
    relations = []
    last_subject = None

    for sent in doc.sents:

        subjects_in_sentence = []

        # --- Detect subjects ---
        for token in sent:

            if token.dep_ in ("nsubj", "nsubjpass"):

                subject_token = token

                # Handle "one of the men"
                if token.lemma_.lower() == "one":
                    for child in token.children:
                        if child.dep_ == "prep" and child.lemma_ == "of":
                            for pobj in child.children:
                                if pobj.dep_ == "pobj":
                                    subject_token = pobj

                subject = clean_entity(noun_phrase(subject_token))

                # pronoun resolution
                if subject in {"they", "them"} and last_subject:
                    subject = last_subject
                else:
                    last_subject = subject

                # ignore generic subjects
                if subject in {"other object", "image", "photo"}:
                    continue

                subjects_in_sentence.append(subject)

                verb = token.head.lemma_.lower()

                # --- Direct objects ---
                for child in token.head.children:
                    if child.dep_ in ("dobj", "obj") and child.pos_ in {"NOUN", "PROPN"}:
                        obj = clean_entity(noun_phrase(child))
                        relations.append((subject, verb, obj))

                # --- Prepositions ---
                for prep in token.head.children:
                    if prep.dep_ == "prep":
                        if prep.lemma_ == "of":
                            continue

                        # in front of
                        if prep.lemma_ == "in":
                            next_token = prep.nbor(1) if prep.i + 1 < len(doc) else None
                            if next_token and next_token.text.lower() == "front":
                                for of_prep in next_token.children:
                                    if of_prep.dep_ == "prep":
                                        for pobj in of_prep.children:
                                            if pobj.pos_ in {"NOUN", "PROPN"}:
                                                obj = clean_entity(noun_phrase(pobj))
                                                relations.append((subject, "in_front_of", obj))
                                continue

                        # normal prepositions
                        for pobj in prep.children:
                            if pobj.pos_ in {"NOUN", "PROPN"}:
                                obj = clean_entity(noun_phrase(pobj))
                                relations.append((subject, prep.lemma_, obj))

                # --- Reciprocal ---
                for prep in token.head.children:
                    if prep.dep_ == "prep":
                        for pobj in prep.children:
                            phrase = " ".join([t.text.lower() for t in pobj.subtree])
                            if phrase in {"each other", "one another"}:
                                relations.append((subject, verb, subject))

        # --- Capture conjoined / gerund verbs (talking, working) ---
        for token in sent:
            if token.pos_ == "VERB" and token.dep_ in {"conj", "advcl", "xcomp"}:
                verb = token.lemma_.lower()
                for subject in subjects_in_sentence:
                    # reciprocal
                    for prep in token.children:
                        if prep.dep_ == "prep":
                            for pobj in prep.children:
                                phrase = " ".join([t.text.lower() for t in pobj.subtree])
                                if phrase in {"each other", "one another"}:
                                    relations.append((subject, verb, subject))
                    # direct objects
                    for child in token.children:
                        if child.dep_ in ("dobj", "obj") and child.pos_ in {"NOUN", "PROPN"}:
                            obj = clean_entity(noun_phrase(child))
                            relations.append((subject, verb, obj))

        # --- Existential: there are X ---
        for token in sent:
            if token.dep_ == "expl" and token.text.lower() == "there":
                for child in token.head.children:
                    if child.dep_ == "attr" and child.pos_ in {"NOUN", "PROPN"}:
                        subject = clean_entity(noun_phrase(child))
                        last_subject = subject
                        for prep in child.children:
                            if prep.dep_ == "prep" and prep.lemma_ != "of":
                                for pobj in prep.children:
                                    if pobj.pos_ in {"NOUN", "PROPN"}:
                                        obj = clean_entity(noun_phrase(pobj))
                                        relations.append((subject, prep.lemma_, obj))
                        # --- Handle lists after "such as" ---
                        for token2 in child.children:
                            if token2.lemma_ == "such":
                                for item in token2.children:
                                    if item.dep_ == "pobj":
                                        obj = clean_entity(noun_phrase(item))
                                        relations.append((subject, "has", obj))
                                        # handle conjoined items
                                        for conj in item.conjuncts:
                                            obj2 = clean_entity(noun_phrase(conj))
                                            relations.append((subject, "has", obj2))

        # --- Including ---
        for token in sent:
            if token.lemma_ == "include":
                head = token.head
                if head.pos_ in {"NOUN", "PROPN"}:
                    subject = clean_entity(noun_phrase(head))
                    for child in token.children:
                        if child.pos_ in {"NOUN", "PROPN"}:
                            obj = clean_entity(noun_phrase(child))
                            relations.append((subject, "include", obj))

    return list(set(relations))


# =========================
# Build Graph
# =========================
def build_caption_graph(caption: str):
    relations = extract_semantic_relations(caption)
    G = nx.DiGraph()

    for s, p, o in relations:
        G.add_node(s)
        G.add_node(o)
        G.add_edge(s, o, label=p)

    fig = plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, k=1.5, seed=42)

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=3000,
        node_color="lightblue",
        font_size=10
    )

    labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    plt.axis("off")
    return fig, relations


# =========================
# Run Semantic Graph
# =========================
def run_semantic_graph(img: Image.Image):
    from imageCaption import run_caption

    caption_dict = run_caption(img)
    caption = caption_dict["answer"]

    fig, relations = build_caption_graph(caption)

    return fig, relations, caption
