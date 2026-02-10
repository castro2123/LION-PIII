import spacy
import re

# =========================
# Configuração spaCy
# =========================
nlp = spacy.load("en_core_web_sm")

# =========================
# Utils
# =========================
def clean_entity(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r'\s+', '_', s)
    s = re.sub(r'[^\w_]', '', s)
    return s

# Palavras genéricas que não devem ser consideradas entidades
GENERIC_WORDS = {"image", "one", "thing", "object", "photo", "background"}

# Mapeamento de preposições
preposition_mapping = {
    "on": "near",
    "in": "near",
    "at": "located",
    "next to": "near",
    "beside": "near",
    "under": "near",
    "above": "near",
    "in front of": "located",
    "behind": "located",
}

# Verbos de comunicação
communication_verbs = {"talk": "talk", "speak": "talk", "chat": "talk", "say": "talk"}

# =========================
# Extração de entidades e relações
# =========================
def extract_entities_and_relations(caption: str):
    doc = nlp(caption)
    entities = set()
    relations = []
    subjects = []

    # Extrai entidades válidas (ignora genéricas)
    for chunk in doc.noun_chunks:
        entity = clean_entity(chunk.text)
        if entity not in GENERIC_WORDS:
            entities.add(entity)

    # Relações verbo + objeto / preposição
    for token in doc:
        if token.pos_ == "VERB":
            subject = None
            for child in token.children:
                if child.dep_ in ("nsubj","nsubjpass"):
                    subj_entity = clean_entity(child.text)
                    if subj_entity not in GENERIC_WORDS:
                        subject = subj_entity
                        subjects.append(subject)
            if not subject:
                continue

            verb_lemma = communication_verbs.get(token.lemma_.lower(), token.lemma_.lower())

            # objeto direto
            for child in token.children:
                if child.dep_ in ("dobj","attr") and child.pos_ in ("NOUN","PROPN"):
                    obj = clean_entity(child.text)
                    if obj not in GENERIC_WORDS:
                        relations.append((subject, verb_lemma, obj))

            # preposições
            for prep in [c for c in token.children if c.dep_=="prep"]:
                mapped = preposition_mapping.get(prep.text.lower(), prep.text.lower())
                for pobj in prep.children:
                    if pobj.pos_ in ("NOUN","PROPN"):
                        obj = clean_entity(pobj.text)
                        if obj not in GENERIC_WORDS:
                            relations.append((subject, mapped, obj))

    # Relações de proximidade sem verbo
    for chunk in doc.noun_chunks:
        head = chunk.root
        for child in head.children:
            if child.dep_ == "prep":
                mapped = preposition_mapping.get(child.text.lower(), child.text.lower())
                for pobj in child.children:
                    obj = clean_entity(pobj.text)
                    subj = clean_entity(chunk.text)
                    if obj not in GENERIC_WORDS and subj not in GENERIC_WORDS:
                        relations.append((subj, mapped, obj))

    # --- Talk apenas entre sujeitos válidos ---
    if len(subjects) > 1:
        for i in range(len(subjects)):
            for j in range(i+1, len(subjects)):
                relations.append((subjects[i], "talk", subjects[j]))
                relations.append((subjects[j], "talk", subjects[i]))

    # Remove duplicados
    relations = list({(s,p,o) for s,p,o in relations})
    
    return sorted(entities), relations

# =========================
# Gera fatos Prolog
# =========================
def to_prolog(caption: str):
    entities, relations = extract_entities_and_relations(caption)

    code = ["% ===== Fatos Prolog gerados automaticamente =====\n"]
    for e in entities:
        code.append(f"entity({e}).")
    code.append("")
    for s,p,o in relations:
        code.append(f"relation({s},{p},{o}).")
    code.append("\n% ----- Regras genéricas -----")
    code.append("talking(X,Y) :- relation(X,talk,Y), relation(Y,talk,X).")
    code.append("near(X,Y) :- relation(X,near,Y); relation(Y,near,X).")
    code.append("located(X,Place) :- relation(X,_,Place).")

    return "\n".join(code), relations
