import time
import gc
import re
import collections
import io
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# ─────────────────────────────────────────────────────────────────
# Paleta de cores
# ─────────────────────────────────────────────────────────────────
_PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#469990", "#dcbeff",
    "#9a6324", "#800000", "#aaffc3", "#808000", "#000075",
    "#ffe119", "#fabed4", "#ffd8b1", "#a9a9a9", "#ff6b6b",
]

_class_color_map: dict[str, str] = {}


def _color_for_class(cls_name: str) -> str:
    if cls_name not in _class_color_map:
        idx = len(_class_color_map) % len(_PALETTE)
        _class_color_map[cls_name] = _PALETTE[idx]
    return _class_color_map[cls_name]


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def _clean_label(text: str) -> str:
    if not text:
        return "object"
    text = str(text).strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return (text[:57] + "...") if len(text) > 60 else (text or "object")


# ─────────────────────────────────────────────────────────────────
# Normalização de labels
# ─────────────────────────────────────────────────────────────────
_LABEL_ALIASES = {
    "vehicle":    "car",  "automobile": "car",   "auto":      "car",
    "human":      "person", "people":  "person", "man":       "person",
    "woman":      "person", "kid":     "person", "child":     "person",
    "bike":       "bicycle", "motorbike": "motorcycle",
    "tv":         "tvmonitor", "monitor": "tvmonitor",
    "couch":      "sofa",
}


def _normalize_label(label: str) -> str:
    l = (label or "").strip().lower()
    return _LABEL_ALIASES.get(l, l)


# ─────────────────────────────────────────────────────────────────
# IoU entre duas bboxes
# ─────────────────────────────────────────────────────────────────
def _iou_boxes(boxA: list, boxB: list) -> float:
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
    areaB = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


# ─────────────────────────────────────────────────────────────────
# AP via interpolação de 11 pontos
# ─────────────────────────────────────────────────────────────────
def _compute_ap(tp_list: list, fp_list: list, n_gt: int) -> float:
    if n_gt == 0 or not tp_list:
        return 0.0
    tp_cum = np.cumsum(tp_list)
    fp_cum = np.cumsum(fp_list)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-9)
    recalls    = tp_cum / (n_gt + 1e-9)
    ap = 0.0
    for thr in np.linspace(0, 1, 11):
        mask = recalls >= thr
        ap += np.max(precisions[mask]) if mask.any() else 0.0
    return ap / 11.0


# ─────────────────────────────────────────────────────────────────
# MODO 1 — Métricas com Ground Truth real
# ─────────────────────────────────────────────────────────────────
def compute_metrics_vs_reference(
    detections: list[dict],
    ground_truth: list[dict],
    iou_threshold: float = 0.5,
) -> dict:
    """
    Precision, Recall, F1, AP por classe + macro/micro médias + mAP@50.
    ground_truth: [{"label": str, "bbox": [x1,y1,x2,y2]}, ...]
    """
    all_classes = set(
        [_normalize_label(d["label"]) for d in detections] +
        [_normalize_label(g["label"]) for g in ground_truth]
    )
    per_class: dict[str, dict] = {}

    for cls in sorted(all_classes):
        cls_dets = sorted(
            [d for d in detections if _normalize_label(d["label"]) == cls],
            key=lambda x: -x.get("confidence", 1.0),
        )
        cls_gt = [g for g in ground_truth if _normalize_label(g["label"]) == cls]
        matched_gt = [False] * len(cls_gt)
        tp_list, fp_list = [], []

        for det in cls_dets:
            best_iou, best_idx = 0.0, -1
            for j, gt in enumerate(cls_gt):
                if matched_gt[j]:
                    continue
                iou = _iou_boxes(det["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou, best_idx = iou, j
            if best_iou >= iou_threshold and best_idx >= 0:
                tp_list.append(1); fp_list.append(0)
                matched_gt[best_idx] = True
            else:
                tp_list.append(0); fp_list.append(1)

        tp = sum(tp_list); fp = sum(fp_list); fn = len(cls_gt) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0.0
        ap = _compute_ap(tp_list, fp_list, len(cls_gt))
        per_class[cls] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
            "ap":        round(ap,        4),
        }

    if per_class:
        macro_p  = float(np.mean([v["precision"] for v in per_class.values()]))
        macro_r  = float(np.mean([v["recall"]    for v in per_class.values()]))
        macro_f1 = float(np.mean([v["f1"]        for v in per_class.values()]))
        mAP      = float(np.mean([v["ap"]        for v in per_class.values()]))
    else:
        macro_p = macro_r = macro_f1 = mAP = 0.0

    total_tp = sum(v["tp"] for v in per_class.values())
    total_fp = sum(v["fp"] for v in per_class.values())
    total_fn = sum(v["fn"] for v in per_class.values())
    micro_p  = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2*micro_p*micro_r / (micro_p+micro_r) if (micro_p+micro_r) > 0 else 0.0

    return {
        "mode": "ground_truth",
        "per_class": per_class,
        "macro_precision": round(macro_p,  4),
        "macro_recall":    round(macro_r,  4),
        "macro_f1":        round(macro_f1, 4),
        "mAP":             round(mAP,      4),
        "micro_precision": round(micro_p,  4),
        "micro_recall":    round(micro_r,  4),
        "micro_f1":        round(micro_f1, 4),
        "total_tp": total_tp, "total_fp": total_fp, "total_fn": total_fn,
    }


# ─────────────────────────────────────────────────────────────────
# MODO 2 — Cross-model consensus (sem GT)
# Constrói GT virtual: deteções presentes em MAIORIA dos modelos
# e calcula Precision/Recall/F1 de cada modelo contra esse consenso.
# ─────────────────────────────────────────────────────────────────
def build_consensus_ground_truth(
    results_per_model: dict[str, tuple[list[dict], float]],
    iou_threshold: float = 0.5,
    min_agreement: float = 0.5,   # fracção mínima de modelos que concordam
) -> list[dict]:
    """
    Para cada deteção de qualquer modelo, verifica quantos outros modelos
    têm uma bbox com IoU >= iou_threshold e mesmo label.
    Se a fracção de modelos em acordo >= min_agreement → entra no GT virtual.
    """
    model_names  = list(results_per_model.keys())
    n_models     = len(model_names)
    if n_models < 2:
        # Com 1 único modelo o consenso seria trivialmente 100% → devolve vazio
        return []

    all_dets: list[dict] = []
    for name, (dets, _) in results_per_model.items():
        for d in dets:
            all_dets.append({**d, "_model": name, "_label_norm": _normalize_label(d["label"])})

    consensus: list[dict] = []
    used = [False] * len(all_dets)

    for i, d in enumerate(all_dets):
        if used[i]:
            continue
        cls = d["_label_norm"]
        # quantos modelos distintos têm esta bbox?
        agreeing_models = {d["_model"]}
        for j, other in enumerate(all_dets):
            if j == i or used[j]:
                continue
            if other["_label_norm"] != cls:
                continue
            if _iou_boxes(d["bbox"], other["bbox"]) >= iou_threshold:
                agreeing_models.add(other["_model"])
                used[j] = True

        used[i] = True
        if len(agreeing_models) / n_models >= min_agreement:
            consensus.append({"label": cls, "bbox": d["bbox"]})

    return consensus


def compute_metrics_consensus(
    results_per_model: dict[str, tuple[list[dict], float]],
    iou_threshold: float = 0.5,
    min_agreement: float = 0.5,
) -> dict:
    """
    Calcula métricas de cada modelo contra o GT de consenso entre modelos.
    Devolve dict com per_model e o GT virtual usado.
    """
    gt_virtual = build_consensus_ground_truth(
        results_per_model, iou_threshold, min_agreement
    )

    per_model: dict[str, dict] = {}
    for name, (dets, _) in results_per_model.items():
        m = compute_metrics_vs_reference(dets, gt_virtual, iou_threshold)
        per_model[name] = m

    return {
        "mode":           "consensus",
        "gt_virtual":     gt_virtual,
        "gt_size":        len(gt_virtual),
        "min_agreement":  min_agreement,
        "per_model":      per_model,
    }


# ─────────────────────────────────────────────────────────────────
# MODO 3 — Métricas de confiança melhoradas (sempre disponível)
# ─────────────────────────────────────────────────────────────────
def compute_confidence_metrics(
    detections: list[dict],
) -> dict:
    """
    Estatísticas de confiança por classe:
    count, mean, min, max, std, e NMS-like duplication rate.
    """
    by_class: dict[str, list] = collections.defaultdict(list)
    for d in detections:
        by_class[_normalize_label(d["label"])].append(d.get("confidence", 1.0))

    result = {}
    for cls, confs in by_class.items():
        arr = np.array(confs)
        result[cls] = {
            "count":     int(len(arr)),
            "mean_conf": round(float(arr.mean()), 4),
            "min_conf":  round(float(arr.min()),  4),
            "max_conf":  round(float(arr.max()),  4),
            "std_conf":  round(float(arr.std()),  4),
            # % de deteções acima de 50% conf
            "high_conf_pct": round(float((arr >= 0.5).mean()), 4),
        }
    return result


# ─────────────────────────────────────────────────────────────────
# Tabela de métricas unificada
# Usa GT real se fornecido, consenso se >1 modelo, confiança sempre
# ─────────────────────────────────────────────────────────────────
def build_metrics_table(
    results_per_model: dict[str, tuple[list[dict], float]],
    ground_truth: Optional[list[dict]] = None,
    iou_threshold: float = 0.5,
) -> tuple[list[dict], str]:
    """
    Devolve (rows, mode) onde mode é "ground_truth" | "consensus" | "confidence".
    rows é a lista de dicts para pd.DataFrame.
    """
    n_models = len(results_per_model)
    rows = []

    # ── MODO 1: GT real ───────────────────────────────────────────
    if ground_truth is not None:
        for name, (dets, ms) in results_per_model.items():
            m = compute_metrics_vs_reference(dets, ground_truth, iou_threshold)
            rows.append({
                "Model":          name,
                "Objects":        len(dets),
                "Time (ms)":      round(ms, 1),
                "Precision ↑":    f"{m['macro_precision']:.1%}",
                "Recall ↑":       f"{m['macro_recall']:.1%}",
                "F1 ↑":           f"{m['macro_f1']:.1%}",
                "mAP@50 ↑":       f"{m['mAP']:.1%}",
                "TP":             m["total_tp"],
                "FP":             m["total_fp"],
                "FN":             m["total_fn"],
            })
        return rows, "ground_truth"

    # ── MODO 2: Consenso entre modelos ───────────────────────────
    if n_models >= 2:
        consensus_result = compute_metrics_consensus(
            results_per_model, iou_threshold, min_agreement=0.5
        )
        gt_size = consensus_result["gt_size"]
        for name, (dets, ms) in results_per_model.items():
            m = consensus_result["per_model"][name]
            rows.append({
                "Model":               name,
                "Objects":             len(dets),
                "Time (ms)":           round(ms, 1),
                "Precision* ↑":        f"{m['macro_precision']:.1%}",
                "Recall* ↑":           f"{m['macro_recall']:.1%}",
                "F1* ↑":               f"{m['macro_f1']:.1%}",
                "mAP@50* ↑":           f"{m['mAP']:.1%}",
                "TP*":                 m["total_tp"],
                "FP*":                 m["total_fp"],
                "FN*":                 m["total_fn"],
                "vs GT virtual":       gt_size,
            })
        return rows, "consensus"

    # ── MODO 3: Confiança (fallback, 1 modelo sem GT) ─────────────
    for name, (dets, ms) in results_per_model.items():
        if dets:
            confs = [d.get("confidence", 1.0) for d in dets]
            rows.append({
                "Model":          name,
                "Objects":        len(dets),
                "Time (ms)":      round(ms, 1),
                "Mean Conf":      f"{np.mean(confs):.1%}",
                "Min Conf":       f"{np.min(confs):.1%}",
                "Max Conf":       f"{np.max(confs):.1%}",
                "Std Conf":       f"{np.std(confs):.3f}",
                "High Conf (≥50%)": f"{np.mean(np.array(confs) >= 0.5):.1%}",
            })
        else:
            rows.append({"Model": name, "Objects": 0, "Time (ms)": round(ms, 1)})
    return rows, "confidence"


def build_per_class_metrics_table(
    results_per_model: dict[str, tuple[list[dict], float]],
    ground_truth: list[dict],
    iou_threshold: float = 0.5,
) -> list[dict]:
    """Tabela de métricas por classe — requer GT (real ou virtual)."""
    rows = []
    for model_name, (dets, _) in results_per_model.items():
        m = compute_metrics_vs_reference(dets, ground_truth, iou_threshold)
        for cls, stats in m["per_class"].items():
            rows.append({
                "Model":     model_name,
                "Class":     cls,
                "TP":        stats["tp"],
                "FP":        stats["fp"],
                "FN":        stats["fn"],
                "Precision": f"{stats['precision']:.1%}",
                "Recall":    f"{stats['recall']:.1%}",
                "F1":        f"{stats['f1']:.1%}",
                "AP@50":     f"{stats['ap']:.1%}",
            })
    return rows


def build_confidence_table(
    results_per_model: dict[str, tuple[list[dict], float]],
) -> list[dict]:
    """Tabela de confiança por classe para cada modelo."""
    rows = []
    for name, (dets, _) in results_per_model.items():
        cm = compute_confidence_metrics(dets)
        for cls, stats in sorted(cm.items()):
            rows.append({
                "Model":           name,
                "Class":           cls,
                "Count":           stats["count"],
                "Mean Conf":       f"{stats['mean_conf']:.1%}",
                "Min Conf":        f"{stats['min_conf']:.1%}",
                "Max Conf":        f"{stats['max_conf']:.1%}",
                "Std":             f"{stats['std_conf']:.3f}",
                "High Conf (≥50%)":f"{stats['high_conf_pct']:.1%}",
            })
    return rows


# ─────────────────────────────────────────────────────────────────
# LION: descrição de uma região recortada
# ─────────────────────────────────────────────────────────────────
def _lion_describe_region(lion, processor, img, bbox, yolo_label, global_tags, device):
    x1, y1, x2, y2 = bbox
    crop = img.crop((x1, y1, x2, y2))
    if crop.width < 20 or crop.height < 20:
        return yolo_label
    try:
        crop_processed = processor(crop)
        question = (
            f"Describe ONLY the main '{yolo_label}' inside the bounding box. "
            f"Do NOT mention background, nearby objects or scene context. "
            f"Use 3 to 6 words starting with '{yolo_label}'."
        )
        out = lion.generate({
            "image": crop_processed.unsqueeze(0).to(device),
            "question": [question], "tags": [global_tags], "category": "image_level",
        })
        raw  = out[0] if isinstance(out, (list, tuple)) else out
        text = str(raw).strip().rstrip(".").lower()
        if yolo_label and not text.startswith(yolo_label):
            text = f"{yolo_label} {text}"
        text = re.sub(rf"\b({re.escape(yolo_label)})(\s+\1)+\b", r"\1", text)
        text = _clean_label(text)
        return text if len(text) >= len(yolo_label) else yolo_label
    except Exception:
        return yolo_label


# ─────────────────────────────────────────────────────────────────
# YOLO: inferência
# ─────────────────────────────────────────────────────────────────
def run_yolo_inference(model, img: Image.Image, conf=0.25, iou=0.45):
    _device = 0 if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    results = model.predict(source=img, conf=conf, iou=iou, device=_device, verbose=False)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    detections = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0].item())
            label  = r.names.get(cls_id, str(cls_id))
            conf_v = float(box.conf[0].item())
            detections.append({
                "label": label, "confidence": conf_v,
                "bbox":  [int(x1), int(y1), int(x2), int(y2)],
                "caption": label,
            })
    gc.collect()
    return detections, elapsed_ms


# ─────────────────────────────────────────────────────────────────
# LION standalone
# ─────────────────────────────────────────────────────────────────
def run_lion_inference(img, conf_threshold=0.18, nms_iou=0.45, use_caption_labels=True):
    from boudingBox import run_lion_inference as run_lion_yolo
    t0 = time.perf_counter()
    _, detections, _ = run_lion_yolo(
        img, conf_threshold=conf_threshold,
        use_caption_labels=use_caption_labels, multi_prompt=True, nms_iou=nms_iou,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    normalized = [{
        "label":      d.get("label", "object"),
        "confidence": float(d.get("confidence", 1.0)),
        "bbox":       d["bbox"],
        "caption":    d.get("caption", d.get("label", "object")),
    } for d in detections]
    gc.collect()
    return normalized, elapsed_ms


# ─────────────────────────────────────────────────────────────────
# LION: enriquece deteções YOLO com captions
# ─────────────────────────────────────────────────────────────────
def enrich_with_lion(detections, img, lion, processor, device):
    if not detections:
        return detections
    try:
        global_tags = lion.generate_tags(img)
    except Exception:
        global_tags = ""
    with torch.no_grad():
        for det in detections:
            det["caption"] = _lion_describe_region(
                lion, processor, img, det["bbox"], det["label"], global_tags, device,
            )
    return detections


# ─────────────────────────────────────────────────────────────────
# Desenho das bounding boxes
# ─────────────────────────────────────────────────────────────────
def draw_yolo_boxes(img, detections, show_conf=True, use_lion_caption=True):
    out  = img.copy().convert("RGBA")
    draw = ImageDraw.Draw(out, "RGBA")
    base = max(img.width, img.height)
    lw   = max(2, base // 400)
    fs   = max(13, base // 60)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fs)
    except Exception:
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        r, g, b = _hex_to_rgb(_color_for_class(det["label"]))
        draw.rectangle([x1, y1, x2, y2], outline=(r, g, b, 255), width=lw)

        # Apenas a class label — sem caption nem confiança
        text = det["label"]
        bb  = draw.textbbox((0, 0), text, font=font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        pad = max(3, fs // 5)
        ty  = y1 - th - pad * 2
        if ty < 0:
            ty = y1 + lw
        draw.rectangle([x1, ty, x1 + tw + pad * 2, ty + th + pad * 2], fill=(r, g, b, 220))
        draw.text((x1 + pad, ty + pad), text, fill="white", font=font)

    return out.convert("RGB")


# ─────────────────────────────────────────────────────────────────
# Tabelas comparativas
# ─────────────────────────────────────────────────────────────────
def build_comparison_table(results_per_model):
    all_classes: set[str] = set()
    counts: dict[str, collections.Counter] = {}
    for model_name, (dets, _) in results_per_model.items():
        c = collections.Counter(_normalize_label(d["label"]) for d in dets)
        counts[model_name] = c
        all_classes.update(c.keys())
    rows = []
    for cls in sorted(all_classes):
        row = {"Class": cls}
        for m in results_per_model:
            row[m] = counts[m].get(cls, 0)
        rows.append(row)
    total_row = {"Class": "TOTAL"}
    for m in results_per_model:
        total_row[m] = len(results_per_model[m][0])
    rows.append(total_row)
    return rows


def build_diff_table(results_per_model):
    class_sets = {
        name: set(_normalize_label(d["label"]) for d in dets)
        for name, (dets, _) in results_per_model.items()
    }
    model_names = list(class_sets.keys())
    all_classes = set().union(*class_sets.values())
    rows = []
    for cls in sorted(all_classes):
        present = [n for n in model_names if cls in class_sets[n]]
        if len(present) < len(model_names):
            row = {"Class": cls}
            for n in model_names:
                row[n] = "✅" if cls in class_sets[n] else "❌"
            rows.append(row)
    return rows


# ─────────────────────────────────────────────────────────────────
# EXPORTAÇÃO PDF
# ─────────────────────────────────────────────────────────────────
def export_results_to_pdf(
    results_per_model: dict,
    annotated_images: dict,
    ground_truth: Optional[list[dict]] = None,
    iou_threshold: float = 0.5,
    title: str = "YOLO Comparison Report",
) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image as RLImage, PageBreak, HRFlowable,
    )
    from reportlab.lib.enums import TA_CENTER

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
        leftMargin=1.5*cm, rightMargin=1.5*cm, topMargin=2*cm, bottomMargin=2*cm)

    styles = getSampleStyleSheet()
    sT  = ParagraphStyle("T",  fontSize=20, spaceAfter=6,  alignment=TA_CENTER, fontName="Helvetica-Bold")
    sH1 = ParagraphStyle("H1", fontSize=13, spaceAfter=4,  spaceBefore=12,      fontName="Helvetica-Bold")
    sCp = ParagraphStyle("Cp", fontSize=8,  spaceAfter=4,  alignment=TA_CENTER, textColor=colors.grey)
    sN  = ParagraphStyle("N",  fontSize=7.5,spaceAfter=6,  textColor=colors.grey)
    sSub= ParagraphStyle("Sb", fontSize=10, alignment=TA_CENTER, textColor=colors.grey)

    story = []

    # Capa
    story.append(Spacer(1, 1.5*cm))
    story.append(Paragraph(title, sT))
    story.append(Paragraph(f"Models: {', '.join(results_per_model.keys())}", sSub))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey, spaceAfter=12))

    # Sumário
    story.append(Paragraph("Detection Summary", sH1))
    story.append(_make_table(
        [["Model", "Objects", "Time (ms)"]] +
        [[n, str(len(d)), f"{ms:.1f}"] for n, (d, ms) in results_per_model.items()]
    ))
    story.append(Spacer(1, 0.4*cm))

    # Métricas principais
    metrics_rows, mode = build_metrics_table(results_per_model, ground_truth, iou_threshold)
    story.append(Paragraph("Model Metrics", sH1))

    mode_labels = {
        "ground_truth": "Metrics computed against provided ground truth annotations.",
        "consensus":    "* Metrics computed against cross-model consensus (GT virtual — detections agreed by ≥50% of models). No ground truth was provided.",
        "confidence":   "No ground truth provided and only one model selected — showing confidence statistics.",
    }
    story.append(Paragraph(mode_labels.get(mode, ""), sN))

    if metrics_rows:
        headers = list(metrics_rows[0].keys())
        story.append(_make_table(
            [headers] + [[str(r.get(h, "")) for h in headers] for r in metrics_rows]
        ))

    # Per-class métricas
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("Per-Class Metrics", sH1))

    if ground_truth is not None:
        gt_for_class = ground_truth
        class_note = "Against provided ground truth."
    elif len(results_per_model) >= 2:
        consensus = build_consensus_ground_truth(results_per_model, iou_threshold)
        gt_for_class = consensus
        class_note = f"Against cross-model consensus ({len(consensus)} virtual GT boxes)."
    else:
        gt_for_class = None
        class_note = "Not available without ground truth or multiple models."

    story.append(Paragraph(class_note, sN))

    if gt_for_class:
        pc_rows = build_per_class_metrics_table(results_per_model, gt_for_class, iou_threshold)
        if pc_rows:
            headers = list(pc_rows[0].keys())
            story.append(_make_table(
                [headers] + [[str(r.get(h, "")) for h in headers] for r in pc_rows]
            ))

    # Confidence table (sempre)
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("Confidence Statistics by Class", sH1))
    conf_rows = build_confidence_table(results_per_model)
    if conf_rows:
        headers = list(conf_rows[0].keys())
        story.append(_make_table(
            [headers] + [[str(r.get(h, "")) for h in headers] for r in conf_rows]
        ))

    # Contagem por classe
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("Detections by Class", sH1))
    cmp_rows = build_comparison_table(results_per_model)
    if cmp_rows:
        headers = list(cmp_rows[0].keys())
        story.append(_make_table(
            [headers] + [[str(r.get(h, "")) for h in headers] for r in cmp_rows]
        ))

    # Divergências
    diff_rows = build_diff_table(results_per_model)
    if diff_rows:
        story.append(Spacer(1, 0.4*cm))
        story.append(Paragraph("Model Divergences", sH1))
        headers = list(diff_rows[0].keys())
        story.append(_make_table(
            [headers] + [[str(r.get(h, "")) for h in headers] for r in diff_rows]
        ))

    # Imagens anotadas
    if annotated_images:
        story.append(PageBreak())
        story.append(Paragraph("Annotated Images", sH1))
        page_w = A4[0] - 3*cm
        n_cols = 2 if len(annotated_images) > 1 else 1
        img_w  = (page_w - (n_cols-1)*0.5*cm) / n_cols

        items = list(annotated_images.items())
        for i in range(0, len(items), n_cols):
            row_items = items[i:i+n_cols]
            row_cells = []
            for name, pil_img in row_items:
                aspect = pil_img.height / pil_img.width
                disp_w = img_w
                disp_h = img_w * aspect
                max_h  = 9*cm
                if disp_h > max_h:
                    disp_h = max_h; disp_w = max_h / aspect
                img_buf = io.BytesIO()
                pil_img.save(img_buf, format="PNG")
                img_buf.seek(0)
                row_cells.append([
                    RLImage(img_buf, width=disp_w, height=disp_h),
                    Paragraph(
                        f"<b>{name}</b> — {len(results_per_model[name][0])} obj · "
                        f"{results_per_model[name][1]:.0f} ms", sCp,
                    ),
                ])
            while len(row_cells) < n_cols:
                row_cells.append([""])
            it = Table([row_cells], colWidths=[img_w+0.2*cm]*n_cols)
            it.setStyle(TableStyle([
                ("VALIGN",  (0,0),(-1,-1),"TOP"), ("ALIGN",(0,0),(-1,-1),"CENTER"),
                ("LEFTPADDING",(0,0),(-1,-1),4),  ("RIGHTPADDING",(0,0),(-1,-1),4),
                ("TOPPADDING",(0,0),(-1,-1),4),   ("BOTTOMPADDING",(0,0),(-1,-1),4),
            ]))
            story.append(it)
            story.append(Spacer(1, 0.3*cm))

    doc.build(story)
    return buf.getvalue()


def _make_table(data, col_widths=None):
    from reportlab.platypus import Table, TableStyle
    from reportlab.lib import colors
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR",     (0,0),(-1,0), colors.white),
        ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(-1,0), 8),
        ("ALIGN",         (0,0),(-1,0), "CENTER"),
        ("BOTTOMPADDING", (0,0),(-1,0), 6),
        ("TOPPADDING",    (0,0),(-1,0), 6),
        ("FONTSIZE",      (0,1),(-1,-1), 7.5),
        ("ALIGN",         (0,1),(-1,-1), "CENTER"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.white, colors.HexColor("#f5f5f5")]),
        ("GRID",          (0,0),(-1,-1), 0.4, colors.HexColor("#cccccc")),
        ("LEFTPADDING",   (0,0),(-1,-1), 5),
        ("RIGHTPADDING",  (0,0),(-1,-1), 5),
        ("TOPPADDING",    (0,1),(-1,-1), 4),
        ("BOTTOMPADDING", (0,1),(-1,-1), 4),
        ("FONTNAME",      (0,-1),(-1,-1), "Helvetica-Bold"),
        ("BACKGROUND",    (0,-1),(-1,-1), colors.HexColor("#e8e8e8")),
    ]))
    return t


# ─────────────────────────────────────────────────────────────────
# Pipeline completo
# ─────────────────────────────────────────────────────────────────
def run_yolo_lion_comparison(
    yolo_models: dict, lion, processor, device: str,
    img: Image.Image, conf=0.25, iou=0.45,
    show_conf=True, use_lion_caption=True,
    include_lion_standalone=False,
    ground_truth: Optional[list[dict]] = None,
) -> dict:
    img = img.convert("RGB")
    results: dict = {}

    for name, model in yolo_models.items():
        dets, ms = run_yolo_inference(model, img, conf=conf, iou=iou)
        if use_lion_caption and lion is not None:
            dets = enrich_with_lion(dets, img, lion, processor, device)
        results[name] = (dets, ms)

    if include_lion_standalone and lion is not None:
        dets, ms = run_lion_inference(img, conf_threshold=conf, nms_iou=iou,
                                      use_caption_labels=use_lion_caption)
        results["LION"] = (dets, ms)

    annotated = {
        name: draw_yolo_boxes(img, dets,
            show_conf=show_conf and name != "LION",
            use_lion_caption=use_lion_caption)
        for name, (dets, _) in results.items()
    }

    metrics_rows, metrics_mode = build_metrics_table(results, ground_truth, iou)

    # Per-class: usa GT real ou consenso automaticamente
    if ground_truth is not None:
        gt_for_class = ground_truth
    elif len(results) >= 2:
        gt_for_class = build_consensus_ground_truth(results, iou)
    else:
        gt_for_class = None

    return {
        "results":            results,
        "annotated":          annotated,
        "comparison_table":   build_comparison_table(results),
        "diff_table":         build_diff_table(results),
        "metrics_table":      metrics_rows,
        "metrics_mode":       metrics_mode,
        "per_class_metrics":  build_per_class_metrics_table(results, gt_for_class, iou) if gt_for_class else None,
        "confidence_table":   build_confidence_table(results),
    }