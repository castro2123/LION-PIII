import gc
import re
import torch
from PIL import Image, ImageDraw, ImageFont
from model_registry import get_models


# ---------------------------------------------------------------------------
# Paleta de cores distinta por objeto
# ---------------------------------------------------------------------------
COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#469990", "#dcbeff",
    "#9a6324", "#800000", "#aaffc3", "#808000", "#000075",
    "#ffe119", "#fabed4", "#ffd8b1", "#a9a9a9", "#ff6b6b",
]


# ---------------------------------------------------------------------------
# Limpeza de labels
# ---------------------------------------------------------------------------
def _clean_label(text: str) -> str:
    if not text:
        return "object"

    text = str(text).strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) > 60:
        text = text[:57] + "..."

    return text or "object"


# ---------------------------------------------------------------------------
# Draw bounding boxes
# Apenas mostra LABEL embaixo
# ---------------------------------------------------------------------------
def _draw_boxes(img: Image.Image, detections: list[dict]) -> Image.Image:

    img_out = img.copy()
    draw = ImageDraw.Draw(img_out)

    base = max(img.width, img.height)

    line_width = max(3, base // 400)
    font_size  = max(16, base // 60)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            font_size
        )
    except Exception:
        font = ImageFont.load_default()

    for idx, det in enumerate(detections):

        x1, y1, x2, y2 = det["bbox"]

        color = COLORS[idx % len(COLORS)]

        # Bounding box
        draw.rectangle(
            [x1, y1, x2, y2],
            outline=color,
            width=line_width
        )

        # Apenas TAG abaixo
        label = _clean_label(det.get("label", "object"))

        bbox_text = draw.textbbox((0, 0), label, font=font)

        tw = bbox_text[2] - bbox_text[0]
        th = bbox_text[3] - bbox_text[1]

        pad = max(4, font_size // 4)

        tx = x1 + ((x2 - x1 - tw) // 2)
        ty = y2 + pad

        # fallback acima
        if ty + th + pad > img.height:
            ty = y1 - th - (pad * 2)

        draw.rectangle(
            [tx - pad, ty - pad, tx + tw + pad, ty + th + pad],
            fill=color
        )

        draw.text(
            (tx, ty),
            label,
            fill="white",
            font=font
        )

    return img_out


# ---------------------------------------------------------------------------
# Filtrar tags
# ---------------------------------------------------------------------------
def _filter_object_tags(
    lion,
    processed_img,
    tags,
    device
) -> list[str]:

    question = (
        "From the image, list ALL physical visible objects or entities. "
        "Include people, vehicles, buildings, furniture, food, tools, "
        "animals, roads, structures, plants, signs and containers. "
        "Do NOT include actions, colors, textures or abstract concepts. "
        "Reply ONLY with comma-separated object names."
    )

    try:

        out = lion.generate({
            "image": processed_img.unsqueeze(0).to(device),
            "question": [question],
            "tags": [tags],
            "category": "image_level",
        })

        raw = out[0] if isinstance(out, (list, tuple)) else out

        words = [
            t.strip().lower()
            for t in re.split(r"[,;\n]+", str(raw))
            if t.strip()
        ]

        seen = set()
        result = []

        for w in words:

            if not w:
                continue

            if len(w) > 40:
                continue

            if w in seen:
                continue

            seen.add(w)
            result.append(w)

        return result

    except Exception:

        if isinstance(tags, (list, tuple)):
            tags_text = ", ".join(str(t) for t in tags)
        else:
            tags_text = str(tags)

        return [
            t.strip().lower()
            for t in re.split(r"[,;]+", tags_text)
            if t.strip()
        ]


# ---------------------------------------------------------------------------
# IOU
# ---------------------------------------------------------------------------
def _iou(a: list[int], b: list[int]) -> float:

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)

    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)

    union = area_a + area_b - inter

    if union <= 0:
        return 0.0

    return inter / union


# ---------------------------------------------------------------------------
# NMS coordenadas
# ---------------------------------------------------------------------------
def _nms_coords(
    coords: list[list[int]],
    iou_threshold: float = 0.85
) -> list[list[int]]:

    kept = []

    for c in coords:

        dominated = any(
            _iou(c, k) > iou_threshold
            for k in kept
        )

        if not dominated:
            kept.append(c)

    return kept


# ---------------------------------------------------------------------------
# NMS deteções
# IMPORTANTE:
# só remove se for MESMA LABEL
# ---------------------------------------------------------------------------
def _nms_simple(
    detections: list[dict],
    iou_threshold: float = 0.45
) -> list[dict]:

    kept = []

    for det in detections:

        dominated = any(
            (
                det["label"] == k["label"]
                and
                _iou(det["bbox"], k["bbox"]) > iou_threshold
            )
            for k in kept
        )

        if not dominated:
            kept.append(det)

    return kept


# ---------------------------------------------------------------------------
# Bounding boxes por tag
# ---------------------------------------------------------------------------
def _get_bboxes_for_tag(
    lion,
    processed_img,
    img: Image.Image,
    tag: str,
    tags,
    device,
) -> list[list[int]]:

    prompts = [

        f"Find all instances of '{tag}' in the image.",

        f"Locate every '{tag}' visible in the image.",

        f"Return bounding boxes for all '{tag}'.",

        # melhor para estruturas grandes
        f"Find large background structures such as '{tag}'.",

        f"Detect the full visible extent of the '{tag}'.",
    ]

    all_coords = []

    for prompt in prompts:

        try:

            output_bbox = lion.generate({
                "image": processed_img.unsqueeze(0).to(device),
                "question": [prompt],
                "tags": [tags],
                "category": "region_level",
            })

            raw = (
                output_bbox[0]
                if isinstance(output_bbox, (list, tuple))
                else output_bbox
            )

            matches = re.findall(
                r"\[([0-9.,\s]+)\]",
                str(raw)
            )

            for b in matches:

                try:

                    coords = [
                        float(v)
                        for v in b.replace(",", " ").split()
                    ]

                    if len(coords) != 4:
                        continue

                    x1 = int(coords[0] * img.width)
                    y1 = int(coords[1] * img.height)
                    x2 = int(coords[2] * img.width)
                    y2 = int(coords[3] * img.height)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    w = x2 - x1
                    h = y2 - y1

                    # mínimo bbox
                    if w < 8 or h < 8:
                        continue

                    area_ratio = (
                        (w * h)
                        / (img.width * img.height)
                    )

                    # permite objetos gigantes
                    if area_ratio > 0.995:
                        continue

                    all_coords.append([x1, y1, x2, y2])

                except Exception:
                    continue

        except Exception:
            continue

    return _nms_coords(
        all_coords,
        iou_threshold=0.85
    )


# ---------------------------------------------------------------------------
# Descrição contextual
# Agora sem mencionar fundo/contexto
# ---------------------------------------------------------------------------
def _describe_region(
    lion,
    processor,
    img,
    bbox,
    tag,
    tags,
    device
) -> str:

    x1, y1, x2, y2 = bbox

    crop = img.crop((x1, y1, x2, y2))

    if crop.width < 20 or crop.height < 20:
        return tag

    try:

        crop_processed = processor(crop)

        question = (
            f"Describe ONLY the main '{tag}' inside the bounding box. "
            f"Do NOT mention background, nearby objects, scene context "
            f"or anything outside the object. "
            f"Use 3 to 6 words starting with '{tag}'."
        )

        out = lion.generate({
            "image": crop_processed.unsqueeze(0).to(device),
            "question": [question],
            "tags": [tags],
            "category": "image_level",
        })

        raw = out[0] if isinstance(out, (list, tuple)) else out

        text = str(raw).strip().rstrip(".").lower()

        if tag and not text.startswith(tag):
            text = f"{tag} {text}"

        text = re.sub(
            rf"\b({re.escape(tag)})(\s+\1)+\b",
            r"\1",
            text
        )

        text = _clean_label(text)

        if len(text) < len(tag):
            return tag

        return text

    except Exception:
        return tag


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def run_lion_inference(
    img: Image.Image,
    conf_threshold: float = 0.18,
    max_classes: int = 80,
    use_caption_labels: bool = True,
    fallback_classes: list[str] | None = None,
    multi_prompt: bool = True,
    nms_iou: float = 0.45,
):

    models = get_models()

    lion = models["lion"]
    processor = models["processor"]
    device = models["device"]

    img = img.convert("RGB")

    processed_img = processor(img)

    with torch.no_grad():

        # ------------------------------------------------------------
        # tags globais
        # ------------------------------------------------------------
        tags = lion.generate_tags(img)

        # ------------------------------------------------------------
        # filtrar objetos físicos
        # ------------------------------------------------------------
        tag_words = _filter_object_tags(
            lion,
            processed_img,
            tags,
            device
        )

        # fallback classes
        if fallback_classes:

            existing = set(tag_words)

            for fc in fallback_classes:

                fc = fc.strip().lower()

                if fc and fc not in existing:
                    tag_words.append(fc)
                    existing.add(fc)

        detections = []

        # ------------------------------------------------------------
        # procurar bbox por tag
        # ------------------------------------------------------------
        for tag in tag_words:

            if multi_prompt:

                bboxes = _get_bboxes_for_tag(
                    lion,
                    processed_img,
                    img,
                    tag,
                    tags,
                    device
                )

            else:

                bboxes = []

            # --------------------------------------------------------
            # captions
            # --------------------------------------------------------
            for bbox in bboxes:

                caption = tag

                if use_caption_labels:

                    caption = _describe_region(
                        lion,
                        processor,
                        img,
                        bbox,
                        tag,
                        tags,
                        device
                    )

                detections.append({
                    "label": tag,
                    "caption": caption,
                    "bbox": bbox,
                })

    # ------------------------------------------------------------
    # NMS final
    # ------------------------------------------------------------
    detections = _nms_simple(
        detections,
        iou_threshold=nms_iou
    )

    # ------------------------------------------------------------
    # draw
    # ------------------------------------------------------------
    img_out = _draw_boxes(
        img,
        detections
    )

    torch.cuda.empty_cache()
    gc.collect()

    return img_out, detections, tags