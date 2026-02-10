# model_registry.py
import torch
import streamlit as st
from ultralytics import YOLO
from models import load_model
from preprocessors.lion_preprocessors import ImageEvalProcessor


@st.cache_resource
def get_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------
    # YOLO clássicos
    # -------------------------
    yolo_det = YOLO("yolov8s.pt")
    yolo_seg = YOLO("yolov8s-seg.pt")

    # -------------------------
    # YOLO-World (open-vocabulary)
    # -------------------------
    yolo_world = YOLO("yolov8s-world.pt")

    # -------------------------
    # LION (caption / tags)
    # -------------------------
    lion = load_model(
        name="lion_t5",
        model_type="flant5xl",
        is_eval=True,
        device=device
    )

    processor = ImageEvalProcessor()

    return {
        "yolo_det": yolo_det,
        "yolo_seg": yolo_seg,
        "yolo_world": yolo_world,   # 🔥 NOVO
        "lion": lion,
        "processor": processor,
        "device": device
    }
