import streamlit as st
import torch
from PIL import Image
from models import load_model
from preprocessors.lion_preprocessors import ImageEvalProcessor


# -------------------------
# Lazy loading + cache
# -------------------------
@st.cache_resource
def load_lion():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = load_model(
        "lion_t5",
        "flant5xl",
        is_eval=True,
        device=device
    )
    processor = ImageEvalProcessor()
    return model, processor, device


# -------------------------
# Função principal
# -------------------------
def run_caption(image_path: str) -> dict:
    lion_model, lion_preprocessor, device = load_lion()

    img = Image.open(image_path).convert("RGB")

    question = "Please describe the objects in the image."
    tags = lion_model.generate_tags(img)
    processed_img = lion_preprocessor(img)

    output = lion_model.generate({
        "image": processed_img.unsqueeze(0).to(device),
        "question": [question],
        "tags": [tags],
        "category": "image_level",
    })

    return {
        "question": question,
        "tags": tags,
        "answer": output[0]
    }

if __name__ == "__main__":
    result = run_caption("images/COCO_train2014_000000024935.jpg")
    print("Question:", result["question"])
    print("Tags:", result["tags"])
    print("Answer:", result["answer"])
