# utils.py
from PIL import Image
import streamlit as st

def show_result_image(img, max_width=500):
    """Mostra imagem mantendo aspect ratio."""
    w, h = img.size
    scale = max_width / w
    new_size = (int(w * scale), int(h * scale))
    st.image(img.resize(new_size), width=max_width)
