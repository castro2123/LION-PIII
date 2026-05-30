# utils.py
from PIL import Image
import streamlit as st

def show_result_image(img, max_width=900):
    """Mostra imagem com boa qualidade, mantendo aspect ratio."""
    w, h = img.size
    if w > max_width:
        scale = max_width / w
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, Image.LANCZOS)  # filtro de alta qualidade
    # Deixar o Streamlit mostrar no tamanho nativo da imagem já redimensionada
    st.image(img, use_container_width=False)
