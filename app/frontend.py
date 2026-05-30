"""
DiffusionCraft Streamlit UI

Run (after starting the API):
    python -m streamlit run app/frontend.py
"""

import streamlit as st
import requests
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="DiffusionCraft", page_icon="🤖")
st.title("✨ DiffusionCraft — Text to Image")

prompt = st.text_input("Prompt", "a beautiful mountain landscape at sunset")
col1, col2 = st.columns(2)
with col1:
    steps = st.slider("Denoising steps", 10, 100, 30)
with col2:
    cfg = st.slider("CFG scale", 1.0, 15.0, 7.5, step= 0.5)

if st.button("Generate", type= "primary"):
    with st.spinner("Generating..."):
        try:
            resp = requests.post(
                "http://localhost:8000/generate",
                json={"prompt": prompt, "steps": steps, "cfg_scale": cfg},
                timeout=120,
            )
            if resp.ok:
                img = Image.open(BytesIO(resp.content))
                st.image(img, caption=prompt, use_container_width=True)
            else:
                st.error(f"API error {resp.status_code}: {resp.text}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot reach API at localhost:8000 — is it running?")
