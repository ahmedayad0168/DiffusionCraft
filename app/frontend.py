import streamlit as st
import requests
from PIL import Image
from io import BytesIO

st.set_page_config(page_title= "DiffusionCraft")
st.title("Text to Image Generation")

prompt = st.text_input("Enter your prompt", "a beautiful mountain landscape")
steps = st.slider("Denoising steps", 10, 100, 50)

if st.button("Generate"):
    with st.spinner("Generating..."):
        resp = requests.post("http://localhost:8000/generate", json= {"prompt": prompt, "steps": steps})
        if resp.ok:
            img = Image.open(BytesIO(resp.content))
            st.image(img, caption= prompt, use_container_width= True)
        else:
            st.error(resp.text)

# python -m streamlit run app/frontend.py