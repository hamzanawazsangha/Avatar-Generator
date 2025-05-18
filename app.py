import streamlit as st
from PIL import Image
import cv2
import numpy as np
import torch
from huggingface_hub import login
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

# ------------------------------
# Hugging Face Token from Streamlit secrets
HF_TOKEN = st.secrets["HF_TOKEN"]

# Login to Hugging Face
login(token=HF_TOKEN)

# ------------------------------
# Load DreamBooth + ControlNet model pipeline
@st.cache_resource(show_spinner=True)
def load_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny", torch_dtype=dtype
    )
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", 
        torch_dtype=torch.float16
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    return pipe

# ------------------------------
# Image preprocessing using OpenCV (Canny edges)
def preprocess_image(image: Image.Image) -> Image.Image:
    image_cv = np.array(image.convert("RGB"))
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    image_cv = cv2.resize(image_cv, (512, 512))

    edges = cv2.Canny(image_cv, 100, 200)

    edges_pil = Image.fromarray(edges)
    edges_pil = edges_pil.convert("RGB")
    return edges_pil

# ------------------------------
# Streamlit UI
st.title("Custom AI Avatar Generator with DreamBooth + ControlNet")

uploaded_file = st.file_uploader("Upload a base image (e.g., face sketch or photo)", type=["png", "jpg", "jpeg"])
prompt = st.text_input("Enter prompt for avatar generation", "A fantasy portrait of a warrior, digital art")

if uploaded_file and prompt:
    input_image = Image.open(uploaded_file)
    st.image(input_image, caption="Uploaded Image", use_container_width=True)

    control_image = preprocess_image(input_image)
    st.image(control_image, caption="Preprocessed ControlNet Input (Edges)", use_container_width=True)

    pipe = load_pipeline()

    with st.spinner("Generating avatar..."):
        output = pipe(prompt=prompt, image=control_image, num_inference_steps=30)
        avatar = output.images[0]
        st.image(avatar, caption="Generated AI Avatar", use_container_width=True)
