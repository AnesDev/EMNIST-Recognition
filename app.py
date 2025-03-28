import torch
import torchvision.transforms as transforms
from src.model_class import CNN 
import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

model = CNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("src/emnist_model.pt", map_location=device))
model.eval()

st.subheader("Draw a character and classify it")


canvas_result = st_canvas(
    fill_color="black",
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

if st.button("Classify Drawing"):
    if canvas_result.image_data is not None:
        img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype(np.uint8))
        img = img.resize((28, 28))
        img = img.convert("L")
        # Preprocess the image
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)
            prediction = torch.argmax(output, dim=1).item()

        st.image(img, caption="Processed Input (28x28)", width=100)
        st.write(f"### Model Prediction: **{prediction}**")
