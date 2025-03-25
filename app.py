import torch
import torchvision.transforms as transforms
from src.model_class import CNN 
import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas


model = CNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("emnist_model.pt", map_location=device))
model.eval()

st.subheader("Test Sample from Dataset")
if st.button("Test Exported Sample"):
    sample_img = Image.open("test_sample.png").convert("L")
    sample_img = sample_img.resize((28, 28))  # Resize to (28, 28)

    # Preprocess the image:
    transform_test = transforms.Compose([
        transforms.ToTensor(),  
        transforms.Normalize((0.5,), (0.5,))
    ])
    img_tensor = transform_test(sample_img).unsqueeze(0)  # Shape: (1, 1, 28, 28)

    with torch.no_grad():
        output = model(img_tensor)
        prediction = torch.argmax(output, dim=1).item()

    st.image(sample_img, caption="Test Sample (28x28)", width=100)
    st.write(f"### Model Prediction: **{prediction}**")