import torch
import torchvision.transforms as transforms
from src.model_class import CNN 
import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

model = CNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("src/model.pt", map_location=device))
model.eval()

st.subheader("Draw a character and classify it")


canvas_result = st_canvas(
    fill_color="black",
    stroke_width=7,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

def idx_to_char(idx):
    label_map = {
        0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
        5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
        10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e',
        15: 'f', 16: 'g', 17: 'h', 18: 'i', 19: 'j',
        20: 'k', 21: 'l', 22: 'm', 23: 'n', 24: 'o',
        25: 'p', 26: 'q', 27: 'r', 28: 's', 29: 't',
        30: 'u', 31: 'v', 32: 'w', 33: 'x', 34: 'y',
        35: 'z',
        36: 'A', 37: 'B', 38: 'C', 39: 'D', 40: 'E',
        41: 'F', 42: 'G', 43: 'H', 44: 'I', 45: 'J',
        46: 'K', 47: 'L', 48: 'M', 49: 'N', 50: 'O',
        51: 'P', 52: 'Q', 53: 'R', 54: 'S', 55: 'T',
        56: 'U', 57: 'V', 58: 'W', 59: 'X', 60: 'Y', 61: 'Z'
    }
    return label_map.get(idx, '?')

if st.button("Classify Drawing"):
    if canvas_result.image_data is not None:
        img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype(np.uint8))
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
        with torch.no_grad():
            output = model(img_tensor)
            prediction = torch.argmax(output, dim=1).item()

        st.write(f"### Model Prediction: **{idx_to_char(prediction)}**")
