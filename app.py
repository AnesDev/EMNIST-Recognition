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
    stroke_width=7,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

emnist_map = {0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9', 10:'A', 11:'B',
       12:'C', 13:'D', 14:'E', 15:'F', 16:'G', 17:'H', 18:'I', 19:'J', 20:'K', 21:'L', 22:'M',
       23:'N', 24:'O', 25:'P', 26:'Q', 27:'R', 28:'S', 29:'T', 30:'U', 31:'V', 32:'W', 33:'X',
       34:'Y', 35:'Z', 36:'a', 37:'b', 38:'d', 39:'e', 40:'f', 41:'g', 42:'h', 43:'n', 44:'q',
       45:'r', 46:'t'}

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
        st.write(f"### Model Prediction: **{emnist_map[prediction]}**")
