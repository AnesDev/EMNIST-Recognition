# 🖊️ Handwritten Character Recognition
🚀 **Current Status:** Model is improving but still needs optimization.🔄  

## 📌 Overview  
This project is a web-based application that allows users to draw characters (digits, uppercase, and lowercase letters), which are then recognized by a **Convolutional Neural Network (CNN)**.  
Built with **PyTorch** and **Streamlit**, the app provides an interactive platform for handwritten character recognition, using a **custom-generated dataset** created from fonts and augmentations.

## ✨ Features  
- 🖌️ **Canvas** to draw digits and letters  
- 📸 **Image preprocessing** for better model input  
- 🧠 **CNN-based classifier** trained on a custom dataset  
- 🌐 **Web-based UI** powered by Streamlit

## 📂 Project Structure  
```bash
Handwritten-Character-Recognition/
│
├── data/
│   ├── dataset/                 # All image data (digit_*, lower_*, upper_*)
│   ├── fonts/                   # Font .ttf files used to generate images
│   ├── augmentations.py         # Augmentation functions
│   ├── generate_images.py       # Script to generate dataset using fonts + augmentations
│   └── dataset.zip              # Zipped dataset folder
│
├── src/
│   ├── model_class.py           # CNN Model definition
│   ├── notebook.ipynb           # Training and evaluation notebook
│   └── model.pt                 # Trained model weights
│
├── app.py                       # Streamlit app
├── requirements.txt             # Python dependencies
└── README.md                    # You're here!
```

## 🏠 Installation  
Clone the repository:  

```bash
git clone https://github.com/AnesDev/EMNIST-Recognition.git
cd EMNIST-Recognition
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Usage  
Run the Streamlit app:

```bash
streamlit run app.py
```

Then, open the localhost URL displayed in the terminal to use the app.

---
Looking forward to your feedback! 🚀✨  

