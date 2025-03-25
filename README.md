# 🖊️ EMNIST Handwritten Text Recognition  

🚀 **Current Status:** Model is improving but still needs optimization.🔄  

## 📌 Overview  
This project is a web-based application that allows users to draw letters or digits, which are then recognized by a Convolutional Neural Network (CNN) trained on the **EMNIST dataset**. Built with **PyTorch** and **Streamlit**, this app provides a simple and interactive way to test handwritten text recognition.  

## ✨ Features  
- 🖌️ **Canvas** to draw letters/digits  
- 📸 **Image preprocessing** for better model input  
- 🧠 **CNN-based classifier** trained on the EMNIST dataset  
- 🌐 **Web-based UI** powered by Streamlit  

## 📂 Project Structure  
```bash
EMNIST-Recognition/
│── data/               # Contains the dataset CSV files
│   ├── train.csv       # Training data
│   ├── test.csv        # Test data
│── src/                # Source code for model and training
│   ├── model_class.py  # CNN Model definition
│   ├── notebook.ipynb  # Jupyter Notebook used for training and exporting model state
│   ├── model.pt        # Saved model state after training
│── app.py              # Streamlit web app
│── requirements.txt    # Dependencies
│── README.md           # Project documentation
```

## 🏗️ Installation  

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