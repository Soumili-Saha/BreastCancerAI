# 🧠 Breast Cancer Detection AI

Welcome to **BreastCancerAI**, a machine learning-powered tool designed to assist in the early detection of breast cancer from medical images. This project leverages deep learning to classify breast images as **Benign** or **Malignant**, helping support healthcare professionals and awareness efforts.

## 🔬 About the Project

This application uses a trained convolutional neural network (CNN) model to predict the likelihood of breast cancer based on uploaded images (e.g., Ultrasound images). It is built using:

- 🧠 **TensorFlow/Keras** for model training
- 🖼️ **Gradio** for an interactive web-based interface
- 🚀 **Hugging Face Spaces** for public deployment

## 🌐 Live Demo

👉 [Try the Model](https://huggingface.co/spaces/SoumiliSaha/BreastCancerAI)

## 🖼️ How to Use

1. Upload a breast ultrasound image.
2. The model will analyze the image and return one of:
   - ✅ **Benign**
   - ⚠️ **Malignant**
   - 😇 **Normal**
3. Use the output to assist diagnosis (not a substitute for professional evaluation).

> ⚠️ **Disclaimer:** This tool is for educational and research purposes only. Always consult a medical professional for actual diagnosis.


Here’s how the interface works:

- Upload a sample image
- View prediction instantly
- Confidence score shown for transparency
- Explains the prediction

  
 ## 📃 Datasets Used

This project was trained using the following publicly available datasets:

- [**Breast Ultrasound Images Dataset (Benign, Malignant, Normal)** – Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)  
  Provided by **Arya Shah**, this dataset contains labeled breast ultrasound images grouped into three classes: **Benign**, **Malignant**, and **Normal**.

- [*UDIAT Breast Ultrasound Dataset** – Kaggle](https://www.kaggle.com/datasets/jarintasnim090/udiat-data)  
  This dataset contains images and corresponding findings from the UDIAT Diagnostic Centre, classified as **benign** or **malignant**, along with image metadata.

