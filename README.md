# 🩺 Breast Cancer Detection AI

Welcome to **BreastCancerAI** — a deep learning-powered tool designed to support early detection of breast cancer through medical image analysis. This project empowers healthcare assistance and awareness using advanced computer vision and explainable AI.

---

## 🔬 About the Project

This application utilizes a **fine-tuned ResNet50 model** enhanced with **Grad-CAM (visual explainability)** and an integrated **attention mechanism** to classify breast ultrasound images into one of the following categories:

- ✅ **Benign**
- ⚠️ **Malignant**
- 😇 **Normal**

### ⚙️ Tech Stack

- 🧠 **TensorFlow/Keras** – for deep learning and model training  
- 📊 **ResNet50 (Pretrained)** – as the base CNN architecture, fine-tuned on domain-specific data  
- 👁️ **Grad-CAM** – to provide visual explanations of the model’s focus regions  
- 🧠 **Attention Layer** – added to prioritize significant regions of the image for improved performance  
- 🖼️ **Gradio** – for building an interactive web interface  
- 🚀 **Hugging Face Spaces** – for public demo deployment  

---

## 🌐 Live Demo

👉 [**Try the Model**](#) – Upload your own ultrasound image and get an instant, explainable prediction.  
_**(Coming soon / Replace # with your demo link)**_

---

## 🖼️ How to Use

1. **Upload** a breast ultrasound image.  
2. The model returns one of:
   - ✅ *Benign*
   - ⚠️ *Malignant*
   - 😇 *Normal*
3. View:
   - 📈 **Confidence score**
   - 🔍 **Grad-CAM heatmap** to understand what the model focused on

> ⚠️ **Disclaimer**: This tool is for educational and research purposes only. It is **not** a substitute for professional medical diagnosis or advice.

---

## 📃 Datasets Used

This model was trained using the following publicly available datasets:

- [**Breast Ultrasound Images Dataset**](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) – Provided by Arya Shah, this dataset contains labeled breast ultrasound images grouped into three classes: *Benign*, *Malignant*, and *Normal*.

- [**UDIAT Breast Ultrasound Dataset**](https://www.kaggle.com/datasets/farhansayyed/udiat-breast-ultrasound-dataset) – Contains annotated images and corresponding findings from the UDIAT Diagnostic Centre, categorized as *Benign* or *Malignant*.

---

## 🤝 Contributions

Feel free to fork, raise issues, or contribute improvements related to performance, explainability, or interface design.

---

## 📜 License

This project is licensed under the MIT License.

