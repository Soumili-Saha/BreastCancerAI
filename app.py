import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from PIL import Image
import io

# Load your model
model = tf.keras.models.load_model("model.h5")

# Define your class names
class_names = ["Benign", "Malignant", "Normal"]  # Change as needed

# Enhanced explanation generator
def generate_detailed_explanation(label, confidence, all_predictions):
    confidence_level = "high" if confidence > 0.8 else "moderate" if confidence > 0.6 else "low"
    
    explanations = {
        "Benign": [
            f"The AI model has identified this ultrasound image as showing a BENIGN LESION with {confidence_level} confidence ({confidence:.1f}%).",
            "Benign lesions are non-cancerous growths that typically do not spread to other parts of the body.",
            "While this suggests a favorable outcome, medical correlation and professional evaluation remain essential for accurate diagnosis."
        ],
        "Malignant": [
            f"The AI analysis indicates a MALIGNANT LESION with {confidence_level} confidence ({confidence:.1f}%).",
            "Malignant lesions are cancerous growths that have the potential to spread to surrounding tissues or other parts of the body.",
            "This finding requires immediate medical attention and further diagnostic evaluation by a qualified healthcare professional."
        ],
        "Normal": [
            f"The ultrasound image appears to show NORMAL TISSUE with {confidence_level} confidence ({confidence:.1f}%).",
            "Normal tissue classification suggests no obvious abnormalities detected in the scanned area.",
            "Regular screening and follow-up as recommended by healthcare providers remain important for ongoing breast health."
        ]
    }
    
    base_explanation = explanations[label]
    
    # Add confidence analysis
    sorted_preds = sorted(zip(class_names, all_predictions), key=lambda x: x[1], reverse=True)
    second_highest = sorted_preds[1]
    
    if second_highest[1] > 0.3:
        base_explanation.append(f"The model also considered {second_highest[0]} as a possibility ({second_highest[1]*100:.1f}% confidence), indicating some uncertainty in the classification.")
    
    # Add general disclaimer
    base_explanation.append("IMPORTANT: This AI analysis is for educational purposes only and should never replace professional medical diagnosis or treatment decisions.")
    
    return " ".join(base_explanation)

# Grad-CAM function
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv5_block3_out"):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Heatmap overlay
def overlay_heatmap(heatmap, image):
    heatmap = cv2.resize(heatmap, (image.width, image.height))
    heatmap = np.uint8(255 * heatmap)
    heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(np.array(image), 0.6, heatmap_img, 0.4, 0)
    return Image.fromarray(superimposed)

# Prediction function
def predict(image):
    resized = image.resize((224, 224))
    array = np.array(resized) / 255.0
    array = np.expand_dims(array, axis=0)
    preds = model.predict(array)[0]
    class_idx = np.argmax(preds)
    label = class_names[class_idx]
    confidence = preds[class_idx] * 100
    
    # Generate detailed explanation
    explanation = generate_detailed_explanation(label, confidence, preds)
    
    heatmap = make_gradcam_heatmap(array, model)
    cam_image = overlay_heatmap(heatmap, image)
    
    # Enhanced plotting with black-pink theme
    plt.style.use('dark_background')
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#0a0a0a')
    
    # Original image
    axs[0].imshow(image)
    axs[0].axis('off')
    axs[0].set_title("Original Image", color='#ff69b4', fontsize=14, fontweight='bold')
    
    # Heatmap overlay
    axs[1].imshow(cam_image)
    axs[1].axis('off')
    axs[1].set_title("AI Attention Heatmap", color='#ff69b4', fontsize=14, fontweight='bold')
    
    # Confidence scores with enhanced styling
    colors = ['#ff1493' if i == class_idx else '#ff69b4' for i in range(len(class_names))]
    bars = axs[2].barh(class_names, preds, color=colors, alpha=0.8)
    axs[2].set_xlim(0, 1)
    axs[2].set_facecolor('#0a0a0a')
    axs[2].tick_params(colors='white')
    axs[2].spines['bottom'].set_color('#ff69b4')
    axs[2].spines['left'].set_color('#ff69b4')
    axs[2].spines['top'].set_visible(False)
    axs[2].spines['right'].set_visible(False)
    
    # Add percentage labels
    for i, v in enumerate(preds):
        axs[2].text(v + 0.02, i, f"{v*100:.1f}%", va='center', color='white', fontweight='bold')
    
    axs[2].set_title("Confidence Scores", color='#ff69b4', fontsize=14, fontweight='bold')
    axs[2].set_xlabel("Confidence Level", color='white', fontweight='bold')
    
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', facecolor='#0a0a0a', dpi=150)
    buf.seek(0)
    plt.close(fig)
    
    return Image.open(buf), explanation

# Custom CSS for black-pink theme
custom_css = """
#component-0, #component-1, #component-2 {
    background: linear-gradient(135deg, #0a0a0a 0%, #1a0a1a 100%) !important;
    border-radius: 15px !important;
    border: 2px solid #ff69b4 !important;
}
.gradio-container {
    background: linear-gradient(135deg, #000000 0%, #1a0a1a 50%, #000000 100%) !important;
    font-family: 'Arial', sans-serif !important;
}
.gr-button {
    background: linear-gradient(45deg, #ff1493, #ff69b4) !important;
    border: none !important;
    color: white !important;
    font-weight: bold !important;
    border-radius: 10px !important;
    transition: all 0.3s ease !important;
}
.gr-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 5px 15px rgba(255, 105, 180, 0.4) !important;
}
.gr-textbox textarea {
    background: rgba(255, 105, 180, 0.1) !important;
    border: 1px solid #ff69b4 !important;
    color: white !important;
    border-radius: 10px !important;
}
h1 {
    color: #ff69b4 !important;
    text-align: center !important;
    text-shadow: 2px 2px 4px rgba(255, 105, 180, 0.5) !important;
    font-size: 2.5em !important;
    margin-bottom: 10px !important;
}
.gr-markdown {
    color: #e0e0e0 !important;
}
.gr-markdown p {
    text-align: center !important;
    font-size: 1.1em !important;
    color: #ff69b4 !important;
}
.gr-image {
    border-radius: 15px !important;
    border: 2px solid #ff69b4 !important;
}
.gr-panel {
    background: rgba(255, 105, 180, 0.05) !important;
    border-radius: 15px !important;
    border: 1px solid #ff69b4 !important;
}
"""

# Gradio app with enhanced UI
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="📸 Upload Breast Ultrasound Image"),
    outputs=[
        gr.Image(type="pil", label="🔍 AI Analysis & Visualization"),
        gr.Textbox(label="📋 Detailed Medical Analysis", lines=6, max_lines=10)
    ],
    title="🩺 Advanced Breast Ultrasound AI Classifier",
    description="""
    <div style='text-align: center; padding: 20px;'>
        <p style='font-size: 1.2em; color: #ff69b4; margin-bottom: 15px;'>
            🤖 Powered by Deep Learning & Grad-CAM Visualization
        </p>
        <p style='color: #e0e0e0; font-size: 1em; line-height: 1.6;'>
            Upload a breast ultrasound image to receive AI-powered analysis with detailed explanations 
            and attention visualization showing which areas the model focused on for its prediction.
        </p>
        <p style='color: #ff69b4; font-size: 0.9em; margin-top: 15px;'>
            ⚠️ For educational and research purposes only - Not for clinical diagnosis
        </p>
    </div>
    """,
    css=custom_css,
    theme=gr.themes.Default(
        primary_hue=gr.themes.colors.pink,
        secondary_hue=gr.themes.colors.purple,
        neutral_hue=gr.themes.colors.gray
    ),
    examples=None,
    allow_flagging="never"
)

demo.launch()