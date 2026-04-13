# AI Image Authenticator 🛡️

A web-based application built with **Streamlit** and **PyTorch** that detects whether an image is Real, AI-Generated, or Manipulated (Deepfake).

## 🚀 Features
- **Real-time Detection**: Uses a Vision Transformer (ViT) fine-tuned for synthetic image detection.
- **Modern UI**: Clean, interactive interface with image previews and dynamic charts.
- **Tri-Fold Analysis**: Classifies images into three distinct categories:
  - ✅ **Real Image**
  - 🤖 **AI-Generated Image**
  - ⚠️ **Real Image (AI Edited/Manipulated)**

## 🛠️ Tech Stack
- **Frontend**: Streamlit
- **Deep Learning**: PyTorch & Transformers (Hugging Face)
- **Visualization**: Plotly & Pandas
- **Image Processing**: Pillow (PIL)

## 📦 Installation & Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

## 📂 Project Structure
```text
ai_image_detector/
├── app.py              # Main Streamlit application and inference logic
├── requirements.txt    # Project dependencies
└── README.md           # Documentation
```

## 📝 Note
The application uses the `umm-maybe/AI-image-detector` model by default. For production environments, consider fine-tuning on the latest datasets (DALL-E 3, Midjourney v6, etc.) to maintain high accuracy against evolving generative AI models.
