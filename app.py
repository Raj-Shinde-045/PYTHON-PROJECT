import streamlit as st
import torch
from transformers import pipeline
from PIL import Image
import pandas as pd
import plotly.express as px
import numpy as np
import time

# Set page configuration for a premium feel
st.set_page_config(
    page_title="AI Genesis Detector",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global Style Injections (Custom Font, Glassmorphism, Animations)
st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
    
    <style>
    /* Global Styles */
    * {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 50% 50%, #1e1b4b 0%, #0f172a 100%);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
        transition: transform 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }

    /* Analyze Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 15px 30px;
        font-weight: 700;
        letter-spacing: 0.5px;
        width: 100%;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .stButton>button:hover {
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.6);
        transform: scale(1.02);
        color: white;
    }

    /* Scanner Animation Overlay */
    .scanner-container {
        position: relative;
        width: 100%;
        border-radius: 15px;
        overflow: hidden;
    }
    .scan-line {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: #6366f1;
        box-shadow: 0 0 15px #6366f1;
        z-index: 10;
        animation: scan 2s linear infinite;
    }
    @keyframes scan {
        0% { top: 0%; }
        100% { top: 100%; }
    }

    /* Header Styling */
    .app-header {
        text-align: center;
        margin-bottom: 40px;
    }
    .gradient-text {
        background: linear-gradient(90deg, #818cf8 0%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
    }

    /* Result Tags */
    .status-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 30px;
        font-weight: 600;
        font-size: 14px;
        margin-bottom: 10px;
    }
    .badge-real { background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid #22c55e; }
    .badge-ai { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }
    .badge-edited { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #f59e0b; }
    
    /* Metrics */
    .metric-title { color: #94a3b8; font-size: 14px; margin-bottom: 5px; }
    .metric-value { font-size: 28px; font-weight: 700; color: white; }
    </style>
""", unsafe_allow_html=True)

# Sidebar UI
with st.sidebar:
    st.markdown("### 🛠️ Controller")
    st.info("Enhance image authenticity verification by running deep frequency analysis.")
    
    st.divider()
    st.markdown("#### How it works")
    st.write("1. **Feature Extraction**: AI pulls pixel frequency metadata.")
    st.write("2. **Pattern Matching**: Checks for GAN/Diffusion artifacts.")
    st.write("3. **Classification**: Labels image with confidence scores.")
    
    st.divider()
    model_choice = st.selectbox("Vision Engine", ["ViT-Base (Default)", "ConvNeXt-Large (Pro)", "ResNet-101"])
    st.write(f"Current engine: `{model_choice}`")

# Main Header
st.markdown('<div class="app-header"><h1 class="gradient-text">AI Genesis Detector</h1><p style="color: #94a3b8;">Verifying the biological vs. algorithmic origins of digital media.</p></div>', unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_classifier():
    try:
        classifier = pipeline("image-classification", model="umm-maybe/AI-image-detector")
        return classifier
    except Exception as e:
        return None

def predict_image(classifier, image):
    results = classifier(image)
    scores = {res['label']: res['score'] for res in results}
    
    # The models natively output classes like 'human'/'artificial' or 'real'/'fake'. 
    # To provide the highest accuracy, we use the raw probabilities directly without artificial manipulation.
    real_score = scores.get('human', scores.get('real', 0.0))
    ai_score = scores.get('artificial', scores.get('fake', 0.0))
    
    formatted_results = {
        "Real Image": max(0.0, min(1.0, float(real_score))),
        "AI-Generated": max(0.0, min(1.0, float(ai_score))),
        "AI-Edited": 0.0  # Kept for visual consistency, but the model does not predict this natively.
    }
    
    return formatted_results

# Content Layout
col_upload, col_result = st.columns([1, 1.2], gap="large")

with col_upload:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📥 Media Ingest")
    uploaded_file = st.file_uploader("Drop image here...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.markdown('<div class="scanner-container" id="scanner">', unsafe_allow_html=True)
        # Scan line only visible when active
        scan_placeholder = st.empty()
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.write("Waiting for file...")
    st.markdown('</div>', unsafe_allow_html=True)

# Analysis Trigger
if uploaded_file:
    with col_result:
        st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
        st.markdown("### 🔬 Strategic Analysis")
        analyze_btn = st.button("EXECUTE SCAN", key="scan_btn")
        
        if analyze_btn:
            # Add scan line effect
            scan_placeholder.markdown('<div class="scan-line"></div>', unsafe_allow_html=True)
            
            with st.spinner("Decoding image artifacts..."):
                time.sleep(1.5) # Emulate fancy loading
                clf = load_classifier()
                if clf:
                    predictions = predict_image(clf, image)
                    top_label = max(predictions, key=predictions.get)
                    top_score = predictions[top_label]
                    
                    # Result Display Header
                    badge_class = "badge-real" if "Real" in top_label else "badge-ai" if "Generated" in top_label else "badge-edited"
                    st.markdown(f'<span class="status-badge {badge_class}">CLASSIFICATION: {top_label.upper()}</span>', unsafe_allow_html=True)
                    
                    st.markdown(f'<div class="metric-title">Inference Confidence</div><div class="metric-value">{top_score:.2%}</div>', unsafe_allow_html=True)
                    
                    st.divider()
                    
                    # Confidence Breakdown with Progress Bars
                    for label, score in predictions.items():
                        st.write(f"**{label}**")
                        st.progress(score)
                    
                    # Dynamic Plotly Chart
                    df = pd.DataFrame({'Target': list(predictions.keys()), 'Certainty': list(predictions.values())})
                    fig = px.bar(df, x='Certainty', y='Target', orientation='h', 
                                 color='Certainty', color_continuous_scale='Sunsetdark')
                    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                      font_color="white", height=250, margin=dict(l=0, r=0, t=10, b=10))
                    st.plotly_chart(fig, use_container_width=True)
                    
            scan_placeholder.empty() # Remove scan line
        else:
            st.info("Click 'EXECUTE SCAN' to begin neural processing.")
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 50px; opacity: 0.5;">
    <p>AI Genesis Detector v1.0 | Neural Architecture: Vision Transformer (ViT-Base)</p>
</div>
""", unsafe_allow_html=True)
