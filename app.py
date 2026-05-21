import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import pickle
import cv2
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="EcoGuard AI - Smart Waste Classifier",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1B5E20;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2E7D32;
        margin-bottom: 1rem;
    }
    .classification-result {
        background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 100%);
        padding: 2rem;
        border-radius: 15px;
        color: #FFFFFF;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .classification-result h1 { color: #FFFFFF !important; }
    .classification-result h2 { color: #A5D6A7 !important; }
    .classification-result h3 { color: #E8F5E9 !important; }
    .recycling-tip {
        background: #F5F5F5;
        padding: 1.2rem;
        border-left: 5px solid #2E7D32;
        margin: 1rem 0;
        border-radius: 8px;
        color: #212121;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    }
    .recycling-tip p { color: #212121; margin: 0; line-height: 1.6; }
    .recycling-tip strong { color: #1B5E20; }
    .stButton > button {
        background: linear-gradient(135deg, #2E7D32 0%, #1B5E20 100%);
        color: #FFFFFF !important;
        padding: 0.7rem 2rem;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        font-size: 1em;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1B5E20 0%, #0D3B0F 100%);
        color: #FFFFFF !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        border: none;
    }
    .challenge-card {
        background: #FFFFFF;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border: 1px solid #E0E0E0;
    }
    .challenge-card h3 { color: #1B5E20; }
    .challenge-card p { color: #424242; }
    .quick-stat-card {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
        height: 100%;
    }
    .quick-stat-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .stat-emoji { font-size: 2.2em; margin-bottom: 8px; }
    .stat-label { font-weight: 600; color: #424242; margin: 5px 0; font-size: 0.9em; text-transform: uppercase; letter-spacing: 0.5px; }
    .stat-value { font-size: 1.3em; font-weight: bold; margin: 8px 0; }
    .stat-subtitle { font-size: 0.85em; opacity: 0.8; margin: 0; }
    .stat-blue { background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%); }
    .stat-blue .stat-value { color: #1565C0; }
    .stat-blue .stat-subtitle { color: #1976D2; }
    .stat-green { background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%); }
    .stat-green .stat-value { color: #2E7D32; }
    .stat-green .stat-subtitle { color: #388E3C; }
    .stat-orange { background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%); }
    .stat-orange .stat-value { color: #E65100; }
    .stat-orange .stat-subtitle { color: #EF6C00; }
    .stat-purple { background: linear-gradient(135deg, #F3E5F5 0%, #E1BEE7 100%); }
    .stat-purple .stat-value { color: #6A1B9A; }
    .stat-purple .stat-subtitle { color: #7B1FA2; }
    .success-message {
        background: #E8F5E9;
        color: #1B5E20;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
    }
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #616161;
        border-top: 1px solid #E0E0E0;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# CLASSIFICATION FUNCTION
# ============================================
def classify_waste_image(image):
    """Improved waste classification using multiple image features"""
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    img = cv2.resize(img_array, (224, 224))
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    avg_rgb = np.mean(img, axis=(0, 1))
    avg_hsv = np.mean(hsv, axis=(0, 1))
    std_rgb = np.std(img, axis=(0, 1))
    
    r, g, b = avg_rgb
    h, s, v = avg_hsv
    
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.mean(edges) / 255.0
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_var = np.var(laplacian)
    
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(gray, kernel, iterations=1)
    eroded = cv2.erode(gray, kernel, iterations=1)
    morph_grad = np.mean(dilated - eroded)
    
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_contours = len(contours)
    total_area = sum(cv2.contourArea(c) for c in contours)
    
    total_pixels = img.shape[0] * img.shape[1]
    
    white_mask = cv2.inRange(img, np.array([200, 200, 200]), np.array([255, 255, 255]))
    white_ratio = np.sum(white_mask > 0) / total_pixels
    dark_mask = cv2.inRange(img, np.array([0, 0, 0]), np.array([50, 50, 50]))
    dark_ratio = np.sum(dark_mask > 0) / total_pixels
    brown_mask = cv2.inRange(hsv, np.array([10, 30, 30]), np.array([30, 255, 200]))
    brown_ratio = np.sum(brown_mask > 0) / total_pixels
    green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    green_ratio = np.sum(green_mask > 0) / total_pixels
    gray_mask = cv2.inRange(hsv, np.array([0, 0, 50]), np.array([180, 30, 200]))
    gray_ratio = np.sum(gray_mask > 0) / total_pixels
    blue_mask = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
    blue_ratio = np.sum(blue_mask > 0) / total_pixels
    red_mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
    red_ratio = (np.sum(red_mask1 > 0) + np.sum(red_mask2 > 0)) / total_pixels
    yellow_mask = cv2.inRange(hsv, np.array([20, 50, 50]), np.array([35, 255, 255]))
    yellow_ratio = np.sum(yellow_mask > 0) / total_pixels
    
    scores = {'plastic': 0, 'paper': 0, 'glass': 0, 'metal': 0, 'organic': 0, 'e-waste': 0}
    
    # PLASTIC
    if s > 35 and v > 80: scores['plastic'] += 20
    if edge_density < 0.12: scores['plastic'] += 15
    if laplacian_var < 500: scores['plastic'] += 10
    if red_ratio > 0.1 or blue_ratio > 0.1 or yellow_ratio > 0.1: scores['plastic'] += 15
    if white_ratio > 0.3 and s < 30: scores['plastic'] += 20
    if 100 < r < 220 and 100 < g < 220 and 100 < b < 220: scores['plastic'] += 10
    
    # PAPER
    if v > 140 and s < 50: scores['paper'] += 25
    if white_ratio > 0.3: scores['paper'] += 20
    if brown_ratio > 0.15 and s < 60: scores['paper'] += 25
    if 0.08 < edge_density < 0.2: scores['paper'] += 15
    if laplacian_var > 300: scores['paper'] += 15
    if morph_grad > 5: scores['paper'] += 10
    if (r > 180 and g > 170 and b > 150) or (r > 140 and g > 120 and b > 90 and r > g > b): scores['paper'] += 15
    
    # GLASS
    if s < 40 and v > 140: scores['glass'] += 20
    if std_rgb[0] > 45 or std_rgb[1] > 45 or std_rgb[2] > 45: scores['glass'] += 20
    if edge_density > 0.14: scores['glass'] += 15
    if laplacian_var > 800: scores['glass'] += 15
    if gray_ratio > 0.2 and s < 25: scores['glass'] += 15
    if white_ratio > 0.4 and s < 20: scores['glass'] += 10
    
    # METAL
    if gray_ratio > 0.4 and s < 35: scores['metal'] += 30
    if abs(r - g) < 25 and abs(g - b) < 25 and abs(r - b) < 25: scores['metal'] += 20
    if std_rgb[0] > 40 or std_rgb[1] > 40: scores['metal'] += 15
    if laplacian_var > 600: scores['metal'] += 15
    if 100 < v < 200 and s < 30: scores['metal'] += 15
    if dark_ratio < 0.05 and white_ratio < 0.3: scores['metal'] += 10
    
    # ORGANIC
    if brown_ratio > 0.15: scores['organic'] += 35
    if green_ratio > 0.1: scores['organic'] += 30
    if yellow_ratio > 0.15 and s > 30: scores['organic'] += 15
    if edge_density > 0.1: scores['organic'] += 15
    if morph_grad > 8: scores['organic'] += 15
    if num_contours > 10 and total_area > 5000: scores['organic'] += 10
    if (r > g and g > b) and s > 20: scores['organic'] += 15
    if 15 < h < 50 and s > 25: scores['organic'] += 10
    
    # E-WASTE
    if dark_ratio > 0.3: scores['e-waste'] += 30
    if edge_density > 0.18: scores['e-waste'] += 25
    if laplacian_var > 1000: scores['e-waste'] += 20
    if num_contours > 20: scores['e-waste'] += 15
    if std_rgb[0] > 55 and std_rgb[1] > 55: scores['e-waste'] += 15
    if r < 100 and g < 100 and b < 100: scores['e-waste'] += 20
    if blue_ratio > 0.05 and dark_ratio > 0.2: scores['e-waste'] += 15
    if red_ratio > 0.05 and dark_ratio > 0.2: scores['e-waste'] += 10
    
    # DISCRIMINATION
    if white_ratio > 0.5 and s < 30:
        scores['organic'] *= 0.3
        scores['paper'] += 20
    if dark_ratio > 0.5:
        scores['organic'] *= 0.2
        scores['e-waste'] += 25
    if blue_ratio > 0.2:
        scores['organic'] *= 0.3
        scores['plastic'] += 15
    if laplacian_var > 900 and s < 30:
        scores['organic'] *= 0.2
        scores['metal'] += 20
    if dark_ratio > 0.3 and num_contours > 15 and edge_density > 0.15:
        scores['e-waste'] += 20
        scores['organic'] *= 0.3
    
    predicted_class = max(scores, key=scores.get)
    total_score = sum(scores.values())
    confidence = scores[predicted_class] / total_score if total_score > 0 else 1/6
    probabilities = {k: v/total_score if total_score > 0 else 1/6 for k, v in scores.items()}
    top_predictions = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:3]
    
    return predicted_class, confidence, probabilities, top_predictions

# ============================================
# ENVIRONMENTAL IMPACT FUNCTIONS
# ============================================
def calculate_environmental_impact(waste_type, weight_kg=1.0):
    impact_factors = {
        'plastic': {'co2_saved': 1.5, 'water_saved': 7.5, 'energy_saved': 5.8, 'landfill_diverted': 0.85},
        'paper': {'co2_saved': 1.0, 'water_saved': 15.0, 'energy_saved': 4.0, 'landfill_diverted': 0.9},
        'glass': {'co2_saved': 0.3, 'water_saved': 1.0, 'energy_saved': 1.2, 'landfill_diverted': 0.95},
        'metal': {'co2_saved': 5.0, 'water_saved': 3.0, 'energy_saved': 8.0, 'landfill_diverted': 0.9},
        'organic': {'co2_saved': 0.5, 'water_saved': 0.0, 'energy_saved': 0.2, 'landfill_diverted': 0.6},
        'e-waste': {'co2_saved': 20.0, 'water_saved': 10.0, 'energy_saved': 50.0, 'landfill_diverted': 0.7}
    }
    impact = impact_factors.get(waste_type.lower(), {'co2_saved': 0.5, 'water_saved': 2.0, 'energy_saved': 2.0, 'landfill_diverted': 0.5})
    return {
        'co2_saved': impact['co2_saved'] * weight_kg,
        'water_saved': impact['water_saved'] * weight_kg,
        'energy_saved': impact['energy_saved'] * weight_kg,
        'landfill_diverted': impact['landfill_diverted'] * weight_kg
    }

def get_recycling_tips(waste_type):
    tips = {
        'plastic': ["Rinse containers before recycling", "Remove caps and labels when possible", "Check local recycling guidelines", "Avoid mixing different types of plastics", "Consider reusing before recycling"],
        'paper': ["Remove plastic windows or tape", "Keep paper dry and clean", "Flatten cardboard boxes", "Don't recycle contaminated paper", "Contain shredded paper in bags"],
        'glass': ["Rinse glass containers thoroughly", "Remove metal lids and caps", "Sort by color if required", "Don't recycle broken glass directly", "Mirrors/ceramics go in general waste"],
        'metal': ["Rinse food cans before recycling", "Remove paper labels when possible", "Crush cans to save space", "Don't recycle aerosol cans unless empty", "Check for deposit return programs"],
        'organic': ["Start composting at home", "Use dedicated compost bin with ventilation", "Balance green and brown materials", "Avoid composting meat/dairy/oily foods", "Use compost in gardens"],
        'e-waste': ["Never dispose in regular trash", "Use certified e-waste recycling centers", "Remove personal data before recycling", "Consider donating working electronics", "Check manufacturer take-back programs"]
    }
    return tips.get(waste_type.lower(), ["Check local recycling guidelines", "Clean the item before recycling", "Separate different materials when possible"])

def create_leaderboard(user_items=0, user_co2_saved=0):
    users = [
        {'Name': 'EcoWarrior', 'Items Recycled': 245, 'CO2 Saved': 367.5, 'Badge': '🏆'},
        {'Name': 'GreenGuardian', 'Items Recycled': 189, 'CO2 Saved': 283.5, 'Badge': '🥇'},
        {'Name': 'RecyclePro', 'Items Recycled': 156, 'CO2 Saved': 234.0, 'Badge': '🥈'},
        {'Name': 'EarthLover', 'Items Recycled': 134, 'CO2 Saved': 201.0, 'Badge': '🥉'},
        {'Name': 'Sustainability', 'Items Recycled': 98, 'CO2 Saved': 147.0, 'Badge': '🌱'},
        {'Name': 'EcoFriend', 'Items Recycled': 87, 'CO2 Saved': 130.5, 'Badge': '🌿'},
        {'Name': 'GreenTechie', 'Items Recycled': 76, 'CO2 Saved': 114.0, 'Badge': '🍃'},
        {'Name': 'PlanetSaver', 'Items Recycled': 65, 'CO2 Saved': 97.5, 'Badge': '🌍'},
        {'Name': 'WasteWarrior', 'Items Recycled': 54, 'CO2 Saved': 81.0, 'Badge': '♻️'},
        {'Name': 'RecycleRookie', 'Items Recycled': 43, 'CO2 Saved': 64.5, 'Badge': '🌱'},
        {'Name': '⭐ You', 'Items Recycled': user_items, 'CO2 Saved': round(user_co2_saved, 2), 'Badge': '🌟'}
    ]
    df = pd.DataFrame(users)
    df = df.sort_values('Items Recycled', ascending=False).reset_index(drop=True)
    df.insert(0, 'Rank', range(1, len(df) + 1))
    return df

# ============================================
# INITIALIZE SESSION STATE
# ============================================
if 'waste_history' not in st.session_state:
    st.session_state.waste_history = []
if 'total_impact' not in st.session_state:
    st.session_state.total_impact = {'co2_saved': 0.0, 'water_saved': 0.0, 'energy_saved': 0.0, 'landfill_diverted': 0.0}

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 100%); 
                padding: 20px; border-radius: 12px; text-align: center; margin-bottom: 20px;">
        <h1 style="color: #FFFFFF; margin: 0; font-size: 1.8em;">🌍 EcoGuard AI</h1>
        <p style="color: #A5D6A7; margin: 5px 0 0 0;">Smart Waste Classifier</p>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.radio("📱 Navigation", ["📸 Classify Waste", "📊 Impact Dashboard", "🎯 Community Challenges", "ℹ️ About"], label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown('<h3 style="color: #1B5E20; margin-bottom: 15px;">🌱 Your Impact</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1: st.metric("Items", len(st.session_state.waste_history))
    with col2: st.metric("CO₂ (kg)", f"{st.session_state.total_impact['co2_saved']:.1f}")
    
    if st.session_state.waste_history:
        st.markdown("---")
        st.markdown('<h4 style="color: #424242;">📝 Recent Activity</h4>', unsafe_allow_html=True)
        emoji_map = {'plastic': '🥤', 'paper': '📄', 'glass': '🫙', 'metal': '🥫', 'organic': '🍎', 'e-waste': '💻'}
        for item in reversed(st.session_state.waste_history[-5:]):
            emoji = emoji_map.get(item['waste_type'], '♻️')
            st.markdown(f"""
            <div style="background: #FAFAFA; padding: 8px 12px; border-radius: 6px; margin: 5px 0; border-left: 3px solid #4CAF50;">
                <span style="font-size: 1.1em;">{emoji}</span>
                <strong style="color: #212121;">{item['waste_type'].title()}</strong>
                <br><small style="color: #757575;">{item['timestamp']}</small>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    if st.button("🔄 Clear All Data", key="sidebar_clear_btn", use_container_width=True):
        st.session_state.waste_history = []
        st.session_state.total_impact = {'co2_saved': 0.0, 'water_saved': 0.0, 'energy_saved': 0.0, 'landfill_diverted': 0.0}
        if 'current_prediction' in st.session_state: del st.session_state.current_prediction
        st.rerun()
    st.caption("🌍 AI For Good Hackathon 2026")

# ============================================
# CLASSIFY WASTE PAGE
# ============================================
if page == "📸 Classify Waste":
    st.markdown("<h1 class='main-header'>🌍 Smart Waste Classification</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h3 style="color: #2E7D32; margin-bottom: 15px;">📤 Upload Waste Image</h3>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose an image of waste material", type=['jpg', 'jpeg', 'png'])
        st.markdown("---")
        st.markdown('<p style="color: #424242; font-weight: 600;">📸 Or take a photo:</p>', unsafe_allow_html=True)
        camera_image = st.camera_input("Take a picture")
        image_to_process = uploaded_file or camera_image
        
        if image_to_process:
            image = Image.open(image_to_process)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            if st.button("🔍 Classify Waste", key="btn_classify", type="primary", use_container_width=True):
                with st.spinner("🔄 Analyzing image features..."):
                    predicted_class, confidence, probabilities, top_predictions = classify_waste_image(image)
                    st.session_state.current_prediction = {
                        'class': predicted_class, 'confidence': confidence,
                        'top_predictions': top_predictions, 'probabilities': probabilities
                    }
    
    with col2:
        if image_to_process and 'current_prediction' in st.session_state:
            pred = st.session_state.current_prediction
            st.markdown('<h3 style="color: #2E7D32; margin-bottom: 15px;">🎯 Classification Results</h3>', unsafe_allow_html=True)
            
            waste_emojis = {'plastic': '🥤', 'paper': '📄', 'glass': '🫙', 'metal': '🥫', 'organic': '🍎', 'e-waste': '💻'}
            emoji = waste_emojis.get(pred['class'], '♻️')
            
            st.markdown(f"""
            <div class="classification-result">
                <h1 style="font-size: 4rem; color: #FFFFFF !important;">{emoji}</h1>
                <h2 style="color: #FFFFFF !important;">{pred['class'].upper()}</h2>
                <h3 style="color: #C8E6C9 !important;">Confidence: {pred['confidence']*100:.1f}%</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown('<p style="color: #424242; font-weight: 600;">📊 Top Predictions:</p>', unsafe_allow_html=True)
            for waste_type, prob in pred['top_predictions']:
                emoji_p = waste_emojis.get(waste_type, '♻️')
                col_a, col_b = st.columns([1, 3])
                with col_a: st.markdown(f"<p style='color: #212121;'>{emoji_p} <strong>{waste_type.title()}</strong></p>", unsafe_allow_html=True)
                with col_b:
                    st.progress(prob)
                    st.caption(f"{prob*100:.1f}%")
            
            st.markdown("---")
            st.markdown('<h3 style="color: #2E7D32;">♻️ Recycling Instructions</h3>', unsafe_allow_html=True)
            for i, tip in enumerate(get_recycling_tips(pred['class']), 1):
                st.markdown(f'<div class="recycling-tip"><p>💡 <strong>Tip {i}:</strong> {tip}</p></div>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown('<h3 style="color: #2E7D32;">🌍 Environmental Impact (per kg)</h3>', unsafe_allow_html=True)
            impact = calculate_environmental_impact(pred['class'], 1.0)
            
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a: st.metric("🌳 CO₂", f"{impact['co2_saved']:.2f} kg")
            with col_b: st.metric("💧 Water", f"{impact['water_saved']:.1f} L")
            with col_c: st.metric("⚡ Energy", f"{impact['energy_saved']:.1f} kWh")
            with col_d: st.metric("🗑️ Landfill", f"{impact['landfill_diverted']:.2f} kg")
            
            st.markdown("---")
            if st.button("✅ Log This Classification", key="btn_log", type="primary", use_container_width=True):
                st.session_state.waste_history.append({
                    'waste_type': pred['class'],
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'confidence': pred['confidence'],
                    'impact': impact
                })
                for key in st.session_state.total_impact:
                    st.session_state.total_impact[key] += impact[key]
                st.balloons()
                st.markdown('<div class="success-message"><strong>✅ Classification logged successfully!</strong></div>', unsafe_allow_html=True)
                del st.session_state.current_prediction

# ============================================
# IMPACT DASHBOARD PAGE (CLEANED - SINGLE VERSION)
# ============================================
elif page == "📊 Impact Dashboard":
    st.markdown("<h1 class='main-header'>📊 Environmental Impact Dashboard</h1>", unsafe_allow_html=True)
    
    col_title, col_clear = st.columns([4, 1])
    with col_title: st.markdown("<h3 style='color: #2E7D32;'>📈 Overview</h3>", unsafe_allow_html=True)
    with col_clear:
        if len(st.session_state.waste_history) > 0:
            if st.button("🗑️ Clear Data", key="clear_dash_btn", type="secondary", use_container_width=True):
                st.session_state.waste_history = []
                st.session_state.total_impact = {'co2_saved': 0.0, 'water_saved': 0.0, 'energy_saved': 0.0, 'landfill_diverted': 0.0}
                if 'current_prediction' in st.session_state: del st.session_state.current_prediction
                st.rerun()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("📦 Items", len(st.session_state.waste_history))
    with col2: st.metric("🌳 CO₂ (kg)", f"{st.session_state.total_impact['co2_saved']:.2f}")
    with col3: st.metric("💧 Water (L)", f"{st.session_state.total_impact['water_saved']:.1f}")
    with col4: st.metric("⚡ Energy (kWh)", f"{st.session_state.total_impact['energy_saved']:.1f}")
    
    has_data = len(st.session_state.waste_history) > 0
    
    if has_data:
        waste_types = [item['waste_type'] for item in st.session_state.waste_history]
        df_waste = pd.DataFrame({'Type': waste_types}).value_counts().reset_index()
        df_waste.columns = ['Type', 'Count']
        
        st.markdown("---")
        st.markdown("<h3 style='color: #2E7D32;'>📊 Visual Analytics</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            fig_pie = px.pie(df_waste, values='Count', names='Type', title='Waste Distribution', color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig_pie, use_container_width=True, key="pie_chart")
        with col2:
            fig_bar = px.bar(df_waste, x='Type', y='Count', title='Items by Type', color='Type', color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig_bar, use_container_width=True, key="bar_chart")
    
    # Detailed Impact Breakdown
    st.markdown("---")
    st.markdown('<div style="background: #E8F5E9; padding: 15px; border-radius: 10px; margin: 20px 0;"><h3 style="margin: 0; color: #1B5E20;">📋 Detailed Impact Breakdown</h3></div>', unsafe_allow_html=True)
    
    if has_data:
        impact_summary = {}
        for item in st.session_state.waste_history:
            wt = item['waste_type']
            if wt not in impact_summary:
                impact_summary[wt] = {'count': 0, 'co2': 0.0, 'water': 0.0, 'energy': 0.0, 'landfill': 0.0}
            impact_summary[wt]['count'] += 1
            impact_summary[wt]['co2'] += item['impact'].get('co2_saved', 0)
            impact_summary[wt]['water'] += item['impact'].get('water_saved', 0)
            impact_summary[wt]['energy'] += item['impact'].get('energy_saved', 0)
            impact_summary[wt]['landfill'] += item['impact'].get('landfill_diverted', 0)
        
        emoji_map = {'plastic': '🥤', 'paper': '📄', 'glass': '🫙', 'metal': '🥫', 'organic': '🍎', 'e-waste': '💻'}
        
        # Build table data
        table_data = []
        total_items = 0
        total_co2 = 0.0
        total_water = 0.0
        total_energy = 0.0
        total_landfill = 0.0
        
        for wt, data in impact_summary.items():
            emoji = emoji_map.get(wt, '♻️')
            table_data.append({
                'Waste Type': f"{emoji} {wt.title()}",
                'Items': str(data['count']),
                'CO₂ (kg)': f"{data['co2']:.2f}",
                'Water (L)': f"{data['water']:.1f}",
                'Energy (kWh)': f"{data['energy']:.1f}",
                'Landfill (kg)': f"{data['landfill']:.2f}"
            })
            total_items += data['count']
            total_co2 += data['co2']
            total_water += data['water']
            total_energy += data['energy']
            total_landfill += data['landfill']
        
        table_data.append({
            'Waste Type': '📊 TOTAL',
            'Items': str(total_items),
            'CO₂ (kg)': f"{total_co2:.2f}",
            'Water (L)': f"{total_water:.1f}",
            'Energy (kWh)': f"{total_energy:.1f}",
            'Landfill (kg)': f"{total_landfill:.2f}"
        })
        
        df_table = pd.DataFrame(table_data)
        st.dataframe(df_table, use_container_width=True, hide_index=True,
            column_config={
                'Waste Type': st.column_config.TextColumn('Waste Type', width='medium'),
                'Items': st.column_config.TextColumn('Items', width='small'),
                'CO₂ (kg)': st.column_config.TextColumn('CO₂ (kg)', width='small'),
                'Water (L)': st.column_config.TextColumn('Water (L)', width='small'),
                'Energy (kWh)': st.column_config.TextColumn('Energy (kWh)', width='small'),
                'Landfill (kg)': st.column_config.TextColumn('Landfill (kg)', width='small'),
            })
        
        # Quick Stats
        st.markdown("---")
        st.markdown('<div style="background: #E8F5E9; padding: 15px; border-radius: 10px; margin: 20px 0;"><h3 style="margin: 0; color: #1B5E20;">🎯 Quick Statistics</h3></div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            most = max(impact_summary.items(), key=lambda x: x[1]['count'])
            emoji = emoji_map.get(most[0], '♻️')
            st.markdown(f'<div class="quick-stat-card stat-blue"><div class="stat-emoji">{emoji}</div><div class="stat-label">Most Recycled</div><div class="stat-value">{most[0].title()}</div><div class="stat-subtitle">{most[1]["count"]} items</div></div>', unsafe_allow_html=True)
        
        with col2:
            best = max(impact_summary.items(), key=lambda x: x[1]['co2'])
            emoji = emoji_map.get(best[0], '♻️')
            st.markdown(f'<div class="quick-stat-card stat-green"><div class="stat-emoji">{emoji}</div><div class="stat-label">Best CO₂ Saver</div><div class="stat-value">{best[0].title()}</div><div class="stat-subtitle">{best[1]["co2"]:.2f} kg</div></div>', unsafe_allow_html=True)
        
        with col3:
            if len(st.session_state.waste_history) > 0:
                dates = [datetime.strptime(item['timestamp'], "%Y-%m-%d %H:%M:%S") for item in st.session_state.waste_history]
                date_range = (max(dates) - min(dates)).days or 1 if len(dates) > 1 else 1
                avg = total_items / date_range
                st.markdown(f'<div class="quick-stat-card stat-orange"><div class="stat-emoji">📊</div><div class="stat-label">Avg Items/Day</div><div class="stat-value">{avg:.1f}</div><div class="stat-subtitle">items per day</div></div>', unsafe_allow_html=True)
        
        with col4:
            score = (total_co2 * 10) + (total_water * 0.1) + (total_energy * 5)
            st.markdown(f'<div class="quick-stat-card stat-purple"><div class="stat-emoji">🌟</div><div class="stat-label">Eco Score</div><div class="stat-value">{score:.0f}</div><div class="stat-subtitle">eco points</div></div>', unsafe_allow_html=True)
        
        # Download
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            export_data = [{'Waste Type': wt.title(), 'Items': data['count'], 'CO2 Saved (kg)': round(data['co2'], 2), 'Water Saved (L)': round(data['water'], 1), 'Energy Saved (kWh)': round(data['energy'], 1), 'Landfill Diverted (kg)': round(data['landfill'], 2)} for wt, data in impact_summary.items()]
            if export_data:
                csv = pd.DataFrame(export_data).to_csv(index=False)
                st.download_button("📥 Download Impact Report (CSV)", data=csv, file_name=f"ecoguard_impact_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv", key="download_btn", use_container_width=True)
    
    else:
        st.info("📦 No items classified yet. Go to **Classify Waste** to start tracking your impact!")
        empty_data = [{'Waste Type': '📊 TOTAL', 'Items': '0', 'CO₂ (kg)': '0.00', 'Water (L)': '0.0', 'Energy (kWh)': '0.0', 'Landfill (kg)': '0.00'}]
        st.dataframe(pd.DataFrame(empty_data), use_container_width=True, hide_index=True)

# ============================================
# COMMUNITY CHALLENGES PAGE
# ============================================
elif page == "🎯 Community Challenges":
    st.markdown("<h1 class='main-header'>🎯 Community Challenges</h1>", unsafe_allow_html=True)
    st.markdown('<h3 style="color: #2E7D32;">🏆 Active Challenges</h3>', unsafe_allow_html=True)
    
    challenges = [
        {"title": "♻️ Recycling Champion", "description": "Classify 50 items this week", "progress": min(len(st.session_state.waste_history) / 50 * 100, 100), "reward": "EcoGuard Badge", "icon": "🏅", "color": "#4CAF50"},
        {"title": "🌍 Carbon Crusader", "description": "Save 100 kg of CO₂", "progress": min(st.session_state.total_impact['co2_saved'] / 100 * 100, 100), "reward": "Planet Protector", "icon": "🌟", "color": "#2196F3"},
        {"title": "💧 Water Warrior", "description": "Save 1000L of water", "progress": min(st.session_state.total_impact['water_saved'] / 1000 * 100, 100), "reward": "Water Guardian", "icon": "💎", "color": "#00BCD4"},
        {"title": "⚡ Energy Saver", "description": "Save 500 kWh of energy", "progress": min(st.session_state.total_impact['energy_saved'] / 500 * 100, 100), "reward": "Energy Expert", "icon": "⚡", "color": "#FF9800"}
    ]
    
    for c in challenges:
        col1, col2, col3 = st.columns([3, 1.5, 1])
        with col1:
            st.markdown(f'<div class="challenge-card"><h3>{c["icon"]} {c["title"]}</h3><p>{c["description"]}</p><div style="background:#E0E0E0; border-radius:10px; height:8px; margin:10px 0;"><div style="background:{c["color"]}; width:{c["progress"]}%; height:100%; border-radius:10px;"></div></div></div>', unsafe_allow_html=True)
        with col2: st.metric("Progress", f"{c['progress']:.1f}%")
        with col3: st.markdown(f'<div style="text-align:center; padding:10px;"><p style="color:#757575; font-size:0.9em;">Reward:</p><p style="font-weight:bold; color:#1B5E20;">{c["reward"]}</p></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown('<h3 style="color: #2E7D32;">🏆 Community Leaderboard</h3>', unsafe_allow_html=True)
    lb = create_leaderboard(len(st.session_state.waste_history), st.session_state.total_impact['co2_saved'])
    st.dataframe(lb, use_container_width=True, hide_index=True,
        column_config={
            'Rank': st.column_config.NumberColumn('Rank', width='small'),
            'Name': st.column_config.TextColumn('Name', width='medium'),
            'Items Recycled': st.column_config.NumberColumn('Items', width='small'),
            'CO2 Saved': st.column_config.NumberColumn('CO₂ (kg)', width='small', format="%.1f"),
            'Badge': st.column_config.TextColumn('Badge', width='small')
        })

# ============================================
# ABOUT PAGE
# ============================================
elif page == "ℹ️ About":
    st.markdown("<h1 class='main-header'>ℹ️ About EcoGuard AI</h1>", unsafe_allow_html=True)
    st.markdown('<div style="background:linear-gradient(135deg,#1B5E20 0%,#2E7D32 100%); padding:2rem; border-radius:15px; color:#FFFFFF; margin:2rem 0;"><h2 style="color:#FFFFFF;">🌍 Our Mission</h2><p style="font-size:1.2em; color:#E8F5E9;">EcoGuard AI tackles the global waste crisis through artificial intelligence and community engagement, making recycling smarter and more rewarding.</p></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div style="background:#FFEBEE; padding:1.5rem; border-radius:10px;"><h3 style="color:#C62828;">🚨 The Problem</h3><ul style="color:#212121; font-size:1.1em;"><li>🌍 <strong>2.01 billion tonnes</strong> of waste annually</li><li>♻️ Only <strong>13.5%</strong> is recycled globally</li><li>🔥 <strong>33%</strong> is dumped or burned</li><li>😕 Confusion about proper recycling</li></ul></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div style="background:#E8F5E9; padding:1.5rem; border-radius:10px;"><h3 style="color:#2E7D32;">💡 Our Solution</h3><ul style="color:#212121; font-size:1.1em;"><li>🤖 AI-Powered Classification</li><li>📊 Real-time Impact Tracking</li><li>🏆 Gamification & Rewards</li><li>📚 Recycling Education</li></ul></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown('<div style="text-align:center; color:#616161;"><p>🛠️ Built with Streamlit, OpenCV, and Plotly</p><p>🏆 AI For Good Hackathon 2026 | ACM-W Track</p></div>', unsafe_allow_html=True)

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown('<div class="footer"><p>🌍 <strong style="color: #1B5E20;">EcoGuard AI</strong> - Making recycling smarter</p><p style="font-size:0.9em; color:#757575;">AI For Good Hackathon 2026 | ACM-W Track</p></div>', unsafe_allow_html=True)