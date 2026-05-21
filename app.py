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
    .classification-result h1 {
        color: #FFFFFF !important;
    }
    .classification-result h2 {
        color: #A5D6A7 !important;
    }
    .classification-result h3 {
        color: #E8F5E9 !important;
    }
    
    .recycling-tip {
        background: #F5F5F5;
        padding: 1.2rem;
        border-left: 5px solid #2E7D32;
        margin: 1rem 0;
        border-radius: 8px;
        color: #212121;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    }
    .recycling-tip p {
        color: #212121;
        margin: 0;
        line-height: 1.6;
    }
    .recycling-tip strong {
        color: #1B5E20;
    }
    
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
    
    .impact-table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
        font-size: 14px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-radius: 10px;
        overflow: hidden;
        background: #FFFFFF;
    }
    .impact-table th {
        background: #2E7D32;
        color: #FFFFFF;
        padding: 14px 12px;
        text-align: left;
        font-weight: 600;
        font-size: 0.9em;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .impact-table td {
        padding: 12px;
        border-bottom: 1px solid #E0E0E0;
        color: #212121;
        background: #FAFAFA;
    }
    .impact-table tr:nth-child(even) td {
        background: #F5F5F5;
    }
    .impact-table tr:hover td {
        background: #E8F5E9;
    }
    .impact-table .total-row td {
        background: #C8E6C9 !important;
        font-weight: bold;
        color: #1B5E20;
        border-top: 2px solid #2E7D32;
    }
    
    .challenge-card {
        background: #FFFFFF;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border: 1px solid #E0E0E0;
    }
    .challenge-card h3 {
        color: #1B5E20;
    }
    .challenge-card p {
        color: #424242;
    }
    
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
    .stat-label { 
        font-weight: 600; 
        color: #424242; 
        margin: 5px 0;
        font-size: 0.9em;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stat-value { 
        font-size: 1.3em; 
        font-weight: bold; 
        margin: 8px 0;
    }
    .stat-subtitle { 
        font-size: 0.85em; 
        opacity: 0.8;
        margin: 0;
    }
    
    .stat-blue { 
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%); 
    }
    .stat-blue .stat-value { color: #1565C0; }
    .stat-blue .stat-subtitle { color: #1976D2; }
    
    .stat-green { 
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%); 
    }
    .stat-green .stat-value { color: #2E7D32; }
    .stat-green .stat-subtitle { color: #388E3C; }
    
    .stat-orange { 
        background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%); 
    }
    .stat-orange .stat-value { color: #E65100; }
    .stat-orange .stat-subtitle { color: #EF6C00; }
    
    .stat-purple { 
        background: linear-gradient(135deg, #F3E5F5 0%, #E1BEE7 100%); 
    }
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
# CLASSIFICATION FUNCTION (BUILT INTO APP.PY)
# ============================================
def classify_waste_image(image):
    """
    Classify waste based on color and texture features from the image
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    # Convert to RGB if needed
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Resize for processing
    img_resized = cv2.resize(img_array, (224, 224))
    
    # Convert to HSV
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_RGB2HSV)
    
    # Calculate features
    avg_rgb = np.mean(img_resized, axis=(0, 1))
    avg_hsv = np.mean(hsv, axis=(0, 1))
    std_rgb = np.std(img_resized, axis=(0, 1))
    
    # Edge detection
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.mean(edges) / 255.0
    
    r, g, b = avg_rgb
    h, s, v = avg_hsv
    
    # Scoring system
    scores = {
        'plastic': 0,
        'paper': 0,
        'glass': 0,
        'metal': 0,
        'organic': 0,
        'e-waste': 0
    }
    
    # PLASTIC: Colorful, smooth surface
    if s > 40 and v > 100:
        scores['plastic'] += 30
    if edge_density < 0.15:
        scores['plastic'] += 20
    if 100 < r < 200 and 100 < g < 200 and 100 < b < 200:
        scores['plastic'] += 15
    if v > 200 and s < 30:  # White plastic
        scores['plastic'] += 15
    
    # PAPER: Light, low saturation, some texture
    if v > 150 and s < 50:
        scores['paper'] += 35
    if edge_density > 0.08:
        scores['paper'] += 20
    if r > 150 and g > 140 and b > 120:
        scores['paper'] += 15
    if (r > g and g > b) and s > 15:  # Brown paper/cardboard
        scores['paper'] += 25
    
    # GLASS: Bright, sharp edges, transparent look
    if v > 150 and s < 40:
        scores['glass'] += 25
    if edge_density > 0.15:
        scores['glass'] += 25
    if std_rgb[0] > 50 or std_rgb[1] > 50 or std_rgb[2] > 50:
        scores['glass'] += 20
    if s < 20 and v > 180:  # Clear glass
        scores['glass'] += 20
    
    # METAL: Gray/silver, shiny, reflective
    if s < 30 and v > 100:
        scores['metal'] += 25
    if abs(r - g) < 20 and abs(g - b) < 20 and abs(r - b) < 20:
        scores['metal'] += 30
    if std_rgb[0] > 40 or std_rgb[1] > 40:
        scores['metal'] += 20
    if 120 < v < 200 and s < 25:
        scores['metal'] += 15
    
    # ORGANIC: Brown/green tones, natural texture
    if (r > g and g > b) or (g > r and g > b):
        scores['organic'] += 35
    if s > 25 and v < 180:
        scores['organic'] += 25
    if edge_density > 0.12:
        scores['organic'] += 20
    if 15 < h < 80 and s > 20:
        scores['organic'] += 20
    if r > 80 and g > 60 and b < 80:  # Brown/green
        scores['organic'] += 15
    
    # E-WASTE: Dark, complex patterns, black
    if v < 100:
        scores['e-waste'] += 35
    if edge_density > 0.2:
        scores['e-waste'] += 30
    if r < 80 and g < 80 and b < 80:
        scores['e-waste'] += 25
    if std_rgb[0] > 50 or std_rgb[1] > 50:
        scores['e-waste'] += 20
    
    # Get prediction
    predicted_class = max(scores, key=scores.get)
    
    # Calculate probabilities
    total_score = sum(scores.values())
    if total_score > 0:
        probabilities = {k: v/total_score for k, v in scores.items()}
    else:
        probabilities = {k: 1/6 for k in scores}
    
    confidence = probabilities[predicted_class]
    
    # Get top 3 predictions
    top_predictions = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:3]
    
    return predicted_class, confidence, probabilities, top_predictions

# ============================================
# ENVIRONMENTAL IMPACT FUNCTIONS
# ============================================
def calculate_environmental_impact(waste_type, weight_kg=1.0):
    """Calculate environmental impact of recycling"""
    impact_factors = {
        'plastic': {'co2_saved': 1.5, 'water_saved': 7.5, 'energy_saved': 5.8, 'landfill_diverted': 0.85},
        'paper': {'co2_saved': 1.0, 'water_saved': 15.0, 'energy_saved': 4.0, 'landfill_diverted': 0.9},
        'glass': {'co2_saved': 0.3, 'water_saved': 1.0, 'energy_saved': 1.2, 'landfill_diverted': 0.95},
        'metal': {'co2_saved': 5.0, 'water_saved': 3.0, 'energy_saved': 8.0, 'landfill_diverted': 0.9},
        'organic': {'co2_saved': 0.5, 'water_saved': 0.0, 'energy_saved': 0.2, 'landfill_diverted': 0.6},
        'e-waste': {'co2_saved': 20.0, 'water_saved': 10.0, 'energy_saved': 50.0, 'landfill_diverted': 0.7}
    }
    
    impact = impact_factors.get(waste_type.lower(), {
        'co2_saved': 0.5, 'water_saved': 2.0, 'energy_saved': 2.0, 'landfill_diverted': 0.5
    })
    
    return {
        'co2_saved': impact['co2_saved'] * weight_kg,
        'water_saved': impact['water_saved'] * weight_kg,
        'energy_saved': impact['energy_saved'] * weight_kg,
        'landfill_diverted': impact['landfill_diverted'] * weight_kg
    }

def get_recycling_tips(waste_type):
    """Get recycling tips for specific waste types"""
    tips = {
        'plastic': [
            "Rinse containers before recycling to prevent contamination",
            "Remove caps and labels when possible",
            "Check local recycling guidelines for plastic types accepted",
            "Avoid mixing different types of plastics",
            "Consider reusing plastic containers before recycling"
        ],
        'paper': [
            "Remove any plastic windows or tape from paper products",
            "Keep paper dry and clean",
            "Flatten cardboard boxes to save space",
            "Don't recycle paper contaminated with food or grease",
            "Shredded paper should be contained in paper bags"
        ],
        'glass': [
            "Rinse glass containers thoroughly",
            "Remove metal lids and caps",
            "Sort by color if required by local facility",
            "Don't recycle broken glass directly - wrap safely",
            "Mirrors and ceramics belong in general waste"
        ],
        'metal': [
            "Rinse food cans before recycling",
            "Remove paper labels when possible",
            "Crush cans to save space",
            "Don't recycle aerosol cans unless empty",
            "Check for deposit return programs in your area"
        ],
        'organic': [
            "Start composting at home for food scraps",
            "Use a dedicated compost bin with proper ventilation",
            "Balance green (nitrogen) and brown (carbon) materials",
            "Avoid composting meat, dairy, or oily foods",
            "Use compost in gardens to enrich soil"
        ],
        'e-waste': [
            "Never dispose of electronics in regular trash",
            "Use certified e-waste recycling centers",
            "Remove personal data from devices before recycling",
            "Consider donating working electronics",
            "Check for manufacturer take-back programs"
        ]
    }
    
    return tips.get(waste_type.lower(), [
        "Check local recycling guidelines",
        "Clean the item before recycling",
        "Separate different materials when possible"
    ])

def create_leaderboard(user_items=0, user_co2_saved=0):
    """Create sample community leaderboard"""
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
    st.session_state.total_impact = {
        'co2_saved': 0.0,
        'water_saved': 0.0,
        'energy_saved': 0.0,
        'landfill_diverted': 0.0
    }

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
    
    page = st.radio(
        "📱 Navigation",
        ["📸 Classify Waste", "📊 Impact Dashboard", "🎯 Community Challenges", "ℹ️ About"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    st.markdown("""
    <h3 style="color: #1B5E20; margin-bottom: 15px;">🌱 Your Impact</h3>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Items", len(st.session_state.waste_history))
    with col2:
        st.metric("CO₂ (kg)", f"{st.session_state.total_impact['co2_saved']:.1f}")
    
    if st.session_state.waste_history:
        st.markdown("---")
        st.markdown("""
        <h4 style="color: #424242;">📝 Recent Activity</h4>
        """, unsafe_allow_html=True)
        
        emoji_map = {
            'plastic': '🥤', 'paper': '📄', 'glass': '🫙',
            'metal': '🥫', 'organic': '🍎', 'e-waste': '💻'
        }
        
        for item in reversed(st.session_state.waste_history[-5:]):
            emoji = emoji_map.get(item['waste_type'], '♻️')
            st.markdown(f"""
            <div style="background: #FAFAFA; padding: 8px 12px; border-radius: 6px; 
                     margin: 5px 0; border-left: 3px solid #4CAF50;">
                <span style="font-size: 1.1em;">{emoji}</span>
                <strong style="color: #212121;">{item['waste_type'].title()}</strong>
                <br><small style="color: #757575;">{item['timestamp']}</small>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption("🌍 AI For Good Hackathon 2026")

# ============================================
# CLASSIFY WASTE PAGE
# ============================================
if page == "📸 Classify Waste":
    st.markdown("<h1 class='main-header'>🌍 Smart Waste Classification</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <h3 style="color: #2E7D32; margin-bottom: 15px;">📤 Upload Waste Image</h3>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose an image of waste material",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of the waste item"
        )
        
        st.markdown("---")
        st.markdown("""
        <p style="color: #424242; font-weight: 600;">📸 Or take a photo:</p>
        """, unsafe_allow_html=True)
        camera_image = st.camera_input("Take a picture")
        
        image_to_process = uploaded_file or camera_image
        
        if image_to_process:
            image = Image.open(image_to_process)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("🔍 Classify Waste", type="primary", use_container_width=True):
                with st.spinner("🔄 Analyzing image features..."):
                    # CLASSIFY THE IMAGE
                    predicted_class, confidence, probabilities, top_predictions = classify_waste_image(image)
                    
                    st.session_state.current_prediction = {
                        'class': predicted_class,
                        'confidence': confidence,
                        'top_predictions': top_predictions,
                        'probabilities': probabilities
                    }
    
    with col2:
        if image_to_process and 'current_prediction' in st.session_state:
            pred = st.session_state.current_prediction
            
            st.markdown("""
            <h3 style="color: #2E7D32; margin-bottom: 15px;">🎯 Classification Results</h3>
            """, unsafe_allow_html=True)
            
            waste_emojis = {
                'plastic': '🥤', 'paper': '📄', 'glass': '🫙',
                'metal': '🥫', 'organic': '🍎', 'e-waste': '💻'
            }
            
            emoji = waste_emojis.get(pred['class'], '♻️')
            
            # Result card
            st.markdown(f"""
            <div class="classification-result">
                <h1 style="font-size: 4rem; color: #FFFFFF !important;">{emoji}</h1>
                <h2 style="color: #FFFFFF !important;">{pred['class'].upper()}</h2>
                <h3 style="color: #C8E6C9 !important;">Confidence: {pred['confidence']*100:.1f}%</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Top predictions
            st.markdown("---")
            st.markdown("""
            <p style="color: #424242; font-weight: 600;">📊 Top Predictions:</p>
            """, unsafe_allow_html=True)
            
            for waste_type, prob in pred['top_predictions']:
                emoji_p = waste_emojis.get(waste_type, '♻️')
                col_a, col_b = st.columns([1, 3])
                with col_a:
                    st.markdown(f"<p style='color: #212121;'>{emoji_p} <strong>{waste_type.title()}</strong></p>", unsafe_allow_html=True)
                with col_b:
                    st.progress(prob)
                    st.caption(f"{prob*100:.1f}%")
            
            # Recycling tips
            st.markdown("---")
            st.markdown("""
            <h3 style="color: #2E7D32;">♻️ Recycling Instructions</h3>
            """, unsafe_allow_html=True)
            
            recycling_info = get_recycling_tips(pred['class'])
            
            for i, tip in enumerate(recycling_info, 1):
                st.markdown(f"""
                <div class="recycling-tip">
                    <p>💡 <strong>Tip {i}:</strong> {tip}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Environmental impact
            st.markdown("---")
            st.markdown("""
            <h3 style="color: #2E7D32;">🌍 Environmental Impact (per kg)</h3>
            """, unsafe_allow_html=True)
            
            impact = calculate_environmental_impact(pred['class'], 1.0)
            
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.metric("🌳 CO₂", f"{impact['co2_saved']:.2f} kg")
            with col_b:
                st.metric("💧 Water", f"{impact['water_saved']:.1f} L")
            with col_c:
                st.metric("⚡ Energy", f"{impact['energy_saved']:.1f} kWh")
            with col_d:
                st.metric("🗑️ Landfill", f"{impact['landfill_diverted']:.2f} kg")
            
            # Log button
            st.markdown("---")
            if st.button("✅ Log This Classification", type="primary", use_container_width=True):
                classification_record = {
                    'waste_type': pred['class'],
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'confidence': pred['confidence'],
                    'impact': impact
                }
                st.session_state.waste_history.append(classification_record)
                
                for key in st.session_state.total_impact:
                    st.session_state.total_impact[key] += impact[key]
                
                st.balloons()
                st.markdown("""
                <div class="success-message">
                    <strong>✅ Classification logged successfully!</strong>
                </div>
                """, unsafe_allow_html=True)
                
                del st.session_state.current_prediction

# ============================================
# IMPACT DASHBOARD PAGE
# ============================================
elif page == "📊 Impact Dashboard":
    st.markdown("<h1 class='main-header'>📊 Environmental Impact Dashboard</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <h3 style="color: #2E7D32; margin-bottom: 15px;">📈 Overview</h3>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📦 Items Classified", len(st.session_state.waste_history))
    with col2:
        st.metric("🌳 CO₂ Saved (kg)", f"{st.session_state.total_impact['co2_saved']:.2f}")
    with col3:
        st.metric("💧 Water Saved (L)", f"{st.session_state.total_impact['water_saved']:.1f}")
    with col4:
        st.metric("⚡ Energy Saved (kWh)", f"{st.session_state.total_impact['energy_saved']:.1f}")
    
    if st.session_state.waste_history:
        # Create some basic charts
        waste_types = [item['waste_type'] for item in st.session_state.waste_history]
        df_waste = pd.DataFrame({'Type': waste_types}).value_counts().reset_index()
        df_waste.columns = ['Type', 'Count']
        
        st.markdown("---")
        st.markdown("""
        <h3 style="color: #2E7D32;">📊 Visual Analytics</h3>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pie = px.pie(df_waste, values='Count', names='Type', 
                           title='Waste Distribution',
                           color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            fig_bar = px.bar(df_waste, x='Type', y='Count', 
                           title='Items by Type',
                           color='Type',
                           color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Detailed Impact Breakdown Table
        st.markdown("---")
        st.markdown("""
        <div style="background: #E8F5E9; padding: 15px; border-radius: 10px; margin: 20px 0;">
            <h3 style="margin: 0; color: #1B5E20;">📋 Detailed Impact Breakdown</h3>
        </div>
        """, unsafe_allow_html=True)
        
        impact_summary = {}
        for item in st.session_state.waste_history:
            waste_type = item['waste_type']
            if waste_type not in impact_summary:
                impact_summary[waste_type] = {
                    'count': 0, 'total_co2': 0.0, 'total_water': 0.0,
                    'total_energy': 0.0, 'total_landfill': 0.0
                }
            impact_summary[waste_type]['count'] += 1
            impact_summary[waste_type]['total_co2'] += item['impact'].get('co2_saved', 0)
            impact_summary[waste_type]['total_water'] += item['impact'].get('water_saved', 0)
            impact_summary[waste_type]['total_energy'] += item['impact'].get('energy_saved', 0)
            impact_summary[waste_type]['total_landfill'] += item['impact'].get('landfill_diverted', 0)
        
        emoji_map = {
            'plastic': '🥤', 'paper': '📄', 'glass': '🫙',
            'metal': '🥫', 'organic': '🍎', 'e-waste': '💻'
        }
        
        html_table = """
        <table class="impact-table">
        <thead>
        <tr>
            <th>Waste Type</th>
            <th>Items</th>
            <th>CO₂ Saved (kg)</th>
            <th>Water Saved (L)</th>
            <th>Energy Saved (kWh)</th>
            <th>Landfill Diverted (kg)</th>
        </tr>
        </thead>
        <tbody>
        """
        
        total_items = 0
        total_co2 = 0.0
        total_water = 0.0
        total_energy = 0.0
        total_landfill = 0.0
        
        for waste_type, data in impact_summary.items():
            emoji = emoji_map.get(waste_type, '♻️')
            html_table += f"""
            <tr>
                <td><strong>{emoji} {waste_type.title()}</strong></td>
                <td>{data['count']}</td>
                <td>{data['total_co2']:.2f}</td>
                <td>{data['total_water']:.1f}</td>
                <td>{data['total_energy']:.1f}</td>
                <td>{data['total_landfill']:.2f}</td>
            </tr>
            """
            total_items += data['count']
            total_co2 += data['total_co2']
            total_water += data['total_water']
            total_energy += data['total_energy']
            total_landfill += data['total_landfill']
        
        html_table += f"""
        <tr class="total-row">
            <td>📊 <strong>TOTAL</strong></td>
            <td><strong>{total_items}</strong></td>
            <td><strong>{total_co2:.2f}</strong></td>
            <td><strong>{total_water:.1f}</strong></td>
            <td><strong>{total_energy:.1f}</strong></td>
            <td><strong>{total_landfill:.2f}</strong></td>
        </tr>
        </tbody>
        </table>
        """
        
        st.markdown(html_table, unsafe_allow_html=True)
        
        # Quick Stats Cards
        st.markdown("---")
        st.markdown("""
        <div style="background: #E8F5E9; padding: 15px; border-radius: 10px; margin: 20px 0;">
            <h3 style="margin: 0; color: #1B5E20;">🎯 Quick Statistics</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if impact_summary:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                most_recycled = max(impact_summary.items(), key=lambda x: x[1]['count'])
                emoji = emoji_map.get(most_recycled[0], '♻️')
                st.markdown(f"""
                <div class="quick-stat-card stat-blue">
                    <div class="stat-emoji">{emoji}</div>
                    <div class="stat-label">Most Recycled</div>
                    <div class="stat-value">{most_recycled[0].title()}</div>
                    <div class="stat-subtitle">{most_recycled[1]['count']} items</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                best_co2 = max(impact_summary.items(), key=lambda x: x[1]['total_co2'])
                emoji = emoji_map.get(best_co2[0], '♻️')
                st.markdown(f"""
                <div class="quick-stat-card stat-green">
                    <div class="stat-emoji">{emoji}</div>
                    <div class="stat-label">Best CO₂ Saver</div>
                    <div class="stat-value">{best_co2[0].title()}</div>
                    <div class="stat-subtitle">{best_co2[1]['total_co2']:.2f} kg</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                if len(st.session_state.waste_history) > 0:
                    dates = [datetime.strptime(item['timestamp'], "%Y-%m-%d %H:%M:%S") 
                            for item in st.session_state.waste_history]
                    if len(dates) > 1:
                        date_range = (max(dates) - min(dates)).days or 1
                        avg_per_day = total_items / date_range
                    else:
                        avg_per_day = total_items
                    
                    st.markdown(f"""
                    <div class="quick-stat-card stat-orange">
                        <div class="stat-emoji">📊</div>
                        <div class="stat-label">Avg Items/Day</div>
                        <div class="stat-value">{avg_per_day:.1f}</div>
                        <div class="stat-subtitle">items per day</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col4:
                env_score = (total_co2 * 10) + (total_water * 0.1) + (total_energy * 5)
                st.markdown(f"""
                <div class="quick-stat-card stat-purple">
                    <div class="stat-emoji">🌟</div>
                    <div class="stat-label">Eco Score</div>
                    <div class="stat-value">{env_score:.0f}</div>
                    <div class="stat-subtitle">eco points</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Download button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            export_data = []
            for waste_type, data in impact_summary.items():
                export_data.append({
                    'Waste Type': waste_type,
                    'Items': data['count'],
                    'CO2 Saved (kg)': round(data['total_co2'], 2),
                    'Water Saved (L)': round(data['total_water'], 1),
                    'Energy Saved (kWh)': round(data['total_energy'], 1),
                    'Landfill Diverted (kg)': round(data['total_landfill'], 2)
                })
            
            export_df = pd.DataFrame(export_data)
            csv = export_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Impact Report (CSV)",
                data=csv,
                file_name=f"ecoguard_impact_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    else:
        st.info("👆 Start classifying waste to see your impact dashboard!")
        
        st.markdown("---")
        st.markdown("""
        <h3 style="color: #2E7D32;">📊 Sample Dashboard Preview</h3>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Items", "0")
        with col2:
            st.metric("CO₂ Saved", "0 kg")
        with col3:
            st.metric("Water Saved", "0 L")
        with col4:
            st.metric("Energy Saved", "0 kWh")

# ============================================
# COMMUNITY CHALLENGES PAGE
# ============================================
elif page == "🎯 Community Challenges":
    st.markdown("<h1 class='main-header'>🎯 Community Challenges</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <h3 style="color: #2E7D32;">🏆 Active Challenges</h3>
    """, unsafe_allow_html=True)
    
    challenges = [
        {"title": "♻️ Recycling Champion", "description": "Classify 50 items this week",
         "progress": min(len(st.session_state.waste_history) / 50 * 100, 100),
         "reward": "EcoGuard Badge", "icon": "🏅", "color": "#4CAF50"},
        {"title": "🌍 Carbon Crusader", "description": "Save 100 kg of CO₂",
         "progress": min(st.session_state.total_impact['co2_saved'] / 100 * 100, 100),
         "reward": "Planet Protector", "icon": "🌟", "color": "#2196F3"},
        {"title": "💧 Water Warrior", "description": "Save 1000L of water",
         "progress": min(st.session_state.total_impact['water_saved'] / 1000 * 100, 100),
         "reward": "Water Guardian", "icon": "💎", "color": "#00BCD4"},
        {"title": "⚡ Energy Saver", "description": "Save 500 kWh of energy",
         "progress": min(st.session_state.total_impact['energy_saved'] / 500 * 100, 100),
         "reward": "Energy Expert", "icon": "⚡", "color": "#FF9800"}
    ]
    
    for challenge in challenges:
        col1, col2, col3 = st.columns([3, 1.5, 1])
        with col1:
            st.markdown(f"""
            <div class="challenge-card">
                <h3>{challenge['icon']} {challenge['title']}</h3>
                <p>{challenge['description']}</p>
                <div style="background: #E0E0E0; border-radius: 10px; height: 8px; margin: 10px 0;">
                    <div style="background: {challenge['color']}; width: {challenge['progress']}%; 
                         height: 100%; border-radius: 10px; transition: width 0.5s;">
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.metric("Progress", f"{challenge['progress']:.1f}%")
        with col3:
            st.markdown(f"""
            <div style="text-align: center; padding: 10px;">
                <p style="color: #757575; font-size: 0.9em;">Reward:</p>
                <p style="font-weight: bold; color: #1B5E20;">{challenge['reward']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Leaderboard
    st.markdown("---")
    st.markdown("""
    <h3 style="color: #2E7D32;">🏆 Community Leaderboard</h3>
    """, unsafe_allow_html=True)
    
    leaderboard_data = create_leaderboard(
        user_items=len(st.session_state.waste_history),
        user_co2_saved=st.session_state.total_impact['co2_saved']
    )
    
    st.dataframe(
        leaderboard_data,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Rank': st.column_config.NumberColumn('Rank', width='small'),
            'Name': st.column_config.TextColumn('Name', width='medium'),
            'Items Recycled': st.column_config.NumberColumn('Items', width='small'),
            'CO2 Saved': st.column_config.NumberColumn('CO₂ (kg)', width='small', format="%.1f"),
            'Badge': st.column_config.TextColumn('Badge', width='small')
        }
    )

# ============================================
# ABOUT PAGE
# ============================================
elif page == "ℹ️ About":
    st.markdown("<h1 class='main-header'>ℹ️ About EcoGuard AI</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 100%); 
              padding: 2rem; border-radius: 15px; color: #FFFFFF; margin: 2rem 0;">
        <h2 style="color: #FFFFFF;">🌍 Our Mission</h2>
        <p style="font-size: 1.2em; color: #E8F5E9;">
            EcoGuard AI tackles the global waste crisis through artificial intelligence 
            and community engagement, making recycling smarter and more rewarding.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: #FFEBEE; padding: 1.5rem; border-radius: 10px;">
            <h3 style="color: #C62828;">🚨 The Problem</h3>
            <ul style="color: #212121; font-size: 1.1em;">
                <li>🌍 <strong>2.01 billion tonnes</strong> of waste annually</li>
                <li>♻️ Only <strong>13.5%</strong> is recycled globally</li>
                <li>🔥 <strong>33%</strong> is dumped or burned</li>
                <li>😕 Confusion about proper recycling</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: #E8F5E9; padding: 1.5rem; border-radius: 10px;">
            <h3 style="color: #2E7D32;">💡 Our Solution</h3>
            <ul style="color: #212121; font-size: 1.1em;">
                <li>🤖 AI-Powered Classification</li>
                <li>📊 Real-time Impact Tracking</li>
                <li>🏆 Gamification & Rewards</li>
                <li>📚 Recycling Education</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #616161;">
        <p>🛠️ Built with Streamlit, OpenCV, and Plotly</p>
        <p>🏆 AI For Good Hackathon 2026 | ACM-W Track</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>🌍 <strong style="color: #1B5E20;">EcoGuard AI</strong> - Making recycling smarter</p>
    <p style="font-size: 0.9em; color: #757575;">AI For Good Hackathon 2026 | ACM-W Track</p>
</div>
""", unsafe_allow_html=True)