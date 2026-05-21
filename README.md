# 🌍 EcoGuard AI - Smart Waste Classification

> **AI-powered waste classification system that makes recycling smarter and tracks environmental impact**
>
> 🏆 Built for the **AI For Good Hackathon 2026** | **ACM-W Track**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ecoguard-ai-c2xudqxekznby3pyzqkq5c.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Live-success.svg)

---

# 📸 Screenshots

<p align="center">
  <img src="screenshots/classification.png" alt="Classification Demo" width="400"/>
  <img src="screenshots/dashboard.png" alt="Impact Dashboard" width="400"/>
</p>

---

# 🎯 Project Overview

EcoGuard AI tackles the global waste management crisis through artificial intelligence and community engagement.

Every year:

- 🌍 **2.01 billion tonnes** of municipal solid waste generated globally
- ♻️ Only **13.5% recycled**
- 🔥 **33% openly dumped or burned**
- 😕 Confusion around proper recycling practices

Our application:

- 🤖 AI-powered waste identification
- 📊 Real-time environmental impact tracking
- 🏆 Gamification for sustainability
- 📚 Educational recycling guidance

---

# ✨ Features

| Feature | Description |
|---------|-------------|
| 📸 Smart Classification | Upload images or use camera to identify 6 waste types |
| ♻️ Recycling Guidance | Get actionable disposal recommendations |
| 📊 Impact Dashboard | Visual analytics for environmental savings |
| 🏆 Community Challenges | Achievements, leaderboards, goals |
| 📚 Educational Content | Learn sustainability practices |

---

# 🚀 Live Demo

👉 **Try it now:**  
https://ecoguard-ai-c2xudqxekznby3pyzqkq5c.streamlit.app

---

# 🛠️ Technology Stack

| Category | Technologies |
|----------|--------------|
| Frontend | Streamlit, HTML/CSS, Plotly |
| Computer Vision | OpenCV, Pillow |
| Machine Learning | Scikit-learn, NumPy |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly Express, Matplotlib |
| Deployment | Streamlit Cloud, GitHub |

---

# 📂 Project Structure

```text
ecoguard-ai/
│
├── app.py
├── requirements.txt
├── .streamlit/
│   └── config.toml
│
├── utils/
│   ├── __init__.py
│   ├── image_processing.py
│   ├── impact_calculator.py
│   └── data_visualizer.py
│
├── models/
│   └── train_model.py
│
├── notebooks/
│   └── model_development.ipynb
│
├── screenshots/
├── README.md
├── LICENSE
└── .gitignore
```

---

# 🚀 Quick Start

## Prerequisites

- Python 3.8+
- pip
- Git (optional)

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/Gokulraj-max/EcoGuard-AI.git
cd EcoGuard-AI
```

### 2. Create Virtual Environment

Linux / macOS:

```bash
python -m venv venv
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Run Application

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

# 🎯 How It Works

## Classification Algorithm

EcoGuard AI uses a **multi-feature color and texture analysis approach**.

### 🔹 Color Space Analysis
- RGB
- HSV
- LAB

### 🔹 Color Ratio Detection
Detects:

- White
- Black
- Brown
- Green
- Blue
- Red
- Yellow

### 🔹 Texture Analysis
- Edge Density
- Laplacian Variance
- Morphological Gradients

### 🔹 Shape Features
- Contour Detection
- Area Analysis

### 🔹 Dominant Color Extraction
- K-Means Clustering

### 🔹 Weighted Scoring
Generates confidence scores across **6 waste categories**

---

# ♻️ Waste Categories

| Category | Emoji | Examples |
|----------|------|----------|
| Plastic | 🥤 | Bottles, containers, bags |
| Paper | 📄 | Cardboard, newspapers |
| Glass | 🫙 | Bottles, jars |
| Metal | 🥫 | Cans, foil |
| Organic | 🍎 | Food scraps |
| E-Waste | 💻 | Electronics, batteries |

---

# 📊 Environmental Impact Tracking

Every classified item contributes to measurable environmental savings.

| Metric | Description |
|--------|------------|
| 🌳 CO₂ Saved | Carbon emissions prevented |
| 💧 Water Saved | Water conserved |
| ⚡ Energy Saved | Energy conserved |
| 🗑️ Landfill Diverted | Waste diverted |

---

# 🏆 Future Improvements

- Deep Learning integration (CNN)
- Mobile Application
- User Accounts
- Global Community Rankings
- Multi-language Support

---

# 🤝 Contributing

Contributions are welcome.

```bash
fork → create branch → commit → push → pull request
```

---

# 📜 License

Distributed under the **MIT License**.

---

# ❤️ Built For

**AI For Good Hackathon 2026**