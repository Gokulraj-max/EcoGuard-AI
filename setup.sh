#!/bin/bash

echo "Setting up EcoGuard AI..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install --upgrade pip
pip install -r requirements.txt

# Create directory structure
mkdir -p data models utils static/images notebooks demos/sample_images

# Train initial model
python models/train_model.py

echo "Setup complete! Run 'streamlit run app.py' to start the application."