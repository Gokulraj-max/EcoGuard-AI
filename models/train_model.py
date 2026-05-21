import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import pickle
from pathlib import Path
import cv2
import os

def create_sample_dataset():
    """
    Create sample training data for demonstration
    In production, this would load actual image data
    """
    n_samples = 1000
    n_features = 128
    
    # Generate synthetic features
    X = np.random.rand(n_samples, n_features)
    
    # Waste categories
    categories = ['plastic', 'paper', 'glass', 'metal', 'organic', 'e-waste']
    y = np.random.choice(categories, n_samples)
    
    return X, y

def train_waste_classifier():
    """
    Train waste classification model
    """
    print("Loading dataset...")
    X, y = create_sample_dataset()
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )
    
    # Train model
    print("Training Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Save model and encoder
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    print("\nSaving model...")
    with open(models_dir / "waste_classifier.pkl", "wb") as f:
        pickle.dump(model, f)
    
    with open(models_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
    
    print("Model training complete!")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': [f'feature_{i}' for i in range(X.shape[1])],
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    return model, label_encoder

def build_cnn_model():
    """
    Build a CNN model for waste classification
    NOTE: Requires TensorFlow to be installed
    Run: pip install tensorflow
    """
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
        from tensorflow.keras.optimizers import Adam
        
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            MaxPooling2D(2, 2),
            
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(6, activation='softmax')  # 6 waste categories
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("CNN model built successfully!")
        return model
        
    except ImportError as e:
        print(f"TensorFlow not installed. CNN model not available.")
        print(f"To install: pip install tensorflow")
        print(f"Error: {e}")
        return None

def train_with_real_images(data_dir="data/train"):
    """
    Train model with real images from directory
    
    Args:
        data_dir: Path to training data directory with subfolders for each class
    """
    print(f"Looking for training images in: {data_dir}")
    
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} not found!")
        print("Creating sample dataset instead...")
        return train_waste_classifier()
    
    X = []
    y = []
    
    categories = ['plastic', 'paper', 'glass', 'metal', 'organic', 'e-waste']
    
    for category in categories:
        category_path = os.path.join(data_dir, category)
        if os.path.exists(category_path):
            images = os.listdir(category_path)
            print(f"Found {len(images)} images for {category}")
            
            for img_name in images[:100]:  # Limit to 100 images per class
                img_path = os.path.join(category_path, img_name)
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, (224, 224))
                        # Extract features (simplified)
                        features = extract_image_features(img)
                        X.append(features)
                        y.append(category)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        else:
            print(f"Category folder not found: {category_path}")
    
    if len(X) == 0:
        print("No images found. Using sample dataset...")
        return train_waste_classifier()
    
    X = np.array(X)
    print(f"Total training samples: {len(X)}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Train model
    print("Training Random Forest classifier on real images...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Save model
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    print("\nSaving model...")
    with open(models_dir / "waste_classifier.pkl", "wb") as f:
        pickle.dump(model, f)
    
    with open(models_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
    
    print("Model training complete!")
    return model, label_encoder

def extract_image_features(img):
    """
    Extract features from an image
    
    Args:
        img: numpy array of image
    
    Returns:
        Feature vector
    """
    features = []
    
    # Resize for consistency
    img = cv2.resize(img, (224, 224))
    
    # Color histograms (32 bins per channel)
    for channel in range(3):
        hist = cv2.calcHist([img], [channel], None, [32], [0, 256])
        features.extend(hist.flatten())
    
    # HSV histogram
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for channel in range(3):
        hist = cv2.calcHist([hsv], [channel], None, [32], [0, 256])
        features.extend(hist.flatten())
    
    # Edge detection features
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.mean(edges) / 255.0
    features.append(edge_density)
    
    # Mean and std of each channel
    for channel in range(3):
        features.append(np.mean(img[:, :, channel]))
        features.append(np.std(img[:, :, channel]))
    
    return np.array(features)

if __name__ == "__main__":
    print("=" * 60)
    print("EcoGuard AI - Model Training")
    print("=" * 60)
    
    # Check if real data exists
    if os.path.exists("data/train"):
        print("\nFound training data directory!")
        train_with_real_images("data/train")
    else:
        print("\nNo training data found. Training with sample data...")
        train_waste_classifier()
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("\nTo train with actual images, organize dataset as:")
    print("data/")
    print("  ├── train/")
    print("  │   ├── plastic/")
    print("  │   ├── paper/")
    print("  │   ├── glass/")
    print("  │   ├── metal/")
    print("  │   ├── organic/")
    print("  │   └── e-waste/")
    print("  └── test/")
    print("      ├── plastic/")
    print("      └── ...")
    print("\nTo build CNN model (requires TensorFlow):")
    print("  pip install tensorflow")
    print("  python -c 'from models.train_model import build_cnn_model; build_cnn_model()'")
    print("=" * 60)