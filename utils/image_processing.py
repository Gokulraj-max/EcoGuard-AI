import cv2
import numpy as np
from PIL import Image
import io

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess image for model input
    
    Args:
        image: PIL Image or numpy array
        target_size: Target size for model input
    
    Returns:
        Preprocessed image array
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to RGB if grayscale
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Resize
    image = cv2.resize(image, target_size)
    
    # Normalize
    image = image.astype('float32') / 255.0
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image

def extract_color_features(image):
    """
    Extract color-based features from image for waste classification
    
    Args:
        image: PIL Image or numpy array
    
    Returns:
        Dictionary containing color features
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to RGB if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Resize for consistent processing
    image_resized = cv2.resize(image, (224, 224))
    
    # Convert to different color spaces
    hsv = cv2.cvtColor(image_resized, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image_resized, cv2.COLOR_RGB2LAB)
    
    # Calculate average colors in different spaces
    avg_rgb = np.mean(image_resized, axis=(0, 1))
    avg_hsv = np.mean(hsv, axis=(0, 1))
    avg_lab = np.mean(lab, axis=(0, 1))
    
    # Calculate color variance (texture indicator)
    std_rgb = np.std(image_resized, axis=(0, 1))
    std_hsv = np.std(hsv, axis=(0, 1))
    
    # Calculate color histograms
    hist_r = cv2.calcHist([image_resized], [0], None, [32], [0, 256]).flatten()
    hist_g = cv2.calcHist([image_resized], [1], None, [32], [0, 256]).flatten()
    hist_b = cv2.calcHist([image_resized], [2], None, [32], [0, 256]).flatten()
    
    # Edge detection for texture analysis
    gray = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.mean(edges) / 255.0
    
    # Dominant color detection
    pixels = image_resized.reshape(-1, 3)
    from collections import Counter
    # Quantize colors
    quantized = (pixels // 32).astype(np.uint8)
    color_tuples = [tuple(pixel) for pixel in quantized]
    color_counts = Counter(color_tuples)
    dominant_colors = color_counts.most_common(5)
    
    # Check for specific color ranges
    is_brown = (avg_hsv[0] > 10 and avg_hsv[0] < 30 and avg_hsv[1] > 30)  # Brown range
    is_green = (avg_hsv[0] > 35 and avg_hsv[0] < 85 and avg_hsv[1] > 30)  # Green range
    is_gray = (avg_hsv[1] < 30 and avg_hsv[2] > 100)  # Gray/silver
    is_white = (avg_hsv[1] < 20 and avg_hsv[2] > 200)  # White/bright
    is_dark = (avg_hsv[2] < 80)  # Dark colors
    is_colorful = (avg_hsv[1] > 50)  # High saturation
    
    features = {
        'avg_rgb': avg_rgb.tolist(),
        'avg_hsv': avg_hsv.tolist(),
        'avg_lab': avg_lab.tolist(),
        'std_rgb': std_rgb.tolist(),
        'std_hsv': std_hsv.tolist(),
        'edge_density': float(edge_density),
        'dominant_colors': [(list(color), count) for color, count in dominant_colors],
        'is_brown': bool(is_brown),
        'is_green': bool(is_green),
        'is_gray': bool(is_gray),
        'is_white': bool(is_white),
        'is_dark': bool(is_dark),
        'is_colorful': bool(is_colorful),
        'brightness': float(avg_hsv[2]),
        'saturation': float(avg_hsv[1]),
        'hue': float(avg_hsv[0])
    }
    
    return features

def augment_image(image):
    """
    Apply data augmentation techniques
    
    Args:
        image: Input image array
    
    Returns:
        List of augmented images
    """
    augmented_images = []
    
    # Original
    augmented_images.append(image)
    
    # Horizontal flip
    augmented_images.append(cv2.flip(image, 1))
    
    # Rotation
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 15, 1)
    augmented_images.append(cv2.warpAffine(image, M, (cols, rows)))
    
    # Rotation -15 degrees
    M = cv2.getRotationMatrix2D((cols/2, rows/2), -15, 1)
    augmented_images.append(cv2.warpAffine(image, M, (cols, rows)))
    
    # Brightness adjustment
    bright = cv2.convertScaleAbs(image, alpha=1.2, beta=0)
    augmented_images.append(bright)
    
    # Darker
    dark = cv2.convertScaleAbs(image, alpha=0.8, beta=0)
    augmented_images.append(dark)
    
    # Blur
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    augmented_images.append(blurred)
    
    return augmented_images

def classify_waste_by_color(image):
    """
    Classify waste type based on image color features
    
    Args:
        image: PIL Image or numpy array
    
    Returns:
        predicted_class: string
        confidence: float
        all_scores: dict of scores for each class
    """
    features = extract_color_features(image)
    
    scores = {
        'plastic': 0,
        'paper': 0,
        'glass': 0,
        'metal': 0,
        'organic': 0,
        'e-waste': 0
    }
    
    r, g, b = features['avg_rgb']
    h, s, v = features['avg_hsv']
    edge_density = features['edge_density']
    
    # PLASTIC: Often colorful, medium saturation, smooth surface
    if features['is_colorful']:
        scores['plastic'] += 30
    if edge_density < 0.15:  # Smooth surface
        scores['plastic'] += 20
    if 50 < s < 200 and v > 80:  # Visible colors
        scores['plastic'] += 15
    if 100 < r < 200 and 100 < g < 200 and 100 < b < 200:
        scores['plastic'] += 10
    if features['is_white']:  # White plastic
        scores['plastic'] += 15
    
    # PAPER: Light colored, low saturation, some texture
    if features['is_white'] or v > 180:
        scores['paper'] += 30
    if s < 40:  # Low saturation
        scores['paper'] += 20
    if 0.1 < edge_density < 0.25:  # Moderate texture
        scores['paper'] += 15
    if features['is_brown']:  # Cardboard
        scores['paper'] += 25
    if r > 150 and g > 140 and b > 120:  # Warm/light tones
        scores['paper'] += 15
    
    # GLASS: Often transparent, high brightness, sharp edges from reflections
    if v > 150 and s < 50:  # Bright, low saturation
        scores['glass'] += 25
    if edge_density > 0.15:  # Sharp edges
        scores['glass'] += 20
    if features['std_rgb'][0] > 40 or features['std_rgb'][1] > 40:  # Reflection variance
        scores['glass'] += 20
    if features['is_gray'] and v > 150:  # Clear glass
        scores['glass'] += 15
    if features['is_green'] and v > 100:  # Green glass
        scores['glass'] += 15
    if features['is_brown'] and v > 100:  # Brown glass
        scores['glass'] += 15
    
    # METAL: Shiny, gray/silver, high brightness variation
    if features['is_gray']:
        scores['metal'] += 35
    if abs(r - g) < 20 and abs(g - b) < 20 and abs(r - b) < 20:  # Gray
        scores['metal'] += 25
    if features['std_rgb'][0] > 35 or features['std_rgb'][1] > 35:  # Reflections
        scores['metal'] += 20
    if s < 30 and 100 < v < 200:  # Low saturation, medium brightness
        scores['metal'] += 15
    if edge_density > 0.1:  # Some edges
        scores['metal'] += 10
    
    # ORGANIC: Brown/green tones, varied texture
    if features['is_brown']:
        scores['organic'] += 35
    if features['is_green']:
        scores['organic'] += 30
    if edge_density > 0.15:  # Natural texture
        scores['organic'] += 20
    if s > 25 and v < 180:  # Natural colors
        scores['organic'] += 15
    if 20 < h < 80 and s > 20:  # Yellow-green range
        scores['organic'] += 20
    if r > g and g > b:  # Brown tones
        scores['organic'] += 15
    
    # E-WASTE: Dark, black/gray, complex patterns, colorful details
    if features['is_dark']:
        scores['e-waste'] += 35
    if edge_density > 0.2:  # Complex patterns
        scores['e-waste'] += 30
    if r < 80 and g < 80 and b < 80:  # Very dark
        scores['e-waste'] += 20
    if features['std_rgb'][0] > 50:  # High variance (circuits)
        scores['e-waste'] += 15
    if features['is_colorful'] and features['is_dark']:  # Dark with colors (LEDs)
        scores['e-waste'] += 20
    
    # Get prediction
    predicted_class = max(scores, key=scores.get)
    max_score = scores[predicted_class]
    
    # Calculate confidence
    total_score = sum(scores.values())
    if total_score > 0:
        confidence = max_score / total_score
    else:
        confidence = 1/6
    
    return predicted_class, confidence, scores

def get_dominant_colors(image, n_colors=5):
    """
    Get dominant colors from image
    
    Args:
        image: numpy array
        n_colors: Number of dominant colors to return
    
    Returns:
        List of (color, percentage) tuples
    """
    from collections import Counter
    
    # Resize for faster processing
    img_small = cv2.resize(image, (50, 50))
    pixels = img_small.reshape(-1, 3)
    
    # Quantize colors
    quantized = (pixels // 32).astype(np.uint8)
    color_tuples = [tuple(pixel) for pixel in quantized]
    
    # Count colors
    color_counts = Counter(color_tuples)
    total_pixels = sum(color_counts.values())
    
    # Get dominant colors with percentages
    dominant = []
    for color, count in color_counts.most_common(n_colors):
        percentage = (count / total_pixels) * 100
        # Convert back to actual RGB values
        actual_color = tuple([c * 32 + 16 for c in color])
        dominant.append((actual_color, percentage))
    
    return dominant

def create_color_histogram(image):
    """
    Create color histogram for visualization
    
    Args:
        image: numpy array
    
    Returns:
        Dictionary with histogram data for each channel
    """
    colors = ('r', 'g', 'b')
    hist_data = {}
    
    for i, color in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        hist_data[color] = hist.flatten().tolist()
    
    return hist_data