"""
Utility modules for EcoGuard AI
"""

from .image_processing import (
    preprocess_image,
    extract_color_features,
    classify_waste_by_color,
    augment_image
)

from .impact_calculator import (
    calculate_environmental_impact,
    get_recycling_tips
)

from .data_visualizer import (
    create_impact_charts,
    create_leaderboard
)

__all__ = [
    'preprocess_image',
    'extract_color_features',
    'classify_waste_by_color',
    'augment_image',
    'calculate_environmental_impact',
    'get_recycling_tips',
    'create_impact_charts',
    'create_leaderboard'
]