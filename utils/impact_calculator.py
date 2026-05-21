import random

def calculate_environmental_impact(waste_type, weight_kg=1.0):
    """
    Calculate environmental impact of recycling different materials
    
    Args:
        waste_type: Type of waste material
        weight_kg: Weight in kilograms
    
    Returns:
        Dictionary with environmental impact metrics
    """
    # Impact factors per kg of recycled material
    impact_factors = {
        'plastic': {
            'co2_saved': 1.5,      # kg CO2
            'water_saved': 7.5,     # liters
            'energy_saved': 5.8,    # kWh
            'landfill_diverted': weight_kg * 0.85  # 85% recyclable
        },
        'paper': {
            'co2_saved': 1.0,
            'water_saved': 15.0,
            'energy_saved': 4.0,
            'landfill_diverted': weight_kg * 0.9
        },
        'glass': {
            'co2_saved': 0.3,
            'water_saved': 1.0,
            'energy_saved': 1.2,
            'landfill_diverted': weight_kg * 0.95
        },
        'metal': {
            'co2_saved': 5.0,
            'water_saved': 3.0,
            'energy_saved': 8.0,
            'landfill_diverted': weight_kg * 0.9
        },
        'organic': {
            'co2_saved': 0.5,
            'water_saved': 0.0,
            'energy_saved': 0.2,
            'landfill_diverted': weight_kg * 0.6
        },
        'e-waste': {
            'co2_saved': 20.0,
            'water_saved': 10.0,
            'energy_saved': 50.0,
            'landfill_diverted': weight_kg * 0.7
        }
    }
    
    # Get impact for specific waste type
    impact = impact_factors.get(waste_type.lower(), {
        'co2_saved': 0.5,
        'water_saved': 2.0,
        'energy_saved': 2.0,
        'landfill_diverted': weight_kg * 0.5
    })
    
    # Scale by weight
    scaled_impact = {
        'co2_saved': impact['co2_saved'] * weight_kg,
        'water_saved': impact['water_saved'] * weight_kg,
        'energy_saved': impact['energy_saved'] * weight_kg,
        'landfill_diverted': impact['landfill_diverted'] * weight_kg
    }
    
    return scaled_impact

def get_recycling_tips(waste_type):
    """
    Get recycling tips for specific waste types
    
    Args:
        waste_type: Type of waste material
    
    Returns:
        List of recycling tips
    """
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