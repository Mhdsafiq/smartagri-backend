import logging
from flask import Blueprint, request, jsonify
import pandas as pd
import joblib
import os
import pickle

logger = logging.getLogger(__name__)

# Load model relative to current file
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'yield_model.pkl')
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    logger.info(f"Loaded yield model from {MODEL_PATH}")
except FileNotFoundError:
    logger.warning(f"Yield model not found at {MODEL_PATH}. Using mock prediction.")
    model = None

# Create Blueprint
yield_bp = Blueprint('yield', __name__)

@yield_bp.route('/predict-yield', methods=['POST'])
def predict_yield():
    try:
        data = request.get_json()
        
        # 1. Validation
        required_fields = ['temperature', 'rainfall', 'humidity', 'crop_type', 'soil_type', 'land_area', 'unit']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400

        # 2. Extract Data
        temperature = float(data['temperature'])
        rainfall = float(data['rainfall'])
        humidity = float(data['humidity'])
        land_area = float(data['land_area'])
        unit_type = data['unit'].lower()
        crop_type = data['crop_type']
        soil_type = data['soil_type']

        # 3. Predict Yield (Using Model or Mock)
        if model:
             # Example Encoding (Would need to match training pre-processing exactly)
             # This is a placeholder. In production, need LabelEncoders saved with model.
             # df = pd.DataFrame([data])
             # encoded = encoder.transform(df)
             # prediction = model.predict(encoded)[0]
             prediction_per_hectare = 5.2 # Placeholder if encoding logic is complex to infer
        else:
             # Mock Logic for Demo if no model file
             base_yield = 4.0 # tons/hectare
             if crop_type == 'Rice': base_yield += 1.2
             if crop_type == 'Wheat': base_yield -= 0.5
             if soil_type == 'Loam': base_yield += 0.8
             prediction_per_hectare = base_yield 

        # 4. Conversion Logic
        # Convert input area to Hectares
        area_in_hectares = 0
        if unit_type == 'hectare':
            area_in_hectares = land_area
        elif unit_type == 'acre':
            area_in_hectares = land_area / 2.47105
        elif unit_type == 'cent':
            area_in_hectares = land_area / 247.105
        else:
            return jsonify({'error': 'Invalid unit type'}), 400

        total_yield_tons = prediction_per_hectare * area_in_hectares
        total_yield_kg = total_yield_tons * 1000

        # 5. Advisory Generation
        advisory = f"Ideally suited for {crop_type}. Ensure irrigation if rainfall drops below {rainfall * 0.8:.0f}mm."
        if temperature > 30:
            advisory += " Watch for heat stress."
        
        return jsonify({
            'yield_per_hectare': f"{prediction_per_hectare:.2f} Tons",
            'total_yield_kg': f"{total_yield_kg:.2f} kg",
            'advisory': advisory
        })

    except Exception as e:
        logger.error(f"Error in yield prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500
