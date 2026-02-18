from flask import Blueprint, request, jsonify
import logging
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import hashlib
import random

# Import the real crop detector
from utils.crop_detector import detect_crop

# Setup Logging
logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'leaf_model.h5')
# Standard Alphabetical Order
CLASS_LABELS = ['Healthy', 'Nitrogen', 'Phosphorus', 'Potassium']

# Load Leaf Deficiency Model
try:
    if os.path.exists(MODEL_PATH):
        leaf_model = tf.keras.models.load_model(MODEL_PATH)
        logger.info(f"Loaded leaf deficiency model from {MODEL_PATH}")
    else:
        logger.warning(f"Leaf model not found at {MODEL_PATH}. Using deterministic mock prediction.")
        leaf_model = None
except Exception as e:
    logger.error(f"Error loading leaf model: {e}")
    leaf_model = None

# Blueprint
fertilizer_bp = Blueprint('fertilizer', __name__)

@fertilizer_bp.route('/predict-fertilizer', methods=['POST'])
def predict_fertilizer():
    try:
        # 1. Validation
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
            
        file = request.files['image']
        try:
            land_area = float(request.form.get('land_area', 1.0) or 1.0)
        except ValueError:
            land_area = 1.0
            
        unit = request.form.get('unit', 'acre').lower()
        
        # 2. Image Preprocessing
        img = Image.open(file.stream).convert('RGB')
        img_for_detection = img.copy()  # Keep a copy for crop detection
        
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # ═══════════════════════════════════════════════════════════════
        # Step A: REAL Crop Detection using MobileNetV2
        # ═══════════════════════════════════════════════════════════════
        crop_result = detect_crop(img_for_detection)
        
        crop_detected = crop_result['crop_detected']
        crop_type = crop_result['crop_name']
        crop_conf = crop_result['confidence']
        crop_category = crop_result['crop_category']
        top_predictions = crop_result.get('top_predictions', [])
        
        logger.info(f"Crop detection: detected={crop_detected}, name={crop_type}, conf={crop_conf:.2f}")

        # ═══════════════════════════════════════════════════════════════
        # Step B: Stop if Crop Not Detected (Low Confidence)
        # ═══════════════════════════════════════════════════════════════
        if not crop_detected:
            # If the crop detector said "No", we stop here.
            # We don't want to show deficiency for an unknown object.
            return jsonify({
                "crop_detected": False,
                "crop_type": "Not Detected",
                "crop_category": "Unknown",
                "crop_confidence": round(crop_conf, 2),
                "message": crop_result.get('message', 'Confidence too low. Please upload a clearer image of a crop leaf.'),
                # Clear out other fields so UI doesn't show them
                "deficiency": None, 
                "fertilizer": None,
                "severity": None,
                "advisory": "Could not identify the crop with sufficient confidence (>80%). Please try again with a better image."
            })

        # ═══════════════════════════════════════════════════════════════
        # Step C: Predict Leaf Deficiency (Only if crop is found)
        # ═══════════════════════════════════════════════════════════════
        LEAF_LABELS = ['Nitrogen', 'Phosphorus', 'Potassium', 'Healthy']
        
        if leaf_model is not None:
            # Use the real leaf model
            preds = leaf_model.predict(img_array)
            idx = np.argmax(preds)
            deficiency = LEAF_LABELS[idx]
            deficiency_conf = float(preds[0][idx])
        else:
            # Pure random mock (User requested random deficiency)
            # Remove deterministic seeding to ensure variety
            # seed_val = int(img_hash[:8], 16)
            # random.seed(seed_val)
            
            deficiency = random.choice(LEAF_LABELS)
            deficiency_conf = 0.85 + (random.random() * 0.14)  # 0.85 to 0.99

        # 4. Filter Low Confidence
        if deficiency_conf < 0.3:
            return jsonify({
                'error': 'Low confidence prediction. Please upload clearer image.',
                'raw_confidence': deficiency_conf,
                'crop_detected': crop_detected,
                'crop_type': crop_type
            }), 400

        # 5. Determine Severity
        severity = 'Mild'
        if deficiency_conf > 0.85:
            severity = 'Severe'
        elif deficiency_conf > 0.6:
            severity = 'Moderate'
            
        # Special case: Healthy plants have no severity
        if deficiency == 'Healthy':
            severity = 'None'

        # 6. Map Fertilizer
        rec_map = {
            'Nitrogen': {'fertilizer': 'Urea', 'base_dose': 50},
            'Phosphorus': {'fertilizer': 'DAP', 'base_dose': 40},
            'Potassium': {'fertilizer': 'MOP', 'base_dose': 30},
            'Healthy': {'fertilizer': None, 'base_dose': 0}
        }
        
        rec_data = next((v for k, v in rec_map.items() if k.lower() in deficiency.lower()), 
                        {'fertilizer': 'General NPK', 'base_dose': 20})
        
        fertilizer_name = rec_data['fertilizer']
        
        # 7. Calculate Quantity
        area_hectares = 0
        if unit == 'hectare': area_hectares = land_area
        elif unit == 'acre': area_hectares = land_area / 2.47
        elif unit == 'cent': area_hectares = land_area / 247.0
        
        total_qty = rec_data['base_dose'] * area_hectares
        
        recommended_qty_str = f"{total_qty:.2f} kg"
        
        if deficiency == 'Healthy':
             fertilizer_name = 'Optional / Maintenance'
             recommended_qty_str = "Maintenance Dose (Optional)"

        # 8. Construct Final JSON
        response = {
            "crop_detected": crop_detected,
            "crop_type": crop_type,
            "crop_category": crop_category,
            "crop_confidence": round(crop_conf, 2),
            "top_predictions": top_predictions,
            "deficiency": f"{deficiency} Deficiency" if deficiency != 'Healthy' else "Healthy",
            "deficiency_confidence": round(deficiency_conf, 2),
            "severity": severity,
            "fertilizer": fertilizer_name,
            "recommended_quantity": recommended_qty_str,
        }
        
        # Add contextual advisory
        if crop_detected:
            if deficiency != 'Healthy':
                response["advisory"] = f"Detected {deficiency} deficiency in {crop_type} ({crop_category}). {severity} severity."
            else:
                response["advisory"] = f"Your {crop_type} ({crop_category}) appears healthy. No treatment required."
        else:
            response["advisory"] = "Could not identify the crop. The deficiency analysis is based on leaf color patterns."
            response["message"] = crop_result.get('message', 'Image does not match any known crop.')
        
        logger.info(f"Prediction: {response}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in fertilizer prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500
