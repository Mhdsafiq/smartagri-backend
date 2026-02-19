"""
Crop Detection using MobileNetV2 (Pre-trained on ImageNet)

ImageNet contains many plant, fruit, and vegetable categories.
This module maps those predictions to recognizable crop names.
If the image doesn't match any known crop/plant, it returns "Not Detected".
"""

import logging
import os
import numpy as np
import tensorflow as tf
from PIL import Image

logger = logging.getLogger(__name__)

# Load MobileNetV2 once at module level (singleton)
_mobilenet_model = None

def _get_model():
    """Lazy-load MobileNetV2 to avoid slow startup."""
    global _mobilenet_model
    if _mobilenet_model is None:
        logger.info("Loading MobileNetV2 model for crop detection...")
        _mobilenet_model = tf.keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=True
        )
        logger.info("MobileNetV2 loaded successfully.")
    return _mobilenet_model

# Load Custom Model once at module level (singleton)
_custom_model = None
_custom_labels = None
_custom_load_attempted = False

def _get_custom_model():
    """Lazy-load Custom Model to avoid slow startup."""
    global _custom_model, _custom_labels, _custom_load_attempted
    
    if _custom_load_attempted:
        return _custom_model, _custom_labels

    _custom_load_attempted = True
    
    try:
        base_path = os.path.dirname(__file__)
        models_dir = os.path.join(base_path, '..', 'models')
        
        custom_model_path_keras = os.path.join(models_dir, 'crop_model.keras')
        custom_model_path_h5 = os.path.join(models_dir, 'crop_model.h5')
        
        custom_model_path = None
        if os.path.exists(custom_model_path_keras):
            custom_model_path = custom_model_path_keras
        elif os.path.exists(custom_model_path_h5):
            custom_model_path = custom_model_path_h5
            
        labels_path = os.path.join(models_dir, 'crop_indices.json')
        
        if custom_model_path and os.path.exists(custom_model_path) and os.path.exists(labels_path):
            # Load custom model
            logger.info(f"Attempting to load custom model from: {custom_model_path}")
            _custom_model = tf.keras.models.load_model(custom_model_path)
            print("🔥 CUSTOM MODEL LOADED:", custom_model_path)

            logger.info("✅ SUCCESS: Custom crop model loaded!")
            
            # Load labels
            import json
            with open(labels_path, 'r') as f:
                _custom_labels = json.load(f) # {0: 'Rice', 1: 'Wheat'}
            logger.info(f"✅ SUCCESS: Custom labels loaded: {list(_custom_labels.values())}")
        else:
            logger.warning(f"Custom model files not found. Checked: {custom_model_path_keras}, {custom_model_path_h5}")
            
    except Exception as e:
        logger.error(f"❌ FAILED to load custom model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        _custom_model = None
        _custom_labels = None

    return _custom_model, _custom_labels


# ─── ImageNet Class → Crop Name Mapping ───────────────────────────────────────
# ImageNet has ~1000 classes. Many are plants, fruits, vegetables.
# We map the relevant ones to user-friendly crop names.
# Format: 'imagenet_class_id': ('Crop Name', 'Category')

IMAGENET_TO_CROP = {
    # ── Fruits ──
    'banana': ('Banana', 'Fruit'),
    'pineapple': ('Pineapple', 'Fruit'),
    'strawberry': ('Strawberry', 'Fruit'),
    'orange': ('Orange', 'Fruit'),
    'lemon': ('Lemon', 'Fruit'),
    'fig': ('Fig', 'Fruit'),
    'pomegranate': ('Pomegranate', 'Fruit'),
    'custard_apple': ('Custard Apple', 'Fruit'),
    'jackfruit': ('Jackfruit', 'Fruit'),
    'Granny_Smith': ('Apple', 'Fruit'),
    'acorn': ('Oak/Acorn', 'Tree'),
    
    # ── Vegetables ──
    'bell_pepper': ('Bell Pepper', 'Vegetable'),
    'cucumber': ('Cucumber', 'Vegetable'),
    'head_cabbage': ('Cabbage', 'Vegetable'),
    'broccoli': ('Broccoli', 'Vegetable'),
    'cauliflower': ('Cauliflower', 'Vegetable'),
    'zucchini': ('Zucchini', 'Vegetable'),
    'spaghetti_squash': ('Squash', 'Vegetable'),
    'butternut_squash': ('Butternut Squash', 'Vegetable'),
    'artichoke': ('Artichoke', 'Vegetable'),
    'cardoon': ('Cardoon', 'Vegetable'),
    'mushroom': ('Mushroom', 'Fungus'),
    'agaric': ('Mushroom', 'Fungus'),
    'ear': ('Corn/Maize', 'Cereal'),  # "ear" in ImageNet refers to ear of corn
    
    # ── Flowers / Plants ──
    'daisy': ('Daisy', 'Flower'),
    'sunflower': ('Sunflower', 'Oilseed Crop'),
    'rapeseed': ('Mustard/Rapeseed', 'Oilseed Crop'),
    'hip': ('Rose Hip', 'Flower'),
    
    # ── Crops & Agricultural ──
    'corn': ('Corn/Maize', 'Cereal'),
    'acorn_squash': ('Squash', 'Vegetable'),
    'hay': ('Rice / Wheat', 'Cereal'), # Re-mapped from 'Hay'
    
    # ── Leaf / Plant structure detections ──
    'leaf_beetle': ('Leaf (with beetle)', 'Plant'),
    'plant': ('Plant', 'Plant'),
    
    # ── Trees ──
    'coconut': ('Coconut', 'Tree Crop'),
    'bamboo': ('Rice / Sugarcane / Bamboo', 'Cereal/Grass'), # Often confused with tall crops
    
    # ── Broad matches for green/crop-like images ──
    'valley': ('Rice / Field Crop', 'Landscape'),  # Farm fields sometimes detected as valley
    'alp': ('Highland Crop Field', 'Landscape'),
    'lakeside': ('Rice Paddy', 'Landscape'), # Wet fields often detected as lakeside
}

# Additional broad keywords that indicate plant/crop content in ImageNet labels
PLANT_KEYWORDS = [
    'flower', 'plant', 'tree', 'leaf', 'seed', 'fruit', 'vegetable',
    'herb', 'grass', 'vine', 'fern', 'moss', 'fungus', 'mushroom',
    'pot', 'garden', 'crop', 'corn', 'wheat', 'rice', 'bean',
    'pea', 'pepper', 'tomato', 'potato', 'onion', 'garlic',
    'cabbage', 'lettuce', 'spinach', 'carrot', 'radish',
    'banana', 'apple', 'orange', 'mango', 'grape', 'berry',
    'melon', 'coconut', 'palm', 'bamboo', 'cactus', 'orchid',
    'tulip', 'rose', 'daisy', 'sunflower', 'lily', 'poppy',
    'rapeseed', 'hay', 'straw', 'field', 'paddy', 'grain'
]

# Confidence threshold for crop detection
CROP_CONFIDENCE_THRESHOLD = 0.10  # 10% - lowered since ImageNet has 1000 classes


def detect_crop(pil_image: Image.Image) -> dict:
    """
    Detect the crop/plant in an image.
    
    Priority:
    1. Custom Trained Model ('models/crop_model.h5')
    2. Generic MobileNetV2 (ImageNet)
    
    Args:
        pil_image: PIL Image object (already opened)
    
    Returns:
        dict with crop details
    """
    try:
        # ─── STRATEGY A: CUSTOM TRAINED MODEL ───
        custom_model, label_map = _get_custom_model()
        
        if custom_model and label_map:
            try:
                # Preprocess (MATCHING YOUR TRAINING: rescale=1./255)
                # Ensure we resize exactly as you did in training (224x224)
                img_resized = pil_image.resize((224, 224))
                img_array = np.array(img_resized, dtype=np.float32)
                
                # CRITICAL FIX: Your model was trained with 1./255 (0 to 1 values)
                # Do NOT use mobilenet_v2.preprocess_input here because it does -1 to 1
                img_array = img_array / 255.0
                
                img_array = np.expand_dims(img_array, axis=0)
                
                # Predict
                predictions = custom_model.predict(img_array)
                predicted_idx = np.argmax(predictions[0])
                confidence = float(predictions[0][predicted_idx])
                
                # LOG ALL PREDICTIONS for debugging
                all_preds = {}
                # Handle label_map keys (might be strings of ints)
                for idx, score in enumerate(predictions[0]):
                    name = label_map.get(str(idx), str(idx))
                    all_preds[name] = float(score)
                
                # Sort and log
                sorted_preds = sorted(all_preds.items(), key=lambda x: x[1], reverse=True)
                logger.info(f"🔍 Full Prediction Distribution: {sorted_preds}")
                
                # STRICT CONFIDENCE CHECK as requested by user
                # If below 80%, show as Not Detected
                CONFIDENCE_THRESHOLD = 0.80 

                if confidence < CONFIDENCE_THRESHOLD:
                    logger.info(f"Low confidence ({confidence:.2f} < {CONFIDENCE_THRESHOLD}). Returning 'Not Detected'.")
                    return {
                        "crop_detected": False,
                        "crop_name": "Not Detected",
                        "crop_category": "Unknown",
                        "confidence": confidence,
                        "top_predictions": [],
                        "message": "Crop not detected (low confidence)."
                    }

                crop_name = label_map.get(str(predicted_idx), "Unknown")
                logger.info(f"✅ Custom Model Prediction: {crop_name} ({confidence:.2f})")

                return {
                    'crop_detected': True,
                    'crop_name': crop_name,
                    'crop_category': 'Custom Trained',
                    'confidence': confidence,
                    'top_predictions': [{'label': crop_name, 'confidence': confidence}]
                }
            except Exception as e:
                logger.error(f"❌ FAILED to use custom model: {e}")
                import traceback
                logger.error(traceback.format_exc())
                # Fallback to generic...
        
        # ─── STRATEGY B: GENERIC MOBILENET V2 ───
        model = _get_model()

        
        # Preprocess for MobileNetV2 (224x224, specific normalization)
        img_resized = pil_image.resize((224, 224))
        img_array = np.array(img_resized, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        # Predict
        predictions = model.predict(img_array, verbose=0)
        
        # Decode top 10 predictions
        decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=10)[0]
        
        # Format top predictions for debugging
        top_preds = [
            {'class_id': cls_id, 'label': label, 'confidence': float(conf)}
            for cls_id, label, conf in decoded
        ]
        
        logger.info(f"Top 5 MobileNetV2 predictions: {top_preds[:5]}")
        
        # Strategy 1: Check direct mapping
        for cls_id, label, conf in decoded:
            if label in IMAGENET_TO_CROP:
                crop_name, category = IMAGENET_TO_CROP[label]
                return {
                    'crop_detected': True,
                    'crop_name': crop_name,
                    'crop_category': category,
                    'confidence': float(conf),
                    'top_predictions': top_preds[:5]
                }
        
        # Strategy 2: Check if any prediction contains plant-related keywords
        for cls_id, label, conf in decoded:
            label_lower = label.lower().replace('_', ' ')
            for keyword in PLANT_KEYWORDS:
                if keyword in label_lower and conf >= CROP_CONFIDENCE_THRESHOLD:
                    # Found a plant-related prediction
                    # Capitalize nicely
                    nice_name = label.replace('_', ' ').title()
                    return {
                        'crop_detected': True,
                        'crop_name': nice_name,
                        'crop_category': 'Plant',
                        'confidence': float(conf),
                        'top_predictions': top_preds[:5]
                    }
        
        # Strategy 3: Check if the top prediction has high enough confidence 
        # and looks like it could be organic/natural
        top_label = decoded[0][1]
        top_conf = float(decoded[0][2])
        
        # Not a crop/plant
        return {
            'crop_detected': False,
            'crop_name': 'Not Detected',
            'crop_category': 'Unknown',
            'confidence': top_conf,
            'top_predictions': top_preds[:5],
            'message': f'Image appears to be: {top_label.replace("_", " ").title()} (not a recognized crop)'
        }
        
    except Exception as e:
        logger.error(f"Crop detection error: {str(e)}")
        return {
            'crop_detected': False,
            'crop_name': 'Not Detected',
            'crop_category': 'Error',
            'confidence': 0.0,
            'top_predictions': [],
            'message': f'Detection failed: {str(e)}'
        }
