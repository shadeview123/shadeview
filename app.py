import os
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, session, flash, g, send_from_directory
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from fpdf import FPDF
from datetime import datetime
import cv2  # For image processing
import numpy as np  # For numerical operations with images
import json  # For parsing Firebase config
import traceback # Import for printing full tracebacks
import uuid # For generating unique filenames

# --- Firestore Imports (Conceptual for Canvas) ---
# from firebase_admin import credentials, firestore, initialize_app
# from google.cloud.firestore import Client as FirestoreClient # For type hinting if using client library

# --- NEW IMPORTS FOR AI (Machine Learning) INTEGRATION ---
from sklearn.neighbors import KNeighborsClassifier
import joblib  # For saving/loading machine learning models
import pandas as pd  # For CSV handling
# ---------------------------------------------------------

# --- NEW IMPORTS FOR DELTA E CALCULATION ---
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000
# ---------------------------------------------

# --- VITA Shade LAB Reference Values (Approximate, can be refined with empirical data) ---
VITA_SHADE_LAB_REFERENCES = {
    "A1": LabColor(lab_l=75.0, lab_a=2.0, lab_b=15.0),
    "A2": LabColor(lab_l=70.0, lab_a=3.0, lab_b=18.0),
    "A3": LabColor(lab_l=65.0, lab_a=4.0, lab_b=21.0),
    "A3.5": LabColor(lab_l=60.0, lab_a=5.0, lab_b=23.0),
    "A4": LabColor(lab_l=55.0, lab_a=6.0, lab_b=25.0),
    "B1": LabColor(lab_l=80.0, lab_a=0.0, lab_b=12.0),
    "B2": LabColor(lab_l=73.0, lab_a=1.0, lab_b=15.0),
    "B3": LabColor(lab_l=67.0, lab_a=2.0, lab_b=18.0),
    "B4": LabColor(lab_l=62.0, lab_a=3.0, lab_b=20.0),
    "C1": LabColor(lab_l=70.0, lab_a=0.0, lab_b=10.0),
    "C2": LabColor(lab_l=64.0, lab_a=1.0, lab_b=12.0),
    "C3": LabColor(lab_l=58.0, lab_a=2.0, lab_b=14.0),
    "C4": LabColor(lab_l=52.0, lab_a=3.0, lab_b=16.0),
    "D2": LabColor(lab_l=68.0, lab_a=2.0, lab_b=14.0),
    "D3": LabColor(lab_l=62.0, lab_a=3.0, lab_b=16.0),
    "D4": LabColor(lab_l=56.0, lab_a=4.0, lab_b=18.0),
}


# --- IMAGE PROCESSING FUNCTIONS (Self-contained for simplicity) ---
def gray_world_white_balance(img):
    """
    Applies Gray World Algorithm for white balancing an image.
    Args:
        img (numpy.ndarray): The input image in BGR format.
    Returns:
        numpy.ndarray: The white-balanced image in BGR format.
    """
    result = img.copy().astype(np.float32)  # Convert to float32 for calculations

    # Calculate average intensity for each channel
    avgB = np.mean(result[:, :, 0])
    avgG = np.mean(result[:, :, 1])
    avgR = np.mean(result[:, :, 2])

    # Calculate overall average gray value
    avgGray = (avgB + avgG + avgR) / 3

    # Apply scaling factor to each channel, clipping at 255
    result[:, :, 0] = np.clip(result[:, :, 0] * (avgGray / avgB), 0, 255)
    result[:, :, 1] = np.clip(result[:, :, 1] * (avgGray / avgG), 0, 255)
    result[:, :, 2] = np.clip(result[:, :, 2] * (avgGray / avgR), 0, 255)
    return result.astype(np.uint8)  # Convert back to uint8 for image display/saving


def clahe_equalization(img):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) for lighting correction.
    Args:
        img (numpy.ndarray): The input image (NumPy array, BGR format).
    Returns:
        numpy.ndarray: The corrected image (NumPy array, BGR format).
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Split channels
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L-channel
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)

    # Merge channels back
    lab_eq = cv2.merge((cl, a, b))

    # Convert back to BGR color space
    img_corrected = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    return img_corrected


# --- END IMAGE PROCESSING FUNCTIONS ---


# ===============================================
# 1. FLASK APPLICATION SETUP & CONFIGURATION
# ===============================================

app = Flask(__name__)
# Secret key from environment variable for production readiness
app.secret_key = os.environ.get("SECRET_KEY", "your_strong_dev_secret_key_12345")

# Define upload and report folders
UPLOAD_FOLDER = 'uploads'
REPORT_FOLDER = 'reports'

# Ensure these directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# Configure Flask app with folder paths
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['REPORT_FOLDER'] = REPORT_FOLDER

# --- Firestore (Simulated for Canvas) ---
app_id = os.environ.get('__app_id', 'default-app-id')
firebase_config_str = os.environ.get('__firebase_config', '{}')
firebase_config = json.loads(firebase_config_str)

db_data = {
    'artifacts': {
        app_id: {
            'users': {},
            'public': {'data': {}}
        }
    }
}
db = db_data

def setup_initial_firebase_globals():
    """
    Sets up conceptual global data for Firestore simulation if needed.
    This runs once at app startup.
    """
    print(f"DEBUG: App ID: {app_id}")
    print(f"DEBUG: Firebase Config (partial): {list(firebase_config.keys())[:3]}...")

setup_initial_firebase_globals()

# ===============================================
# ADDED: Route to serve uploaded files statically
# ===============================================
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files from the UPLOAD_FOLDER."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# ===============================================
# 2. DATABASE INITIALIZATION & HELPERS (Firestore)
# ===============================================

def get_firestore_collection(path_segments):
    """Navigates the simulated Firestore structure to get a collection."""
    current_level = db_data
    for segment in path_segments:
        if segment not in current_level:
            current_level[segment] = {}
        current_level = current_level[segment]
    return current_level


def get_firestore_document(path_segments):
    """Navigates the simulated Firestore structure to get a document."""
    collection = get_firestore_collection(path_segments[:-1])
    doc_id = path_segments[-1]
    return collection.get(doc_id)


def set_firestore_document(path_segments, data):
    """Sets a document in the simulated Firestore."""
    collection = get_firestore_collection(path_segments[:-1])
    doc_id = path_segments[-1]
    collection[doc_id] = data
    print(f"DEBUG: Simulated Firestore set: {os.path.join(*path_segments)}")


def add_firestore_document(path_segments, data):
    """Adds a document with auto-generated ID in the simulated Firestore."""
    collection = get_firestore_collection(path_segments)
    doc_id = str(np.random.randint(100000, 999999))  # Simulate auto-ID
    collection[doc_id] = data
    print(f"DEBUG: Simulated Firestore added: {os.path.join(*path_segments)}/{doc_id}")
    return doc_id  # Return the simulated ID


def get_firestore_documents_in_collection(path_segments, query_filters=None):
    """Gets documents from a simulated Firestore collection, with basic filtering."""
    collection = get_firestore_collection(path_segments)
    results = []
    for doc_id, doc_data in collection.items():
        if query_filters:
            match = True
            for field, value in query_filters.items():
                if doc_data.get(field) != value:
                    match = False
                    break
            if match:
                results.append(doc_data)
        else:
            results.append(doc_data)

    if results and 'timestamp' in results[0]:
        results.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    return results


# ===============================================
# 3. AUTHENTICATION HELPERS (Adapted for Firestore)
# ===============================================

@app.before_request
def load_logged_in_user():
    """Loads the logged-in user into Flask's g object for the current request.
    Uses session for persistence across requests.
    """
    if 'user_id' not in session:
        initial_auth_token = os.environ.get('__initial_auth_token')
        if initial_auth_token:
            session['user_id'] = initial_auth_token.split(':')[-1]
            session['user'] = {'id': session['user_id'], 'username': f"User_{session['user_id'][:8]}"}
            print(f"DEBUG: Initializing session user from token: {session['user']['username']}")
        else:
            session['user_id'] = 'anonymous-' + str(np.random.randint(100000, 999999))
            session['user'] = {'id': session['user_id'], 'username': f"AnonUser_{session['user_id'][-6:]}"}
            print(f"DEBUG: Initializing session user to anonymous: {session['user']['username']}")

    g.user_id = session.get('user_id')
    g.user = session.get('user')
    g.firestore_user_id = g.user_id


def login_required(view):
    """Decorator to protect routes that require a logged-in user (not anonymous)."""
    import functools

    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if g.user is None or 'anonymous' in g.user_id:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return view(**kwargs)

    return wrapped_view


# ===============================================
# 4. CORE HELPER FUNCTIONS (Image Correction, Shade Detection, PDF Generation, Enhanced Simulated AI)
# ===============================================

def map_l_to_shade_rule_based(l_value_100_scale):
    """
    Maps L-value (0-100 scale) to a VITA shade.
    Adjusted thresholds for a more granular mapping.
    """
    if l_value_100_scale > 85:
        return "B1+"
    elif l_value_100_scale > 80:
        return "B1"
    elif l_value_100_scale > 75:
        return "A1"
    elif l_value_100_scale > 70:
        return "B2"
    elif l_value_100_scale > 65:
        return "A2"
    elif l_value_100_scale > 60:
        return "D2"
    elif l_value_100_scale > 55:
        return "C1"
    elif l_value_100_scale > 50:
        return "A3"
    elif l_value_100_scale > 45:
        return "D3"
    elif l_value_100_scale > 40:
        return "B3"
    elif l_value_100_scale > 35:
        return "C2"
    elif l_value_100_scale > 30:
        return "A3.5"
    elif l_value_100_scale > 25:
        return "D4"
    elif l_value_100_scale > 20:
        return "C3"
    else:
        return "C4"

def match_shade_with_delta_e(target_lab_color):
    """
    Compares a target LabColor to predefined VITA shade LAB references
    and returns the closest VITA shade based on Delta E 2000.
    """
    min_delta_e = float('inf')
    best_shade = "N/A"
    for shade, ref_lab in VITA_SHADE_LAB_REFERENCES.items():
        delta_e = delta_e_cie2000(target_lab_color, ref_lab)
        if delta_e < min_delta_e:
            min_delta_e = delta_e
            best_shade = shade
    return best_shade, min_delta_e

# --- AI Model Setup (Loading Data from CSV & Training/Loading) ---
MODEL_FILENAME = "shade_classifier_model.pkl"
DATASET_FILENAME = "tooth_shades_simulated.csv"


def train_model():
    """Train a new KNN model using the CSV file and save it."""
    if not os.path.exists(DATASET_FILENAME):
        print(f"ERROR: Dataset '{DATASET_FILENAME}' is missing. Cannot train model.")
        return None

    try:
        df = pd.read_csv(DATASET_FILENAME)
        if df.empty:
            print(f"ERROR: Dataset '{DATASET_FILENAME}' is empty. Cannot train model.")
            return None

        X = df[['incisal_l', 'middle_l', 'cervical_l']].values
        y = df['overall_shade'].values
        print(f"DEBUG: Training data shape={X.shape}, classes={np.unique(y)}")

        model_to_train = KNeighborsClassifier(n_neighbors=3)
        model_to_train.fit(X, y)
        joblib.dump(model_to_train, MODEL_FILENAME)
        print(f"DEBUG: Model trained and saved to {MODEL_FILENAME}")
        return model_to_train
    except Exception as e:
        print(f"ERROR: Failed to train model: {e}")
        return None


def load_or_train_model():
    """Load existing model or train a new one."""
    if os.path.exists(MODEL_FILENAME):
        try:
            loaded_model = joblib.load(MODEL_FILENAME)
            print(f"DEBUG: Loaded pre-trained shade model from {MODEL_FILENAME}")
            return loaded_model
        except Exception as e:
            print(f"WARNING: Could not load model from {MODEL_FILENAME}: {e}. Attempting to retrain.")
            return train_model()
    else:
        print(f"DEBUG: No existing model found at {MODEL_FILENAME}. Attempting to train new model.")
        return train_model()


shade_classifier_model = load_or_train_model()

# =========================================================
# ENHANCED: Placeholder AI Modules for Advanced Analysis
# =========================================================

# New simulated AI-based device and light correction function
def perform_ai_correction_for_lab(lab_values_255_scale, brightness_estimate, phone_model="SimulatedPhoneX"):
    """
    Simulates AI-based device and light correction on LAB values.
    In a real scenario, this would be a trained ML model (XGBoost/LightGBM)
    that learns the typical LAB shifts from different devices/lighting
    to match true reference values.
    For this simulation, it applies a small, brightness-dependent adjustment.

    Args:
        lab_values_255_scale (np.ndarray): Array of [L, a, b] values (0-255 scale).
        brightness_estimate (float): A measure of image brightness (e.g., mean L-channel).
        phone_model (str): Simulated phone model (for conceptual input to AI).

    Returns:
        np.ndarray: "Corrected" LAB values, still on 0-255 scale, as if an AI model
                    had adjusted them for device/light variations.
    """
    corrected_lab = np.copy(lab_values_255_scale).astype(np.float32)

    # Simulate adjustment based on brightness and device (simple example)
    # A real model would learn complex, multi-dimensional shifts
    brightness_factor = (brightness_estimate / 255.0) - 0.5 # Normalize to -0.5 to 0.5 range
    
    # Simulate a slight L-channel compensation for very bright/dark images
    corrected_lab[0] -= brightness_factor * 20 # Adjust L by up to +/- 10 (on 255 scale)
    
    # Simulate minor 'a' and 'b' adjustments based on "phone model" or just random noise
    if "PhoneX" in phone_model:
        corrected_lab[1] += np.random.uniform(-2, 2) # small a-channel variation
        corrected_lab[2] += np.random.uniform(-1, 1) # small b-channel variation
    else:
        corrected_lab[1] += np.random.uniform(-1, 1)
        corrected_lab[2] += np.random.uniform(-0.5, 0.5)

    return np.clip(corrected_lab, 0, 255).astype(np.uint8)


def detect_face_features(image_np_array):
    """
    ENHANCED PLACEHOLDER: Simulates detailed face feature extraction.
    Now attempts to derive more nuanced skin tone (including undertones),
    detailed lip color, and eye contrast based on average color properties
    and simple statistical analysis of the input image.
    """
    print("DEBUG: Simulating detailed Face Detection and Feature Extraction with color analysis...")

    img_lab = cv2.cvtColor(image_np_array, cv2.COLOR_BGR2LAB)
    avg_l = np.mean(img_lab[:, :, 0])
    avg_a = np.mean(img_lab[:, :, 1])
    avg_b = np.mean(img_lab[:, :, 2])

    skin_tone_category = "Medium"
    skin_undertone = "Neutral"

    if avg_l > 75:
        skin_tone_category = "Light"
    elif avg_l > 60:
        skin_tone_category = "Medium"
    elif avg_l > 45:
        skin_tone_category = "Dark"
    else:
        skin_tone_category = "Very Dark"

    if avg_b > 15 and avg_a > 8:
        skin_undertone = "Warm (Golden/Peach)"
    elif avg_b < 0 and avg_a < 5:
        skin_undertone = "Cool (Pink/Blue)"
    elif avg_b >= 0 and avg_a >= 5 and avg_a <= 8 and avg_b <= 15:
        skin_undertone = "Neutral"
    elif avg_b > 5 and avg_a < 5:
        skin_undertone = "Olive (Greenish)"

    simulated_skin_tone = f"{skin_tone_category} with {skin_undertone} undertones"

    simulated_lip_color = "Natural Pink"
    if avg_a > 20 and avg_l < 60:
        simulated_lip_color = "Deep Rosy Red"
    elif avg_a > 15 and avg_l >= 60:
        simulated_lip_color = "Bright Coral"
    elif avg_b < 5 and avg_l < 50:
        simulated_lip_color = "Subtle Mauve/Berry"
    elif avg_l < 50 and avg_a < 10 and avg_b < 10:
        simulated_lip_color = "Muted Nude/Brown"
    elif avg_l > 70 and avg_a < 10:
        simulated_lip_color = "Pale Nude"

    l_channel = img_lab[:, :, 0]
    p10, p90 = np.percentile(l_channel, [10, 90])
    contrast_spread = p90 - p10
    eye_contrast_sim = "Medium"
    if contrast_spread > 40:
        eye_contrast_sim = "High (Distinct Features)"
    elif contrast_spread < 20:
        eye_contrast_sim = "Low (Soft Features)"

    return {
        "skin_tone": simulated_skin_tone,
        "lip_color": simulated_lip_color,
        "eye_contrast": eye_contrast_sim,
        "facial_harmony_score": round(np.random.uniform(0.7, 0.95), 2),
    }


def segment_and_analyze_teeth(image_np_array):
    """
    ENHANCED PLACEHOLDER: Simulates advanced tooth segmentation and shade analysis.
    Provides more detailed simulated insights on tooth condition and stain presence.
    """
    print("DEBUG: Simulating detailed Tooth Segmentation and Analysis...")

    img_lab = cv2.cvtColor(image_np_array, cv2.COLOR_BGR2LAB)

    avg_l = np.mean(img_lab[:, :, 0])
    avg_a = np.mean(img_lab[:, :, 1])
    avg_b = np.mean(img_lab[:, :, 2])

    if avg_l > 78:
        simulated_overall_shade = "B1 (High Brightness)"
    elif avg_l > 73:
        simulated_overall_shade = "A1 (Natural Brightness)"
    elif avg_l > 68:
        simulated_overall_shade = "A2 (Medium Brightness)"
    elif avg_l > 63:
        simulated_overall_shade = "B2 (Slightly Darker)"
    elif avg_l > 58:
        simulated_overall_shade = "C1 (Moderate Darkness)"
    elif avg_l > 53:
        simulated_overall_shade = "C2 (Noticeable Darkness)"
    elif avg_l > 48:
        simulated_overall_shade = "A3 (Darker, Reddish Tint)"
    else:
        simulated_overall_shade = "C3 (Very Dark)"

    tooth_condition_sim = "Normal & Healthy Appearance"
    if avg_b > 20 and avg_l < 70:
        tooth_condition_sim = "Mild Discoloration (Yellowish)"
    elif avg_b > 25 and avg_l < 60:
        tooth_condition_sim = "Moderate Discoloration (Strong Yellow)"
    elif avg_l < 55 and avg_a > 10:
        tooth_condition_sim = "Pronounced Discoloration (Brown/Red)"
    elif avg_l < 60 and avg_b < 0:
        tooth_condition_sim = "Greyish Appearance"

    l_std_dev = np.std(img_lab[:, :, 0])
    stain_presence_sim = "None detected"
    if l_std_dev > 25 and avg_l > 60:
        stain_presence_sim = "Possible light surface stains"
    elif l_std_dev > 35 and avg_l < 60:
        stain_presence_sim = "Moderate localized stains"

    decay_presence_sim = "No visible signs of decay"
    if np.random.rand() < 0.05:
        decay_presence_sim = "Potential small carious lesion (simulated - consult professional)"

    return {
        "overall_lab": {"L": float(avg_l), "a": float(avg_a), "b": float(avg_b)},
        "simulated_overall_shade": simulated_overall_shade,
        "tooth_condition": tooth_condition_sim,
        "stain_presence": stain_presence_sim,
        "decay_presence": decay_presence_sim,
    }


def aesthetic_shade_suggestion(facial_features, tooth_analysis):
    """
    ENHANCED PLACEHOLDER: Simulates an aesthetic mapping model with more context.
    Suggestions are now more specific, considering simulated skin/lip tones.
    Confidence is now more dynamic based on harmony score and conditions.
    """
    print("DEBUG: Simulating detailed Aesthetic Mapping and Shade Suggestion...")

    suggested_shade = "No specific aesthetic suggestion (Simulated)"
    aesthetic_confidence = "Low"
    recommendation_notes = "This is a simulated aesthetic suggestion. Consult a dental specialist for personalized cosmetic planning based on your unique facial features and desired outcome. Advanced AI for aesthetics is complex and evolving."

    current_shade = tooth_analysis.get('simulated_overall_shade', '')
    skin_tone = facial_features.get('skin_tone', '').lower()
    lip_color = facial_features.get('lip_color', '').lower()
    facial_harmony_score = facial_features.get('facial_harmony_score', 0.5)

    if "warm" in skin_tone:
        if "b1" in current_shade or "a1" in current_shade:
            suggested_shade = "Optimal Match (Simulated - Warm Undertone)"
            aesthetic_confidence = "Very High"
            recommendation_notes = "Your simulated warm skin undertone harmonizes exceptionally well with this bright shade, suggesting an optimal match. Consider maintaining this shade."
        elif "c3" in current_shade or "c2" in current_shade or "a3" in current_shade:
            suggested_shade = "B1 or A2 (Simulated - Brightening for Warm Undertone)"
            aesthetic_confidence = "High"
            recommendation_notes = "Your simulated warm skin undertone would be beautifully complemented by a brighter, slightly warmer tooth shade like B1 or A2. Consider professional whitening for a more radiant smile."
        else:
            aesthetic_confidence = "Medium"

    elif "cool" in skin_tone:
        if "a1" in current_shade or "b1" in current_shade:
            suggested_shade = "Optimal Match (Simulated - Cool Undertone)"
            aesthetic_confidence = "Very High"
            recommendation_notes = "This shade provides excellent contrast and harmony with your simulated cool skin undertone, suggesting an optimal match. A very crisp and bright appearance."
        elif "a3" in current_shade or "b2" in current_shade or "d" in current_shade:
            suggested_shade = "A1 or B1 (Simulated - Brightening for Cool Undertone)"
            aesthetic_confidence = "High"
            recommendation_notes = "With your simulated cool skin undertone, a crisp, bright shade like A1 or B1 could enhance your overall facial harmony. Avoid overly yellow shades for best results."
        else:
            aesthetic_confidence = "Medium"

    elif "neutral" in skin_tone:
        if "b1" in current_shade or "a1" in current_shade or "a2" in current_shade:
            suggested_shade = "Balanced Brightness (Simulated)"
            aesthetic_confidence = "High"
            recommendation_notes = "Your neutral skin tone offers great versatility. This shade provides a balanced and natural bright smile. Options for further brightening or warmth can be explored."
        else:
            aesthetic_confidence = "Medium"

    elif "olive" in skin_tone:
        if "a2" in current_shade or "b2" in current_shade:
            suggested_shade = "Enhanced Natural (Simulated - Olive Tone)"
            aesthetic_confidence = "Medium"
            recommendation_notes = "For a simulated olive skin tone, a balanced brightening to A2 can provide a natural yet enhanced smile. Be mindful of shades that pull too much yellow or grey."
        elif "a1" in current_shade or "b1" in current_shade:
            suggested_shade = "Significant Brightening (Simulated - Olive Tone)"
            aesthetic_confidence = "High"
            recommendation_notes = "While your current shade provides a natural look, shades like A1 or B1 could offer a more noticeable brightening effect while maintaining harmony with your olive tone."
        else:
            aesthetic_confidence = "Low"

    if facial_harmony_score >= 0.90:
        if aesthetic_confidence == "Low":
            aesthetic_confidence = "Medium"
        elif aesthetic_confidence == "Medium":
            aesthetic_confidence = "High"
    elif facial_harmony_score >= 0.80 and aesthetic_confidence == "Low":
        aesthetic_confidence = "Medium"

    if aesthetic_confidence == "Very High":
        pass
    elif aesthetic_confidence == "High":
        pass
    elif aesthetic_confidence == "Medium":
        if "Balanced Brightness" not in suggested_shade and "Enhanced Natural" not in suggested_shade:
            recommendation_notes = "This shade offers a natural and pleasing appearance. For more significant changes, a dental consultation is recommended."
    else:
        suggested_shade = "Consult Dental Specialist (Simulated)"
        recommendation_notes = "Based on the simulated analysis, a personalized consultation with a dental specialist is highly recommended for tailored cosmetic planning due to the complexity of aesthetic matching."


    return {
        "suggested_aesthetic_shade": suggested_shade,
        "aesthetic_confidence": aesthetic_confidence,
        "recommendation_notes": recommendation_notes
    }


def calculate_confidence(delta_e_value):
    """
    Calculates a simple confidence score based on the Delta E value.
    Lower Delta E means higher confidence.
    This is a simplified, illustrative confidence score.
    """
    # Max possible Delta E is around 100 for very different colors.
    # We want confidence to be high for low Delta E, dropping as Delta E increases.
    # Let's say dE < 1 is excellent, < 3 is good, < 5 is acceptable, > 5 is low confidence.
    if delta_e_value <= 1.0:
        return 98 # Excellent match, very high confidence
    elif delta_e_value <= 3.0:
        return 90 # Good match, high confidence
    elif delta_e_value <= 5.0:
        return 80 # Acceptable match, medium confidence
    elif delta_e_value <= 10.0:
        return 65 # Borderline, low confidence
    else:
        return 50 # Poor match, very low confidence


def detect_shades_from_image(image_path):
    """
    Performs lighting correction, white balance, extracts features,
    and then uses the pre-trained ML model for overall tooth shade detection.
    Also, provides rule-based shades for individual zones for UI consistency,
    and now includes Delta E 2000 matching for more precise shade suggestions.
    """
    print(f"DEBUG: Starting image processing for {image_path}")
    
    # Initialize variables to prevent NameError in case of early return or exceptions
    face_features = {}
    tooth_analysis = {}
    aesthetic_suggestion = {}

    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"ERROR: cv2.imread returned None for image at {image_path}. File might be missing, corrupted, or not an image.")
            return {
                "incisal": "N/A", "middle": "N/A", "cervical": "N/A",
                "overall_ml_shade": "N/A",
                "face_features": face_features, "tooth_analysis": tooth_analysis, "aesthetic_suggestion": aesthetic_suggestion,
                "delta_e_matched_shades": {}, "accuracy_confidence": {"overall": "N/A", "notes": ""}
            }
        print(f"DEBUG: Image loaded successfully. Shape: {img.shape}, Type: {img.dtype}")

        if img.size == 0:
            print(f"ERROR: Image loaded but is empty (0 size) for {image_path}.")
            return {
                "incisal": "N/A", "middle": "N/A", "cervical": "N/A",
                "overall_ml_shade": "N/A",
                "face_features": face_features, "tooth_analysis": tooth_analysis, "aesthetic_suggestion": aesthetic_suggestion,
                "delta_e_matched_shades": {}, "accuracy_confidence": {"overall": "N/A", "notes": ""}
            }

        # --- Apply Image Pre-processing (Gray World + CLAHE) ---
        img_wb = gray_world_white_balance(img)
        print("DEBUG: Gray world white balance applied.")
        img_corrected = clahe_equalization(img_wb) # Apply CLAHE after white balance
        print("DEBUG: Lighting correction applied (CLAHE).")

        # --- Call Enhanced Placeholder AI modules (now wrapped in try-except for robustness) ---
        try:
            face_features = detect_face_features(img_corrected)
            tooth_analysis = segment_and_analyze_teeth(img_corrected)
            aesthetic_suggestion = aesthetic_shade_suggestion(face_features, tooth_analysis)
            print("DEBUG: Simulated AI modules executed.")
        except Exception as ai_module_error:
            print(f"WARNING: An error occurred during simulated AI module execution: {ai_module_error}")
            traceback.print_exc() # Print full traceback for the AI module error
            # face_features, tooth_analysis, aesthetic_suggestion remain as empty dicts initialized above


        # --- Conceptual Tooth Region Detection (Step 3) ---
        # For this simulation, we assume the central part of the image contains the tooth.
        # In a real app, this would be precise segmentation.
        height, width, _ = img_corrected.shape
        print(f"DEBUG: Corrected image dimensions: Height={height}, Width={width}")

        min_height_for_slicing = 30
        if height < min_height_for_slicing:
            print(f"ERROR: Image height ({height} pixels) is too small for zonal slicing. Minimum required: {min_height_for_slicing} pixels. Cannot perform detailed shade detection.")
            return {
                "incisal": "Error", "middle": "Error", "cervical": "Error",
                "overall_ml_shade": "Error - Image Too Small",
                "face_features": face_features, "tooth_analysis": tooth_analysis, "aesthetic_suggestion": aesthetic_suggestion,
                "delta_e_matched_shades": {}, "accuracy_confidence": {"overall": "N/A", "notes": "Image too small for detailed analysis."}
            }

        # Crop central 60% horizontally (simulated tooth region)
        # This helps exclude lips/gums more effectively than full width slices
        crop_start_x = int(width * 0.20)
        crop_end_x = int(width * 0.80)
        cropped_tooth_region = img_corrected[:, crop_start_x:crop_end_x, :]

        if cropped_tooth_region.size == 0:
            print(f"ERROR: Cropped tooth region is empty. Check image dimensions and cropping logic.")
            return {
                "incisal": "Error", "middle": "Error", "cervical": "Error",
                "overall_ml_shade": "Error - Cropped Zone Empty",
                "face_features": face_features, "tooth_analysis": tooth_analysis, "aesthetic_suggestion": aesthetic_suggestion,
                "delta_e_matched_shades": {}, "accuracy_confidence": {"overall": "N/A", "notes": "Cropped tooth region is empty."}
            }

        # Redefine zones from the *cropped* tooth region
        # This simulates focusing only on the tooth
        cropped_height, cropped_width, _ = cropped_tooth_region.shape

        incisal_zone = cropped_tooth_region[0:int(cropped_height*0.3), :, :]
        middle_zone = cropped_tooth_region[int(cropped_height*0.3):int(cropped_height*0.7), :, :]
        cervical_zone = cropped_tooth_region[int(cropped_height*0.7):cropped_height, :, :]
        print("DEBUG: Tooth region cropped and zones sliced.")

        if incisal_zone.size == 0 or middle_zone.size == 0 or cervical_zone.size == 0:
            print(f"WARNING: One or more sliced image zones (after cropping) are empty. Incisal size: {incisal_zone.size}, Middle size: {middle_zone.size}, Cervical size: {cervical_zone.size}")
            # Attempt to use overall image if zones are problematic
            avg_incisal_lab_cv = np.mean(cv2.cvtColor(cropped_tooth_region, cv2.COLOR_BGR2LAB).reshape(-1, 3), axis=0)
            avg_middle_lab_cv = avg_incisal_lab_cv
            avg_cervical_lab_cv = avg_incisal_lab_cv
            print("DEBUG: Fallback: Using overall cropped region for all zones due to empty slices.")
        else:
            # Convert zones to LAB for full L, a, b values (0-255 scale)
            incisal_lab_full = cv2.cvtColor(incisal_zone, cv2.COLOR_BGR2LAB)
            middle_lab_full = cv2.cvtColor(middle_zone, cv2.COLOR_BGR2LAB)
            cervical_lab_full = cv2.cvtColor(cervical_zone, cv2.COLOR_BGR2LAB)
            print("DEBUG: Zones converted to LAB color space (OpenCV 0-255 scale).")

            # Calculate average L, a, b values for each zone (0-255 scale)
            avg_incisal_lab_cv = np.mean(incisal_lab_full.reshape(-1, 3), axis=0)
            avg_middle_lab_cv = np.mean(middle_lab_full.reshape(-1, 3), axis=0)
            avg_cervical_lab_cv = np.mean(cervical_lab_full.reshape(-1, 3), axis=0)
            print(f"DEBUG: Average LAB values (OpenCV 0-255 scale): Incisal={avg_incisal_lab_cv}, Middle={avg_middle_lab_cv}, Cervical={avg_cervical_lab_cv}")

        # --- AI-Based Device & Light Correction (Simulated - Step 5) ---
        # Get conceptual brightness and phone model (can be replaced by real EXIF/JS data)
        brightness_estimate = np.mean(cv2.cvtColor(img_corrected, cv2.COLOR_BGR2GRAY))
        phone_model_simulated = "Android_Pixel_7" # This would come from EXIF or JS in a real app

        avg_incisal_lab_corrected = perform_ai_correction_for_lab(avg_incisal_lab_cv, brightness_estimate, phone_model_simulated)
        avg_middle_lab_corrected = perform_ai_correction_for_lab(avg_middle_lab_cv, brightness_estimate, phone_model_simulated)
        avg_cervical_lab_corrected = perform_ai_correction_for_lab(avg_cervical_lab_cv, brightness_estimate, phone_model_simulated)
        overall_avg_lab_cv_initial = np.mean([avg_incisal_lab_cv, avg_middle_lab_cv, avg_cervical_lab_cv], axis=0)
        overall_avg_lab_corrected = perform_ai_correction_for_lab(overall_avg_lab_cv_initial, brightness_estimate, phone_model_simulated)
        print("DEBUG: Simulated AI-based device & light correction applied to LAB values.")

        # Normalize L values to 0-100 scale for ML prediction and rule-based mapping (using the corrected values)
        avg_incisal_l_100 = avg_incisal_lab_corrected[0] / 2.55
        avg_middle_l_100 = avg_middle_lab_corrected[0] / 2.55
        avg_cervical_l_100 = avg_cervical_lab_corrected[0] / 2.55
        print(f"DEBUG: Average L values (0-100 scale, after AI correction): Incisal={avg_incisal_l_100:.2f}, Middle={avg_middle_l_100:.2f}, Cervical={avg_cervical_l_100:.2f}")

        # --- Delta E Matching (Step 6) ---
        # Convert corrected OpenCV LAB (0-255 for L, a, b) to Colormath LabColor (0-100 for L, -128 to 127 for a,b)
        incisal_lab_colormath = LabColor(
            lab_l=np.clip(avg_incisal_lab_corrected[0] / 2.55, 0, 100),
            lab_a=np.clip(avg_incisal_lab_corrected[1] - 128, -128, 127),
            lab_b=np.clip(avg_incisal_lab_corrected[2] - 128, -128, 127)
        )
        middle_lab_colormath = LabColor(
            lab_l=np.clip(avg_middle_lab_corrected[0] / 2.55, 0, 100),
            lab_a=np.clip(avg_middle_lab_corrected[1] - 128, -128, 127),
            lab_b=np.clip(avg_middle_lab_corrected[2] - 128, -128, 127)
        )
        cervical_lab_colormath = LabColor(
            lab_l=np.clip(avg_cervical_lab_corrected[0] / 2.55, 0, 100),
            lab_a=np.clip(avg_cervical_lab_corrected[1] - 128, -128, 127),
            lab_b=np.clip(avg_cervical_lab_corrected[2] - 128, -128, 127)
        )
        
        overall_lab_colormath = LabColor(
            lab_l=np.clip(overall_avg_lab_corrected[0] / 2.55, 0, 100),
            lab_a=np.clip(overall_avg_lab_corrected[1] - 128, -128, 127),
            lab_b=np.clip(overall_avg_lab_corrected[2] - 128, -128, 127)
        )
        print(f"DEBUG: Colormath LAB values (after AI correction): Incisal={incisal_lab_colormath}, Middle={middle_lab_colormath}, Cervical={cervical_lab_colormath}, Overall={overall_lab_colormath}")

        # Perform Delta E matching for each zone and overall
        incisal_delta_e_shade, incisal_min_delta = match_shade_with_delta_e(incisal_lab_colormath)
        middle_delta_e_shade, middle_min_delta = match_shade_with_delta_e(middle_lab_colormath)
        cervical_delta_e_shade, cervical_min_delta = match_shade_with_delta_e(cervical_lab_colormath)
        overall_delta_e_shade, overall_min_delta = match_shade_with_delta_e(overall_lab_colormath)
        print(f"DEBUG: Delta E matched shades: Overall={overall_delta_e_shade}, Incisal={incisal_delta_e_shade}, Middle={middle_delta_e_shade}, Cervical={cervical_delta_e_shade}")

        # --- ML Prediction (using 0-100 L values from AI-corrected data) ---
        overall_ml_shade = "Model Error"
        if shade_classifier_model is not None:
            features_for_ml_prediction = np.array([[avg_incisal_l_100, avg_middle_l_100, avg_cervical_l_100]])
            overall_ml_shade = shade_classifier_model.predict(features_for_ml_prediction)[0]
            print(f"DEBUG: Features for ML: {features_for_ml_prediction}")
            print(f"DEBUG: Predicted Overall Shade (ML): {overall_ml_shade}")
        else:
            print("WARNING: AI model not loaded/trained. Cannot provide ML shade prediction.")

        # --- Calculate Confidence (Step 7) ---
        overall_accuracy_confidence = calculate_confidence(overall_min_delta)
        confidence_notes = f"Confidence based on Delta E 2000 value of {overall_min_delta:.2f}. Lower dE means higher confidence in the color match."


        detected_shades = {
            "incisal": map_l_to_shade_rule_based(avg_incisal_l_100),
            "middle": map_l_to_shade_rule_based(avg_middle_l_100),
            "cervical": map_l_to_shade_rule_based(avg_cervical_l_100),
            "overall_ml_shade": overall_ml_shade,

            "delta_e_matched_shades": {
                "overall": overall_delta_e_shade,
                "overall_delta_e": round(float(overall_min_delta), 2),
                "incisal": incisal_delta_e_shade,
                "incisal_delta_e": round(float(incisal_min_delta), 2),
                "middle": middle_delta_e_shade,
                "middle_delta_e": round(float(middle_min_delta), 2),
                "cervical": cervical_delta_e_shade,
                "cervical_delta_e": round(float(cervical_min_delta), 2),
            },
            "face_features": face_features, # Now safe
            "tooth_analysis": tooth_analysis, # Now safe
            "aesthetic_suggestion": aesthetic_suggestion, # Now safe
            "accuracy_confidence": { # New key for overall accuracy confidence
                "overall_percentage": overall_accuracy_confidence,
                "notes": confidence_notes
            }
        }
        return detected_shades

    except Exception as e:
        print(f"CRITICAL ERROR during shade detection: {e}")
        traceback.print_exc()
        return {
            "incisal": "Error", "middle": "Error", "cervical": "Error",
            "overall_ml_shade": "Error",
            "face_features": {}, # Set to empty dicts in the error case as well
            "tooth_analysis": {},
            "aesthetic_suggestion": {},
            "delta_e_matched_shades": {},
            "accuracy_confidence": {"overall": "Error", "notes": f"Processing failed: {e}"}
        }


def generate_pdf_report(patient_name, shades, image_path, filepath):
    """Generates a PDF report with detected shades and the uploaded image."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)

    pdf.cell(200, 10, txt="Shade View - Tooth Shade Analysis Report", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, txt=f"Patient Name: {patient_name}", ln=True)
    pdf.cell(0, 10, txt=f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', size=14)
    pdf.cell(0, 10, txt="Detected Shades (L-value Rule-based / ML):", ln=True)
    pdf.set_font("Arial", size=12)
    if "overall_ml_shade" in shades and shades["overall_ml_shade"] != "N/A":
        pdf.cell(0, 7, txt=f"  - Overall AI Prediction (ML): {shades['overall_ml_shade']}", ln=True)

    pdf.cell(0, 7, txt=f"  - Incisal Zone (Rule-based): {shades['incisal']}", ln=True)
    pdf.cell(0, 7, txt=f"  - Middle Zone (Rule-based): {shades['middle']}", ln=True)
    pdf.cell(0, 7, txt=f"  - Cervical Zone (Rule-based): {shades['cervical']}", ln=True)
    
    pdf.ln(8)
    pdf.set_font("Arial", 'B', size=14)
    # Changed 'ΔE' to 'dE' to avoid UnicodeEncodeError in FPDF
    pdf.cell(0, 10, txt="Delta E 2000 Matched Shades (Perceptual Match):", ln=True)
    pdf.set_font("Arial", size=12)
    delta_e_shades = shades.get("delta_e_matched_shades", {})
    if delta_e_shades:
        pdf.cell(0, 7, txt=f"  - Overall Delta E Match: {delta_e_shades.get('overall', 'N/A')} (dE: {delta_e_shades.get('overall_delta_e', 'N/A'):.2f})", ln=True)
        pdf.cell(0, 7, txt=f"  - Incisal Zone Delta E Match: {delta_e_shades.get('incisal', 'N/A')} (dE: {delta_e_shades.get('incisal_delta_e', 'N/A'):.2f})", ln=True)
        pdf.cell(0, 7, txt=f"  - Middle Zone Delta E Match: {delta_e_shades.get('middle', 'N/A')} (dE: {delta_e_shades.get('middle_delta_e', 'N/A'):.2f})", ln=True)
        pdf.cell(0, 7, txt=f"  - Cervical Zone Delta E Match: {delta_e_shades.get('cervical', 'N/A')} (dE: {delta_e_shades.get('cervical_delta_e', 'N/A'):.2f})", ln=True)
    else:
        pdf.cell(0, 7, txt="  - Delta E matching data not available.", ln=True)


    # New: Accuracy Confidence Score
    pdf.ln(8)
    pdf.set_font("Arial", 'B', size=14)
    pdf.cell(0, 10, txt="Shade Detection Accuracy Confidence:", ln=True)
    pdf.set_font("Arial", size=12)
    accuracy_conf = shades.get("accuracy_confidence", {})
    if accuracy_conf and accuracy_conf.get("overall_percentage") != "N/A":
        pdf.cell(0, 7, txt=f"  - Overall Confidence: {accuracy_conf.get('overall_percentage', 'N/A')}%", ln=True)
        pdf.multi_cell(0, 7, txt=f"  - Notes: {accuracy_conf.get('notes', 'N/A')}")
    else:
        pdf.cell(0, 7, txt="  - Confidence data not available or processing error.", ln=True)


    pdf.ln(8)
    pdf.set_font("Arial", 'B', size=13)
    pdf.cell(0, 10, txt="Advanced AI Insights (Simulated):", ln=True)
    pdf.set_font("Arial", size=11)

    tooth_analysis = shades.get("tooth_analysis", {})
    if tooth_analysis:
        pdf.cell(0, 7, txt="  -- Tooth Analysis --", ln=True)
        pdf.cell(0, 7, txt=f"  - Simulated Overall Shade (Detailed): {tooth_analysis.get('simulated_overall_shade', 'N/A')}",
                 ln=True)
        pdf.cell(0, 7, txt=f"  - Simulated Condition: {tooth_analysis.get('tooth_condition', 'N/A')}", ln=True)
        pdf.cell(0, 7, txt=f"  - Simulated Stain Presence: {tooth_analysis.get('stain_presence', 'N/A')}", ln=True)
        pdf.cell(0, 7, txt=f"  - Simulated Decay Presence: {tooth_analysis.get('decay_presence', 'N/A')}", ln=True)
        l_val = tooth_analysis.get('overall_lab', {}).get('L', 'N/A')
        a_val = tooth_analysis.get('overall_lab', {}).get('a', 'N/A')
        b_val = tooth_analysis.get('overall_lab', {}).get('b', 'N/A')
        if all(isinstance(v, (int, float)) for v in [l_val, a_val, b_val]):
            pdf.cell(0, 7, txt=f"  - Simulated Overall LAB: L={l_val:.2f}, a={a_val:.2f}, b={b_val:.2f}", ln=True)
        else:
            pdf.cell(0, 7, txt=f"  - Simulated Overall LAB: L={l_val}, a={a_val}, b={b_val}", ln=True)

    pdf.ln(3)
    face_features = shades.get("face_features", {})
    if face_features:
        pdf.cell(0, 7, txt="  -- Facial Aesthetics Analysis --", ln=True)
        pdf.cell(0, 7, txt=f"  - Simulated Skin Tone: {face_features.get('skin_tone', 'N/A')}", ln=True)
        pdf.cell(0, 7, txt=f"  - Simulated Lip Color: {face_features.get('lip_color', 'N/A')}", ln=True)
        pdf.cell(0, 7, txt=f"  - Simulated Eye Contrast: {face_features.get('eye_contrast', 'N/A')}", ln=True)
        harmony_score = face_features.get('facial_harmony_score', 'N/A')
        if isinstance(harmony_score, (int, float)):
            pdf.cell(0, 7, txt=f"  - Simulated Facial Harmony Score: {harmony_score:.2f}", ln=True)
        else:
            pdf.cell(0, 7, txt=f"  - Simulated Facial Harmony Score: {harmony_score}", ln=True)

    pdf.ln(3)
    aesthetic_suggestion = shades.get("aesthetic_suggestion", {})
    if aesthetic_suggestion:
        pdf.cell(0, 7, txt="  -- Aesthetic Shade Suggestion --", ln=True)
        pdf.cell(0, 7, txt=f"  - Suggested Shade: {aesthetic_suggestion.get('suggested_aesthetic_shade', 'N/A')}",
                 ln=True)
        pdf.cell(0, 7, txt=f"  - Confidence: {aesthetic_suggestion.get('aesthetic_confidence', 'N/A')}", ln=True)
        pdf.multi_cell(0, 7, txt=f"  - Notes: {aesthetic_suggestion.get('recommendation_notes', 'N/A')}")

    pdf.ln(10)

    try:
        if os.path.exists(image_path):
            pdf.cell(0, 10, txt="Uploaded Image:", ln=True)
            if pdf.get_y() > 200:
                pdf.add_page()
            img_cv = cv2.imread(image_path)
            if img_cv is not None:
                h_img, w_img, _ = img_cv.shape
                max_w_pdf = 180
                w_pdf = min(w_img, max_w_pdf)
                h_pdf = h_img * (w_pdf / w_img)

                if pdf.get_y() + h_pdf + 10 > pdf.h - pdf.b_margin:
                    pdf.add_page()

                img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                temp_image_path = "temp_pdf_image.png"
                cv2.imwrite(temp_image_path, img_rgb)
                
                pdf.image(temp_image_path, x=pdf.get_x(), y=pdf.get_y(), w=w_pdf, h=h_pdf)
                pdf.ln(h_pdf + 10)
                os.remove(temp_image_path)
            else:
                pdf.cell(0, 10, txt="Note: Image could not be loaded for embedding.", ln=True)

        else:
            pdf.cell(0, 10, txt="Note: Uploaded image file not found for embedding.", ln=True)
    except Exception as e:
        print(f"Error adding image to PDF: {e}")
        pdf.cell(0, 10, txt="Note: An error occurred while embedding the image in the report.", ln=True)

    pdf.set_font("Arial", 'I', size=9)
    pdf.multi_cell(0, 6,
                   txt="DISCLAIMER: This report is based on simulated AI analysis for demonstration purposes only. It is not intended for clinical diagnosis, medical advice, or professional cosmetic planning. Always consult with a qualified dental or medical professional for definitive assessment, diagnosis, and treatment.",
                   align='C')
    pdf.output(filepath)


# ===============================================
# 5. ROUTES (Adapted for Firestore)
# ===============================================
@app.route('/')
def home():
    """Renders the home/landing page."""
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handles user login (Simulated for Canvas)."""
    if g.user and 'anonymous' not in g.user['id']:
        flash(f"You are already logged in as {g.user['username']}.", 'info')
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        error = None

        if not username or not password:
            error = 'Username and password are required.'

        if error is None:
            simulated_user_id = 'user_' + username.lower().replace(' ', '_')
            session['user_id'] = simulated_user_id
            session['user'] = {'id': simulated_user_id, 'username': username}
            flash(f'Simulated login successful for {username}!', 'success')
            print(f"DEBUG: Simulated login for user: {username} (ID: {session['user_id']})")
            return redirect(url_for('dashboard'))
        flash(error, 'danger')

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handles user registration (Simulated for Canvas)."""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        error = None

        if not username:
            error = 'Username is required.'
        elif not password:
            error = 'Password is required.'

        if error is None:
            flash(f"Simulated registration successful for {username}. You can now log in!", 'success')
            return redirect(url_for('login'))
        flash(error, 'danger')

    return render_template('register.html')

@app.route('/logout')
def logout():
    """Handles user logout."""
    session.clear()
    flash('You have been logged out.', 'info')
    print(f"DEBUG: User logged out. Session cleared.")
    return redirect(url_for('home'))

@app.route('/dashboard')
@login_required
def dashboard():
    """Renders the user dashboard, displaying past reports."""
    reports_path = ['artifacts', app_id, 'users', g.firestore_user_id, 'reports']
    user_reports = get_firestore_documents_in_collection(reports_path)
    user_reports.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

    current_date_formatted = datetime.now().strftime('%Y-%m-%d')

    return render_template('dashboard.html',
                           reports=user_reports,
                           user=g.user,
                           current_date=current_date_formatted)


@app.route('/save_patient_data', methods=['POST'])
@login_required
def save_patient_data():
    """Handles saving new patient records to Firestore and redirects to image upload page."""
    op_number = request.form['op_number']
    patient_name = request.form['patient_name']
    age = request.form['age']
    sex = request.form['sex']
    record_date = request.form['date']
    user_id = g.user['id']

    patients_collection_path = ['artifacts', app_id, 'users', user_id, 'patients']

    existing_patients = get_firestore_documents_in_collection(patients_collection_path, query_filters={'op_number': op_number, 'user_id': user_id})
    if existing_patients:
        flash('OP Number already exists for another patient under your account. Please use a unique OP Number or select from recent entries.', 'error')
        return redirect(url_for('dashboard'))

    try:
        patient_data = {
            'user_id': user_id,
            'op_number': op_number,
            'patient_name': patient_name,
            'age': int(age),
            'sex': sex,
            'record_date': record_date,
            'created_at': datetime.now().isoformat()
        }

        add_firestore_document(patients_collection_path, patient_data)

        flash('Patient record saved successfully (to Firestore)! Now upload an image.', 'success')
        return redirect(url_for('upload_page', op_number=op_number))
    except Exception as e:
        flash(f'Error saving patient record to Firestore: {e}', 'error')
        return redirect(url_for('dashboard'))


@app.route('/upload_page/<op_number>')
@login_required
def upload_page(op_number):
    """Renders the dedicated image upload page for a specific patient."""
    user_id = g.user['id']

    patients_collection_path = ['artifacts', app_id, 'users', user_id, 'patients']
    patient = None
    all_patients = get_firestore_documents_in_collection(patients_collection_path, query_filters={'op_number': op_number, 'user_id': user_id})
    if all_patients:
        patient = all_patients[0]

    if patient is None:
        flash('Patient not found or unauthorized access.', 'error')
        return redirect(url_for('dashboard'))

    return render_template('upload_page.html', op_number=op_number, patient_name=patient['patient_name'])


@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_file():
    """Handles image upload, shade detection, and PDF report generation."""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        file = request.files['file']
        op_number_from_form = request.form.get('op_number')
        patient_name = request.form.get('patient_name', 'Unnamed Patient')

        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            file_ext = os.path.splitext(filename)[1]
            unique_filename = str(uuid.uuid4()) + file_ext
            
            original_image_path = os.path.join(UPLOAD_FOLDER, unique_filename)
            file.save(original_image_path)
            flash('Image uploaded successfully!', 'success')

            detected_shades = detect_shades_from_image(original_image_path)

            if (detected_shades.get("overall_ml_shade") == "Error" and
                detected_shades.get("delta_e_matched_shades", {}).get("overall") == "N/A"):
                flash("Error processing image for shade detection. Please try another image.", 'danger')
                if os.path.exists(original_image_path):
                    os.remove(original_image_path)
                return redirect(url_for('upload_page', op_number=op_number_from_form))

            report_filename = f"report_{patient_name.replace(' ', '')}{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            report_filepath = os.path.join(REPORT_FOLDER, report_filename)
            generate_pdf_report(patient_name, detected_shades, original_image_path, report_filepath)
            flash('PDF report generated!', 'success')

            formatted_analysis_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            report_data = {
                'patient_name': patient_name,
                'op_number': op_number_from_form,
                'original_image': unique_filename,
                'report_filename': report_filename,
                'detected_shades': detected_shades,
                'timestamp': datetime.now().isoformat(),
                'user_id': g.firestore_user_id
            }
            reports_collection_path = ['artifacts', app_id, 'users', g.firestore_user_id, 'reports']
            add_firestore_document(reports_collection_path, report_data)

            return render_template('report.html',
                                   patient_name=patient_name,
                                   shades=detected_shades,
                                   image_filename=unique_filename,
                                   report_filename=report_filename,
                                   analysis_date=formatted_analysis_date)

    flash("Please select a patient from the dashboard to upload an image.", 'info')
    return redirect(url_for('dashboard'))


@app.route('/download_report/<filename>')
@login_required
def download_report(filename):
    """Allows downloading of generated PDF reports."""
    return send_from_directory(REPORT_FOLDER, filename, as_attachment=True)

# Main entry point
if __name__ == '__main__':
    if shade_classifier_model is None:
        print("CRITICAL: Machine Learning model could not be loaded or trained. Shade prediction will not work.")
    app.run(debug=True)
