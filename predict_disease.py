# predict_disease.py

import joblib
import os
import re

# Load model and encoder
model = joblib.load("model/rf_disease_model.pkl")
encoder = joblib.load("model/label_encoder.pkl")
features_order = model.feature_names_in_

# Clean and normalize input text
def clean_text(text):
    return re.sub(r'[^a-zA-Z\s]', '', text.lower())

def extract_symptoms_from_text(text):
    """
    Extract known symptoms from free-text input using simple keyword matching.
    """
    cleaned_text = clean_text(text)
    matched = [symptom for symptom in features_order if symptom.replace('_', ' ') in cleaned_text]
    return matched

def predict_disease_from_text(input_text):
    """
    Predict disease from free text input describing symptoms.
    """
    extracted = extract_symptoms_from_text(input_text)

    # Convert to binary feature vector
    input_vector = [1 if feature in extracted else 0 for feature in features_order]

    # Predict
    pred_encoded = model.predict([input_vector])[0]
    pred_label = encoder.inverse_transform([pred_encoded])[0]

    return {
        'extracted_symptoms': extracted,
        'predicted_disease': pred_label
    }
