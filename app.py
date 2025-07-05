# app.py

import streamlit as st
import joblib
import os
import re
import pandas as pd
from fuzzywuzzy import fuzz, process

# === Load model and data ===
model = joblib.load(os.path.join("model", "rf_disease_model.pkl"))
encoder = joblib.load(os.path.join("model", "label_encoder.pkl"))
features_order = joblib.load(os.path.join("model", "feature_names.pkl"))
treatment_df = pd.read_csv(os.path.join("data", "ayurvedic treatment.csv"))

# === Utility Functions ===

def clean_text(text):
    return re.sub(r'[^a-zA-Z\s]', ' ', text.lower())

def extract_symptoms_from_text(text, threshold=80):
    cleaned_text = clean_text(text)
    tokens = cleaned_text.split()
    matched_symptoms = set()
    for token in tokens:
        match, score = process.extractOne(token, features_order, scorer=fuzz.partial_ratio)
        if score >= threshold:
            matched_symptoms.add(match)
    return list(matched_symptoms)

def predict_disease_from_symptoms(symptoms):
    input_vector = [1 if feature in symptoms else 0 for feature in features_order]
    prediction = model.predict([input_vector])[0]
    predicted_label = encoder.inverse_transform([prediction])[0]
    return predicted_label

def get_treatments_by_fuzzy_match(input_text, threshold=80):
    input_text = input_text.lower()
    possible_matches = treatment_df['Disease'].dropna().unique()
    matches = []
    for disease in possible_matches:
        score = fuzz.partial_ratio(input_text, str(disease).lower())
        if score >= threshold:
            matches.append((disease, score))
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches

def get_treatment(disease_name):
    row = treatment_df[treatment_df['Disease'].str.lower() == disease_name.lower()]
    if not row.empty:
        return row.iloc[0]['Treatment']
    return None

# === Streamlit UI ===

st.set_page_config(page_title="Ayurvedic Disease Prediction", page_icon="🌿")
st.title("🌿 Ayurvedic Disease Prediction App")
st.markdown("Enter your symptoms (e.g. *fever, nausea, body pain*) in plain language.")

user_input = st.text_area("Symptoms:", height=150)

if st.button("Predict Disease"):
    if not user_input.strip():
        st.warning("⚠️ Please enter your symptoms.")
    else:
        extracted_symptoms = extract_symptoms_from_text(user_input)

        if not extracted_symptoms:
            st.error("❌ No symptoms matched. Try using simpler or more descriptive terms.")
        else:
            st.checkbox("✅ Extracted Symptoms:", value=True, disabled=True)
            st.write(", ".join(extracted_symptoms))

            # === Only One Symptom Entered ===
            if len(extracted_symptoms) == 1:
                st.warning("⚠️ Only one symptom detected. This is not enough for accurate disease prediction.")

                matches = get_treatments_by_fuzzy_match(extracted_symptoms[0])
                if matches:
                    st.success("🌿 Showing treatments for symptoms like: " + extracted_symptoms[0])
                    for disease, _ in matches:
                        treatment = get_treatment(disease)
                        if treatment:
                            st.markdown(f"**Disease/Symptom:** {disease}")
                            st.markdown(f"🌱 **Treatment:** {treatment}")
                            break
                else:
                    st.error("❌ No Ayurvedic treatment found for this symptom.")
            else:
                # === Predict disease from multiple symptoms ===
                predicted_disease = predict_disease_from_symptoms(extracted_symptoms)
                st.success(f"🧠 Predicted Disease: **{predicted_disease}**")

                matches = get_treatments_by_fuzzy_match(predicted_disease)
                if matches:
                    st.markdown("🌿 **Matching Ayurvedic Treatments:**")
                    for disease, _ in matches:
                        treatment = get_treatment(disease)
                        if treatment:
                            st.markdown(f"**Disease/Symptom:** {disease}")
                            st.markdown(f"🌱 **Treatment:** {treatment}")
                            break
                else:
                    # Fallback: try matching based on each symptom
                    fallback_treatments = []
                    for sym in extracted_symptoms:
                        symptom_matches = get_treatments_by_fuzzy_match(sym)
                        if symptom_matches:
                            disease, _ = symptom_matches[0]
                            treatment = get_treatment(disease)
                            if treatment:
                                fallback_treatments.append((disease, treatment))

                    if fallback_treatments:
                        st.markdown("🌿 **Fallback Treatment Suggestions:**")
                        for disease, treatment in fallback_treatments:
                            st.markdown(f"**Symptom:** {disease}")
                            st.markdown(f"🌱 **Treatment:** {treatment}")
                    else:
                        st.warning("⚠️ No Ayurvedic treatment found.")


# At the end of st.button("Predict Disease") logic
# ...

# === Ayurvedic Disclaimer ===
st.markdown("---")
st.info("🧘 **Note:** These are Ayurvedic suggestions. They generally have no side effects, but this is not a 100% accurate diagnosis. Please consult a certified medical professional for confirmation.")
