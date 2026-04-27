import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("model/model.pkl")

# UI
st.title("🧠 AI Health Assistant")
st.caption("Enter your symptoms and get AI-based health insights")

# Input
user_input = st.text_area("Describe your symptoms", height=100)

# Button
if st.button("Analyze"):

    text = user_input.lower()

    # Feature extraction (UPDATED)

    fever_level = 2 if "high fever" in text else 1 if "fever" in text else 0
    cough_type = 2 if "wet cough" in text else 1 if "cough" in text else 0
    headache_level = 2 if "severe headache" in text else 1 if "headache" in text else 0
    fatigue_level = 2 if "very tired" in text else 1 if "tired" in text else 0

    body_pain = 1 if "body pain" in text else 0
    sore_throat = 1 if "sore throat" in text else 0
    runny_nose = 1 if "runny nose" in text else 0

    # Duration
    if "week" in text:
        duration_level = 2
    elif "day" in text:
        duration_level = 1
    else:
        duration_level = 0

    # Create DataFrame
    input_data = pd.DataFrame([[
        fever_level, cough_type, headache_level, fatigue_level,
        body_pain, sore_throat, runny_nose, duration_level
    ]], columns=[
        "fever_level","cough_type","headache_level","fatigue_level",
        "body_pain","sore_throat","runny_nose","duration_level"
    ])

    # Prediction
    prediction = model.predict(input_data)
    probabilities = model.predict_proba(input_data)

    disease = prediction[0]
    confidence = max(probabilities[0]) * 100

    # Severity logic (based on levels)
    symptom_score = fever_level + cough_type + headache_level + fatigue_level + body_pain

    if symptom_score <= 2:
        severity = "Mild"
    elif symptom_score <= 5:
        severity = "Moderate"
    else:
        severity = "Severe"

    # Guidance
    guidance = {
        "Flu": {
            "advice": "Rest, fluids, avoid cold exposure",
            "meds": "Paracetamol, Cough syrup"
        },
        "Cold": {
            "advice": "Warm fluids, rest",
            "meds": "Decongestants"
        },
        "Viral Fever": {
            "advice": "Hydration + rest",
            "meds": "Paracetamol"
        },
        "Headache": {
            "advice": "Rest, reduce screen time",
            "meds": "Pain relievers"
        },
        "Allergy": {
            "advice": "Avoid allergens",
            "meds": "Antihistamines"
        }
    }

    # Output
    st.divider()
    st.markdown(f"### 🩺 Condition: {disease}")
    st.markdown(f"**Confidence:** {confidence:.2f}%")

    # Severity UI
    if severity == "Mild":
        st.success(f"Severity: {severity} ✅")
    elif severity == "Moderate":
        st.warning(f"Severity: {severity} ⚠️")
    else:
        st.error(f"Severity: {severity} 🚨")

    st.subheader("💡 Advice")
    st.write(guidance[disease]["advice"])

    st.subheader("💊 Medication Categories")
    st.write(guidance[disease]["meds"])

    st.warning("⚠️ Educational tool only, not medical advice.")