import joblib
import pandas as pd

model = joblib.load("model/model.pkl")

user_input = input("Describe your symptoms: ").lower()

# 🔥 SMART FEATURE EXTRACTION

# Fever
if "high fever" in user_input:
    fever_level = 2
elif "mild fever" in user_input or "fever" in user_input:
    fever_level = 1
else:
    fever_level = 0

# Cough
if "wet cough" in user_input:
    cough_type = 2
elif "dry cough" in user_input:
    cough_type = 1
elif "cough" in user_input:
    cough_type = 1
else:
    cough_type = 0

# Headache
if "severe headache" in user_input:
    headache_level = 2
elif "headache" in user_input:
    headache_level = 1
else:
    headache_level = 0

# Fatigue
if "very tired" in user_input:
    fatigue_level = 2
elif "tired" in user_input or "fatigue" in user_input:
    fatigue_level = 1
else:
    fatigue_level = 0

# Binary symptoms
body_pain = 1 if "body pain" in user_input else 0
sore_throat = 1 if "sore throat" in user_input else 0
runny_nose = 1 if "runny nose" in user_input else 0

# 🔥 FALLBACK LOGIC
if "cold" in user_input:
    cough_type = max(cough_type, 1)
    runny_nose = 1

if "flu" in user_input:
    fever_level = max(fever_level, 2)
    body_pain = 1

# Duration
if "week" in user_input:
    duration_level = 2
elif "day" in user_input:
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

print("\n--- Result ---")
print("Condition:", disease)
print(f"Confidence: {confidence:.2f}%")