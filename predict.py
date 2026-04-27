import joblib
import pandas as pd

model = joblib.load("model/model.pkl")

user_input = input("Describe your symptoms: ").lower()

# Feature extraction (SMART version)

fever_level = 2 if "high fever" in user_input else 1 if "fever" in user_input else 0

cough_type = 2 if "wet cough" in user_input else 1 if "cough" in user_input else 0

headache_level = 2 if "severe headache" in user_input else 1 if "headache" in user_input else 0

fatigue_level = 2 if "very tired" in user_input else 1 if "tired" in user_input else 0

body_pain = 1 if "body pain" in user_input else 0
sore_throat = 1 if "sore throat" in user_input else 0
runny_nose = 1 if "runny nose" in user_input else 0

# Duration
if "week" in user_input:
    duration_level = 2
elif "days" in user_input:
    duration_level = 1
else:
    duration_level = 0

input_data = pd.DataFrame([[
    fever_level, cough_type, headache_level, fatigue_level,
    body_pain, sore_throat, runny_nose, duration_level
]], columns=[
    "fever_level","cough_type","headache_level","fatigue_level",
    "body_pain","sore_throat","runny_nose","duration_level"
])

prediction = model.predict(input_data)
probabilities = model.predict_proba(input_data)

disease = prediction[0]
confidence = max(probabilities[0]) * 100

print("\nCondition:", disease)
print(f"Confidence: {confidence:.2f}%")