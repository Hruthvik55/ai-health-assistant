import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# 🔥 Expanded dataset (important)
data = {
    "text": [
        "high fever and cough",
        "fever with body pain and fatigue",
        "high temperature and weakness",
        "runny nose and cough",
        "dry cough and sore throat",
        "blocked nose and mild cough",
        "very tired with high fever",
        "fever and extreme fatigue",
        "severe headache",
        "headache with tiredness",
        "migraine and fatigue",
        "runny nose sneezing and cough",
        "allergy cough no fever",
        "sore throat and runny nose",
        "wet cough with fever",
        "body ache with fever",
        "mild fever and cough",
        "weakness and headache",
        "nasal congestion and cough",
        "cough with no fever",
        "extreme tiredness and fever",
        "headache only",
        "cold and runny nose",
        "dry cough no fever",
        "fever for 3 days and fatigue"
    ],
    "disease": [
        "Flu","Flu","Flu",
        "Cold","Cold","Cold",
        "Viral Fever","Viral Fever",
        "Headache","Headache","Headache",
        "Allergy","Allergy","Allergy",
        "Flu","Flu","Cold",
        "Headache","Cold","Cold",
        "Viral Fever","Headache",
        "Cold","Cold","Flu"
    ]
}

df = pd.DataFrame(data)

# 🔥 NLP PIPELINE (BEST PRACTICE)
model = Pipeline([
    ("vectorizer", TfidfVectorizer(stop_words='english')),
    ("classifier", MultinomialNB())
])
# Train
model.fit(df["text"], df["disease"])

# Save
joblib.dump(model, "model/model.pkl")

print("✅ NLP model trained successfully!")