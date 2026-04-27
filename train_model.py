import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load dataset
data = pd.read_csv("data/dataset.csv")

# Features
X = data.drop("disease", axis=1)

# Target
y = data["disease"]

# Model
model = DecisionTreeClassifier()

# Train
model.fit(X, y)

# Save
joblib.dump(model, "model/model.pkl")

# Accuracy
accuracy = model.score(X, y)
print(f"Model trained successfully! Accuracy: {accuracy * 100:.2f}%")