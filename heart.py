# ensure joblib is present in the notebook’s environment

# Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# Load Dataset
data = pd.read_csv("heart.csv")

# Separate Features and Target
X = data.drop("target", axis=1)   # Change "target" if your column name is different
y = data["target"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Check Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Save BOTH model and scaler together
joblib.dump((model, scaler), "heart_model.pkl")

print("heart_model.pkl file created successfully!")