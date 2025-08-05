import os
import shutil
import pandas as pd
import numpy as np
import subprocess
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Install Kaggle (only needed once)
try:
    import kaggle
except ImportError:
    subprocess.run(["pip", "install", "kaggle"])

# Set up Kaggle API key (Ensure kaggle.json is placed manually in ~/.kaggle/)
KAGGLE_PATH = os.path.expanduser("~/.kaggle")
KAGGLE_JSON = "kaggle.json"  # Place it in the script directory manually

if not os.path.exists(KAGGLE_PATH):
    os.makedirs(KAGGLE_PATH)

shutil.copy(KAGGLE_JSON, os.path.join(KAGGLE_PATH, "kaggle.json"))
os.chmod(os.path.join(KAGGLE_PATH, "kaggle.json"), 0o600)  # Set permission

# Download dataset from Kaggle (update with actual dataset name)
subprocess.run(["kaggle", "datasets", "download", "-d", "your-dataset-name"])
subprocess.run(["unzip", "dataset-file.zip"])  # Unzip dataset

# Load dataset
dataset = pd.read_csv("Crop_recommendation.csv")

# Check dataset information
print(dataset.head())
print("Dataset shape:", dataset.shape)
print("Missing values:\n", dataset.isnull().sum())

# Split features and labels
X = dataset.iloc[:, :-1]  # Features
y = dataset.iloc[:, -1]   # Labels

# Split into train (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Predict new crop
new_features = np.array([[36, 58, 25, 28.66024, 59.31891, 8.399136, 36.9263]])
predicted_crop = model.predict(new_features)
print("Predicted Crop:", predicted_crop[0])
