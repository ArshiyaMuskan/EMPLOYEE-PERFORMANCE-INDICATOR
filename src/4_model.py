import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------
# PATH HANDLING (IMPORTANT FIX)
# ------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH = os.path.join(BASE_DIR, "data", "cleaned_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Create folders if not exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------
# LOAD DATA
# ------------------------------
print("📂 Loading data...")
data = pd.read_csv(DATA_PATH)

# Features & target
X = data.drop("Performance", axis=1)
y = data["Performance"]

# ------------------------------
# TRAIN TEST SPLIT
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------
# MODEL TRAINING
# ------------------------------
print("🤖 Training model...")
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# ------------------------------
# PREDICTION & EVALUATION
# ------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.2f}")

# ------------------------------
# CONFUSION MATRIX (SAVE ONLY)
# ------------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm")
plt.title("Confusion Matrix")

cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path)
plt.close()

print(f"📊 Confusion matrix saved at: {cm_path}")

# ------------------------------
# SAVE MODEL
# ------------------------------
model_path = os.path.join(MODEL_DIR, "performance_model.pkl")
joblib.dump(model, model_path)

print(f"💾 Model saved at: {model_path}")

print("\n🚀 MODEL TRAINING COMPLETED SUCCESSFULLY!")