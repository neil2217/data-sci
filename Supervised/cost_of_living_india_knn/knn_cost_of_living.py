import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

# --- Create model directory ---
model_dir = "model_files"
os.makedirs(model_dir, exist_ok=True)

# --- 1. Load dataset ---
try:
    df = pd.read_csv("cost_of_living.csv")
    print("Dataset loaded successfully.")
except FileNotFoundError:
    raise SystemExit("Error: 'cost_of_living.csv' not found.")

# --- 2. Data cleaning ---
df.replace("N/A", np.nan, inplace=True)
drop_cols = [
    "City", "Region", "Language_Diversity_Score", "Safety_Index",
    "Crime_Index", "Rent_as_Percentage_of_Total", "Affordability_Index"
]
df_cleaned = df.dropna().drop(columns=drop_cols)
print("Data cleaned and preprocessed.")

# --- 3. Features & target ---
X = df_cleaned.drop("City_Tier", axis=1)
y = df_cleaned["City_Tier"]

feature_names = X.columns.tolist()
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# --- 4. Train/test split & scaling ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 5. Train KNN model ---
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
print("KNN model trained successfully with k=5.")

# --- 6. Evaluate ---
accuracy = accuracy_score(y_test, knn.predict(X_test_scaled))
print(f"Model Accuracy: {accuracy:.2f}")

# --- 7. Export model & preprocessors ---
joblib.dump(knn, os.path.join(model_dir, "knn_model.pkl"))
joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
joblib.dump(label_encoder, os.path.join(model_dir, "label_encoder.pkl"))
joblib.dump(feature_names, os.path.join(model_dir, "feature_names.pkl"))

print(f"Model and files saved to '{model_dir}'. Ready for Flask application.")
