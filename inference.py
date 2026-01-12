import os
import joblib
import pandas as pd
import numpy as np
from extract_features import extract_features_from_excel

# =====================================
# PATHS
# =====================================

MODEL_PATH  = "Models/svm_model.pkl"
SCALER_PATH = "Models/scaler.pkl"

DATA_FOLDER = "Data"   # <-- provide folder path only

# =====================================
# LOAD MODEL
# =====================================

svm_model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# =====================================
# INFERENCE ON ONE FILE
# =====================================

def run_inference_on_file(excel_path):
    try:
        features = extract_features_from_excel(excel_path)
        features = scaler.transform(features)
        prediction = svm_model.predict(features)[0]
        return prediction, None
    except Exception as e:
        return None, str(e)

# =====================================
# PROCESS ALL EXCEL FILES IN FOLDER
# =====================================

def run_inference_on_folder(folder_path):

    if not os.path.isdir(folder_path):
        print(f"❌ Invalid folder path: {folder_path}")
        return

    files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(".xlsx")
    ])

    if not files:
        print("❌ No Excel files found in folder")
        return

    print("\n===== FAULT DETECTION RESULTS =====\n")

    for file in files:
        file_path = os.path.join(folder_path, file)

        prediction, error = run_inference_on_file(file_path)

        if error is not None:
            print(f"{file} → ❌ ERROR: {error}")
            continue

        if prediction == 0:
            print(f"{file} → ⚠️ FAULT DETECTED (Stuck-at-Zero)")
        else:
            print(f"{file} → ✅ NORMAL")

    print("\n===== INFERENCE COMPLETED =====\n")

# =====================================
# MAIN
# =====================================

if __name__ == "__main__":
    run_inference_on_folder(DATA_FOLDER)
