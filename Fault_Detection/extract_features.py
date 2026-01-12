# -*- coding: utf-8 -*-
"""
Feature extraction for HAlpha plasma signal
(mean of means, variance of means)

Author: Harsh
"""

import pandas as pd
import numpy as np

# =====================================
# PARAMETERS (MUST MATCH TRAINING)
# =====================================

SHEET_NAME = "HAlpha"

START_TIME = 0.0
TOTAL_SAMPLES = 1500

WINDOW_SIZE = 100
HOP_SIZE = 80

# =====================================
# FEATURE EXTRACTION FUNCTION
# =====================================

def extract_features_from_excel(excel_path):
    """
    Extracts features from a single Excel file.

    Returns
    -------
    features : numpy.ndarray of shape (1, 2)
        [[mean_mean, var_mean]]
    """

    # -----------------------------
    # LOAD EXCEL
    # -----------------------------
    try:
        df = pd.read_excel(excel_path, sheet_name=SHEET_NAME, header=None)
    except Exception as e:
        raise ValueError(f"HAlpha sheet not found: {e}")

    # -----------------------------
    # HANDLE HEADER INCONSISTENCY
    # -----------------------------
    # If first row contains strings like "Time", "Signal"
    if isinstance(df.iloc[0, 0], str):
        df = df.iloc[1:].reset_index(drop=True)

    df.columns = ["Time_ms", "Signal"]

    # -----------------------------
    # SORT BY TIME
    # -----------------------------
    df.sort_values("Time_ms", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # -----------------------------
    # FIND 0 ms
    # -----------------------------
    if START_TIME not in df["Time_ms"].values:
        raise ValueError("0 ms not found in Time column")

    start_idx = df.index[df["Time_ms"] == START_TIME][0]

    if start_idx + TOTAL_SAMPLES > len(df):
        raise ValueError("Not enough samples after 0 ms")

    signal = df.loc[
        start_idx : start_idx + TOTAL_SAMPLES - 1,
        "Signal"
    ].values

    # -----------------------------
    # WINDOWING
    # -----------------------------
    window_means = []

    for start in range(0, TOTAL_SAMPLES - WINDOW_SIZE + 1, HOP_SIZE):
        window = signal[start : start + WINDOW_SIZE]
        window_means.append(np.mean(window))

    window_means = np.array(window_means)

    # -----------------------------
    # FINAL FEATURES (USED BY SVM)
    # -----------------------------
    mean_mean = np.mean(window_means)
    var_mean  = np.var(window_means)

    return np.array([[mean_mean, var_mean]])

# =====================================
# TEST (OPTIONAL)
# =====================================

if __name__ == "__main__":
    test_file = "example.xlsx"  # change path if testing

    try:
        feats = extract_features_from_excel(test_file)
        print("Extracted Features:")
        print(f"mean_mean : {feats[0,0]:.6e}")
        print(f"var_mean  : {feats[0,1]:.6e}")
    except Exception as err:
        print("Error:", err)
