import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import os
import joblib

from extract_features import extract_features_from_excel

# =====================================
# PATHS
# =====================================

MODEL_PATH  = "Models/svm_model.pkl"
SCALER_PATH = "Models/scaler.pkl"

# =====================================
# LOAD MODEL
# =====================================

svm_model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# =====================================
# GLOBALS
# =====================================

data_folder = None

# =====================================
# GUI FUNCTIONS
# =====================================

def select_folder():
    global data_folder
    folder = filedialog.askdirectory(title="Select Folder Containing Excel Files")
    if folder:
        data_folder = folder
        folder_label.config(text=f"Data Folder:\n{folder}", fg="cyan")

def clear_output():
    output_box.delete(1.0, tk.END)

def run_detection():
    if not data_folder:
        messagebox.showerror("Error", "Please select a data folder first!")
        return

    files = sorted([
        f for f in os.listdir(data_folder)
        if f.lower().endswith(".xlsx")
    ])

    if not files:
        messagebox.showerror("Error", "No Excel files found in selected folder!")
        return

    output_box.insert(tk.END, "===== Fault Detection Started =====\n\n")
    output_box.update_idletasks()

    for file in files:
        file_path = os.path.join(data_folder, file)

        try:
            features = extract_features_from_excel(file_path)
            features = scaler.transform(features)
            prediction = svm_model.predict(features)[0]

            if prediction == 0:
                output_box.insert(
                    tk.END,
                    f"{file}  →  FAULTY (Stuck-at-Zero)\n",
                    "fault"
                )
            else:
                output_box.insert(
                    tk.END,
                    f"{file}  →  NORMAL\n",
                    "normal"
                )

        except Exception as e:
            output_box.insert(
                tk.END,
                f"{file}  →  ERROR: {str(e)}\n",
                "error"
            )

        # --- FORCE GUI UPDATE AFTER EACH FILE ---
        output_box.see(tk.END)
        output_box.update_idletasks()

        # Optional: slow down for visibility (remove if not needed)
        root.after(150)

    output_box.insert(tk.END, "\n===== Detection Completed =====\n")
    output_box.see(tk.END)

# =====================================
# GUI SETUP
# =====================================

root = tk.Tk()
root.title("Plasma Fault Detection System")
root.geometry("700x500")
root.configure(bg="#1e1e1e")

# -------------------------------------
# TITLE
# -------------------------------------

title_label = tk.Label(
    root,
    text="Plasma Fault Detection (SVM)",
    font=("Arial", 20, "bold"),
    bg="#1e1e1e",
    fg="white"
)
title_label.pack(pady=10)

# -------------------------------------
# FOLDER SELECTION
# -------------------------------------

select_btn = tk.Button(
    root,
    text="Select Data Folder",
    font=("Arial", 12),
    command=select_folder,
    bg="#007acc",
    fg="white",
    width=20
)
select_btn.pack(pady=5)

folder_label = tk.Label(
    root,
    text="No folder selected",
    font=("Arial", 10),
    bg="#1e1e1e",
    fg="gray",
    wraplength=650,
    justify="center"
)
folder_label.pack(pady=5)

# -------------------------------------
# CONTROL BUTTONS
# -------------------------------------

btn_frame = tk.Frame(root, bg="#1e1e1e")
btn_frame.pack(pady=10)

start_btn = tk.Button(
    btn_frame,
    text="START Detection",
    font=("Arial", 12, "bold"),
    command=run_detection,
    bg="green",
    fg="white",
    width=15
)
start_btn.grid(row=0, column=0, padx=10)

clear_btn = tk.Button(
    btn_frame,
    text="CLEAR Window",
    font=("Arial", 12),
    command=clear_output,
    bg="orange",
    fg="black",
    width=15
)
clear_btn.grid(row=0, column=1, padx=10)

# -------------------------------------
# OUTPUT WINDOW
# -------------------------------------

output_box = scrolledtext.ScrolledText(
    root,
    width=80,
    height=15,
    font=("Consolas", 10),
    bg="black",
    fg="white"
)
output_box.pack(padx=10, pady=10)

# -------------------------------------
# TEXT COLOR TAGS
# -------------------------------------

output_box.tag_config("fault", foreground="red")
output_box.tag_config("normal", foreground="light green")
output_box.tag_config("error", foreground="yellow")

# -------------------------------------
# RUN GUI
# -------------------------------------

root.mainloop()
