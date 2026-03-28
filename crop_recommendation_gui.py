import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# Load Dataset
# -------------------------------
try:
    data = pd.read_csv("crop_recommendation.csv")   # Kaggle dataset
except FileNotFoundError:
    messagebox.showerror("Error", "Dataset 'crop_recommendation.csv' not found!")
    exit()

X = data.drop("label", axis=1)
y = data["label"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# GUI Application (Tkinter)
# -------------------------------
def predict_crop():
    try:
        N = float(entry_N.get())
        P = float(entry_P.get())
        K = float(entry_K.get())
        temp = float(entry_temp.get())
        humidity = float(entry_humidity.get())
        ph = float(entry_ph.get())
        rainfall = float(entry_rainfall.get())
        
        features = [[N, P, K, temp, humidity, ph, rainfall]]
        prediction = model.predict(features)[0]
        
        messagebox.showinfo("Recommended Crop", f"🌱 You should grow: {prediction}")
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values!")

# Create Main Window
root = tk.Tk()
root.title("Crop Recommendation System")
root.geometry("400x500")
root.config(bg="#f0f5f5")

# Labels and Entry Fields
tk.Label(root, text="Crop Recommendation System", font=("Arial", 16, "bold"), bg="#f0f5f5").pack(pady=10)

frame = tk.Frame(root, bg="#f0f5f5")
frame.pack(pady=10)

labels = ["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)",
          "Temperature (°C)", "Humidity (%)", "pH", "Rainfall (mm)"]

entries = []
for i, label in enumerate(labels):
    tk.Label(frame, text=label, font=("Arial", 12), bg="#f0f5f5").grid(row=i, column=0, sticky="w", pady=5)
    entry = tk.Entry(frame, font=("Arial", 12))
    entry.grid(row=i, column=1, pady=5)
    entries.append(entry)

entry_N, entry_P, entry_K, entry_temp, entry_humidity, entry_ph, entry_rainfall = entries

# Predict Button
tk.Button(root, text="Recommend Crop", command=predict_crop, font=("Arial", 14), bg="#4CAF50", fg="white").pack(pady=20)

root.mainloop()
