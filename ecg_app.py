import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import wfdb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.utils.multiclass import unique_labels
from collections import Counter

from preprocess_utils import (
    add_athlete_noise, preprocess_ecg,
    detect_r_peaks, extract_beats, compute_hrv
)
from model_utils import build_model

st.set_page_config(layout='wide')
st.title("🏃 Real-Time ECG Monitoring and Arrhythmia Detection for Athletes")

RECORD = '100'
FS = 360

@st.cache_data
def load_ecg_data():
    wfdb.dl_database('mitdb', dl_dir='mitdb', records=[RECORD])
    record = wfdb.rdrecord(f'mitdb/{RECORD}')
    return record.p_signal[:, 0], record.fs

signal, fs = load_ecg_data()
athlete_signal = add_athlete_noise(signal)
filtered_signal = preprocess_ecg(athlete_signal)

# ECG PLOT
st.subheader("📈 ECG Signals")
fig, axs = plt.subplots(2, 1, figsize=(12, 4))
axs[0].plot(signal[:1000])
axs[0].set_title("Original ECG")
axs[1].plot(filtered_signal[:1000])
axs[1].set_title("Filtered Athlete ECG")
st.pyplot(fig)

# R-Peak Detection and HRV
r_peaks = detect_r_peaks(filtered_signal, fs)
beats = extract_beats(filtered_signal, r_peaks, fs)
hrv = compute_hrv(r_peaks, fs)

st.subheader("💓 Extracted Beats and HRV")
fig2 = plt.figure(figsize=(12, 3))
for i in range(min(5, len(beats))):
    plt.plot(np.linspace(-0.3, 0.3, beats.shape[1]), beats[i])
plt.title("First 5 Heartbeats")
st.pyplot(fig2)
st.json(hrv)

# Load annotations
ann = wfdb.rdann(f'mitdb/{RECORD}', 'atr')
symbol_map = {'N': 'normal', 'V': 'ventricular'}
X, y = [], []

for i, r in enumerate(r_peaks):
    if i >= len(beats): continue
    nearest_idx = np.argmin(np.abs(ann.sample - r))
    label = symbol_map.get(ann.symbol[nearest_idx], 'unknown')
    if label in ['normal', 'ventricular']:
        X.append(beats[i])
        y.append(label)

if len(X) < 2:
    st.error("❌ Not enough labeled beats for training.")
    st.stop()

X = np.array(X)[..., np.newaxis]
le = LabelEncoder()
y_enc = le.fit_transform(y)
y_cat = to_categorical(y_enc)

# Show class distribution
class_counts = Counter(y_enc)
st.write("📊 Class distribution:", {le.inverse_transform([k])[0]: v for k, v in class_counts.items()})

# Stratified or fallback split
if all(count >= 2 for count in class_counts.values()):
    stratify_arg = y_enc
    st.info("✅ Stratified train-test split applied.")
else:
    stratify_arg = None
    st.warning("⚠ Stratified split disabled due to low sample count.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=stratify_arg
)

# Build and train model
model = build_model((X_train.shape[1], 1))
with st.spinner("Training CNN-LSTM model..."):
    history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test), verbose=0)
st.success("✅ Model training complete")

# Evaluate model
y_true = np.argmax(y_test, axis=1)
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)

available_labels = unique_labels(y_true, y_pred_labels)
available_names = le.inverse_transform(available_labels)

if len(available_labels) == 1:
    st.warning(f"⚠ Only one class ('{available_names[0]}') present in test set.")
    accuracy = np.mean(y_true == y_pred_labels)
    st.metric("Accuracy", f"{accuracy * 100:.2f}%")
else:
    report = classification_report(
        y_true,
        y_pred_labels,
        labels=available_labels,
        target_names=available_names,
        output_dict=True,
        zero_division=0
    )
    st.subheader("📋 Classification Report")
    st.json(report)

# Plot training performance
fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history.history['accuracy'], label='Train')
ax1.plot(history.history['val_accuracy'], label='Validation')
ax1.set_title("Accuracy"); ax1.legend()

ax2.plot(history.history['loss'], label='Train')
ax2.plot(history.history['val_loss'], label='Validation')
ax2.set_title("Loss"); ax2.legend()
st.pyplot(fig3)