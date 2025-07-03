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

st.set_page_config(layout="wide", page_title="Athlete ECG Monitor", page_icon="💓")

# Sidebar controls
st.sidebar.title("⚙ Settings")
record_id = st.sidebar.selectbox("Choose ECG Record", ['100', '101', '102'])
noise_level = st.sidebar.slider("Add Athlete Noise", 0.0, 0.2, 0.05, step=0.01)
show_hrv = st.sidebar.checkbox("Show HRV Metrics", value=True)

st.title("💓 Real-Time ECG Monitoring and Arrhythmia Detection for Athletes")

@st.cache_data
def load_ecg_data(record):
    wfdb.dl_database('mitdb', dl_dir='mitdb', records=[record])
    record_data = wfdb.rdrecord(f'mitdb/{record}')
    return record_data.p_signal[:, 0], record_data.fs

signal, fs = load_ecg_data(record_id)
athlete_signal = add_athlete_noise(signal, noise_level=noise_level)
filtered_signal = preprocess_ecg(athlete_signal)

r_peaks = detect_r_peaks(filtered_signal, fs)
beats = extract_beats(filtered_signal, r_peaks, fs)
hrv = compute_hrv(r_peaks, fs)

# Annotation loading
ann = wfdb.rdann(f'mitdb/{record_id}', 'atr')
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

class_counts = Counter(y_enc)
st.sidebar.write("📊 Class counts:", {le.inverse_transform([k])[0]: v for k, v in class_counts.items()})

if all(count >= 2 for count in class_counts.values()):
    stratify_arg = y_enc
    st.sidebar.success("✅ Stratified split applied")
else:
    stratify_arg = None
    st.sidebar.warning("⚠ Stratified split disabled")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=stratify_arg
)

model = build_model((X_train.shape[1], 1))
with st.spinner("⏳ Training CNN-LSTM model..."):
    history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test), verbose=0)
st.success("✅ Model training complete")

y_true = np.argmax(y_test, axis=1)
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
available_labels = unique_labels(y_true, y_pred_labels)
available_names = le.inverse_transform(available_labels)

tab1, tab2, tab3 = st.tabs(["📈 ECG Signals", "🧠 Model Performance", "💓 HRV Analysis"])

with tab1:
    st.subheader("ECG: Original vs Filtered")
    fig, axs = plt.subplots(2, 1, figsize=(12, 4))
    axs[0].plot(signal[:1000])
    axs[0].set_title("Original ECG")
    axs[1].plot(filtered_signal[:2000])
    plt.subplots_adjust(hspace=0.5)
    axs[1].set_title("Filtered Athlete ECG")
    st.pyplot(fig)

    st.subheader("Extracted Heartbeats")
    fig2 = plt.figure(figsize=(12, 3))
    for i in range(min(5, len(beats))):
        plt.plot(np.linspace(-0.3, 0.3, beats.shape[1]), beats[i])
    plt.title("First 5 Heartbeats")
    st.pyplot(fig2)

with tab2:
    st.subheader("Model Training Metrics")
    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title("Accuracy"); ax1.legend()
    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title("Loss"); ax2.legend()
    st.pyplot(fig3)

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

with tab3:
    st.subheader("HRV Metrics")
    if show_hrv:
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean RR (ms)", f"{hrv['mean_rr']:.1f}")
        col2.metric("SDNN", f"{hrv['sdnn']:.1f}")
        col3.metric("RMSSD", f"{hrv['rmssd']:.1f}")
        st.write("Full HRV Summary:")
        st.json(hrv)