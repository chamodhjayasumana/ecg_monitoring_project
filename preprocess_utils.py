
import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter
import heartpy as hp

def add_athlete_noise(signal, noise_level=0.05):
    noise = np.random.normal(0, noise_level * np.std(signal), len(signal))
    return signal + noise

def preprocess_ecg(signal, fs=360):
    nyq = 0.5 * fs
    low, high = 0.5 / nyq, 45.0 / nyq
    b, a = butter(4, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)
    filtered = savgol_filter(filtered, 21, 2)
    baseline = savgol_filter(filtered, 501, 3)
    return filtered - baseline

def detect_r_peaks(signal, fs=360):
    working_data, _ = hp.process(signal, sample_rate=fs)
    return working_data['peaklist']

def extract_beats(signal, r_peaks, fs=360):
    window_size = int(0.6 * fs)
    half_window = window_size // 2
    beats = [signal[r - half_window : r + half_window] for r in r_peaks
             if r - half_window >= 0 and r + half_window < len(signal)]
    return np.array(beats)

def compute_hrv(r_peaks, fs=360):
    rr_intervals = np.diff(r_peaks) / fs * 1000
    return {
        'mean_rr': np.mean(rr_intervals),
        'sdnn': np.std(rr_intervals),
        'rmssd': np.sqrt(np.mean(np.square(np.diff(rr_intervals)))),
        'nn50': np.sum(np.abs(np.diff(rr_intervals)) > 50),
        'pnn50': np.sum(np.abs(np.diff(rr_intervals)) > 50) / len(rr_intervals)
    }
