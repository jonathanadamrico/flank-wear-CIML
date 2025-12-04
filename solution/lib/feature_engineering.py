# feature_engineering.py

import numpy as np
import scipy.stats as stats
from scipy.signal import butter, filtfilt, welch
from scipy.fft import fft
import pandas as pd

def subtract_features_from_cut01(features_df, set_list, cut01_features, ignore_cols=('cut_no', 'valid_sensor_data', 'AE_median_diff_cut01')):
    features_cut01_df = features_df.copy()
    cuts_per_set = len(features_df) // len(set_list)
    print(f'{cuts_per_set} cuts per set')
    
    diff_features = [x for x in features_df.columns if x not in ignore_cols]
    features_cut01_df[diff_features] = features_cut01_df[diff_features].astype('float32')
    for set_no in set_list:
        
        value_to_subtract = cut01_features.loc[set_no - 1, diff_features].astype('float32') 
        
        # Perform the relative difference calculation (x - xhat) / xhat
        features_cut01_df.loc[(set_no - 1) * cuts_per_set:(set_no) * cuts_per_set - 1, diff_features] = \
            (features_cut01_df.loc[(set_no - 1) * cuts_per_set:(set_no) * cuts_per_set - 1, diff_features] - value_to_subtract) / \
             np.where(value_to_subtract == 0, value_to_subtract + 1e-12, value_to_subtract)
             
    return features_cut01_df

def detect_cutting_window(signal, window_size=10000, threshold=0.05):
    """
    Detect cutting window from signal using RMS thresholding.
    Returns start and end indices of cutting region.
    
    Parameters:
        signal: 1D array (e.g., AE or accel magnitude)
        window_size: size of moving window in samples
        threshold: RMS threshold for activity detection
    
    Returns:
        start_idx, end_idx
    """
    signal = np.abs(signal)
    n = len(signal)
    
    # Moving RMS
    rms = np.sqrt(np.convolve(signal**2, np.ones(window_size)/window_size, mode='same'))

    # Normalize RMS if needed
    norm_rms = (rms - np.min(rms)) / (np.max(rms) - np.min(rms) + 1e-12)

    # Binary activity mask
    active = norm_rms > threshold

    # Find cutting start and end
    indices = np.where(active)[0]
    if len(indices) == 0:
        return 0, n  # fallback if nothing detected

    start_idx = indices[0]
    end_idx = indices[-1]

    return start_idx, end_idx




def detect_cutting_window_vectorized(signal, threshold=0.05):
    """
    Fastest possible cutting window detection using simple thresholding.
    No loops, fully vectorized. Returns first and last sample above threshold.
    
    Parameters:
        signal: 1D array
        threshold: threshold on normalized absolute magnitude (0 to 1)
    
    Returns:
        start_idx, end_idx
    """
    signal = np.abs(signal)
    norm_signal = signal / (np.max(signal) + 1e-12)

    # Create binary mask where signal is "active"
    active = norm_signal > threshold

    if not np.any(active):
        return 0, len(signal)

    indices = np.where(active)[0]
    return indices[0], indices[-1]



def detect_cutting_window_from_load(controller_df, load_threshold=2):
    
    if controller_df["mainSpndLoad"].max() <= load_threshold:
        load_threshold = 0
    
    # First occurrence of value > load_threshold
    start_time = controller_df.loc[controller_df["mainSpndLoad"]>load_threshold, "start_step"].iloc[0]
    
    # Last occurence of value > load_threshold
    end_time = controller_df.loc[controller_df["mainSpndLoad"]>load_threshold, "end_step"].iloc[-1]
    
    return start_time, end_time

def bandpass(sig, fs, low=10, high=10000, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, sig)

# Utility: extract basic features from one signal
def extract_time_features(signal, name=''):
    signal = np.asarray(signal)
    features = {
        f'{name}_mean': np.mean(signal),
        f'{name}_std': np.std(signal),
        f'{name}_var': np.var(signal),
        f'{name}_rms': np.sqrt(np.mean(signal**2)),
        f'{name}_ptp': np.ptp(signal),  # peak-to-peak
        f'{name}_max': np.max(signal),
        f'{name}_min': np.min(signal),
        f'{name}_skew': stats.skew(signal),
        f'{name}_kurtosis': stats.kurtosis(signal),
        f'{name}_crest_factor': np.max(np.abs(signal)) / np.sqrt(np.mean(signal**2) + 1e-12),  # avoid divide by 0
        f'{name}_entropy': stats.entropy(np.histogram(signal, bins=50, density=True)[0] + 1e-6)
    }
    return features

# Utility: frequency domain features (FFT-based)
def extract_freq_features(signal, fs, name=''):
    # FFT and magnitude spectrum
    n = len(signal)
    fft_vals = np.abs(fft(signal))
    fft_freqs = np.fft.fftfreq(n, d=1/fs)

    # Only positive freqs
    mask = fft_freqs > 0
    fft_vals = fft_vals[mask]
    fft_freqs = fft_freqs[mask]

    # Normalize power
    power = fft_vals**2
    power /= np.sum(power + 1e-12)

    features = {
        f'{name}_spectral_centroid': np.sum(fft_freqs * power),
        f'{name}_spectral_bandwidth': np.sqrt(np.sum(((fft_freqs - np.sum(fft_freqs * power))**2) * power)),
        f'{name}_spectral_entropy': -np.sum(power * np.log2(power + 1e-12)),
        f'{name}_dominant_freq': fft_freqs[np.argmax(fft_vals)],
    }
    return features

# AE-specific features
def extract_ae_features(ae_signal, threshold=0.5, name='AE'):
    ae_signal = np.asarray(ae_signal)
    above_thresh = abs(ae_signal) > threshold

    features = {
        f'{name}_median': np.median(ae_signal),
        f'{name}_peak_count': np.sum(above_thresh),
        f'{name}_signal_energy': np.sum(ae_signal**2),
        #f'{name}_duration_above_thresh': np.sum(ae_signal > 0.5), 
    }
    return features

# sensor feature extraction
def extract_sensor_features(accel_x, accel_y, accel_z, ae, fs_accel=25600, fs_ae=25600):
    features = {}

    # Time and frequency features from accelerometer
    for signal, axis in zip([accel_x, accel_y, accel_z], ['x', 'y', 'z']):
        features.update(extract_time_features(signal, name=f'accel_{axis}'))
        features.update(extract_freq_features(signal, fs=fs_accel, name=f'accel_{axis}'))

    # AE time and freq features
    features.update(extract_time_features(ae, name='AE'))
    features.update(extract_freq_features(ae, fs=fs_ae, name='AE'))
    features.update(extract_ae_features(ae, threshold=max(abs(ae))*0.5))

    return features

def extract_controller_features(controller_df, col_names):
    features = {}
    for col in col_names:
        features[f'{col}_max'] = (controller_df[col] * controller_df["mainSpndLoad"]).max()
        features[f'{col}_sum'] = (controller_df[col] * controller_df["mainSpndLoad"]).sum()
        features[f'{col}_mean'] = (controller_df[col] * controller_df["mainSpndLoad"]).mean()
        
    features['mainSpndLoad_max'] = controller_df["mainSpndLoad"].max()
    features['mainSpndLoad_sum'] = controller_df["mainSpndLoad"].sum()
    features['mainSpndLoad_mean'] = controller_df["mainSpndLoad"].mean()
    features['cut_no'] = controller_df.loc[controller_df["step_no"]==10, "cut_no"].iloc[0]
        
    return features

def get_features(controller_df, sensor_df):
    fs = 25600
    start_time, end_time = detect_cutting_window_from_load(controller_df, load_threshold=2)
    
    features = {}
    features['effective_cut_time'] = (end_time - start_time).total_seconds()
    
    accel_x = bandpass(sensor_df["Acceleration X (g)"], fs, 50, 7000)
    accel_y = bandpass(sensor_df["Acceleration Y (g)"], fs, 50, 7000)
    accel_z = bandpass(sensor_df["Acceleration Z (g)"], fs, 50, 7000)
    ae = bandpass(sensor_df["AE (V)"], fs, 5000, 12700)
    
    col_names = ["X_load","Y_load","Z_load","feedrate","mainSpndSpd"]
    
    sensor_features = extract_sensor_features(accel_x, accel_y, accel_z, ae)
    controller_features = extract_controller_features(controller_df, col_names)
    
    features = sensor_features | controller_features | features
    
    return features
    