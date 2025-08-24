import os
import numpy as np
import random

def create_simulated_signal(path, is_ideal):
    dir_list = os.listdir(path)
    ideal = ""

    if is_ideal:
        ideal = "_ideal"

    for dir in dir_list:
        print("./squigulator --ideal " + path + "/" + dir + " -o " + "signal_output/test0/" + dir[:-3] + ideal + ".blow5 -n 10")


# ensures each element is a 1-D vector 

def to_vector_sequence(arr):
    return np.asarray(arr, dtype=np.float32).reshape(-1, 1)


def batch_normalize_windows(signal, target_len, stride, normalize=True):
    n_windows = (len(signal) - target_len) // stride + 1
    norm_windows = []

    for w in range(n_windows):
        start = w * stride
        end = start + target_len

        if end > len(signal):
            break

        window = signal[start:end]
        
        if normalize:
            window = (window - np.mean(window)) / (np.std(window) + 1e-8)

        norm_windows.append(window)

    return np.array(norm_windows)


# Remove low-variance (flat) regions at the start and end of a 1D signal.
def trim_padding(signal: np.ndarray, 
                 window_size: int = 5, 
                 flat_threshold: float = 1e-3):
    padded = np.pad(signal, (window_size//2, window_size//2), mode='edge')
    var = np.array([np.var(padded[i:i + window_size]) for i in range(len(signal))])
    non_flat = np.where(var > flat_threshold)[0]

    if non_flat.size == 0:
        return signal.copy()

    start_idx = non_flat[0]
    end_idx = non_flat[-1] + 1

    return signal[start_idx:end_idx]


def count_base_content(sequence):
    base_counts = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
    for base in sequence:
        base = base.upper()
        if base in base_counts:
            base_counts[base] += 1

    percentage = []
    percentage.append(base_counts['A'] / len(sequence) * 100)
    percentage.append(base_counts['C'] / len(sequence) * 100)
    percentage.append(base_counts['G'] / len(sequence) * 100)
    percentage.append(base_counts['T'] / len(sequence) * 100)

    return base_counts, percentage


def create_random_sequence(length):
    return ''.join(random.choices("AGCT", k=length))

import pywt

from scipy.ndimage import median_filter
from scipy.stats import linregress

def correct_signal_baseline(signal, window_size=1000):
        # Can optimise with parallel programming mp (multiprocessing)
        corrected_signal = signal
        window_size = int(len(signal) * 0.2) 
        estimated_baseline = median_filter(signal, size=window_size, mode="reflect")

        x = np.arange(len(estimated_baseline))
        corrected_signal = signal - estimated_baseline

        return corrected_signal


def denoise_signal_wavelet(signal, wavelet='db4', level=4, thresholding='soft'):
    # Perform wavelet decomposition
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # Estimate noise level (Median Absolute Deviation method)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745

    # Set universal threshold
    threshold = sigma * np.sqrt(2 * np.log(len(signal))) * 1.2

    # Apply thresholding to detail coefficients
    new_coeffs = [coeffs[0]] + [pywt.threshold(c, threshold, mode=thresholding) for c in coeffs[1:]]

    # Reconstruct the denoised signal
    denoised_signal = pywt.waverec(new_coeffs, wavelet)

    return denoised_signal