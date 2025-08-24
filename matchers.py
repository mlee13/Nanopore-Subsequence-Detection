import time
import numpy as np
from numba import njit
from fastdtw import fastdtw
from dtaidistance import dtw
from scipy.spatial.distance import euclidean

import utils as u


###########################
#  DTW scoring functions  #
###########################

def score_dtw_match(best_distance, distances, n_windows):
    distances = np.array(distances)
    mean = np.mean(distances)
    std  = np.std(distances) + 1e-8

    # 1. z-score (same as before)
    z        = (mean - best_distance) / std
    z_scaled = ((z + 3.5) / 7) ** 2 * 100
    z_scaled = np.clip(z_scaled, 0, 100)

    # 2. margin vs. top-10 %
    sorted_d = np.sort(distances)
    k = max(2, int(0.1 * len(sorted_d))) 
    top10 = sorted_d[:k]
    if k <= 2:
        margin_score = 100.0
    else:
        top10_rest_mean = np.mean(top10[1:])
        margin = top10_rest_mean - top10[0]
        margin_ratio = margin / (top10_rest_mean + 1e-8)
        margin_score = np.clip(margin_ratio * 100, 0, 100)

    base_score = 0.75 * z_scaled + 0.25 * margin_score

    # 3. penalise large search space
    correction = (np.log10(n_windows + 1) / 4) * 1.25
    adjusted   = base_score / correction
    return float(np.clip(adjusted, 0, 100))


# match ~100% non-match 70-80%
# zscore quadratic + percentile
def zscore_quadratic_score(best_distance, distances):
    mean = np.mean(distances)
    std = np.std(distances) + 1e-8
    z = (mean - best_distance) / std
    base_score = ((z + 3) / 6) ** 2 * 100
    base_score = np.clip(base_score, 0, 100)
    # return the best_score as float here for just zscore quadratic

    # Penalize if best_distance is not in bottom 10%
    rank = np.sum(distances < best_distance)
    percentile = rank / len(distances)

    if percentile > 0.1:
        base_score *= 0.7  # adjust this weight as needed

    return float(np.clip(base_score, 0, 100))


# match 35-40% non-match 15~20%
# Margin + Percentile
def margin_score(best_distance, distances):
    sorted_d = np.sort(distances)
    if len(sorted_d) < 2:
        return 100.0
    margin = sorted_d[1] - sorted_d[0]
    ratio = margin / (sorted_d[1] + 1e-8)
    # For just using margin score without boosting
    # return float(np.clip(ratio * 100, 0, 100))
    base = ratio * 100

    # Soft boost if margin is small but distance is also very low
    if best_distance < np.percentile(distances, 10):
        base += 15  # lift up good matches that are otherwise not unique

    return float(np.clip(base, 0, 100))


###########################
#         Matchers        #
# 1. Euclidean dist       #
# 2. NCC                  #
# 3. DTW (banded, fastdtw)#
# 4. Logistic Regression? #
###########################

def euclidean_distance_match(target: np.ndarray, signal: np.ndarray):
    signal = (signal - np.mean(signal)) / np.std(signal)

    m, n = len(target), len(signal)
    min_dist = float('inf')
    best_pos = 0

    for i in range(n - m + 1):
        window = signal[i:i + m]
        dist = np.linalg.norm(target - window)
        if dist < min_dist:
            min_dist = dist
            best_pos = i

    # Convert distance to probability using exponential decay
    scale = np.linalg.norm(target)  # normalize to self-distance
    probability = np.exp(-min_dist / (scale + 1e-6)) * 100
    return probability, best_pos


def ncc_match(target: np.ndarray, signal: np.ndarray):
    m, n = len(target), len(signal)

    max_score = -np.inf
    best_pos = 0

    for i in range(n - m + 1):
        window = signal[i:i + m]
        w_mean = np.mean(window)
        w_std = np.std(window) + 1e-6
        w_norm = (window - w_mean) / w_std

        # Make sure both window and taget is z-score normalized 
        # (zero mean and unit variance) for score in range [-1,1]
        score = np.dot(target, w_norm) / m  # cosine similarity 
        if score > max_score:
            max_score = score
            best_pos = i

    probability = np.clip(max_score, 0.0, 1.0) * 100.0
    return probability, best_pos


def dtw_distance_pruned_with_band(target, window, band_percent=0.1):
    N = len(target)
    band = max(1, int(N * band_percent))
    return dtw.distance(
        target,
        window,
        use_c=True,
        use_pruning=True,
        window=band
    )

@njit
def fastdtw_subsequence_match(target, test_signal, radius=1, stride=10, scale_window=True):
    start_time = time.time()
    start_cpu = time.process_time()

    len_target = len(target)
    n_windows = (len(test_signal) - len_target) // stride + 1

    distances = []
    positions = []

    norm_windows = u.batch_normalize_windows(test_signal, len_target, stride, scale_window)

    target_seq = u.to_vector_sequence(target)
    target_flat  = np.asarray(target, dtype=float)

    for w, window in enumerate(norm_windows):
        if len(test_signal) > 0:
            window_flat  = np.asarray(window, dtype=float).ravel()
            distance = dtw_distance_pruned_with_band(target_flat, window_flat, band_percent=0.2)
        else:
            distance, _ = fastdtw(target_seq, window.reshape(-1, 1), radius=radius, dist=euclidean)

        distances.append(distance)
        positions.append(w * stride)

    distances = np.array(distances)
    best_idx = np.argmin(distances)
    best_distance = distances[best_idx]
    best_index = positions[best_idx]

    # distances_norm = normalize_distances(np.array(distances), len_target)
    # best_idx = np.argmin(distances_norm)
    # best_distance = distances_norm[best_idx]
    # best_index = positions[best_idx]

    match_probability = score_dtw_match(best_distance, distances, n_windows)

    end_time = time.time()
    end_cpu = time.process_time()

    elapsed_time = end_time - start_time
    cpu_time = end_cpu - start_cpu

    print(f"Elapsed time: {elapsed_time:.4f} seconds")
    print(f"CPU time:     {cpu_time:.4f} seconds")

    return best_distance, best_index, match_probability
