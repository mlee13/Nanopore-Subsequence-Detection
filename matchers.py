import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

import utils as u


###########################
#  DTW scoring functions  #
###########################

# match 80% non-match 50-60%
# zscore + margin
def score_dtw_match(best_distance, distances):
    distances = np.array(distances)
    mean = np.mean(distances)
    std = np.std(distances) + 1e-8

    # Boosted z-score component (exponential stretch)
    z = (mean - best_distance) / std
    z_scaled = ((z + 3) / 6) ** 2.5 * 100  # steeper than quadratic
    z_scaled = np.clip(z_scaled, 0, 100)

    # Margin component (smaller weight)
    sorted_d = np.sort(distances)
    if len(sorted_d) < 2:
        margin_score = 100.0
    else:
        margin = sorted_d[1] - sorted_d[0]
        margin_ratio = margin / (sorted_d[1] + 1e-8)
        margin_score = np.clip(margin_ratio * 100, 0, 100)

    return float(0.75 * z_scaled + 0.25 * margin_score)


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

    t_norm = (target - np.mean(target)) / (np.std(target) + 1e-6)

    max_score = -np.inf
    best_pos = 0

    for i in range(n - m + 1):
        window = signal[i:i + m]
        w_mean = np.mean(window)
        w_std = np.std(window) + 1e-6
        w_norm = (window - w_mean) / w_std

        # Make sure both window and taget is z-score normalized 
        # (zero mean and unit variance) for score in range [-1,1]
        score = np.dot(t_norm, w_norm) / m  # cosine similarity
        if score > max_score:
            max_score = score
            best_pos = i

    probability = np.clip(max_score, 0.0, 1.0) * 100.0
    return probability, best_pos


# TODO: can you just change the scoring function here to make it correct
# or should this just be fn for best_subsequence_match?
def fastdtw_subsequence_match(target, test_signal, radius=1, stride=10, scale_window=True):
    len_target = len(target)
    n_windows = (len(test_signal) - len_target) // stride + 1

    if scale_window:
        target = (target - np.mean(target)) / (np.std(target) + 1e-8)

    distances = []
    positions = []

    for w in range(n_windows):
        start = w * stride
        end = start + len_target

        if end > len(test_signal):
            break

        window = test_signal[start:end]

        if scale_window:
            window = (window - np.mean(window)) / (np.std(window) + 1e-8)

        target_seq = u.to_vector_sequence(target)
        window_seq = u.to_vector_sequence(window)

        distance, _ = fastdtw(target_seq, window_seq, radius=radius, dist=euclidean)

        distances.append(distance)
        positions.append(start)

    distances = np.array(distances)
    best_idx = np.argmin(distances)
    best_distance = distances[best_idx]
    best_index = positions[best_idx]

    match_probability = score_dtw_match(best_distance, distances)

    return best_distance, best_index, match_probability
