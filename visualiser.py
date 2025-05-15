import pyslow5
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from scipy.stats import linregress
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import pywt
import os

def correct_signal_baseline(signal, window_size=200):
        # Can optimise with parallel programming mp (multiprocessing)
        corrected_signal = signal
        estimated_baseline = median_filter(signal, size=window_size, mode="reflect")

        x = np.arange(len(estimated_baseline))
        baseline_slope, _, _, _, _ = linregress(x, estimated_baseline)

        # TODO: should i always remove baseline (for offset) or check the slope (for only drift)
        if abs(baseline_slope) > 0.0025:
            corrected_signal = signal - estimated_baseline

        return corrected_signal


def denoise_signal_wavelet(signal, wavelet='db4', level=4, thresholding='soft'):
    # Perform wavelet decomposition
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # Estimate noise level (Median Absolute Deviation method)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745

    # Set universal threshold
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))

    # Apply thresholding to detail coefficients
    new_coeffs = [coeffs[0]] + [pywt.threshold(c, threshold, mode=thresholding) for c in coeffs[1:]]

    # Reconstruct the denoised signal
    denoised_signal = pywt.waverec(new_coeffs, wavelet)

    return denoised_signal


def plot_blow5_signal(
    blow5_path: str,
    read_id: str = None,
    num_reads: int = 1
) -> None:
    """
    Open a BLOW5 file, fetch 'num_reads' reads (or the specified read_id),
    convert the raw signal to picoamperes, and plot each signal trace.

    Parameters
    ----------
    blow5_path : str
        Path to the .blow5 file.
    read_id : str, optional
        Specific read ID to plot. If None, plots the first 'num_reads' reads.
    num_reads : int
        Number of reads to plot (only used if read_id is None).
    """

    # Open the file in read-only mode. An index will be built or loaded automatically.
    s5 = pyslow5.Open(blow5_path, 'r')  # :contentReference[oaicite:0]{index=0}

    if read_id is not None:
        # Fetch a single read by ID
        recs = [s5.read(read_id)]
    else:
        # Iterate sequentially over reads
        recs = []
        it = s5.seq_reads()  # generator over all reads
        for _ in range(num_reads):
            try:
                recs.append(next(it))
            except StopIteration:
                break

    for rec in recs:
        # raw = np.array(rec['signal'], dtype=np.int16)
        raw = trim_padding(np.array(rec['signal'], dtype=np.int16))

        # raw = denoise_signal_wavelet(correct_signal_baseline(np.array(rec['signal'], dtype=np.int16)))

        # Retrieve metadata needed for conversion (defined in the SLOW5 spec)
        digitisation = rec['digitisation']
        range_        = rec['range']
        offset        = rec['offset']
        samp_rate     = rec['sampling_rate']

        print(rec['read_id'])
        print("raw signal size: ", len(raw))
        print("samp rate: ", samp_rate)
        print("")

        # Convert to picoamperes: pA = (raw + offset) * (range / digitisation)
        # (see SLOW5 format specification) :contentReference[oaicite:1]{index=1}
        current_pA = (raw + offset) * (range_ / digitisation)

        # Build a time axis in seconds
        time_s = np.arange(len(current_pA)) / samp_rate

        # Plot
        plt.figure(figsize=(10, 4))
        plt.plot(time_s, current_pA)
        plt.xlabel('Time (s)')
        plt.ylabel('Current (pA)')
        plt.title(f"Read ID: {rec['read_id']}")
        plt.tight_layout()
        plt.show()

    s5.close()


def create_run_cmds(path, is_ideal):
    # /home/ml4320/squigulator-v0.4.0/signal_input/test1
    dir_list = os.listdir(path)
    ideal = ""

    if is_ideal:
        ideal = "_ideal"

    for dir in dir_list:
        print("./squigulator --ideal " + path + "/" + dir + " -o " + "signal_output/test0/" + dir[:-3] + ideal + ".blow5 -n 10")


import numpy as np

def trim_padding(signal: np.ndarray, 
                 window_size: int = 5, 
                 flat_threshold: float = 1e-3):
    """
    Remove low-variance (flat) regions at the start and end of a 1D signal.
    """
    padded = np.pad(signal, (window_size//2, window_size//2), mode='edge')
    var = np.array([np.var(padded[i:i + window_size]) for i in range(len(signal))])
    non_flat = np.where(var > flat_threshold)[0]
    if non_flat.size == 0:
        return signal.copy()
    start_idx = non_flat[0]
    end_idx = non_flat[-1] + 1
    return signal[start_idx:end_idx]

import numpy as np
from scipy.signal import correlate

def euclidean_distance_score(target: np.ndarray, signal: np.ndarray):
    """
    Slide target over signal and compute Euclidean distance.
    Return probability based on inverted, normalized distance.

    Parameters
    ----------
    target : np.ndarray
        Shorter signal
    signal : np.ndarray
        Longer signal

    Returns
    -------
    probability : float
        Match confidence [0, 100]
    best_pos : int
        Start index of best match
    """
    m, n = len(target), len(signal)
    if m > n:
        raise ValueError("Target must be shorter than signal")

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


def ncc_match_probability(target: np.ndarray, signal: np.ndarray):
    """
    Compute the probability (0–100%) that `target` exists within `signal`
    using Normalized Cross-Correlation (NCC), comparing both forward and reversed directions.

    Parameters
    ----------
    target : np.ndarray
        1D array of the shorter signal.
    signal : np.ndarray
        1D array of the longer signal.

    Returns
    -------
    probability : float
        Similarity score scaled to [0, 100], where 100 means a perfect match.
    best_pos : int
        Starting index in `signal` where the best alignment occurs.
    orientation : str
        'forward' if the best match uses the forward orientation of `target`,
        or 'reverse' if the reversed `target` matches better.
    """
    m, n = len(target), len(signal)
    if m > n:
        raise ValueError("Target length must be <= signal length")

    # Normalize the target
    t_norm = (target - np.mean(target)) / (np.std(target) + 1e-6)
    t_rev = t_norm[::-1]
    sum_t = np.sum(t_norm)
    sum_t2 = np.sum(t_norm ** 2)

    # Compute sliding statistics on the signal
    window = np.ones(m)
    sum_s = np.convolve(signal, window, mode='valid')
    sum_s2 = np.convolve(signal ** 2, window, mode='valid')
    mean_s = sum_s / m
    var_s = sum_s2 / m - mean_s ** 2
    std_s = np.sqrt(np.clip(var_s, 1e-12, None))

    # Forward correlation
    raw_corr_fwd = np.convolve(signal, t_norm[::-1], mode='valid')
    numerator_fwd = raw_corr_fwd - mean_s * sum_t
    denominator_fwd = std_s * np.sqrt(sum_t2)
    ncc_fwd = numerator_fwd / (denominator_fwd + 1e-12)

    # Reverse correlation
    raw_corr_rev = np.convolve(signal, t_rev[::-1], mode='valid')
    sum_t_rev = np.sum(t_rev)
    numerator_rev = raw_corr_rev - mean_s * sum_t_rev
    denominator_rev = std_s * np.sqrt(sum_t2)
    ncc_rev = numerator_rev / (denominator_rev + 1e-12)

    # Find best match
    idx_fwd = np.argmax(ncc_fwd)
    max_fwd = ncc_fwd[idx_fwd]
    idx_rev = np.argmax(ncc_rev)
    max_rev = ncc_rev[idx_rev]

    if max_rev > max_fwd:
        best_prob = max_rev
        best_pos = idx_rev
        orientation = 'reverse'
    else:
        best_prob = max_fwd
        best_pos = idx_fwd
        orientation = 'forward'

    # Clamp to [0, 1] and convert to percentage
    best_prob = np.clip(best_prob, 0.0, 1.0)
    return best_prob * 100.0, best_pos, orientation

import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def zscore_prob(best_distance, distances):
    mean = np.mean(distances)
    std = np.std(distances) + 1e-8
    z = (mean - best_distance) / std
    return float(100 / (1 + np.exp(-z)))  # sigmoid-scaled probability

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

        distance, _ = fastdtw(target, window, radius=radius, dist=euclidean)
        distances.append(distance)
        positions.append(start)

    distances = np.array(distances)
    best_idx = np.argmin(distances)
    best_distance = distances[best_idx]
    best_index = positions[best_idx]
    match_probability = zscore_prob(best_distance, distances)

    return best_distance, best_index, match_probability


def dtw_distance_banded(a: np.ndarray, b: np.ndarray, band: int = None, dist_func=np.abs):
    """
    Compute DTW distance between a and b with a Sakoe-Chiba band constraint.

    Parameters
    ----------
    a : np.ndarray
        Target signal (length m)
    b : np.ndarray
        Signal to compare (length n)
    band : int
        Maximum allowable warping (|i - j| <= band). If None, defaults to 10% of length of a.
    dist_func : callable
        Function to compute local distance (default: absolute difference)

    Returns
    -------
    total_cost : float
        Final DTW cost
    cost_matrix : np.ndarray
        DTW cost matrix
    path : list of tuples
        Optimal alignment path
    """
    m, n = len(a), len(b)
    if band is None:
        band = max(int(m * 0.1), 1)

    cost = np.full((m + 1, n + 1), np.inf)
    cost[0, 0] = 0

    for i in range(1, m + 1):
        j_start = max(1, i - band)
        j_end = min(n + 1, i + band + 1)
        for j in range(j_start, j_end):
            d = dist_func(a[i - 1] - b[j - 1])
            cost[i, j] = d + min(
                cost[i - 1, j],    # insertion
                cost[i, j - 1],    # deletion
                cost[i - 1, j - 1] # match
            )

    # Backtrack
    i, j = m, n
    path = [(i - 1, j - 1)]
    while i > 1 or j > 1:
        directions = []
        if i > 1: directions.append((cost[i - 1, j], i - 1, j))
        if j > 1: directions.append((cost[i, j - 1], i, j - 1))
        if i > 1 and j > 1: directions.append((cost[i - 1, j - 1], i - 1, j - 1))
        if not directions:
            break
        _, i, j = min(directions)
        path.append((i - 1, j - 1))
    path.reverse()

    return cost[m, n], cost[1:, 1:], path


def dtw_banded_match_probability(target: np.ndarray, signal: np.ndarray, scale=600.0, band: int = None):
    """
    Estimate the probability that `target` exists inside `signal` using Sakoe-Chiba banded DTW.

    Parameters
    ----------
    target : np.ndarray
        Target signal (shorter, clean signal).
    signal : np.ndarray
        Raw signal (longer, possibly noisy/dwell-varied).
    scale : float
        Controls sharpness of exponential decay for probability scaling.
    band : int
        Sakoe-Chiba band width. If None, defaults to 10% of target length.

    Returns
    -------
    probability : float
        Match confidence [0, 100]
    best_start : int
        Start index of best match in signal
    best_end : int
        End index of best match in signal
    """
    m, n = len(target), len(signal)
    if m > n:
        raise ValueError("Target must be shorter than signal")

    min_cost = float('inf')
    best_start, best_end = 0, 0

    # Slide target across signal
    window_stride = max(1, m // 10)
    for start in range(0, n - m + 1, window_stride):
        window = signal[start:start + m + m // 2]  # allow some stretching
        cost, _, _ = dtw_distance_banded(target, window, band=band)
        if cost < min_cost:
            min_cost = cost
            best_start = start
            best_end = start + len(window)

    print("Best DTW cost:", min_cost)

    probability = np.exp(-min_cost / scale) * 100
    return probability, best_start, best_end


def ncc_match_probability_debug(target: np.ndarray, signal: np.ndarray):
    m, n = len(target), len(signal)
    if m > n:
        raise ValueError("Target longer than signal")

    t_norm = (target - np.mean(target)) / (np.std(target) + 1e-6)
    t_rev = t_norm[::-1]
    sum_t = np.sum(t_norm)
    sum_t2 = np.sum(t_norm ** 2)

    window = np.ones(m)
    sum_s = np.convolve(signal, window, mode='valid')
    sum_s2 = np.convolve(signal**2, window, mode='valid')
    mean_s = sum_s / m
    var_s = sum_s2 / m - mean_s**2
    std_s = np.sqrt(np.clip(var_s, 1e-12, None))

    raw_corr_fwd = np.convolve(signal, t_norm[::-1], mode='valid')
    numerator_fwd = raw_corr_fwd - mean_s * sum_t
    denominator_fwd = std_s * np.sqrt(sum_t2)
    ncc_fwd = numerator_fwd / (denominator_fwd + 1e-12)

    plt.plot(ncc_fwd)
    plt.title("NCC Forward")
    plt.xlabel("Position")
    plt.ylabel("Score")
    plt.show()

    print("NCC max (forward):", np.max(ncc_fwd))
    print("NCC mean (forward):", np.mean(ncc_fwd))
    print("Denominator min/max (fwd):", np.min(denominator_fwd), np.max(denominator_fwd))

    return np.clip(np.max(ncc_fwd), 0, 1) * 100


# Import numpy (was missing in the environment)
import numpy as np

def ncc_match_probability_safe(target: np.ndarray, signal: np.ndarray):
    """
    Compute the probability (0–100%) that `target` exists within `signal`
    using properly normalized cross-correlation (NCC).
    
    Each sliding window of the signal is z-score normalized before computing
    cosine similarity with the normalized target, ensuring NCC stays in [-1, 1].

    Parameters
    ----------
    target : np.ndarray
        1D array of the shorter signal (target).
    signal : np.ndarray
        1D array of the longer signal.

    Returns
    -------
    probability : float
        Similarity score scaled to [0, 100], where 100 means a perfect match.
    best_pos : int
        Starting index in `signal` where the best alignment occurs.
    orientation : str
        Always 'forward' (for consistency with other matchers).
    """
    m, n = len(target), len(signal)
    if m > n:
        raise ValueError("Target must be shorter than signal")

    # Normalize the target to zero mean and unit variance
    t_norm = (target - np.mean(target)) / (np.std(target) + 1e-6)

    max_score = -np.inf
    best_pos = 0

    for i in range(n - m + 1):
        window = signal[i:i + m]
        w_mean = np.mean(window)
        w_std = np.std(window) + 1e-6
        w_norm = (window - w_mean) / w_std

        score = np.dot(t_norm, w_norm) / m  # cosine similarity
        if score > max_score:
            max_score = score
            best_pos = i

    probability = np.clip(max_score, 0.0, 1.0) * 100.0
    return probability, best_pos, "forward"

def compute_dtw(signal1, signal2, radius=1):
    distance, path = fastdtw(signal1, signal2, radius=radius, dist=euclidean)
    return distance, path


import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# def zscore_prob2(best_distance, distances):
#     mean = np.mean(distances)
#     std = np.std(distances) + 1e-8
#     z = (mean - best_distance) / std
#     prob = 100 / (1 + np.exp(-z))  # sigmoid
#     return max(0.0, min(prob, 100.0))  # Clip to valid range


def best_match_confidence(best_distance, distances, dist_cutoff=200):
    mean = np.mean(distances)
    std = np.std(distances) + 1e-8
    z = (mean - best_distance) / std
    prob = (z + 3) / 6 * 100
    prob = np.clip(prob, 0, 100)
    if best_distance > dist_cutoff:
        prob *= 0.5
    return float(prob)

def relative_match_prob(best_distance, distances):
    """
    Returns a confidence score based on:
    - Z-score of best distance (linear scaled)
    - Margin between best and second-best match
    """
    mean = np.mean(distances)
    std = np.std(distances) + 1e-8
    z = (mean - best_distance) / std
    zscore_component = (z + 3) / 6 * 100

    sorted_d = np.sort(distances)
    if len(sorted_d) < 2:
        margin_component = 100.0
    else:
        margin = sorted_d[1] - sorted_d[0]
        margin_ratio = margin / (sorted_d[1] + 1e-8)
        margin_component = margin_ratio * 100

    final_prob = 0.5 * zscore_component + 0.5 * margin_component
    return float(np.clip(final_prob, 0, 100))



def fastdtw_subsequence_match2(target, test_signal, radius=1, stride=10, scale_window=True):
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

        distance, _ = fastdtw(target, window, radius=radius, dist=euclidean)
        distances.append(distance)
        positions.append(start)

    distances = np.array(distances)
    best_idx = np.argmin(distances)
    best_distance = distances[best_idx]
    best_index = positions[best_idx]

    match_probability = relative_match_prob(best_distance, distances)

    return best_distance, best_index, match_probability



def calculate_prob(
    target_path: str,
    signal_path: str,
    num_reads: int = 1,
    read_id: str = None):
    # Open the file in read-only mode. An index will be built or loaded automatically.
    s5 = pyslow5.Open(target_path, 'r')  # :contentReference[oaicite:0]{index=0}
    
    # Iterate sequentially over reads
    recs = []
    it = s5.seq_reads()  # generator over all reads
    for _ in range(num_reads):
        try:
            recs.append(next(it))
        except StopIteration:
            break

    s5_signal = pyslow5.Open(signal_path, 'r')  # :contentReference[oaicite:0]{index=0}

    # Iterate sequentially over reads
    recs_signal = []
    it_signal = s5_signal.seq_reads()  # generator over all reads
    for _ in range(num_reads):
        try:
            recs_signal.append(next(it_signal))
        except StopIteration:
            break
    
    print(target_path)
    print(signal_path)

    for rec, rec_signal in zip(recs, recs_signal):
        target = trim_padding(np.array(rec['signal'], dtype=np.int16))
        signal = np.array(rec_signal['signal'], dtype=np.int16)

        # print(target)
        # print(signal)

        signal = (signal - np.mean(signal)) / np.std(signal)
        target = (target - np.mean(target)) / np.std(target)

        # prob = ncc_match_probability_debug((target), signal)
        # print(f"Match probability: {prob:.2f}%")
        
        # This works!
        # prob, pos, orient = ncc_match_probability_safe((target), signal)
        # print(f"Match probability: {prob:.2f}% at position {pos}, orientation: {orient}")

        # dist, _ = compute_dtw(target, signal)
        # print(f"Fast DTW Dist: {dist:.2f}")

        # prob_dtw, start, end = dtw_banded_match_probability((target), signal)
        # print(f"DTW Match probability: {prob_dtw:.2f}% at position {start}, until: {end}")
        
        # prob_dtw_fast, start_fast, end_fast = fastdtw_match_probability((target), signal)
        best_distance, best_index, match_prob = fastdtw_subsequence_match2(target, signal, radius=2, stride=10)
        print(f"Fast DTW Match probability: {match_prob:.2f}% at position {best_index}, with dist: {best_distance}")
        # print("")


if __name__ == "__main__":
    # simulate matching segments
    # signal = np.concatenate([np.random.randn(500), np.sin(np.linspace(0, 5*np.pi, 100)), np.random.randn(500)])
    # target = np.sin(np.linspace(0, 5*np.pi, 100))

    # # normalize both
    # signal = (signal - np.mean(signal)) / np.std(signal)
    # target = (target - np.mean(target)) / np.std(target)

    # cost, path = fastdtw(target, signal[500:600], dist=euclidean)
    # print("FastDTW cost:", cost)
    # print("Estimated probability:", np.exp(-cost / 100.0) * 100)

    out_path = "/home/ml4320/squigulator-v0.4.0/signal_output/test0/"

    calculate_prob("/home/ml4320/squigulator-v0.4.0/signal_output/random250_40_ideal.blow5",
                   "/home/ml4320/squigulator-v0.4.0/signal_output/random250_ideal.blow5", 10)

    # for signal in os.listdir(out_path):
    #     plot_blow5_signal(out_path + signal, num_reads=5)

    # plot_blow5_signal("/home/ml4320/squigulator-v0.4.0/signal_output/acgt_tripled_ideal.blow5", num_reads=10)
    # create_run_cmds("/home/ml4320/squigulator-v0.4.0/signal_input/test0", True)
