import pyslow5
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from scipy.stats import linregress
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import pywt
import matchers as m
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


def zscore_prob(best_distance, distances):
    mean = np.mean(distances)
    std = np.std(distances) + 1e-8
    z = (mean - best_distance) / std
    return float(100 / (1 + np.exp(-z)))  # sigmoid-scaled probability


def dtw_banded_match_probability(target: np.ndarray, signal: np.ndarray, scale=600.0, band: int = None):
    """
    Estimate the probability that `target` exists inside `signal` using Sakoe-Chiba banded DTW.

    Parameters
    ----------
    scale : float
        Controls sharpness of exponential decay for probability scaling.
    band : int
        Sakoe-Chiba band width. If None, defaults to 10% of target length.
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


# low percent for matches ~70% and lower percent for non-matches 40~50%
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


# ensures each element is a 1-D vector
def to_vector_sequence(arr):
    return [[float(v)] for v in arr]  


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

        target_seq = to_vector_sequence(target)
        window_seq = to_vector_sequence(window)
        distance, _ = fastdtw(target_seq, window_seq, radius=radius, dist=euclidean)


        # distance, _ = fastdtw(target, window, radius=radius, dist=euclidean)
        distances.append(distance)
        positions.append(start)

    distances = np.array(distances)
    best_idx = np.argmin(distances)
    best_distance = distances[best_idx]
    best_index = positions[best_idx]

    match_probability = relative_match_prob(best_distance, distances)

    return best_distance, best_index, match_probability

def dtw_windowed_presence_prob(
    target: np.ndarray,
    test_signal: np.ndarray,
    stretch: float = 1.5,
    stride_factor: float = 0.2,
    max_cost: float = 0.9,
    scale: bool = True
) -> float:
    """
    Return the probability (0–100%) that `target` exists somewhere in `test_signal`
    by sliding a window L = stretch*len(target) and computing a windowed‐band DTW score.

    Args:
      target: 1D array, your template signal.
      test_signal: 1D array, the longer signal to search.
      stretch: max expected dilation factor (e.g. 1.5 = allow 50% stretching).
      stride_factor: fraction of window size to slide each step (e.g. 0.2 = 20%).
      max_cost: per‐sample DTW cost that maps to 0% probability. Tune on non‐matches.
      scale: whether to z‐normalize each window + target.

    Returns:
      prob: float in [0,100], high means “target likely present.”
    """
    N = len(target)
    L = int(np.ceil(N * stretch))                        # window length
    stride = max(1, int(L * stride_factor))              # how far to slide
    band_radius = int(np.floor((stretch - 1) / 2 * N))    # Sakoe–Chiba band half‐width

    # Pre‐normalize target once if desired
    if scale:
        target = (target - target.mean()) / (target.std() + 1e-8)
    t_seq = to_vector_sequence(target)

    best_prob = 0.0
    for start in range(0, len(test_signal) - L + 1, stride):
        window = test_signal[start : start + L]
        if scale:
            window = (window - window.mean()) / (window.std() + 1e-8)
        w_seq = to_vector_sequence(window)

        # FastDTW with a constrained band (Sakoe–Chiba)
        dist, _ = fastdtw(t_seq, w_seq, radius=band_radius, dist=euclidean)

        # per‐point cost and absolute similarity → 0–100%
        per_pt = dist / N
        score = max(0.0, 1 - per_pt / max_cost) * 100
        best_prob = max(best_prob, score)

    return best_prob


def calibrate_max_cost(target, nonmatch_signals, stretch=1.5, stride_factor=0.2, radius=None, scale=True):
    N = len(target)
    L = int(np.ceil(N * stretch))
    stride = max(1, int(L * stride_factor))
    if radius is None:
        radius = int(np.floor((stretch - 1) / 2 * N))

    if scale:
        target = (target - np.mean(target)) / (np.std(target) + 1e-8)
    t_seq = to_vector_sequence(target)

    all_per_point_costs = []

    nonmatch_signals_extracted = map(lambda x: np.array(x['signal'], dtype=np.int16), nonmatch_signals)

    for signal in nonmatch_signals_extracted:
        for start in range(0, len(signal) - L + 1, stride):
            window = signal[start:start + L]
            if scale:
                window = (window - np.mean(window)) / (np.std(window) + 1e-8)
            w_seq = to_vector_sequence(window)

            dist, _ = fastdtw(t_seq, w_seq, radius=radius, dist=euclidean)
            per_point = dist / N
            all_per_point_costs.append(per_point)

    max_cost_95 = np.percentile(all_per_point_costs, 95)
    return max_cost_95, all_per_point_costs

import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def probability_of_target_in_test_dtw(target_signal, test_signal, radius=1, num_noise_samples=10, noise_std_factor=0.5):
    """
    Calculates the probability of a target signal existing within a test signal using FastDTW.
    Compares the minimum DTW distance to the distribution of distances with random noise.

    Args:
        target_signal (np.ndarray): 1D numpy array representing the target signal.
        test_signal (np.ndarray): 1D numpy array representing the test signal.
        radius (int): Radius parameter for FastDTW.
        num_noise_samples (int): Number of random noise signals to generate for comparison.
        noise_std_factor (float): Factor to scale the standard deviation of the noise
                                   relative to the test signal's standard deviation.

    Returns:
        float: Probability (between 0 and 1) that the target signal exists within the test signal.
    """
    n_target = len(target_signal)
    n_test = len(test_signal)

    if n_target > n_test:
        return 0.0

    min_dist = float('inf')
    all_distances = []

    for i in range(n_test - n_target + 1):
        segment = test_signal[i : i + n_target]

        target_signal1_1d = to_vector_sequence(target_signal)
        segment_1d = to_vector_sequence(segment)

        distance, _ = fastdtw(target_signal1_1d, segment_1d, dist=euclidean, radius=radius)
        all_distances.append(distance)
        min_dist = min(min_dist, distance)

    all_distances = np.array(all_distances)

    # Generate DTW distances with random noise
    noise_distances = []
    test_std = np.std(test_signal)
    for _ in range(num_noise_samples):
        noise = np.random.normal(0, noise_std_factor * test_std, n_target)
        
        target_signal_1d = to_vector_sequence(target_signal)
        noise_1d = to_vector_sequence(noise)

        dist_noise, _ = fastdtw(target_signal_1d, noise_1d, dist=euclidean, radius=radius)
        noise_distances.append(dist_noise)
    noise_distances = np.array(noise_distances)

    # Calculate the probability based on how much smaller the minimum distance is
    # compared to the noise distances.
    if np.std(noise_distances) < 1e-6: # Avoid division by zero if noise distances are all the same
        if min_dist < np.mean(noise_distances):
            probability = 0.9
        else:
            probability = 0.1
    else:
        # Calculate a z-score-like value: how many standard deviations below the mean
        # of the noise distances is our minimum distance?
        z_score = (np.mean(noise_distances) - min_dist) / np.std(noise_distances)

        # Use a sigmoid function to map the z-score to a probability between 0 and 1
        probability = 1 / (1 + np.exp(-z_score))

    return np.clip(probability, 0.0, 1.0)


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

    # max_cost_95, all_per_p_costs = calibrate_max_cost(trim_padding(np.array(recs[0]['signal'], dtype=np.int16)), recs_signal)
    # print(f"max cost 95: {max_cost_95}, all per point costs: {all_per_p_costs} ")


    for rec, rec_signal in zip(recs, recs_signal):
        target = trim_padding(np.array(rec['signal'], dtype=np.int16))
        signal = np.array(rec_signal['signal'], dtype=np.int16)

        # print(target)
        # print(signal)

        signal = (signal - np.mean(signal)) / np.std(signal)
        target = (target - np.mean(target)) / np.std(target)

        # prob = ncc_match_probability_debug((target), signal)
        # print(f"Match probability: {prob:.2f}%")

        # prob, pos = euclidean_distance_score((target), signal)
        # print(f"Euclidean Match probability: {prob:.2f}% at position {pos}")
        
        # This works!
        # prob, pos = ncc_match_probability_safe((target), signal)
        # print(f"Match probability: {prob:.2f}% at position {pos}")

        # all probs between 50-60% 
        # prob_dtw, start, end = dtw_banded_match_probability((target), signal)
        # print(f"DTW Match probability: {prob_dtw:.2f}% at position {start}, until: {end}")
        
        # final = probability_of_target_in_test_dtw(target, signal)
        # print(f"NEW FastDTW Match probability: {final:.2f}% ")

        best_distance, best_index, match_prob = fastdtw_subsequence_match2(target, signal, radius=2, stride=10)
        print(f"Fast DTW Match probability: {match_prob:.2f}% at position {best_index}, with dist: {best_distance}")
        # print("")


if __name__ == "__main__":

    # Data from the table
    lengths = [1000, 5000, 10000, 20000, 30000]
    dtw_match = [97.4, 74.1, 68.5, 64.8, 62.5]
    dtw_nonmatch = [71.5, 60.5, 57.2, 55.8, 55.1]
    gaps = [m - n for m, n in zip(dtw_match, dtw_nonmatch)]

    import matplotlib.pyplot as plt

    # Plot both match and non-match as lines
    plt.figure(figsize=(10, 6))
    plt.plot(lengths, dtw_match, marker='o', label='Match', color='tab:blue')
    plt.plot(lengths, dtw_nonmatch, marker='o', label='Non-Match', color='tab:orange')

    # Add match and non-match score annotations
    for x, match_y, nonmatch_y in zip(lengths, dtw_match, dtw_nonmatch):
        plt.text(x, match_y + 1, f'{match_y:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='tab:blue')
        plt.text(x, nonmatch_y - 1, f'{nonmatch_y:.1f}', ha='center', va='top', fontsize=9, fontweight='bold', color='tab:orange')

    # Highlight gaps with vertical lines and annotate them
    for x, y1, y2 in zip(lengths, dtw_match, dtw_nonmatch):
        plt.vlines(x, y2, y1, color='gray', linestyle='dashed', alpha=0.7)
        gap = y1 - y2
        plt.text(x, (y1 + y2) / 2, f'{gap:.1f}', ha='center', va='center', fontsize=10, color='black', backgroundcolor='white')

    # Axis and layout
    plt.xlabel('Test Signal Length (samples)')
    plt.ylabel('Mean DTW Score')
    plt.title('Average DTW Scores & Match–Non-Match Gap Trends with Test Signal Length')
    plt.xticks(lengths, [f'{l:,}' for l in lengths])  # Custom x-tick labels with commas
    plt.grid(True, color='lightgray')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # simulate matching segments
    # signal = np.concatenate([np.random.randn(500), np.sin(np.linspace(0, 5*np.pi, 100)), np.random.randn(500)])
    # target = np.sin(np.linspace(0, 5*np.pi, 100))

    # # normalize both
    # signal = (signal - np.mean(signal)) / np.std(signal)
    # target = (target - np.mean(target)) / np.std(target)

    # cost, path = fastdtw(target, signal[500:600], dist=euclidean)
    # print("FastDTW cost:", cost)
    # print("Estimated probability:", np.exp(-cost / 100.0) * 100)

    # out_path = "/home/ml4320/squigulator-v0.4.0/signal_output/test0/"

    # calculate_prob("/Users/yminsele/IdeaProjects/tempFYP/signal_output/random250_rand40_ideal.blow5",
    #                "/Users/yminsele/IdeaProjects/tempFYP/signal_output/random250_ideal.blow5", 10)

    # calculate_prob("/Users/yminsele/IdeaProjects/tempFYP/signal_output/acgt_five_50_ideal.blow5",
                #    "/Users/yminsele/IdeaProjects/tempFYP/signal_output/acgt_five_ideal.blow5", 10)

    # for signal in os.listdir(out_path):
    #     plot_blow5_signal(out_path + signal, num_reads=5)

    # plot_blow5_signal("/home/ml4320/squigulator-v0.4.0/signal_output/acgt_tripled_ideal.blow5", num_reads=10)
    # create_run_cmds("/home/ml4320/squigulator-v0.4.0/signal_input/test0", True)
