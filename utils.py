import os
import numpy as np


def create_run_cmds(path, is_ideal):
    # /home/ml4320/squigulator-v0.4.0/signal_input/test1
    dir_list = os.listdir(path)
    ideal = ""

    if is_ideal:
        ideal = "_ideal"

    for dir in dir_list:
        print("./squigulator --ideal " + path + "/" + dir + " -o " + "signal_output/test0/" + dir[:-3] + ideal + ".blow5 -n 10")

# ensures each element is a 1-D vector
def to_vector_sequence(arr):
    return [[float(v)] for v in arr]  

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


def count_base_content(sequence):
    base_counts = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
    for base in sequence:
        base = base.upper()
        if base in base_counts:
            base_counts[base] += 1
            # print(base.upper())

    percentage = []
    percentage.append(base_counts['A'] / len(sequence) * 100)
    percentage.append(base_counts['C'] / len(sequence) * 100)
    percentage.append(base_counts['G'] / len(sequence) * 100)
    percentage.append(base_counts['T'] / len(sequence) * 100)

    return base_counts, percentage
