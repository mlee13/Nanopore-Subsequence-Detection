import pyslow5
import numpy as np

import matchers as m
import visualisers as v
import utils as u

from dtaidistance import dtw


def calculate_prob(
    target_path: str,
    signal_path: str,
    num_reads: int = 1,
    read_id: str = None):

    s5 = pyslow5.Open(target_path, 'r')
    s5_signal = pyslow5.Open(signal_path, 'r')
    
    # Iterate sequentially over target reads
    # TODO: add read_id back again
    recs = []
    recs_signal = []

    print(target_path)
    print(signal_path)

    it = s5.seq_reads()  
    for _ in range(num_reads):
        try:
            recs.append(next(it))
        except StopIteration:
            break

    # Iterate sequentially over signal reads
    it_signal = s5_signal.seq_reads()  
    for _ in range(num_reads):
        try:
            recs_signal.append(next(it_signal))
        except StopIteration:
            break

    signal_unnormalised = np.array(recs_signal[0]['signal'], dtype=np.int16)
    signal = (signal_unnormalised - np.mean(signal_unnormalised)) / np.std(signal_unnormalised)

    for rec in recs:
        target_ncc = np.array(rec['signal'], dtype=np.int16)
        target = (target_ncc - np.mean(target_ncc)) / np.std(target_ncc)

        prob, pos = m.euclidean_distance_match((target), signal)
        print(f"Euclidean Match probability: {prob:.2f}% at position {pos}")

        prob, pos = m.ncc_match((target), signal_unnormalised)
        print(f"NCC Match probability: {prob:.2f}% at position {pos}")

        best_distance, best_index, match_prob = m.fastdtw_subsequence_match(target, signal_unnormalised, radius=1, stride=5, scale_window=True)
        print(f"Fast DTW Match probability: {match_prob:.2f}% at position {best_index}, with dist: {best_distance}")


if __name__ == "__main__":
    out_path = "/home/ml4320/squigulator-v0.4.0/signal_output/test0/"
    # TODO change to ask for target and signal paths here instead hard coding
    # In no particular order - comment out the function that is needed 

    # 1. Calculating Probability (main)
    # print("Using ideal target")
    # calculate_prob("tempFYP/signal_output/non-matches/random250_rand40_ideal.blow5",
    #                "tempFYP/test_main/test_signals/random_250.blow5", 1)
    
    print("Using clean target")
    calculate_prob("signal_output/non-matches/random250_non_match_20_2_clean.blow5",
                   "signal_output/test_signals/random250_clean.blow5", 1)
    
    # 2. Plot all signals in directory
    # for signal in os.listdir(out_path):
    #     v.plot_blow5_signal(out_path + signal, num_reads=1)

    # 3. Plot two signals overlayed (with chosen align start position)
    # v.plot_aligned_overlay("tempFYP/signal_output/non-matches/random250_rand40_ideal.blow5",
    #                "tempFYP/test_main/test_signals/random_250.blow5", 5)
    
    # 4. Plot a signal
    # v.plot_blow5_signal("tempFYP/signal_output/test_signals/random250_fullcontig.blow5")

    # 5. To generate Randon sequence 
    # print(u.create_random_sequence(20))

    # 6. Examine Base content 
    # count, percentages = u.count_base_content("ctaggggtcagtctacggcgttggtagcgcactggccgattgcgcaagcctgatgctggatgggcatctgagcagggtttgaccgctgcacggtaggtcatggctggtgacggctcgctgtttcgtgcagcttggcgctttttgatggcgtggtttgtgatggctgttcccatggtggttctgtacagcggctttcggtccgggagggggtttgtcgcctccaggtctcgtgactccgctttttcgtgtg")
    # print(f"Base content: {count} with percentages: {percentages}")

    # 7. Simulate signals using the input fast5 file 
    # u.create_simulated_signal("/home/ml4320/squigulator-v0.4.0/signal_input/test0", True)
