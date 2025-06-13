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
    # TODO: decide if we want to run target on all signal reads or just one
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

    # for rec, rec_signal in zip(recs, recs_signal):
    #     target = u.trim_padding(np.array(rec['signal'], dtype=np.int16))
    #     signal = np.array(rec_signal['signal'], dtype=np.int16)

    #     # print(target)
    #     # print(signal)

    #     # Normalizing - could be extracted in utils
    #     signal = (signal - np.mean(signal)) / np.std(signal)
    #     target = (target - np.mean(target)) / np.std(target)

    #     # prob, pos = m.euclidean_distance_match((target), signal)
    #     # print(f"Euclidean Match probability: {prob:.2f}% at position {pos}")
        
    #     # # This works!
    #     prob, pos = m.ncc_match((target), signal)
    #     print(f"Match probability: {prob:.2f}% at position {pos}")

    #     # prob_dtw, start, end = dtw_banded_match_probability((target), signal)
    #     # print(f"DTW Match probability: {prob_dtw:.2f}% at position {start}, until: {end}")
        
    #     best_distance, best_index, match_prob = m.fastdtw_subsequence_match(target, signal, radius=2, stride=10)
    #     print(f"Fast DTW Match probability: {match_prob:.2f}% at position {best_index}, with dist: {best_distance}")
    #     # print("")

if __name__ == "__main__":
    out_path = "/home/ml4320/squigulator-v0.4.0/signal_output/test0/"
    # could ask for target and signal path here

    # calculate_prob("/Users/yminsele/IdeaProjects/tempFYP/test_main/non_match_signals/long_10000_target_1.blow5",
    #                "/Users/yminsele/IdeaProjects/tempFYP/test_main/test_signals/long_random_10000_clean.blow5", 10)

    # print("Using ideal target")
    # calculate_prob("tempFYP/signal_output/non-matches/random250_rand40_ideal.blow5",
    #                "tempFYP/test_main/test_signals/random_250.blow5", 1)
    
    # print("Using clean target")
    # calculate_prob("signal_output/non-matches/random250_non_match_20_2_clean.blow5",
    #                "signal_output/test_signals/random250_clean.blow5", 1)
    
    # for signal in os.listdir(out_path):
    #     v.plot_blow5_signal(out_path + signal, num_reads=1)

    # v.plot_overlay("tempFYP/signal_output/non-matches/random250_rand40_ideal.blow5",
    #                "tempFYP/test_main/test_signals/random_250.blow5", 5)
    
    v.plot_blow5_signal("tempFYP/test_main/test_signals/random_250_ideal.blow5")
    v.plot_blow5_signal("tempFYP/test_main/test_signals/random_250_ideal_reads.blow5", num_reads=3)

    # v.plot_blow5_signal("tempFYP/signal_output/test_signals/random250_fullcontig.blow5")

    # count, percentages = u.count_base_content("ctaggggtcagtctacggcgttggtagcgcactggccgattgcgcaagcctgatgctggatgggcatctgagcagggtttgaccgctgcacggtaggtcatggctggtgacggctcgctgtttcgtgcagcttggcgctttttgatggcgtggtttgtgatggctgttcccatggtggttctgtacagcggctttcggtccgggagggggtttgtcgcctccaggtctcgtgactccgctttttcgtgtg")
    # print(f"Base content: {count} with percentages: {percentages}")

    # u.create_run_cmds("/home/ml4320/squigulator-v0.4.0/signal_input/test0", True)

    # print(u.create_random_sequence(20))