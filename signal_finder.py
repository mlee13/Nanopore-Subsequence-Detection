import pyslow5
import numpy as np

import matchers as m
import visualisers as v
import utils as u


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

    signal = np.array(recs_signal[0]['signal'], dtype=np.int16)
    signal = (signal - np.mean(signal)) / np.std(signal)

    for rec in recs:
        target = u.trim_padding(np.array(rec['signal'], dtype=np.int16))
        target = (target - np.mean(target)) / np.std(target)

        prob, pos = m.ncc_match((target), signal)
        print(f"NCC Match probability: {prob:.2f}% at position {pos}")

        best_distance, best_index, match_prob = m.fastdtw_subsequence_match(target, signal, radius=2, stride=10)
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

    # calculate_prob("/Users/yminsele/IdeaProjects/tempFYP/signal_output/random250_40_ideal.blow5",
    #                "/Users/yminsele/IdeaProjects/tempFYP/signal_output/random250_fullcontig.blow5", 10)

    # calculate_prob("/Users/yminsele/IdeaProjects/tempFYP/signal_output/acgt_five_50_ideal.blow5",
    #                "/Users/yminsele/IdeaProjects/tempFYP/signal_output/acgt_five_fullcontig.blow5", 10)
    
    # for signal in os.listdir(out_path):
    #     v.plot_blow5_signal(out_path + signal, num_reads=1)

    # v.plot_blow5_signal("/Users/yminsele/IdeaProjects/tempFYP/signal_output/kmer_test.blow5", num_reads=10)

    count, percentages = u.count_base_content("ctaggggtcagtctacggcgttggtagcgcactggccgattgcgcaagcctgatgctggatgggcatctgagcagggtttgaccgctgcacggtaggtcatggctggtgacggctcgctgtttcgtgcagcttggcgctttttgatggcgtggtttgtgatggctgttcccatggtggttctgtacagcggctttcggtccgggagggggtttgtcgcctccaggtctcgtgactccgctttttcgtgtg")
    print(f"Base content: {count} with percentages: {percentages}")

    # u.create_run_cmds("/home/ml4320/squigulator-v0.4.0/signal_input/test0", True)
