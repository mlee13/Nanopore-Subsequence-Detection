import random
import subprocess
import os
import csv
import pyslow5
import numpy as np
import utils as u
import matchers as m
from Bio import SeqIO


# === VARIABLES ===
squigulator_path = "./squigulator"  # Path to Squigulator binary

# Target details
match_input_seq_dir = "test_main/match_sequences"
match_output_signal_dir = "test_main/match_signals"

non_match_input_seq_dir = "test_main/non_match_sequences"
non_match_output_signal_dir = "test_main/non_match_signals"


# ===== CONFIGS ===== should modify each run!!!
num_targets = 1
target_min_length = 65
target_max_length = 65
target_bias_length = 0
match_mode = "non-match" # Options are "match" or anything else is non-match

curr_last_target_idx = 37 # last idx of target in match/non-match directory

# Test signal details
test_sequence_path = "signal_input/test_signals/random250.fa"
test_signal_path = "signal_output/test_signals/random250_clean.blow5"

# Results
results_csv = f"test_main/results/random250_non_match_scores.csv"
# results_csv = f"test_main/results/match_scores_{curr_last_target_idx + 1}-{curr_last_target_idx + num_targets}.csv"

#====================

def read_fasta_sequence(filepath):
    record = next(SeqIO.parse(filepath, "fasta"))
    return str(record.seq)

# === FUNCTION TO GENERATE MATCHING DNA SEQUENCE ===
def generate_matching_subsequence(length, test_seq):
    if length > len(test_seq):
        raise ValueError("Requested length is longer than the test sequence.")

    start_index = random.randint(0, len(test_seq) - length)

    while (start_index + length >= len(test_seq)):
        start_index = random.randint(0, len(test_seq) - length)

    return test_seq[start_index:start_index + length]

# === FUNCTION TO GENERATE RANDOM DNA SEQUENCE ===
def generate_non_matching_subsequence(length, test_seq):
    if length > len(test_seq):
        raise ValueError("Requested length is longer than the test sequence.")
    
    bases = ['A', 'C', 'G', 'T']
    seq = ''.join(random.choices(bases, k=length))

    while (seq in test_seq):
        seq = ''.join(random.choices(bases, k=length))

    return seq

# === FUNCTION TO WRITE FASTA FILE ===
def write_fasta(sequence, file_path, id=-1, filename=""):
    if id != -1:
        filename = "TARGET_{id}"

    with open(file_path, 'w') as f:
        f.write(f">{filename}\n")
        f.write(sequence + "\n")

# === FUNCTION TO CALL SQUIGULATOR ===
def run_squigulator(input_path, output_path, ideal="ideal"):
    command = [
        squigulator_path,
        f"--{ideal}",
        "--full-contig",
        input_path,
        "-o", output_path
    ]
    subprocess.run(command, check=True)

def compute_match_probability(algoCode, target_path, signal_path):
    s5_target = pyslow5.Open(target_path, 'r')
    s5_test_signal = pyslow5.Open(signal_path, 'r')

    target_record = next(s5_target.seq_reads())
    test_signal_record = next(s5_test_signal.seq_reads())

    test_signal =  np.array(test_signal_record['signal'], dtype=np.int16)
    test_signal = (test_signal - np.mean(test_signal)) / np.std(test_signal)
    
    target_ncc = np.array(target_record['signal'], dtype=np.int16)
    target = (target_ncc - np.mean(target_ncc)) / np.std(target_ncc)

    algoCode = algoCode.upper()

    if algoCode == "EUC":
        prob, best_idx = m.euclidean_distance_match(target, test_signal)
        # print(f"Euclidean Match probability: {prob:.2f}% at position {pos}")
    elif algoCode == "NCC":
        prob, best_idx = m.ncc_match(target_ncc, test_signal)
        # print(f"NCC Match probability: {prob:.2f}% at position {pos}")
    elif algoCode == "DTW":
        best_distance, best_idx, prob = m.fastdtw_subsequence_match(target, test_signal, radius=2, stride=10)
        # print(f"Fast DTW Match probability: {match_prob:.2f}% at position {best_index}, with dist: {best_distance}")
    else:
        print("HUH")

    return prob, best_idx

def get_match_outcome(ncc_score, dtw_score):
    if (ncc_score >= 80 or dtw_score >= 65 or 
        (ncc_score >= 70 and dtw_score >= 60)):
        return "match"
    
    if (dtw_score > 55 and dtw_score < 65):
        return "undefined"

    return "non-match"

def biased_random(min_val, max_val, bias, strength=3):
    if not (min_val <= bias <= max_val):
        raise ValueError("Bias must be within min and max.")

    bias_normalized = (bias - min_val) / (max_val - min_val)
    u = random.random()

    if u < bias_normalized:
        u = (u / bias_normalized) ** strength * bias_normalized
    else:
        u = 1 - ((1 - u) / (1 - bias_normalized)) ** strength * (1 - bias_normalized)

    # Scale to [min, max] and convert to integer
    value = int(round(min_val + u * (max_val - min_val)))
    return min(max(value, min_val), max_val)  # Clamp in case of rounding edge cases


def run_experiment():
    if (match_mode == "match"):
        input_seq_dir = match_input_seq_dir
        output_signal_dir = match_output_signal_dir
    else:
        input_seq_dir = non_match_input_seq_dir
        output_signal_dir = non_match_output_signal_dir

    os.makedirs(input_seq_dir, exist_ok=True)
    os.makedirs(output_signal_dir, exist_ok=True)
    csv_file_exists = os.path.exists(results_csv)

    # Create csv file to store results
    with open(results_csv, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write header only if file is new
        if not csv_file_exists:
            writer.writerow(["RunID", "Filename", "TargetLength",
                            "EucScore", "NCCScore", "DTWScore",
                            "EucAlign", "NCCAlign", "DTWAlign",
                            "ExpectedLabel", "ActualLabel"])

        # Iterate for number of targets we want to create
        for i in range(curr_last_target_idx + 1, curr_last_target_idx + num_targets + 1):

            if target_bias_length != 0:
                target_length = biased_random(target_min_length, target_max_length, target_bias_length)
            else:
                target_length = random.randint(target_min_length, target_max_length)

            # Create Target Sequence
            test_seq = read_fasta_sequence(test_sequence_path)

            if (match_mode == "match"):
                target_seq = generate_matching_subsequence(target_length, test_seq)
            else:
                target_seq = generate_non_matching_subsequence(target_length, test_seq)
            
            # Initialise name for files (target)
            filename = f"target_{i}.fa"
            fasta_path = os.path.join(input_seq_dir, filename)
            signal_path = os.path.join(output_signal_dir, f"target_{i}.blow5")

            # Create fasta, run squigulator on it
            write_fasta(target_seq, fasta_path, id=i)
            run_squigulator(fasta_path, signal_path)

            # Get match info and write to results file
            eucScore, eucAlign = compute_match_probability("EUC", signal_path, test_signal_path)
            NCCScore, NCCAlign = compute_match_probability("NCC", signal_path, test_signal_path)
            DTWScore, DTWAlign = compute_match_probability("DTW", signal_path, test_signal_path)

            match_outcome = get_match_outcome(NCCScore, DTWScore)
            
            writer.writerow([i, filename, target_length, eucScore, NCCScore, DTWScore,
                             eucAlign, NCCAlign, DTWAlign, match_mode, match_outcome])
            
            print(f"[{i}] {filename} | Target Length: {target_length} | "
                f"Euclidean Score: {eucScore:.2f}, NCC Score: {NCCScore:.2f}, DTW Score: {DTWScore:.2f} | "
                f"Euclidean Align: {eucAlign}, NCC Align: {NCCAlign}, DTW Align: {DTWAlign} | "
                f"Mode: {match_mode} | Outcome: {match_outcome}")
        
        print(f"DONE! Created {num_targets} targets in {match_mode}.")
        print(f"See results at: {results_csv}")


def generate_test_signal(length, filename, filedir_seq, filedir_signal):
    # Create random sequence
    signal_seq = u.create_random_sequence(length)
    
    # Initialise name for files (target)
    filename = f"{filename}.fa"
    fasta_path = os.path.join(filedir_seq, filename)
    signal_path = os.path.join(filedir_signal, f"{filename}_clean.blow5")

    # Create fasta, run squigulator on it
    write_fasta(signal_seq, fasta_path, filename=filename)
    run_squigulator(fasta_path, signal_path, ideal="ideal-amp")


# === MAIN PIPELINE ===
if __name__ == "__main__":

    confirm_idx = input("Please confirm that you have updated the curr_last_target_idx. " \
    f"\nIs {curr_last_target_idx} still the correct last target index for {match_mode}?: (y/n)")

    confirm_result_csv = input("\nPlease confirm the location to save the results. " \
    f"\nDo you want to save the {match_mode} results at {results_csv}?: (y/n)")

    if ((confirm_idx.lower() == 'y' or confirm_idx.lower() == 'yes') 
        and (confirm_result_csv.lower() == 'y' or confirm_result_csv.lower() == 'yes')):
        run_experiment()

    # generate_test_signal(3000)