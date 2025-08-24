import pyslow5
import numpy as np
import matplotlib.pyplot as plt

import utils as u

def plot_blow5_signal(
    blow5_path: str,
    read_id: str = None,
    num_reads: int = 1
) -> None:
    
    s5 = pyslow5.Open(blow5_path, 'r')  

    if read_id is not None:
        recs = [s5.read(read_id)]
    else:
        recs = []
        it = s5.seq_reads()
        for _ in range(num_reads):
            try:
                recs.append(next(it))
            except StopIteration:
                break

    for rec in recs:
        raw = np.array(rec['signal'], dtype=np.int16)

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
        current_pA = (raw + offset) * (range_ / digitisation)

        # Build a time axis in seconds
        time_s = np.arange(len(current_pA)) / samp_rate

        # Plot
        plt.figure(figsize=(10, 4))
        plt.plot(time_s, current_pA)
        plt.xlabel('Time (s)')
        plt.ylabel('Current (pA)')
        # plt.title(f"Read ID: {rec['read_id']}")
        plt.tight_layout()
        plt.show()

    s5.close()


def plot_aligned_overlay(
    blow5_path: str,
    blow5_path2: str,
    align_start: int = 0
) -> None:
    
    s5 = pyslow5.Open(blow5_path, 'r')  
    s52 = pyslow5.Open(blow5_path2, 'r')  

    it = s5.seq_reads() 
    rec = (next(it))

    it2 = s52.seq_reads() 
    rec2 = (next(it2))

    raw = np.array(rec['signal'], dtype=np.int16)
    raw2 = np.array(rec2['signal'], dtype=np.int16)

    digitisation = rec['digitisation']
    range_        = rec['range']
    offset        = rec['offset']
    samp_rate     = rec['sampling_rate']

    print(rec['read_id'])
    print("raw signal size: ", len(raw))
    print("samp rate: ", samp_rate)
    print("")

    # Convert to picoamperes: pA = (raw + offset) * (range / digitisation)
    current_pA = (raw + offset) * (range_ / digitisation)

    # Build a time axis in seconds
    time_s = np.arange(len(current_pA)) / samp_rate

    plt.figure(figsize=(10, 4))

    # test signal 2 is shown in full
    plt.plot(range(len(raw2)), raw2, label="Test Signal", linewidth=2)

    # target signal 1 is overlayed starting at align_start
    plt.plot(range(align_start, align_start + len(raw)), raw, label="Target Signal", linestyle='--', linewidth=2)

    plt.title(f"Signal Overlap Starting at Position {align_start}")
    plt.xlabel("Position Index (in the number of samples)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.grid(False)
    plt.show()


    s5.close()