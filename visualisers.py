import pyslow5
import numpy as np
import matplotlib.pyplot as plt

import utils as u

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