#!/usr/bin/env python3
"""
BeiDou B1C Acquisition Test using an IQ binary file
Reads complex IQ samples from a bin file and runs B1C acquisition.
"""

import argparse
import numpy as np

from beidou_b1c_signal_generator import BeiDouB1CSignalGenerator
from beidou_b1c_acquisition import BeiDouB1CAcquisition


def main():
    parser = argparse.ArgumentParser(description="BeiDou B1C acquisition from IQ file")
    parser.add_argument("--file", required=True, help="Path to IQ binary file (interleaved I/Q)")
    parser.add_argument("--format", choices=["8", "16"], default="16", help="Bit depth of file (8 or 16)")
    parser.add_argument("--fs", type=float, default=3.2e6, help="Sampling frequency in Hz (default 3.2e6)")
    parser.add_argument("--if", dest="intermediate_freq", type=float, default=0.0, help="Intermediate frequency in Hz (default 0)")
    parser.add_argument("--ms", type=float, default=50.0, help="Milliseconds of data to load (default 50 ms)")
    parser.add_argument("--threshold", type=float, default=2.0, help="Acquisition threshold (default 2.0)")
    parser.add_argument("--search-band-khz", type=float, default=14.0, help="Search band around IF in kHz (default 14)")
    parser.add_argument("--prn-max", type=int, default=50, help="Max PRN to search (default 50)")
    args = parser.parse_args()

    gen = BeiDouB1CSignalGenerator(sampling_freq=args.fs, IF=args.intermediate_freq, duration=10)

    num_samples = int(args.fs * (args.ms / 1000.0))
    signal = gen.read_bin_file(args.file, format_bits=int(args.format), max_samples=num_samples)

    acq = BeiDouB1CAcquisition(
        sampling_freq=args.fs,
        IF=args.intermediate_freq,
        acq_satellite_list=list(range(1, args.prn_max + 1)),
        acq_search_band=args.search_band_khz,
        acq_threshold=args.threshold,
    )

    sat_peaks, detected_prns = acq.process_samples(signal)
    acq_results = acq.acquisition(signal, args.threshold)

    print("\n" + "=" * 50)
    print("BEIDOU B1C ACQUISITION RESULTS")
    print("=" * 50)
    if detected_prns:
        print("Detected PRNs:", detected_prns)
        for prn in detected_prns:
            idx = prn - 1
            print(f"PRN {prn:2d}: peak={sat_peaks[idx]:.2f} carr={acq_results['carrFreq'][idx]:.1f} Hz phase={int(acq_results['codePhase'][idx])}")
    else:
        print("No satellites detected. Try increasing ms, lowering threshold, or adjusting IF/search band.")


if __name__ == "__main__":
    main()


