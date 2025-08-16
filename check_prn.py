#!/usr/bin/env python3
"""
Quick PRN Checker for gpssim.bin
Simple script to check PRN codes from gpssim.bin file with configurable parameters
"""

import numpy as np
from gps_acquisition import GPSAcquisition
from gps_signal_generator import GPSSignalGenerator

def check_prn_from_file(filename="gpssim.bin", 
                       duration_ms=50, 
                       threshold=1.5, 
                       sampling_freq=2.6e6,
                       format_bits=16):
    """
    Check PRN codes from a GPS signal file
    
    Args:
        filename (str): Path to the GPS signal file
        duration_ms (int): Duration to read in milliseconds
        threshold (float): Acquisition threshold
        sampling_freq (float): Sampling frequency in Hz
        format_bits (int): Bit depth of the file (8 or 16)
    """
    print(f"Checking PRN codes from {filename}")
    print(f"Duration: {duration_ms}ms, Threshold: {threshold}")
    print("=" * 50)
    
    # Create generator and acquisition objects
    generator = GPSSignalGenerator(sampling_freq=sampling_freq)
    generator.acquisition.acq_threshold = threshold
    
    try:
        # Read signal data
        acq_samples = int(sampling_freq * duration_ms / 1000)
        signal = generator.read_bin_file(filename, format_bits=format_bits, max_samples=acq_samples)
        
        # Perform acquisition
        sat_peaks, detected_prns = generator.acquisition.process_samples(signal)
        
        # Display results
        if detected_prns:
            print(f"✅ Detected {len(detected_prns)} satellites:")
            for prn in detected_prns:
                peak_idx = prn - 1
                peak_value = sat_peaks[peak_idx]
                print(f"  PRN {prn:2d}: Peak = {peak_value:.2f}")
        else:
            print("❌ No satellites detected")
            
            # Show top 3 peaks for debugging
            peak_indices = np.argsort(sat_peaks)[::-1][:3]
            print("Top 3 peaks:")
            for i, peak_idx in enumerate(peak_indices):
                prn = peak_idx + 1
                peak_value = sat_peaks[peak_idx]
                print(f"  {i+1}. PRN {prn:2d}: {peak_value:.2f}")
        
        return detected_prns, sat_peaks
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return [], []

if __name__ == "__main__":
    # Quick check with default parameters
    detected_prns, peaks = check_prn_from_file()
    
    # You can also call with different parameters:
    # detected_prns, peaks = check_prn_from_file(duration_ms=100, threshold=1.0)
    # detected_prns, peaks = check_prn_from_file(duration_ms=20, threshold=2.0)
