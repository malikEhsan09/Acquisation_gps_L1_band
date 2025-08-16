#!/usr/bin/env python3
"""
GPS Signal Acquisition Test using gpssim.bin
Reads GPS signals from gpssim.bin file and tests acquisition with clear PRN code display
"""

import numpy as np
from gps_acquisition import GPSAcquisition
from gps_signal_generator import GPSSignalGenerator

def main():
    print("GPS Signal Acquisition Test using gpssim.bin")
    print("=" * 50)
    
    # Create signal generator for reading files
    print("\n1. Creating GPS Signal Generator...")
    generator = GPSSignalGenerator(
        sampling_freq=2.6e6,  # 2.6 MHz sampling rate
        duration=5,           # 5 seconds duration
        output_bits=16        # 16-bit output
    )
    
    # Lower the acquisition threshold for better detection
    generator.acquisition.acq_threshold = 3.0  # Increased from 1.5 to reduce false positives
    
    # Read GPS signal from gpssim.bin file
    print("\n2. Reading GPS signal from gpssim.bin...")
    try:
        # Read more data for better acquisition (50ms instead of 11ms)
        acq_samples = int(generator.sampling_freq * 0.050)  # 50ms of data
        signal = generator.read_bin_file("gpssim.bin", format_bits=16, max_samples=acq_samples)
        
        print(f"   Successfully loaded {len(signal)} samples")
        print(f"   Signal duration: {len(signal) / generator.sampling_freq:.3f} seconds")
        
    except FileNotFoundError:
        print("❌ ERROR: gpssim.bin file not found!")
        print("   Please ensure gpssim.bin is in the current directory")
        return
    except Exception as e:
        print(f"❌ ERROR reading file: {e}")
        return
    
    # Test acquisition on the loaded signal
    print("\n3. Testing acquisition on gpssim.bin signal...")
    sat_peaks, detected_prns = generator.acquisition.process_samples(signal)
    
    # Display results clearly
    print("\n" + "=" * 50)
    print("ACQUISITION RESULTS FROM gpssim.bin")
    print("=" * 50)
    
    if detected_prns:
        print(f"✅ SUCCESS: Detected {len(detected_prns)} satellites!")
        print(f"\nDetected PRN codes:")
        for prn in detected_prns:
            peak_idx = prn - 1
            peak_value = sat_peaks[peak_idx]
            print(f"  PRN {prn:2d}: Peak = {peak_value:.2f}")
        
        # Show all peak metrics for reference
        print(f"\nAll PRN peak metrics (threshold: {generator.acquisition.acq_threshold:.1f}):")
        for prn in range(1, 33):
            peak_idx = prn - 1
            peak_value = sat_peaks[peak_idx]
            if peak_value > generator.acquisition.acq_threshold:
                print(f"  PRN {prn:2d}: {peak_value:.2f} ✅")
            else:
                print(f"  PRN {prn:2d}: {peak_value:.2f}")
        
    else:
        print("❌ FAILED: No satellites detected")
        print("   This might indicate:")
        print("   - Signal too weak")
        print("   - Acquisition threshold too high")
        print("   - File format mismatch")
        print("   - No GPS signals in the file")
        
        # Show top 5 peak metrics for debugging
        print(f"\nTop 5 PRN peak metrics:")
        peak_indices = np.argsort(sat_peaks)[::-1][:5]
        for i, peak_idx in enumerate(peak_indices):
            prn = peak_idx + 1
            peak_value = sat_peaks[peak_idx]
            print(f"  {i+1}. PRN {prn:2d}: {peak_value:.2f}")
    
    # Show acquisition parameters
    print("\n" + "=" * 50)
    print("ACQUISITION PARAMETERS")
    print("=" * 50)
    print(f"Sampling frequency: {generator.sampling_freq / 1e6:.1f} MHz")
    print(f"Acquisition threshold: {generator.acquisition.acq_threshold}")
    print(f"Search band: ±{generator.acquisition.acq_search_band} kHz")
    print(f"Satellites searched: {len(generator.acquisition.acq_satellite_list)}")
    print(f"Signal duration used: {len(signal) / generator.sampling_freq * 1000:.1f} ms")
    
    print("\n" + "=" * 50)
    print("Test completed!")
    print("=" * 50)

if __name__ == "__main__":
    main()
