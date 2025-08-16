#!/usr/bin/env python3
"""
GPS Detection Validation Script
Compares expected satellites from gps-sdr-sim output with actual detection results
"""

import numpy as np
from gps_acquisition import GPSAcquisition
from gps_signal_generator import GPSSignalGenerator

def validate_detection(expected_prns, filename="gpssim.bin", threshold=3.0):
    """
    Validate detection against expected PRN list
    
    Args:
        expected_prns (list): List of expected PRN numbers
        filename (str): GPS signal file path
        threshold (float): Acquisition threshold
    """
    print(f"Validating detection against expected PRNs: {expected_prns}")
    print("=" * 60)
    
    # Create acquisition objects
    generator = GPSSignalGenerator(sampling_freq=2.6e6)
    generator.acquisition.acq_threshold = threshold
    
    try:
        # Read signal data (50ms)
        acq_samples = int(generator.sampling_freq * 0.050)
        signal = generator.read_bin_file(filename, format_bits=16, max_samples=acq_samples)
        
        # Perform acquisition
        sat_peaks, detected_prns = generator.acquisition.process_samples(signal)
        
        # Analysis
        print(f"Expected PRNs: {sorted(expected_prns)}")
        print(f"Detected PRNs: {sorted(detected_prns)}")
        print(f"Threshold used: {threshold}")
        print()
        
        # Check for correct detections
        correct_detections = [prn for prn in detected_prns if prn in expected_prns]
        missed_detections = [prn for prn in expected_prns if prn not in detected_prns]
        false_positives = [prn for prn in detected_prns if prn not in expected_prns]
        
        # Display results
        print("DETECTION ANALYSIS:")
        print("-" * 30)
        print(f"✅ Correct detections ({len(correct_detections)}): {sorted(correct_detections)}")
        
        if missed_detections:
            print(f"❌ Missed detections ({len(missed_detections)}): {sorted(missed_detections)}")
        else:
            print("✅ No missed detections")
            
        if false_positives:
            print(f"⚠️  False positives ({len(false_positives)}): {sorted(false_positives)}")
        else:
            print("✅ No false positives")
        
        # Calculate accuracy metrics
        total_expected = len(expected_prns)
        total_detected = len(detected_prns)
        correct_count = len(correct_detections)
        
        if total_expected > 0:
            detection_rate = correct_count / total_expected * 100
            precision = correct_count / total_detected * 100 if total_detected > 0 else 0
            print()
            print("ACCURACY METRICS:")
            print("-" * 20)
            print(f"Detection Rate: {detection_rate:.1f}% ({correct_count}/{total_expected})")
            print(f"Precision: {precision:.1f}% ({correct_count}/{total_detected})")
            
            if detection_rate == 100 and precision == 100:
                print("🎯 PERFECT DETECTION!")
            elif detection_rate >= 90 and precision >= 90:
                print("✅ EXCELLENT DETECTION!")
            elif detection_rate >= 80 and precision >= 80:
                print("👍 GOOD DETECTION!")
            else:
                print("⚠️  NEEDS IMPROVEMENT")
        
        # Show peak values for detected satellites
        print()
        print("PEAK VALUES FOR DETECTED SATELLITES:")
        print("-" * 40)
        for prn in sorted(detected_prns):
            peak_idx = prn - 1
            peak_value = sat_peaks[peak_idx]
            status = "✅" if prn in expected_prns else "⚠️"
            print(f"  PRN {prn:2d}: {peak_value:.2f} {status}")
        
        return correct_detections, missed_detections, false_positives
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return [], [], []

if __name__ == "__main__":
    # Expected PRNs from your gps-sdr-sim output
    expected_prns = [5, 10, 12, 13, 14, 15, 18, 20, 23, 24, 28]
    
    # Test with different thresholds
    thresholds = [2.0, 2.5, 3.0, 3.5, 4.0]
    
    for threshold in thresholds:
        print(f"\n{'='*60}")
        print(f"TESTING WITH THRESHOLD = {threshold}")
        print(f"{'='*60}")
        validate_detection(expected_prns, threshold=threshold)
        print()
