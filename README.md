# GPS Signal Acquisition and Generation

Clean Python implementation of GPS signal acquisition and generation, converted from MATLAB code.

## Files

- **`gps_acquisition.py`** - GPS signal acquisition implementation
- **`gps_signal_generator.py`** - GPS signal generation and SDR file creation
- **`test_gps.py`** - Test script that reads from gpssim.bin and shows PRN codes
- **`check_prn.py`** - Quick PRN checker for gpssim.bin file
- **`gpssim.bin`** - GPS signal data file (2.2GB)
- **`requirements.txt`** - Python dependencies (only numpy)
- **`setup.py`** - Virtual environment setup script

## Quick Start

### 1. Setup Virtual Environment

```bash
# Run setup script
python setup.py

# Activate virtual environment
# Windows:
gps_env\Scripts\activate
# Linux/Mac:
source gps_env/bin/activate
```

### 2. Run Test with gpssim.bin

```bash
python test_gps.py
```

This will:
- Read GPS signals from `gpssim.bin` file
- Test acquisition and show detected PRN codes
- Display acquisition parameters and results

### 3. Quick PRN Check

```bash
python check_prn.py
```

For quick checking with different parameters:
```python
from check_prn import check_prn_from_file

# Check with different duration and threshold
detected_prns, peaks = check_prn_from_file(duration_ms=100, threshold=1.0)
```

## Usage

### Read GPS Signal from File

```python
from gps_signal_generator import GPSSignalGenerator

# Create generator
generator = GPSSignalGenerator(sampling_freq=2.6e6)

# Read from binary file
signal = generator.read_bin_file("gpssim.bin", format_bits=16, max_samples=130000)
```

### Test Acquisition

```python
from gps_acquisition import GPSAcquisition

# Create acquisition object
acq = GPSAcquisition(sampling_freq=2.6e6, acq_threshold=1.5)

# Process samples
sat_peaks, detected_prns = acq.process_samples(signal)
print(f"Detected PRNs: {detected_prns}")
```

## Output

The test script will show:
```
✅ SUCCESS: Detected 6 satellites!

Detected PRN codes:
  PRN  4: Peak = 1.51
  PRN 11: Peak = 1.81
  PRN 12: Peak = 1.63
  PRN 22: Peak = 2.26
  PRN 29: Peak = 1.56
  PRN 32: Peak = 2.39
```

## Features

- **GPS Signal Acquisition**: Complete C/A code generation and correlation
- **File Reading**: Read GPS signals from binary files (8-bit or 16-bit format)
- **Real Data Support**: Works with actual GPS signal recordings
- **Configurable Parameters**: Adjustable threshold, duration, and sampling frequency
- **Clean Output**: Shows detected PRN codes clearly in console
- **Minimal Dependencies**: Only requires numpy

## File Format Support

The system supports reading GPS signal files in the following formats:
- **16-bit interleaved I/Q**: Signed 16-bit integers, I and Q samples interleaved
- **8-bit interleaved I/Q**: Signed 8-bit integers, I and Q samples interleaved

## Testing with Real Data

The system is designed to work with real GPS signal recordings:

1. Use the provided `gpssim.bin` file (2.2GB GPS signal recording)
2. The system will automatically detect GPS satellites present in the signal
3. Adjust acquisition parameters as needed for different signal conditions

The acquisition threshold can be tuned based on signal quality:
- Lower threshold (1.0-1.5): More sensitive, may detect weak signals
- Higher threshold (2.0-3.0): More selective, reduces false detections
