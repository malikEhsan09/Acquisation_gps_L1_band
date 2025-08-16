import numpy as np
import os
from gps_acquisition import GPSAcquisition

class GPSSignalGenerator:
    """
    GPS Signal Generator - Core functionality for GPS signal generation
    """
    
    def __init__(self, 
                 sampling_freq=2.6e6,
                 gps_freq=1575.42e6,
                 IF=0e6,
                 duration=10,
                 output_bits=16):
        """
        Initialize GPS Signal Generator
        
        Args:
            sampling_freq (float): Sampling frequency in Hz
            gps_freq (float): GPS L1 frequency in Hz
            IF (float): Intermediate frequency in Hz
            duration (int): Duration in seconds
            output_bits (int): Output bit depth (1, 8, or 16)
        """
        self.sampling_freq = sampling_freq
        self.gps_freq = gps_freq
        self.IF = IF
        self.duration = duration
        self.output_bits = output_bits
        
        # GPS constants
        self.code_freq_basis = 1.023e6
        self.code_length = 1023
        
        # Generate CA codes table
        self.ca_codes_table = self.make_ca_table()
        
        # Initialize acquisition object for testing
        self.acquisition = GPSAcquisition(sampling_freq=sampling_freq)
    
    def read_bin_file(self, filename, format_bits=16, max_samples=None):
        """
        Read GPS signal from binary file
        
        Args:
            filename (str): Path to binary file
            format_bits (int): Bit depth of the file (8 or 16)
            max_samples (int): Maximum number of samples to read (None for all)
            
        Returns:
            numpy.ndarray: Complex signal samples
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} not found")
        
        file_size = os.path.getsize(filename)
        print(f"Reading file: {filename}")
        print(f"File size: {file_size} bytes")
        
        if format_bits == 8:
            # 8-bit format: signed bytes, interleaved I/Q
            dtype = np.int8
            samples_per_complex = 2
        elif format_bits == 16:
            # 16-bit format: signed 16-bit integers, interleaved I/Q
            dtype = np.int16
            samples_per_complex = 2
        else:
            raise ValueError("format_bits must be 8 or 16")
        
        # Calculate number of complex samples
        total_complex_samples = file_size // (format_bits // 8 * samples_per_complex)
        
        if max_samples is not None:
            total_complex_samples = min(total_complex_samples, max_samples)
        
        print(f"Reading {total_complex_samples} complex samples")
        
        # Read raw data
        with open(filename, 'rb') as f:
            raw_data = np.fromfile(f, dtype=dtype, count=total_complex_samples * samples_per_complex)
        
        # Reshape to separate I and Q
        if len(raw_data) % 2 != 0:
            raw_data = raw_data[:-1]  # Remove last sample if odd number
        
        i_samples = raw_data[0::2].astype(np.float64)
        q_samples = raw_data[1::2].astype(np.float64)
        
        # Normalize to [-1, 1] range
        if format_bits == 8:
            i_samples /= 127.0
            q_samples /= 127.0
        elif format_bits == 16:
            i_samples /= 32767.0
            q_samples /= 32767.0
        
        # Combine into complex signal
        signal = i_samples + 1j * q_samples
        
        print(f"Signal loaded: {len(signal)} complex samples")
        print(f"Signal duration: {len(signal) / self.sampling_freq:.2f} seconds")
        print(f"Signal power: {np.mean(np.abs(signal)**2):.6f}")
        
        return signal
    
    def generate_ca_code(self, prn):
        """Generate C/A code for given PRN"""
        # G2 shift array for different PRNs
        g2s = [5, 6, 7, 8, 17, 18, 139, 140, 141, 251,
               252, 254, 255, 256, 257, 258, 469, 470, 471, 472,
               473, 474, 509, 512, 513, 514, 515, 516, 859, 860,
               861, 862]
        
        g2shift = g2s[prn - 1]  # PRN is 1-indexed
        
        # Generate G1 code
        g1 = np.zeros(1023)
        reg = -np.ones(10)
        
        for i in range(1023):
            g1[i] = reg[9]
            save_bit = reg[2] * reg[9]
            reg[1:10] = reg[0:9]
            reg[0] = save_bit
        
        # Generate G2 code
        g2 = np.zeros(1023)
        reg = -np.ones(10)
        
        for i in range(1023):
            g2[i] = reg[9]
            save_bit = reg[1] * reg[2] * reg[5] * reg[7] * reg[8] * reg[9]
            reg[1:10] = reg[0:9]
            reg[0] = save_bit
        
        # Shift G2 code
        g2 = np.concatenate([g2[1023-g2shift:1023], g2[0:1023-g2shift]])
        
        # Form C/A code
        ca_code = -(g1 * g2)
        
        return ca_code
    
    def make_ca_table(self):
        """Generate CA codes for all 32 satellites"""
        samples_per_code = int(round(self.sampling_freq / (self.code_freq_basis / self.code_length)))
        ca_codes_table = np.zeros((32, samples_per_code))
        
        ts = 1 / self.sampling_freq
        tc = 1 / self.code_freq_basis
        
        for prn in range(1, 33):
            ca_code = self.generate_ca_code(prn)
            
            # Digitizing
            code_value_index = np.ceil((ts * np.arange(1, samples_per_code + 1)) / tc) - 1
            code_value_index = code_value_index.astype(int)
            code_value_index[-1] = 1022
            
            ca_codes_table[prn - 1, :] = ca_code[code_value_index]
        
        return ca_codes_table
    
    def generate_satellite_signal(self, prn, doppler_freq=0, code_phase=0, amplitude=1.0):
        """Generate signal for a single satellite"""
        total_samples = int(self.sampling_freq * self.duration)
        samples_per_code = int(round(self.sampling_freq / (self.code_freq_basis / self.code_length)))
        
        # Generate time array
        t = np.arange(total_samples) / self.sampling_freq
        
        # Generate carrier signal
        carrier_freq = self.IF + doppler_freq
        carrier = np.exp(1j * 2 * np.pi * carrier_freq * t)
        
        # Generate C/A code sequence
        ca_code = self.ca_codes_table[prn - 1, :]
        num_codes_needed = int(np.ceil(total_samples / samples_per_code)) + 1
        long_ca_code = np.tile(ca_code, num_codes_needed)
        
        # Apply code phase offset
        long_ca_code = np.roll(long_ca_code, code_phase)
        
        # Truncate to desired length
        long_ca_code = long_ca_code[:total_samples]
        
        # Combine carrier and code
        signal = amplitude * carrier * long_ca_code
        
        return signal
    
    def generate_multi_satellite_signal(self, satellite_configs):
        """Generate signal with multiple satellites"""
        total_samples = int(self.sampling_freq * self.duration)
        combined_signal = np.zeros(total_samples, dtype=np.complex128)
        
        for config in satellite_configs:
            prn = config.get('prn', 1)
            doppler = config.get('doppler', 0)
            phase = config.get('phase', 0)
            amplitude = config.get('amplitude', 1.0)
            
            sat_signal = self.generate_satellite_signal(prn, doppler, phase, amplitude)
            combined_signal += sat_signal
        
        # Add noise
        noise_power = 0.1
        noise = np.random.normal(0, np.sqrt(noise_power/2), total_samples) + \
                1j * np.random.normal(0, np.sqrt(noise_power/2), total_samples)
        combined_signal += noise
        
        return combined_signal
    
    def save_to_bin_file(self, signal, filename="gpssim.bin"):
        """Save signal to binary file"""
        # Normalize signal
        signal = signal / np.max(np.abs(signal))
        
        if self.output_bits == 1:
            # 1-bit format: pack 4 samples into 1 byte
            i_bits = (np.real(signal) > 0).astype(np.uint8)
            q_bits = (np.imag(signal) > 0).astype(np.uint8)
            
            packed = np.zeros(len(signal) // 4, dtype=np.uint8)
            for i in range(0, len(signal), 4):
                if i + 3 < len(signal):
                    byte_val = (i_bits[i] << 7) | (q_bits[i] << 6) | \
                              (i_bits[i+1] << 5) | (q_bits[i+1] << 4) | \
                              (i_bits[i+2] << 3) | (q_bits[i+2] << 2) | \
                              (i_bits[i+3] << 1) | q_bits[i+3]
                    packed[i // 4] = byte_val
            
            with open(filename, 'wb') as f:
                f.write(packed.tobytes())
                
        elif self.output_bits == 8:
            # 8-bit format: signed bytes
            i_samples = np.clip(np.real(signal) * 127, -127, 127).astype(np.int8)
            q_samples = np.clip(np.imag(signal) * 127, -127, 127).astype(np.int8)
            
            # Interleave I and Q samples
            interleaved = np.empty(2 * len(signal), dtype=np.int8)
            interleaved[0::2] = i_samples
            interleaved[1::2] = q_samples
            
            with open(filename, 'wb') as f:
                f.write(interleaved.tobytes())
                
        elif self.output_bits == 16:
            # 16-bit format: signed 16-bit integers
            i_samples = np.clip(np.real(signal) * 32767, -32767, 32767).astype(np.int16)
            q_samples = np.clip(np.imag(signal) * 32767, -32767, 32767).astype(np.int16)
            
            # Interleave I and Q samples
            interleaved = np.empty(2 * len(signal), dtype=np.int16)
            interleaved[0::2] = i_samples
            interleaved[1::2] = q_samples
            
            with open(filename, 'wb') as f:
                f.write(interleaved.tobytes())
        
        print(f"Signal saved to {filename}")
        print(f"File size: {os.path.getsize(filename)} bytes")
    
    def test_acquisition_on_generated_signal(self, signal):
        """Test acquisition on the generated signal"""
        print("\nTesting acquisition on generated signal...")
        
        # Take first 11ms for acquisition
        acq_samples = int(self.sampling_freq * 0.011)
        test_signal = signal[:acq_samples]
        
        # Process with acquisition
        sat_peaks, detected_prns = self.acquisition.process_samples(test_signal)
        
        print(f"Acquisition threshold: {self.acquisition.acq_threshold}")
        print(f"Detected satellites (PRN): {detected_prns}")
        
        # Show peak metrics for detected satellites
        if detected_prns:
            print("Peak metrics for detected satellites:")
            for prn in detected_prns:
                peak_idx = prn - 1
                print(f"  PRN {prn}: {sat_peaks[peak_idx]:.2f}")
        
        return sat_peaks, detected_prns
