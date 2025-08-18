import numpy as np
import time
import threading
from collections import deque
import bladerf
import bladerf._bladerf as _bladerf  # Import the internal module with enums

from gps_acquisition import GPSAcquisition

class RealTimeGPSAcquisition:
    """
    Real-time GPS acquisition using BladeRF
    """
    
    def __init__(self, 
                 center_freq=1575.42e6,  # GPS L1 frequency
                 sample_rate=2.4e6,      # 2.4 MHz sample rate
                 gain=60,                # RX gain in dB
                 buffer_size=8192,       # BladeRF buffer size
                 num_buffers=16,         # Number of buffers
                 processing_interval=1.0, # Processing interval in seconds
                 acq_threshold=2.0):     # GPS acquisition threshold
        """
        Initialize real-time GPS acquisition
        
        Args:
            center_freq (float): GPS L1 center frequency (Hz)
            sample_rate (float): Sampling rate (Hz) 
            gain (float): RX gain (dB)
            buffer_size (int): BladeRF buffer size
            num_buffers (int): Number of BladeRF buffers
            processing_interval (float): How often to process data (seconds)
            acq_threshold (float): GPS acquisition threshold
        """
        
        self.center_freq = center_freq
        self.sample_rate = sample_rate
        self.gain = gain
        self.buffer_size = buffer_size
        self.num_buffers = num_buffers
        self.processing_interval = processing_interval
        
        # Initialize GPS acquisition engine
        self.gps_acq = GPSAcquisition(
            sampling_freq=sample_rate,
            IF=0,  # Direct sampling at baseband
            acq_threshold=acq_threshold
        )
        
        # Data buffer - GPS acquisition needs at least 10ms of data for fine frequency search
        # Initial acquisition uses 2ms, but fine frequency search needs 10ms
        samples_per_code = int(round(sample_rate / (1.023e6 / 1023)))  # ~2346 samples per 1ms code
        self.samples_needed = int(12 * samples_per_code)  # 12ms worth of samples for safety
        print(f"GPS processing needs {self.samples_needed} samples ({self.samples_needed/sample_rate*1000:.1f}ms)")
        
        # Use a larger circular buffer to ensure we always have enough data
        self.data_buffer = deque(maxlen=self.samples_needed * 5)  # Keep 60ms buffer
        
        # Threading control
        self.running = False
        self.rx_thread = None
        self.processing_thread = None
        
        # BladeRF device
        self.sdr = None
        self.rx_ch = None
        
        # Statistics
        self.samples_received = 0
        self.last_processing_time = time.time()
        
    def initialize_bladerf(self):
        """Initialize BladeRF device"""
        try:
            # Open BladeRF device
            self.sdr = bladerf.BladeRF()
            print(f"Opened BladeRF: {self.sdr.get_board_name()}")
            print(f"FPGA Version: {self.sdr.get_fpga_version()}")
            print(f"Firmware Version: {self.sdr.get_fw_version()}")
            
            # Get RX channel
            self.rx_ch = self.sdr.Channel(bladerf.CHANNEL_RX(0))
            
            # Configure RX channel
            self.rx_ch.frequency = int(self.center_freq)
            self.rx_ch.sample_rate = int(self.sample_rate)
            self.rx_ch.bandwidth = int(self.sample_rate * 0.8)  # 80% of sample rate
            self.rx_ch.gain = self.gain
            
            print(f"RX Frequency: {self.rx_ch.frequency / 1e6:.3f} MHz")
            print(f"RX Sample Rate: {self.rx_ch.sample_rate / 1e6:.3f} MHz")
            print(f"RX Bandwidth: {self.rx_ch.bandwidth / 1e6:.3f} MHz")
            print(f"RX Gain: {self.rx_ch.gain} dB")
            
            # Setup synchronous stream - using the correct _bladerf enums
            self.sdr.sync_config(
                layout=_bladerf.ChannelLayout.RX_X1,  # Use _bladerf.ChannelLayout.RX_X1
                fmt=_bladerf.Format.SC16_Q11,         # Use _bladerf.Format.SC16_Q11
                num_buffers=self.num_buffers,
                buffer_size=self.buffer_size,
                num_transfers=8,
                stream_timeout=3500
            )
            
            print("BladeRF configured successfully")
            return True
            
        except Exception as e:
            print(f"Error initializing BladeRF: {e}")
            return False
    
    def receive_samples(self):
        """Continuous sample reception thread"""
        if self.sdr is None or self.rx_ch is None:
            print("BladeRF not initialized")
            return
            
        # Create receive buffer
        bytes_per_sample = 4  # 2 bytes I + 2 bytes Q = 4 bytes per complex sample
        buf = bytearray(self.buffer_size * bytes_per_sample)
        
        # Enable RX
        print("Starting BladeRF reception...")
        self.rx_ch.enable = True
        
        try:
            while self.running:
                # Receive samples
                self.sdr.sync_rx(buf, self.buffer_size)
                
                # Convert buffer to numpy array
                samples = np.frombuffer(buf, dtype=np.int16)
                
                # Convert to complex IQ (interleaved I,Q,I,Q...)
                i_samples = samples[0::2]  # Even indices are I
                q_samples = samples[1::2]  # Odd indices are Q
                
                # Scale from 12-bit ADC range to normalized complex
                # BladeRF uses Q11 format, so divide by 2048 (2^11)
                complex_samples = (i_samples + 1j * q_samples) / 2048.0
                
                # Store in buffer
                self.data_buffer.extend(complex_samples)
                self.samples_received += len(complex_samples)
                
                # Print reception status every second
                current_time = time.time()
                if current_time - self.last_processing_time > 1.0:
                    print(f"Received {self.samples_received} samples, "
                          f"Buffer size: {len(self.data_buffer)}, "
                          f"Max amplitude: {np.max(np.abs(complex_samples)):.3f}")
                    self.last_processing_time = current_time
                    
        except Exception as e:
            print(f"Error in sample reception: {e}")
        finally:
            # Disable RX
            self.rx_ch.enable = False
            print("BladeRF reception stopped")
    
    def process_gps_data(self):
        """GPS processing thread"""
        print("Starting GPS processing...")
        
        while self.running:
            try:
                # Check if we have enough samples
                if len(self.data_buffer) >= self.samples_needed:
                    # Get exactly the right number of samples for processing
                    samples_list = list(self.data_buffer)
                    
                    # Take exactly samples_needed from the end of the buffer
                    samples = np.array(samples_list[-self.samples_needed:])
                    
                    # Verify we have the correct number of samples
                    if len(samples) != self.samples_needed:
                        print(f"Warning: Expected {self.samples_needed} samples, got {len(samples)}")
                        time.sleep(0.1)
                        continue
                    
                    print(f"Processing {len(samples)} samples ({len(samples)/self.sample_rate*1000:.1f}ms)")
                    
                    # Process GPS acquisition
                    start_time = time.time()
                    sat_peaks, detected_prns = self.gps_acq.process_samples(samples)
                    processing_time = time.time() - start_time
                    
                    # Print results
                    current_time = time.strftime("%H:%M:%S", time.localtime())
                    print(f"\n[{current_time}] GPS Acquisition Results:")
                    print(f"Processing time: {processing_time:.3f}s")
                    
                    if detected_prns:
                        print(f"🛰️  Detected satellites (PRN): {detected_prns}")
                        
                        # Get detailed results for detected satellites (run acquisition once)
                        acq_results = self.gps_acq.acquisition(samples, self.gps_acq.acq_threshold)
                        
                        for prn in detected_prns:
                            peak_metric = sat_peaks[prn - 1]
                            carr_freq = acq_results['carrFreq'][prn - 1]
                            code_phase = acq_results['codePhase'][prn - 1]
                            print(f"  PRN {prn:2d}: Peak={peak_metric:.2f}, "
                                  f"Freq={carr_freq:+6.1f}Hz, Phase={code_phase:4.0f}")
                    else:
                        print("❌ No satellites detected")
                    
                    # Show top 5 peak metrics for debugging
                    top_indices = np.argsort(sat_peaks)[-5:][::-1]
                    print("Top 5 peak metrics:")
                    for idx in top_indices:
                        prn = idx + 1
                        status = "✓" if sat_peaks[idx] > self.gps_acq.acq_threshold else " "
                        print(f"  {status} PRN {prn:2d}: {sat_peaks[idx]:.2f}")
                    
                    print("-" * 60)
                else:
                    current_buffer_ms = len(self.data_buffer) / self.sample_rate * 1000
                    needed_ms = self.samples_needed / self.sample_rate * 1000
                    print(f"Waiting for data... Buffer has {len(self.data_buffer)} samples ({current_buffer_ms:.1f}ms), need {self.samples_needed} samples ({needed_ms:.1f}ms)")
                
                # Sleep for processing interval
                time.sleep(self.processing_interval)
                
            except Exception as e:
                print(f"Error in GPS processing: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1)
    
    def start(self):
        """Start real-time acquisition"""
        if not self.initialize_bladerf():
            return False
            
        self.running = True
        
        # Start reception thread
        self.rx_thread = threading.Thread(target=self.receive_samples)
        self.rx_thread.daemon = True
        self.rx_thread.start()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_gps_data)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        print("Real-time GPS acquisition started")
        print("Press Ctrl+C to stop")
        
        return True
    
    def stop(self):
        """Stop real-time acquisition"""
        print("Stopping real-time acquisition...")
        self.running = False
        
        if self.rx_thread and self.rx_thread.is_alive():
            self.rx_thread.join(timeout=2)
            
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2)
            
        if self.sdr:
            try:
                if self.rx_ch:
                    self.rx_ch.enable = False
                self.sdr.close()
            except:
                pass
                
        print("Real-time acquisition stopped")

def main():
    """Main function"""
    # Configuration
    config = {
        'center_freq': 1575.42e6,    # GPS L1 frequency
        'sample_rate': 2.4e6,        # 2.4 MHz sample rate
        'gain': 60,                  # Start with 60dB gain
        'buffer_size': 8192,         # BladeRF buffer size
        'num_buffers': 16,           # Number of buffers
        'processing_interval': 2.0,   # Process every 2 seconds
        'acq_threshold': 2.0         # GPS acquisition threshold
    }
    
    # Create acquisition instance
    gps_acq = RealTimeGPSAcquisition(**config)
    
    try:
        if gps_acq.start():
            # Keep running until interrupted
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nReceived interrupt signal")
    finally:
        gps_acq.stop()

if __name__ == "__main__":
    main() 