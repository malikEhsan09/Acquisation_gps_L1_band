#!/usr/bin/env python3
"""
Real-time GPS Signal Acquisition using KrakenSDR/RTL-SDR
- Transmission: BladeRF (L1 C/A GPS signals)
- Reception: KrakenSDR/RTL-SDR with 5 antennas (8-bit samples)
- Stores IQ data in circular buffer and performs real-time GPS acquisition
"""

import numpy as np
import time
import threading
import signal
import sys
from datetime import datetime
from collections import deque
import queue

try:
    from rtlsdr import RtlSdr
    RTLSDR_AVAILABLE = True
except ImportError as e:
    print("❌ RTL-SDR Python library not available!")
    print("Install with: pip install pyrtlsdr[lib]")
    print(f"Error: {e}")
    RTLSDR_AVAILABLE = False

from gps_acquisition import GPSAcquisition


class CircularIQBuffer:
    """Thread-safe circular buffer for 8-bit IQ samples"""
    
    def __init__(self, max_samples):
        self.max_samples = max_samples
        self.buffer = np.zeros(max_samples, dtype=np.complex64)
        self.write_ptr = 0
        self.read_ptr = 0
        self.count = 0
        self.lock = threading.Lock()
    
    def write(self, samples):
        """Write samples to buffer (thread-safe)"""
        with self.lock:
            samples = np.asarray(samples, dtype=np.complex64)
            n_samples = len(samples)
            
            # Handle wraparound
            if self.write_ptr + n_samples <= self.max_samples:
                # No wraparound needed
                self.buffer[self.write_ptr:self.write_ptr + n_samples] = samples
                self.write_ptr = (self.write_ptr + n_samples) % self.max_samples
            else:
                # Wraparound needed
                first_chunk = self.max_samples - self.write_ptr
                second_chunk = n_samples - first_chunk
                
                self.buffer[self.write_ptr:] = samples[:first_chunk]
                self.buffer[:second_chunk] = samples[first_chunk:]
                self.write_ptr = second_chunk
            
            # Update count (saturate at max_samples)
            self.count = min(self.count + n_samples, self.max_samples)
    
    def read(self, n_samples):
        """Read n_samples from buffer (thread-safe)"""
        with self.lock:
            if self.count < n_samples:
                return None
            
            samples = np.zeros(n_samples, dtype=np.complex64)
            
            # Calculate read position (most recent data)
            start_pos = (self.write_ptr - n_samples) % self.max_samples
            
            if start_pos + n_samples <= self.max_samples:
                # No wraparound needed
                samples = self.buffer[start_pos:start_pos + n_samples].copy()
            else:
                # Wraparound needed
                first_chunk = self.max_samples - start_pos
                second_chunk = n_samples - first_chunk
                
                samples[:first_chunk] = self.buffer[start_pos:]
                samples[second_chunk:] = self.buffer[:second_chunk]
            
            return samples
    
    def available_samples(self):
        """Get number of available samples"""
        with self.lock:
            return self.count


class KrakenRTLSDRGPS:
    """Real-time GPS acquisition using KrakenSDR/RTL-SDR"""
    
    def __init__(self,
                center_freq=1575.42e6,  # GPS L1 frequency
                sample_rate=2.4e6,      # Sample rate (Hz)
                gain='auto',            # RTL-SDR gain ('auto' or dB)
                device_index=0,         # RTL-SDR device index (0-4 for KrakenSDR)
                buffer_size_ms=5000,    # Buffer size in milliseconds
                acq_threshold=2.5,      # GPS acquisition threshold
                bandwidth=None,         # Tuner bandwidth (Hz). Defaults to sample_rate
                rx_offset_khz=0.0):     # RX frequency offset to avoid DC spike (kHz)
        
        if not RTLSDR_AVAILABLE:
            raise ImportError("RTL-SDR Python library not available")
        
        self.center_freq = center_freq
        self.sample_rate = sample_rate
        self.gain = gain
        self.device_index = device_index
        self.acq_threshold = acq_threshold
        self.bandwidth = bandwidth if bandwidth is not None else sample_rate
        # Apply RX frequency offset (in Hz). Positive moves RF up; IF becomes offset
        self.if_hz = float(rx_offset_khz) * 1e3
        
        # Buffer settings (8-bit samples as requested)
        self.buffer_size_samples = int(sample_rate * buffer_size_ms / 1000)
        self.iq_buffer = CircularIQBuffer(self.buffer_size_samples)
        
        # GPS acquisition engine
        self.gps_acquisition = GPSAcquisition(
            sampling_freq=sample_rate,
            IF=self.if_hz,  # Set IF to RX offset
            acq_threshold=acq_threshold
        )
        
        # RTL-SDR device
        self.sdr = None
        self.receiving = False
        self.receive_thread = None
        
        # Processing settings
        # GPS acquisition needs at least ~10 code periods plus margin
        self.samples_per_code = int(round(self.sample_rate / (1.023e6 / 1023)))
        # Use 12 code periods to ensure enough data regardless of code phase
        self.samples_per_acquisition = int(12 * self.samples_per_code)
        
        # Statistics
        self.total_samples_received = 0
        self.last_detection_time = time.time()
        self.prev_detected_prns = set()
        # Power gate (normalized IQ). Detections are ignored below this average power.
        self.min_signal_power = 1e-5  # Lowered from 1e-4 to 1e-5 for weak signals
        
        # Signal handler for clean shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        print(f"\n🛑 Stopping GPS acquisition...")
        self.stop_reception()
        sys.exit(0)
    
    def setup_rtlsdr(self):
        """Setup RTL-SDR device"""
        try:
            print("🔧 Setting up RTL-SDR/KrakenSDR...")
            
            # List available devices
            try:
                device_serials = RtlSdr.get_device_serial_addresses()
                device_count = len(device_serials)
                print(f"   Found {device_count} RTL-SDR devices")
                
                # List device serials
                for i, serial in enumerate(device_serials):
                    print(f"      Device {i}: {serial}")
                    
            except Exception as e:
                print(f"   Warning: Could not enumerate devices: {e}")
                device_count = 0
            
            if device_count == 0:
                print("❌ No RTL-SDR devices found!")
                return False
            
            if self.device_index >= device_count:
                print(f"❌ Device index {self.device_index} not available!")
                return False
            
            # Open device
            self.sdr = RtlSdr(device_index=self.device_index)
            
            # Configure device
            # Apply RX LO offset if requested
            tune_freq = self.center_freq + self.if_hz
            self.sdr.center_freq = tune_freq
            self.sdr.sample_rate = self.sample_rate
            
            # Set tuner bandwidth if supported
            try:
                self.sdr.set_bandwidth(self.bandwidth)
            except Exception:
                pass
            
            # Gain / AGC
            try:
                if isinstance(self.gain, str) and self.gain == 'auto':
                    # Enable AGC for auto
                    self.sdr.set_manual_gain_enabled(False)
                else:
                    self.sdr.set_manual_gain_enabled(True)
                    self.sdr.gain = float(self.gain)
            except Exception:
                # Fallback to direct assignment
                self.sdr.gain = self.gain
            
            try:
                device_serial = device_serials[self.device_index]
                print(f"   Device {self.device_index}: {device_serial}")
            except:
                print(f"   Device {self.device_index}: Unknown")
            print(f"   Frequency: {self.sdr.center_freq/1e6:.3f} MHz (tuned)")
            if abs(self.if_hz) > 0:
                print(f"   IF (RX offset): {self.if_hz:+.0f} Hz; Baseband target: {self.center_freq/1e6:.3f} MHz")
            print(f"   Sample Rate: {self.sdr.sample_rate/1e6:.3f} MHz")
            try:
                bw = self.sdr.get_bandwidth()
                print(f"   Bandwidth: {bw/1e6:.3f} MHz")
            except Exception:
                print(f"   Bandwidth: {self.bandwidth/1e6:.3f} MHz (requested)")
            print(f"   Gain: {self.sdr.gain} dB")
            
            print("✅ RTL-SDR setup complete!")
            return True
            
        except Exception as e:
            print(f"❌ RTL-SDR setup failed: {e}")
            return False
    
    def _receive_samples_callback(self, samples, context):
        """RTL-SDR callback for sample reception"""
        # Convert 8-bit samples to complex (RTL-SDR provides uint8)
        # RTL-SDR samples are interleaved I,Q,I,Q... as uint8
        try:
            # Some backends may already provide complex64; handle both
            if np.iscomplexobj(samples):
                complex_samples = samples.astype(np.complex64)
            else:
                # Ensure samples are uint8 array
                if samples.dtype != np.uint8:
                    samples = samples.astype(np.uint8)
                
                # Convert to complex samples and normalize to -1 to +1 range
                # RTL-SDR uses 8-bit unsigned integers (0-255)
                i_samples = samples[0::2].astype(np.float32) - 127.5
                q_samples = samples[1::2].astype(np.float32) - 127.5
                
                # Normalize to approximately -1 to +1 range
                complex_samples = (i_samples + 1j * q_samples) / 127.5
            
            # Store in circular buffer
            self.iq_buffer.write(complex_samples)
            self.total_samples_received += len(complex_samples)
            
        except Exception as e:
            if self.receiving:
                print(f"⚠️ Sample processing error: {e}")
    
    def _receive_samples_async(self):
        """Asynchronous sample reception using RTL-SDR async mode"""
        try:
            print("📡 Starting asynchronous sample reception...")
            
            # Start async read with callback
            # RTL-SDR will continuously call our callback with samples
            self.sdr.read_samples_async(
                callback=self._receive_samples_callback,
                num_samples=8192,  # Samples per callback
                context=None
            )
            
        except Exception as e:
            if self.receiving:
                print(f"❌ Async reception error: {e}")
    
    def start_reception(self):
        """Start continuous sample reception"""
        if not self.setup_rtlsdr():
            return False
        
        print("🚀 Starting GPS signal reception...")
        print(f"   Buffer size: {self.buffer_size_samples} samples ({self.buffer_size_samples/self.sample_rate:.1f}s)")
        print(f"   Using 8-bit samples from RTL-SDR")
        print(f"   Device: RTL-SDR #{self.device_index}")
        
        self.receiving = True
        
        # Start async reception in separate thread
        self.receive_thread = threading.Thread(target=self._receive_samples_async)
        self.receive_thread.daemon = True
        self.receive_thread.start()
        
        # Wait for buffer to start filling
        print("⏳ Waiting for buffer to fill...")
        start_time = time.time()
        while self.iq_buffer.available_samples() < self.samples_per_acquisition:
            time.sleep(0.1)
            if time.time() - start_time > 10:
                print("❌ Timeout waiting for samples!")
                return False
        
        print(f"✅ Reception started! Buffer: {self.iq_buffer.available_samples()} samples")
        return True
    
    def stop_reception(self):
        """Stop sample reception"""
        if self.receiving:
            print("🛑 Stopping sample reception...")
            self.receiving = False
            
            if self.sdr:
                try:
                    self.sdr.cancel_read_async()
                    self.sdr.close()
                except:
                    pass
            
            if self.receive_thread and self.receive_thread.is_alive():
                self.receive_thread.join(timeout=2.0)
            
            print("📡 Reception stopped")
    
    def process_current_samples(self):
        """Process current buffer contents for GPS acquisition"""
        # Get samples from buffer
        samples = self.iq_buffer.read(self.samples_per_acquisition)
        
        if samples is None:
            return [], [], 0
        
        # Process with GPS acquisition
        try:
            sat_peaks, detected_prns = self.gps_acquisition.process_samples(samples)
            signal_power = np.mean(np.abs(samples)**2)
            
            # Debug: Show top 5 peak metrics
            if len(sat_peaks) > 0:
                top_indices = np.argsort(sat_peaks)[-5:][::-1]
                print("   Top 5 peak metrics:")
                for idx in top_indices:
                    prn = idx + 1
                    status = "✓" if sat_peaks[idx] > self.gps_acquisition.acq_threshold else " "
                    print(f"     {status} PRN {prn:2d}: {sat_peaks[idx]:.2f}")
            
            return sat_peaks, detected_prns, signal_power
        except Exception as e:
            print(f"❌ GPS processing error: {e}")
            import traceback
            traceback.print_exc()
            return [], [], 0
    
    def run_single_acquisition(self):
        """Run a single GPS acquisition"""
        print("🛰️ REAL-TIME GPS ACQUISITION (KrakenSDR/RTL-SDR)")
        print("="*70)
        print("Reception: RTL-SDR/KrakenSDR with 8-bit samples")
        print("Transmission: BladeRF with L1 C/A GPS signals")
        print("="*70)
        
        if not self.start_reception():
            return False
        
        try:
            print(f"\n🔍 Processing GPS signals...")
            print(f"   Sample Rate: {self.sample_rate/1e6:.1f} MHz")
            print(f"   Center Frequency: {self.center_freq/1e6:.3f} MHz")
            print(f"   Device: RTL-SDR #{self.device_index}")
            print(f"   Buffer Size: {self.buffer_size_samples} samples")
            print(f"   Acquisition Length: {self.samples_per_acquisition} samples")
            
            # Process samples
            sat_peaks, detected_prns, signal_power = self.process_current_samples()
            
            # Display results
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"\n" + "="*70)
            print("📊 GPS ACQUISITION RESULTS")
            print("="*70)
            print(f"Analysis completed at: {timestamp}")
            print(f"Signal power: {signal_power:.6f}")
            print(f"Total samples received: {self.total_samples_received:,}")
            print(f"Threshold: {self.gps_acquisition.acq_threshold}")
            
            if detected_prns:
                print(f"\n✅ SUCCESS: Detected {len(detected_prns)} GPS satellites!")
                print(f"\n🛰️ DETECTED SATELLITES:")
                
                for prn in detected_prns:
                    peak_metric = sat_peaks[prn - 1]
                    print(f"   PRN {prn:2d}: Peak Metric = {peak_metric:.2f}")
                
                prn_str = ", ".join([f"PRN{prn}" for prn in detected_prns])
                print(f"\n📡 Summary: {prn_str}")
                
            else:
                available_samples = self.iq_buffer.available_samples()
                print(f"\n❌ NO GPS SATELLITES DETECTED")
                print(f"\nBuffer status:")
                print(f"   Available samples: {available_samples:,}")
                print(f"   Signal power: {signal_power:.6f}")
                
                print(f"\nPossible reasons:")
                print(f"   - No GPS transmission from BladeRF")
                print(f"   - Signal too weak (try different antenna/gain)")
                print(f"   - Threshold too high (current: {self.gps_acquisition.acq_threshold})")
                print(f"   - Wrong frequency offset")
            
            print(f"\n" + "="*70)
            return True
            
        finally:
            self.stop_reception()
    
    def run_continuous_acquisition(self, update_interval=2.0):
        """Run continuous GPS acquisition with real-time updates"""
        print(f"\n🔄 Starting continuous GPS monitoring...")
        print(f"   Update interval: {update_interval} seconds")
        print(f"   Threshold: {self.gps_acquisition.acq_threshold}")
        print("   Press Ctrl+C to stop")
        print("   " + "="*60)
        
        if not self.start_reception():
            return False
        
        try:
            acquisition_count = 0
            detection_history = {}
            
            while True:
                # Wait for enough samples
                while self.iq_buffer.available_samples() < self.samples_per_acquisition:
                    time.sleep(0.1)
                    print(f"   Waiting for samples... Buffer: {self.iq_buffer.available_samples()}/{self.samples_per_acquisition}")
                
                acquisition_count += 1
                current_time = datetime.now().strftime('%H:%M:%S')
                print(f"\n[Acquisition #{acquisition_count}] - {current_time}")
                
                # Process current samples
                start_time = time.time()
                sat_peaks, detected_prns, signal_power = self.process_current_samples()
                processing_time = time.time() - start_time
                
                print(f"   Processing time: {processing_time:.3f}s")
                print(f"   Signal power: {signal_power:.6f}")
                
                # Update detection history
                for prn in detected_prns:
                    if prn not in detection_history:
                        detection_history[prn] = 0
                    detection_history[prn] += 1
                
                # Display results
                available_samples = self.iq_buffer.available_samples()
                total_received = self.total_samples_received
                
                # Power gate to reduce false positives
                power_gate = (signal_power >= self.min_signal_power)
                if detected_prns and power_gate:
                    # Confirmed detections across two consecutive acquisitions
                    confirmed_prns = sorted(list(set(detected_prns).intersection(self.prev_detected_prns)))
                    self.prev_detected_prns = set(detected_prns)
                    
                    if confirmed_prns:
                        prn_str = ", ".join([f"PRN{prn}" for prn in confirmed_prns])
                        print(f"   ✅ {len(confirmed_prns)} satellites (confirmed): {prn_str}")
                        
                        # Get detailed results for confirmed satellites
                        acq_results = self.gps_acquisition.acquisition(
                            self.iq_buffer.read(self.samples_per_acquisition), 
                            self.gps_acquisition.acq_threshold
                        )
                        
                        # Show detailed info for confirmed satellites with Doppler sanity check
                        max_dopp_hz = max(20000.0, float(self.gps_acquisition.acq_search_band) * 1000.0)
                        for prn in confirmed_prns:
                            peak_metric = sat_peaks[prn - 1]
                            carr_freq = acq_results['carrFreq'][prn - 1]
                            code_phase = acq_results['codePhase'][prn - 1]
                            if abs(carr_freq - self.if_hz) > max_dopp_hz:
                                print(f"      PRN {prn:2d}: Rejected (Doppler |{carr_freq-self.if_hz:.0f}|>{max_dopp_hz:.0f} Hz)")
                                continue
                            print(f"      PRN {prn:2d}: Peak={peak_metric:.2f}, "
                                f"Freq={carr_freq:+6.1f}Hz, Phase={code_phase:4.0f}")
                        self.last_detection_time = time.time()
                    else:
                        print("   ℹ️ Candidates detected; awaiting confirmation next cycle")
                elif detected_prns and not power_gate:
                    print("   ⚠️ Power below gate; ignoring detections")
                    self.prev_detected_prns = set()
                else:
                    time_since_last = time.time() - self.last_detection_time
                    print(f"   ❌ No satellites detected")
                    print(f"   ⏱️ Time since last detection: {time_since_last:.1f}s")
                
                print(f"   📡 Buffer: {available_samples:,} samples, Total RX: {total_received:,}")
                print("-" * 60)
                
                time.sleep(update_interval)
                
        except KeyboardInterrupt:
            print(f"\n⚠️ Stopped by user")
            
            if detection_history:
                print(f"\n📊 Detection Summary:")
                for prn, count in sorted(detection_history.items()):
                    print(f"   PRN{prn}: {count} detections")
            else:
                print(f"\n❌ No GPS signals detected during monitoring")
        
        finally:
            self.stop_reception()
        
        return True


def main():
    """Main function"""
    print("🛰️ Real-time GPS Signal Acquisition")
    print("Reception: KrakenSDR/RTL-SDR (8-bit samples)")
    print("Transmission: BladeRF (L1 C/A GPS)")
    print("="*70)
    
    # Configuration
    config = {
        'center_freq': 1575.42e6,    # GPS L1 frequency
        'sample_rate': 2.4e6,        # Sample rate (Hz)
        'gain': 'auto',              # Auto gain (or specify dB value)
        'device_index': 0,           # RTL-SDR device index (0-4 for KrakenSDR)
        'buffer_size_ms': 5000,      # 5 second buffer
        'acq_threshold': 2.5,        # GPS acquisition threshold
        'bandwidth': None,           # Defaults to sample_rate
        'rx_offset_khz': 0.0         # Apply RX LO offset to avoid DC spike
    }
    
    # Parse command line arguments
    mode = "continuous"  # Default to continuous mode
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg == "single":
                mode = "single"
            elif arg == "continuous":
                mode = "continuous"
            elif arg.startswith("device="):
                config['device_index'] = int(arg.split("=")[1])
            elif arg.startswith("gain="):
                gain_val = arg.split("=")[1]
                config['gain'] = 'auto' if gain_val == 'auto' else float(gain_val)
            elif arg.startswith("rate="):
                config['sample_rate'] = float(arg.split("=")[1]) * 1e6
                # default bandwidth to sample rate if not set later
                if config['bandwidth'] in (None, 0):
                    config['bandwidth'] = config['sample_rate']
            elif arg.startswith("bandwidth="):
                # in MHz
                config['bandwidth'] = float(arg.split("=")[1]) * 1e6
            elif arg.startswith("offset_khz="):
                config['rx_offset_khz'] = float(arg.split("=")[1])
            elif arg.startswith("threshold="):
                config['acq_threshold'] = float(arg.split("=")[1])
    
    print(f"Mode: {mode}")
    print(f"Device: RTL-SDR #{config['device_index']}")
    print(f"Sample Rate: {config['sample_rate']/1e6:.1f} MHz")
    print(f"Gain: {config['gain']}")
    print(f"Threshold: {config['acq_threshold']}")
    
    # Create GPS acquisition system
    try:
        gps_system = KrakenRTLSDRGPS(**config)
        
        if mode == "single":
            success = gps_system.run_single_acquisition()
        else:
            success = gps_system.run_continuous_acquisition(update_interval=1.0)  # 1 second updates
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    print("Usage:")
    print("  python kraken_rtlsdr_gps_acquisition.py                    # Continuous monitoring (default)")
    print("  python kraken_rtlsdr_gps_acquisition.py single             # Single acquisition")
    print("  python kraken_rtlsdr_gps_acquisition.py device=1           # Use RTL-SDR device #1")
    print("  python kraken_rtlsdr_gps_acquisition.py gain=40            # Set gain to 40 dB")
    print("  python kraken_rtlsdr_gps_acquisition.py rate=2.6           # 2.6 MHz sample rate")
    print("  python kraken_rtlsdr_gps_acquisition.py bandwidth=2.6      # 2.6 MHz tuner bandwidth")
    print("  python kraken_rtlsdr_gps_acquisition.py offset_khz=250     # +250 kHz RX LO offset (IF)")
    print("  python kraken_rtlsdr_gps_acquisition.py threshold=1.5      # Lower threshold")
    print("  python kraken_rtlsdr_gps_acquisition.py device=1 gain=35 rate=2.6 bandwidth=2.6 offset_khz=250 threshold=1.5")
    print()
    
    sys.exit(main()) 