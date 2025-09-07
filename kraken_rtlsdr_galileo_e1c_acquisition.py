#!/usr/bin/env python3
"""
Real-time Galileo E1C (pilot) Signal Acquisition using KrakenSDR/RTL-SDR
- Reception: RTL-SDR/KrakenSDR (8-bit samples)
- Acquisition: Galileo E1C (primary) using FFT-based search

Mirrors the GPS flow in kraken_rtlsdr_gps_acquisition.py.
"""

import numpy as np
import time
import threading
import signal
import sys
from datetime import datetime

try:
    from rtlsdr import RtlSdr
    RTLSDR_AVAILABLE = True
except ImportError as e:
    print("❌ RTL-SDR Python library not available!")
    print("Install with: pip install pyrtlsdr[lib]")
    print(f"Error: {e}")
    RTLSDR_AVAILABLE = False

from galileo_e1c_acquisition import GalileoE1CAcquisition


class CircularIQBuffer:
    """Thread-safe circular buffer for 8-bit IQ samples"""

    def __init__(self, max_samples):
        self.max_samples = max_samples
        self.buffer = np.zeros(max_samples, dtype=np.complex64)
        self.write_ptr = 0
        self.count = 0
        self.lock = threading.Lock()

    def write(self, samples):
        with self.lock:
            samples = np.asarray(samples, dtype=np.complex64)
            n_samples = len(samples)

            if self.write_ptr + n_samples <= self.max_samples:
                self.buffer[self.write_ptr:self.write_ptr + n_samples] = samples
                self.write_ptr = (self.write_ptr + n_samples) % self.max_samples
            else:
                first_chunk = self.max_samples - self.write_ptr
                second_chunk = n_samples - first_chunk
                self.buffer[self.write_ptr:] = samples[:first_chunk]
                self.buffer[:second_chunk] = samples[first_chunk:]
                self.write_ptr = second_chunk

            self.count = min(self.count + n_samples, self.max_samples)

    def read_latest(self, n_samples):
        with self.lock:
            if self.count < n_samples:
                return None
            start_pos = (self.write_ptr - n_samples) % self.max_samples
            if start_pos + n_samples <= self.max_samples:
                return self.buffer[start_pos:start_pos + n_samples].copy()
            out = np.zeros(n_samples, dtype=np.complex64)
            first_chunk = self.max_samples - start_pos
            second_chunk = n_samples - first_chunk
            out[:first_chunk] = self.buffer[start_pos:]
            out[first_chunk:] = self.buffer[:second_chunk]
            return out

    def available_samples(self):
        with self.lock:
            return self.count


class KrakenRTLSDRGalileoE1C:
    def __init__(self,
                 center_freq=1575.42e6,
                 sample_rate=2.4e6,
                 gain=40.0,
                 device_index=0,
                 buffer_size_ms=5000,
                 acq_threshold=1.1,
                 bandwidth=None,
                 rx_offset_khz=0.0):
        if not RTLSDR_AVAILABLE:
            raise ImportError("RTL-SDR Python library not available")

        self.center_freq = center_freq
        self.sample_rate = sample_rate
        self.gain = gain
        self.device_index = device_index
        self.bandwidth = bandwidth if bandwidth is not None else sample_rate
        self.if_hz = float(rx_offset_khz) * 1e3

        self.buffer_size_samples = int(sample_rate * buffer_size_ms / 1000)
        self.iq_buffer = CircularIQBuffer(self.buffer_size_samples)

        self.acq = GalileoE1CAcquisition(sampling_freq=sample_rate, IF=self.if_hz, acq_threshold=acq_threshold)

        self.sdr = None
        self.receiving = False
        self.receive_thread = None

        # For E1C primary acquisition, need ~ 8 ms for margin
        self.samples_per_code = int(round(self.sample_rate / (1.023e6 / 4092)))
        self.samples_per_acquisition = int(12 * self.samples_per_code)

        self.total_samples_received = 0
        self.last_detection_time = time.time()
        self.prev_detected_prns = set()
        self.min_signal_power = 5e-6  # Lower threshold for better detection

        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        print("\n🛑 Stopping Galileo E1C acquisition...")
        self.stop_reception()
        sys.exit(0)

    def setup_rtlsdr(self):
        try:
            print("🔧 Setting up RTL-SDR/KrakenSDR for Galileo E1C...")
            serials = []
            try:
                serials = RtlSdr.get_device_serial_addresses()
                print(f"   Found {len(serials)} RTL-SDR devices")
                for i, s in enumerate(serials):
                    print(f"      Device {i}: {s}")
            except Exception as e:
                print(f"   Warning: Could not enumerate devices: {e}")

            if len(serials) == 0:
                print("❌ No RTL-SDR devices found!")
                return False
            if self.device_index >= len(serials):
                print(f"❌ Device index {self.device_index} not available!")
                return False

            self.sdr = RtlSdr(device_index=self.device_index)
            tune_freq = self.center_freq + self.if_hz
            self.sdr.center_freq = tune_freq
            self.sdr.sample_rate = self.sample_rate
            try:
                self.sdr.set_bandwidth(self.bandwidth)
            except Exception:
                pass
            # Set gain properly
            try:
                if isinstance(self.gain, str) and self.gain == 'auto':
                    self.sdr.set_manual_gain_enabled(False)
                    print(f"   Gain: Auto (AGC enabled)")
                else:
                    print(f"   Setting gain to {self.gain} dB...")
                    self.sdr.set_manual_gain_enabled(True)
                    self.sdr.gain = float(self.gain)
                    actual_gain = self.sdr.get_gain()
                    print(f"   Gain: {actual_gain} dB (manual) - Requested: {self.gain} dB")
            except Exception as e:
                print(f"   Error: Could not set gain: {e}")
                # Fallback to auto gain
                self.sdr.set_manual_gain_enabled(False)
                print(f"   Gain: Auto (AGC enabled) - Fallback due to error")

            print(f"   Frequency: {self.sdr.center_freq/1e6:.3f} MHz (tuned)")
            if abs(self.if_hz) > 0:
                print(f"   IF offset: {self.if_hz:+.0f} Hz")
            print(f"   Sample Rate: {self.sdr.sample_rate/1e6:.3f} MHz")
            print("✅ RTL-SDR setup complete!")
            return True
        except Exception as e:
            print(f"❌ RTL-SDR setup failed: {e}")
            return False

    def _receive_samples_callback(self, samples, context):
        try:
            if np.iscomplexobj(samples):
                complex_samples = samples.astype(np.complex64)
            else:
                if samples.dtype != np.uint8:
                    samples = samples.astype(np.uint8)
                i_s = samples[0::2].astype(np.float32) - 127.5
                q_s = samples[1::2].astype(np.float32) - 127.5
                complex_samples = (i_s + 1j * q_s) / 127.5
            self.iq_buffer.write(complex_samples)
            self.total_samples_received += len(complex_samples)
        except Exception as e:
            if self.receiving:
                print(f"⚠️ Sample processing error: {e}")

    def _receive_samples_async(self):
        try:
            print("📡 Starting asynchronous sample reception...")
            self.sdr.read_samples_async(
                callback=self._receive_samples_callback,
                num_samples=8192,
                context=None
            )
        except Exception as e:
            if self.receiving:
                print(f"❌ Async reception error: {e}")

    def start_reception(self):
        if not self.setup_rtlsdr():
            return False
        print("🚀 Starting Galileo E1C signal reception...")
        print(f"   Buffer size: {self.buffer_size_samples} samples ({self.buffer_size_samples/self.sample_rate:.1f}s)")
        self.receiving = True
        self.receive_thread = threading.Thread(target=self._receive_samples_async)
        self.receive_thread.daemon = True
        self.receive_thread.start()
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
        if self.receiving:
            print("🛑 Stopping sample reception...")
            self.receiving = False
            if self.sdr:
                try:
                    self.sdr.cancel_read_async()
                    self.sdr.close()
                except Exception:
                    pass
            if self.receive_thread and self.receive_thread.is_alive():
                self.receive_thread.join(timeout=2.0)
            print("📡 Reception stopped")

    def _process_once(self):
        samples = self.iq_buffer.read_latest(self.samples_per_acquisition)
        if samples is None:
            return [], [], 0.0
        try:
            sat_peaks, detected_prns = self.acq.process_samples(samples)
            signal_power = np.mean(np.abs(samples) ** 2)
            if len(sat_peaks) > 0:
                top_indices = np.argsort(sat_peaks)[-5:][::-1]
                print("   Top 5 peak metrics:")
                for idx in top_indices:
                    prn = idx + 1
                    status = "✓" if sat_peaks[idx] > self.acq.acq_threshold else " "
                    print(f"     {status} PRN {prn:2d}: {sat_peaks[idx]:.2f}")
            return sat_peaks, detected_prns, signal_power
        except Exception as e:
            print(f"❌ E1C processing error: {e}")
            return [], [], 0.0

    def run_single(self):
        print("🛰️ REAL-TIME GALILEO E1C ACQUISITION (KrakenSDR/RTL-SDR)")
        print("=" * 70)
        if not self.start_reception():
            return False
        try:
            sat_peaks, detected_prns, signal_power = self._process_once()
            timestamp = datetime.now().strftime("%H:%M:%S")
            print("\n" + "=" * 70)
            print("📊 E1C ACQUISITION RESULTS")
            print("=" * 70)
            print(f"Analysis completed at: {timestamp}")
            print(f"Signal power: {signal_power:.6f}")
            print(f"Threshold: {self.acq.acq_threshold}")
            if detected_prns:
                print(f"\n✅ SUCCESS: Detected {len(detected_prns)} Galileo E1C satellites!")
                for prn in detected_prns:
                    peak_metric = sat_peaks[prn - 1]
                    print(f"   PRN {prn:2d}: Peak Metric = {peak_metric:.2f}")
            else:
                print("\n❌ NO GALILEO E1C SATELLITES DETECTED")
            print("\n" + "=" * 70)
            return True
        finally:
            self.stop_reception()

    def run_continuous(self, update_interval=1.0):
        print("\n🔄 Starting continuous Galileo E1C monitoring...")
        if not self.start_reception():
            return False
        try:
            acquisition_count = 0
            while True:
                while self.iq_buffer.available_samples() < self.samples_per_acquisition:
                    time.sleep(0.1)
                acquisition_count += 1
                print(f"\n[Acquisition #{acquisition_count}]")
                t0 = time.time()
                sat_peaks, detected_prns, signal_power = self._process_once()
                dt = time.time() - t0
                print(f"   Processing time: {dt:.3f}s; Power: {signal_power:.6f}")
                power_gate = (signal_power >= self.min_signal_power)
                if detected_prns and power_gate:
                    # More lenient confirmation: if we have high peaks, confirm immediately
                    high_confidence_prns = [prn for prn in detected_prns if sat_peaks[prn-1] > 2.0]
                    confirmed = sorted(list(set(detected_prns).intersection(self.prev_detected_prns)))
                    
                    # Add high confidence detections to confirmed list
                    for prn in high_confidence_prns:
                        if prn not in confirmed:
                            confirmed.append(prn)
                    
                    self.prev_detected_prns = set(detected_prns)
                    if confirmed:
                        prn_str = ", ".join([f"PRN{p}" for p in confirmed])
                        print(f"   ✅ {len(confirmed)} satellites (confirmed): {prn_str}")
                    else:
                        print("   ℹ️ Candidates detected; awaiting confirmation")
                elif detected_prns and not power_gate:
                    print("   ⚠️ Power below gate; ignoring detections")
                    self.prev_detected_prns = set()
                else:
                    print("   ❌ No satellites detected")
                print("-" * 60)
                time.sleep(update_interval)
        except KeyboardInterrupt:
            print("\n⚠️ Stopped by user")
        finally:
            self.stop_reception()
        return True


def main():
    print("🛰️ Real-time Galileo E1C Signal Acquisition")
    print("Reception: KrakenSDR/RTL-SDR (8-bit samples)")
    print("=" * 70)

    config = {
        'center_freq': 1575.42e6,
        'sample_rate': 2.4e6,
        'gain': 'auto',
        'device_index': 0,
        'buffer_size_ms': 5000,
        'acq_threshold': 2.0,
        'bandwidth': None,
        'rx_offset_khz': 0.0
    }

    mode = "continuous"
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg == "single":
                mode = "single"
            elif arg == "continuous":
                mode = "continuous"
            elif arg.startswith("device="):
                config['device_index'] = int(arg.split("=")[1])
            elif arg.startswith("gain="):
                val = arg.split("=")[1]
                config['gain'] = 'auto' if val == 'auto' else float(val)
            elif arg.startswith("rate="):
                config['sample_rate'] = float(arg.split("=")[1]) * 1e6
                if config['bandwidth'] in (None, 0):
                    config['bandwidth'] = config['sample_rate']
            elif arg.startswith("bandwidth="):
                config['bandwidth'] = float(arg.split("=")[1]) * 1e6
            elif arg.startswith("offset_khz="):
                config['rx_offset_khz'] = float(arg.split("=")[1])
            elif arg.startswith("threshold="):
                config['acq_threshold'] = float(arg.split("=")[1])

    try:
        sys_obj = KrakenRTLSDRGalileoE1C(**config)
        if mode == "single":
            ok = sys_obj.run_single()
        else:
            ok = sys_obj.run_continuous(update_interval=1.0)
        return 0 if ok else 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    print("Usage:")
    print("  python kraken_rtlsdr_galileo_e1c_acquisition.py                    # Continuous monitoring (default)")
    print("  python kraken_rtlsdr_galileo_e1c_acquisition.py single             # Single acquisition")
    print("  python kraken_rtlsdr_galileo_e1c_acquisition.py device=1           # Use RTL-SDR device #1")
    print("  python kraken_rtlsdr_galileo_e1c_acquisition.py gain=40            # Set gain to 40 dB")
    print("  python kraken_rtlsdr_galileo_e1c_acquisition.py rate=2.6           # 2.6 MHz sample rate")
    print("  python kraken_rtlsdr_galileo_e1c_acquisition.py bandwidth=2.6      # 2.6 MHz tuner bandwidth")
    print("  python kraken_rtlsdr_galileo_e1c_acquisition.py offset_khz=250     # +250 kHz RX LO offset (IF)")
    print("  python kraken_rtlsdr_galileo_e1c_acquisition.py threshold=1.5      # Lower threshold")
    print()
    sys.exit(main())


