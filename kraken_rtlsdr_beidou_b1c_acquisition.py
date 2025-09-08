#!/usr/bin/env python3
"""
Real-time BeiDou B1C (pilot) Acquisition using KrakenSDR/RTL-SDR
- Transmission: BladeRF (B1C signals), e.g., file playback at 3.2 Msps
- Reception: KrakenSDR/RTL-SDR (8-bit IQ) centered at 1575.42 MHz
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
    print("❌ RTL-SDR Python library not available! Install: pip install pyrtlsdr[lib]")
    print(f"Error: {e}")
    RTLSDR_AVAILABLE = False

from beidou_b1c_acquisition import BeidouB1CAcquisition


class CircularIQBuffer:
    def __init__(self, max_samples):
        self.max_samples = max_samples
        self.buffer = np.zeros(max_samples, dtype=np.complex64)
        self.write_ptr = 0
        self.count = 0
        self.lock = threading.Lock()

    def write(self, samples):
        with self.lock:
            samples = np.asarray(samples, dtype=np.complex64)
            n = len(samples)
            end = self.write_ptr + n
            if end <= self.max_samples:
                self.buffer[self.write_ptr:end] = samples
                self.write_ptr = end % self.max_samples
            else:
                first = self.max_samples - self.write_ptr
                second = n - first
                self.buffer[self.write_ptr:] = samples[:first]
                self.buffer[:second] = samples[first:]
                self.write_ptr = second
            self.count = min(self.count + n, self.max_samples)

    def read(self, n):
        with self.lock:
            if self.count < n:
                return None
            start = (self.write_ptr - n) % self.max_samples
            if start + n <= self.max_samples:
                return self.buffer[start:start+n].copy()
            first = self.max_samples - start
            second = n - first
            out = np.empty(n, dtype=np.complex64)
            out[:first] = self.buffer[start:]
            out[first:] = self.buffer[:second]
            return out

    def available(self):
        with self.lock:
            return self.count

    def read_latest(self, n):
        return self.read(n)


class KrakenRTLSDRBeidouB1C:
    def __init__(self,
                 center_freq=1575.42e6,
                 sample_rate=3.2e6,
                 gain='auto',
                 device_index=0,
                 buffer_size_ms=5000,
                 acq_threshold=1.3,
                 bandwidth=None,
                 rx_offset_khz=0.0,
                 channel='pilot'):
        if not RTLSDR_AVAILABLE:
            raise ImportError("RTL-SDR Python library not available")

        self.center_freq = center_freq
        self.sample_rate = sample_rate
        self.gain = gain
        self.device_index = device_index
        self.acq_threshold = acq_threshold
        self.bandwidth = bandwidth if bandwidth is not None else sample_rate
        self.if_hz = float(rx_offset_khz) * 1e3

        self.buffer_size_samples = int(sample_rate * buffer_size_ms / 1000)
        self.iq_buffer = CircularIQBuffer(self.buffer_size_samples)

        self.b1c_acq = BeidouB1CAcquisition(
            sampling_freq=sample_rate,
            IF=self.if_hz,
            acq_threshold=acq_threshold,
            channel=channel
        )

        self.samples_per_code = int(round(self.sample_rate / (1.023e6 / 10230)))
        self.samples_per_acquisition = int(12 * self.samples_per_code)

        self.sdr = None
        self.receiving = False
        self.receive_thread = None
        self.total_samples_received = 0
        self.prev_detected = set()
        self.last_detection_time = time.time()
        self.min_signal_power = 1e-5

        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        print("\n🛑 Stopping B1C acquisition...")
        self.stop()
        sys.exit(0)

    def setup(self):
        try:
            print("🔧 Setting up RTL-SDR/KrakenSDR for BeiDou B1C...")
            serials = []
            try:
                serials = RtlSdr.get_device_serial_addresses()
                print(f"   Found {len(serials)} RTL-SDR devices")
                for i, s in enumerate(serials):
                    print(f"      Device {i}: {s}")
            except Exception as e:
                print(f"   Warning: Could not enumerate devices: {e}")
            if len(serials) == 0 or self.device_index >= len(serials):
                print("❌ No suitable RTL-SDR device found")
                return False

            self.sdr = RtlSdr(device_index=self.device_index)
            tune_freq = self.center_freq + self.if_hz
            self.sdr.center_freq = tune_freq
            self.sdr.sample_rate = self.sample_rate
            try:
                self.sdr.set_bandwidth(self.bandwidth)
            except Exception:
                pass
            try:
                if isinstance(self.gain, str) and self.gain == 'auto':
                    self.sdr.set_manual_gain_enabled(False)
                else:
                    self.sdr.set_manual_gain_enabled(True)
                    self.sdr.gain = float(self.gain)
            except Exception:
                self.sdr.gain = self.gain

            print(f"✅ Tuned: {self.sdr.center_freq/1e6:.3f} MHz, SR: {self.sdr.sample_rate/1e6:.3f} MHz, BW: {(self.bandwidth or 0)/1e6:.3f} MHz")
            return True
        except Exception as e:
            print(f"❌ RTL-SDR setup failed: {e}")
            return False

    def _rx_cb(self, samples, _):
        try:
            if np.iscomplexobj(samples):
                cs = samples.astype(np.complex64)
            else:
                if samples.dtype != np.uint8:
                    samples = samples.astype(np.uint8)
                i = samples[0::2].astype(np.float32) - 127.5
                q = samples[1::2].astype(np.float32) - 127.5
                cs = (i + 1j * q) / 127.5
            self.iq_buffer.write(cs)
            self.total_samples_received += len(cs)
        except Exception as e:
            if self.receiving:
                print(f"⚠️ RX error: {e}")

    def _rx_async(self):
        try:
            self.sdr.read_samples_async(callback=self._rx_cb, num_samples=8192, context=None)
        except Exception as e:
            if self.receiving:
                print(f"❌ Async reception error: {e}")

    def start(self):
        if not self.setup():
            return False
        self.receiving = True
        self.receive_thread = threading.Thread(target=self._rx_async, daemon=True)
        self.receive_thread.start()
        print("⏳ Waiting for buffer to fill...")
        t0 = time.time()
        while self.iq_buffer.available() < self.samples_per_acquisition:
            time.sleep(0.1)
            if time.time() - t0 > 10:
                print("❌ Timeout waiting for samples")
                return False
        print("✅ Reception started")
        return True

    def stop(self):
        if self.receiving:
            print("🛑 Stopping reception...")
            self.receiving = False
            try:
                if self.sdr:
                    self.sdr.cancel_read_async()
                    self.sdr.close()
            except Exception:
                pass
            if self.receive_thread and self.receive_thread.is_alive():
                self.receive_thread.join(timeout=2.0)
            print("📡 Reception stopped")

    def process_once(self):
        samples = self.iq_buffer.read_latest(self.samples_per_acquisition)
        if samples is None:
            return [], [], 0.0
        try:
            sat_peaks, detected_prns = self.b1c_acq.process_samples(samples)
            power = float(np.mean(np.abs(samples) ** 2))
            if len(sat_peaks) > 0:
                top_indices = np.argsort(sat_peaks)[-5:][::-1]
                print("   Top 5 peak metrics:")
                for idx in top_indices:
                    prn = idx + 1
                    status = "✓" if sat_peaks[idx] > self.b1c_acq.acq_threshold else " "
                    print(f"     {status} PRN {prn:2d}: {sat_peaks[idx]:.2f}")
            return sat_peaks, detected_prns, power
        except Exception as e:
            print(f"❌ Processing error: {e}")
            return [], [], 0.0

    def run_single(self):
        print("🛰️ REAL-TIME BeiDou B1C ACQUISITION (KrakenSDR/RTL-SDR)")
        print("=" * 70)
        if not self.start():
            return False
        try:
            sat_peaks, detected_prns, power = self.process_once()
            ts = datetime.now().strftime('%H:%M:%S')
            print("\n" + "=" * 70)
            print("📊 B1C ACQUISITION RESULTS")
            print("=" * 70)
            print(f"Time: {ts}")
            print(f"Signal power: {power:.6f}")
            print(f"Threshold: {self.b1c_acq.acq_threshold}")
            if detected_prns:
                print(f"✅ Detected {len(detected_prns)} BeiDou B1C satellites: {', '.join(['PRN'+str(p) for p in detected_prns])}")
                for prn in detected_prns:
                    print(f"   PRN {prn:2d}: Peak Metric = {sat_peaks[prn-1]:.2f}")
            else:
                print("❌ No B1C satellites detected")
            print("=" * 70)
            return True
        finally:
            self.stop()

    def run_continuous(self, update_interval=1.0):
        print("\n🔄 Starting continuous BeiDou B1C monitoring...")
        if not self.start():
            return False
        try:
            acquisition_count = 0
            while True:
                while self.iq_buffer.available() < self.samples_per_acquisition:
                    time.sleep(0.1)
                acquisition_count += 1
                print(f"\n[Acquisition #{acquisition_count}]")
                t0 = time.time()
                sat_peaks, detected_prns, power = self.process_once()
                dt = time.time() - t0
                print(f"   Processing time: {dt:.3f}s; Power: {power:.6f}")

                power_gate = (power >= self.min_signal_power)
                if detected_prns and power_gate:
                    high_confidence_prns = [prn for prn in detected_prns if sat_peaks[prn-1] > max(2.0, self.b1c_acq.acq_threshold + 0.5)]
                    confirmed = sorted(list(set(detected_prns).intersection(self.prev_detected)))
                    for prn in high_confidence_prns:
                        if prn not in confirmed:
                            confirmed.append(prn)
                    self.prev_detected = set(detected_prns)
                    if confirmed:
                        prn_str = ", ".join([f"PRN{p}" for p in confirmed])
                        print(f"   ✅ {len(confirmed)} satellites (confirmed): {prn_str}")
                        self.last_detection_time = time.time()
                    else:
                        print("   ℹ️ Candidates detected; awaiting confirmation")
                elif detected_prns and not power_gate:
                    print("   ⚠️ Power below gate; ignoring detections")
                    self.prev_detected = set()
                else:
                    print("   ❌ No satellites detected")
                time.sleep(update_interval)
        except KeyboardInterrupt:
            print("\n⚠️ Stopped by user")
        finally:
            self.stop()
        return True


def main():
    config = {
        'center_freq': 1575.42e6,
        'sample_rate': 3.2e6,
        'gain': 'auto',
        'device_index': 0,
        'buffer_size_ms': 5000,
        'acq_threshold': 1.3,
        'bandwidth': 3.2e6,
        'rx_offset_khz': 0.0,
        'channel': 'pilot',
    }

    mode = 'continuous'
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg == 'single':
                mode = 'single'
            elif arg == 'continuous':
                mode = 'continuous'
            elif arg.startswith('device='):
                config['device_index'] = int(arg.split('=')[1])
            elif arg.startswith('gain='):
                val = arg.split('=')[1]
                config['gain'] = 'auto' if val == 'auto' else float(val)
            elif arg.startswith('rate='):
                config['sample_rate'] = float(arg.split('=')[1]) * 1e6
            elif arg.startswith('bandwidth='):
                config['bandwidth'] = float(arg.split('=')[1]) * 1e6
            elif arg.startswith('offset_khz='):
                config['rx_offset_khz'] = float(arg.split('=')[1])
            elif arg.startswith('threshold='):
                config['acq_threshold'] = float(arg.split('=')[1])
            elif arg.startswith('channel='):
                config['channel'] = arg.split('=')[1]

    print("🛰️ KrakenSDR BeiDou B1C Acquisition")
    print(f"Mode: {mode}, Device: {config['device_index']}, SR: {config['sample_rate']/1e6:.1f} MHz, BW: {(config['bandwidth'] or 0)/1e6:.1f} MHz, Gain: {config['gain']}, Channel: {config['channel']}")

    try:
        sys_obj = KrakenRTLSDRBeidouB1C(**config)
        if mode == 'single':
            ok = sys_obj.run_single()
        else:
            ok = sys_obj.run_continuous(update_interval=1.0)
        return 0 if ok else 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == '__main__':
    # print("Usage:")
    # print("  python kraken_rtlsdr_beidou_b1c_acquisition.py                # Continuous")
    # print("  python kraken_rtlsdr_beidou_b1c_acquisition.py single         # Single run")
    print("  python kraken_rtlsdr_beidou_b1c_acquisition.py device=1 gain=40 rate=3.2 bandwidth=3.2 offset_khz=0 threshold=2.0")
    print()
    sys.exit(main())


