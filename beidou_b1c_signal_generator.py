import numpy as np
from beidou_b1c_acquisition import BeiDouB1CAcquisition


class BeiDouB1CSignalGenerator:
    """
    BeiDou B1C (pilot) signal generator for testing acquisition.

    - Generates complex baseband signal with primary B1C pilot code
      (assumed 10230 chips) suitable for coarse acquisition testing.
    - Optional BOC(1,1)-like shaping can be applied for realism.
    - Secondary code and full TMBOC details are omitted for simplicity,
      mirroring the Galileo E1C approach used in this project.
    """

    def __init__(self,
                 sampling_freq=3.2e6,
                 IF=0.0,
                 duration=10,
                 apply_boc_shaping=False):
        self.sampling_freq = float(sampling_freq)
        self.IF = float(IF)
        self.duration = int(duration)
        self.apply_boc_shaping = bool(apply_boc_shaping)

        # B1C constants
        self.code_freq_basis = 1.023e6
        self.primary_code_length_chips = 10230

        # Reuse acquisition class for consistent primary-code replicas
        self.acq = BeiDouB1CAcquisition(sampling_freq=self.sampling_freq, IF=self.IF)

    def _generate_boc11_subcarrier(self, num_samples):
        # Simple BOC(1,1) square subcarrier at 1.023 MHz
        f_sc = 1.023e6
        t = np.arange(num_samples) / self.sampling_freq
        return np.sign(np.sin(2.0 * np.pi * f_sc * t)).astype(np.float64)

    def generate_satellite_signal(self, prn, doppler_hz=0.0, code_phase=0, amplitude=1.0):
        total_samples = int(self.sampling_freq * self.duration)
        samples_per_code = int(round(self.sampling_freq / (self.code_freq_basis / self.primary_code_length_chips)))

        # Primary code replica (digitized to sampling grid from acquisition table)
        primary = self.acq.ca_codes_table[prn - 1, :].copy()
        num_codes_needed = int(np.ceil(total_samples / samples_per_code)) + 1
        long_primary = np.tile(primary, num_codes_needed)

        # Apply code phase (in samples)
        code_phase_samples = int(code_phase) % samples_per_code
        long_primary = np.roll(long_primary, code_phase_samples)
        long_primary = long_primary[:total_samples]

        # Optional BOC(1,1) shaping
        if self.apply_boc_shaping:
            boc = self._generate_boc11_subcarrier(total_samples)
        else:
            boc = 1.0

        # Carrier/Doppler at IF
        t = np.arange(total_samples) / self.sampling_freq
        carrier = np.exp(1j * 2.0 * np.pi * (self.IF + doppler_hz) * t)

        signal = amplitude * carrier * (long_primary * boc)
        return signal.astype(np.complex128)

    def generate_multi_satellite_signal(self, satellite_configs):
        total_samples = int(self.sampling_freq * self.duration)
        combined = np.zeros(total_samples, dtype=np.complex128)

        for cfg in satellite_configs:
            prn = int(cfg.get('prn', 1))
            doppler = float(cfg.get('doppler', 0.0))
            phase = int(cfg.get('phase', 0))
            amplitude = float(cfg.get('amplitude', 1.0))
            sat = self.generate_satellite_signal(prn, doppler, phase, amplitude)
            combined += sat

        # Add white Gaussian noise
        noise_power = 0.05
        noise = (np.random.normal(0, np.sqrt(noise_power/2), total_samples) +
                 1j * np.random.normal(0, np.sqrt(noise_power/2), total_samples))
        combined += noise
        return combined


