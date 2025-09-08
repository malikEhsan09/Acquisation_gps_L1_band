import numpy as np


class BeiDouB1CAcquisition:
    """
    BeiDou B1C (pilot) acquisition engine.

    Notes:
    - B1C pilot primary code length assumed: 10230 chips at 1.023 Mcps (≈10.0 ms)
      This uses a Weil-sequence scaffold over the prime p = 10231 so that the
      sequence length is p-1 = 10230. It is deterministic and per-PRN unique.
      For lab testing with the included signal generator, this is sufficient.
      To acquire real BeiDou satellites, replace `generate_primary_code` with
      official B1C pilot code generation per the ICD or a lookup table.
    - Acquisition structure mirrors the existing Galileo E1C implementation.
    """

    def __init__(self,
                 sampling_freq=3.2e6,
                 IF=0.0,
                 acq_satellite_list=None,
                 acq_search_band=14.0,
                 acq_threshold=2.0): 
        self.sampling_freq = float(sampling_freq)
        self.IF = float(IF)
        self.acq_threshold = float(acq_threshold)
        self.acq_search_band = float(acq_search_band)  # kHz, around IF

        # Typical BDS-3 PRNs span roughly 1..63. Use 1..50 as default search set.
        if acq_satellite_list is None:
            self.acq_satellite_list = list(range(1, 51))
        else:
            self.acq_satellite_list = list(acq_satellite_list)

        # B1C primary code constants
        self.code_freq_basis = 1.023e6
        self.primary_code_length_chips = 10230
        self.prime_p = 10231  # prime for Weil construction

        # Precompute primary code table (digitized at sampling_freq)
        self.ca_codes_table = self._make_primary_table()

    # -----------------------------
    # Primary code generation (Weil scaffold)
    # -----------------------------
    def _legendre_symbol(self, a, p):
        a_mod = a % p
        if a_mod == 0:
            return 0
        val = pow(a_mod, (p - 1) // 2, p)
        return -1 if val == p - 1 else 1

    def _weil_sequence(self, d, p0):
        p = self.prime_p
        seq = np.empty(self.primary_code_length_chips, dtype=np.int8)
        for k in range(self.primary_code_length_chips):
            x = (k + 1) % p
            x_shift = (x + p0) % p
            Lx = self._legendre_symbol(x_shift, p)
            Lxd = self._legendre_symbol((x_shift + d) % p, p)
            v = Lx * Lxd
            if v == 0:
                v = 1
            seq[k] = 1 if v > 0 else -1
        return seq.astype(np.int8)

    def _get_weil_parameters_for_prn(self, prn):
        p = self.prime_p
        d = (137 * prn + 59) % (p - 1)
        if d == 0:
            d = 1
        p0 = (311 * prn + 97) % p
        return int(d), int(p0)

    def generate_primary_code(self, prn):
        if prn < 1 or prn > 200:
            raise ValueError(f"PRN {prn} not supported (valid range: 1-200)")
        d, p0 = self._get_weil_parameters_for_prn(prn)
        return self._weil_sequence(d, p0).astype(np.int8)

    def _make_primary_table(self):
        samples_per_code = int(round(self.sampling_freq / (self.code_freq_basis / self.primary_code_length_chips)))
        table = np.zeros((max(self.acq_satellite_list), samples_per_code), dtype=np.float64)

        ts = 1.0 / self.sampling_freq
        tc = 1.0 / self.code_freq_basis

        for prn in self.acq_satellite_list:
            prn_seq = self.generate_primary_code(prn).astype(np.float64)
            code_value_index = np.ceil((ts * np.arange(1, samples_per_code + 1)) / tc) - 1
            code_value_index = code_value_index.astype(int)
            code_value_index[code_value_index >= self.primary_code_length_chips] = self.primary_code_length_chips - 1
            table[prn - 1, :] = prn_seq[code_value_index]

        return table

    # -----------------------------
    # Acquisition
    # -----------------------------
    def acquisition(self, long_signal, acq_threshold):
        samples_per_code = int(round(self.sampling_freq / (self.code_freq_basis / self.primary_code_length_chips)))

        # Use two chunks of one code period each for robustness
        signal1 = long_signal[0:samples_per_code]
        signal2 = long_signal[samples_per_code:2 * samples_per_code]

        signal0_dc = long_signal - np.mean(long_signal)

        ts = 1.0 / self.sampling_freq
        phase_points = np.arange(samples_per_code) * 2.0 * np.pi * ts

        number_of_frq_bins = int(round(self.acq_search_band * 2.0)) + 1

        acq_results = {
            'carrFreq': np.zeros(80),
            'codePhase': np.zeros(80),
            'peakMetric': np.zeros(80)
        }

        for prn in self.acq_satellite_list:
            ca_code_freq_dom = np.conj(np.fft.fft(self.ca_codes_table[prn - 1, :]))

            results = np.zeros((number_of_frq_bins, samples_per_code))
            frq_bins = np.zeros(number_of_frq_bins)

            for frq_bin_index in range(number_of_frq_bins):
                frq_bins[frq_bin_index] = (
                    self.IF - (self.acq_search_band / 2.0) * 1000.0 + 0.5e3 * frq_bin_index
                )

                sig_carr = np.exp(1j * frq_bins[frq_bin_index] * phase_points)

                sig1 = sig_carr * signal1
                sig2 = sig_carr * signal2

                iq_freq_dom1 = np.fft.fft(sig1)
                iq_freq_dom2 = np.fft.fft(sig2)

                conv_code_iq1 = iq_freq_dom1 * ca_code_freq_dom
                conv_code_iq2 = iq_freq_dom2 * ca_code_freq_dom

                acq_res1 = np.abs(np.fft.ifft(conv_code_iq1)) ** 2
                acq_res2 = np.abs(np.fft.ifft(conv_code_iq2)) ** 2

                if np.max(acq_res1) > np.max(acq_res2):
                    results[frq_bin_index, :] = acq_res1
                else:
                    results[frq_bin_index, :] = acq_res2

            # Peak detection
            peak_size, frequency_bin_index = self._find_peak(results)
            _, code_phase = self._find_peak(results.T)

            # Peak metric (exclude +/- one chip zone around peak)
            samples_per_code_chip = int(round(self.sampling_freq / self.code_freq_basis))
            exclude_range_index1 = int(code_phase - samples_per_code_chip)
            exclude_range_index2 = int(code_phase + samples_per_code_chip)

            if exclude_range_index1 < 2:
                code_phase_range = np.concatenate([
                    np.arange(exclude_range_index2, samples_per_code),
                    np.arange(exclude_range_index1)
                ])
            elif exclude_range_index2 >= samples_per_code:
                code_phase_range = np.concatenate([
                    np.arange(exclude_range_index2 - samples_per_code, exclude_range_index1)
                ])
            else:
                code_phase_range = np.concatenate([
                    np.arange(exclude_range_index1),
                    np.arange(exclude_range_index2, samples_per_code)
                ])

            code_phase_range = code_phase_range[
                (code_phase_range >= 0) & (code_phase_range < samples_per_code)
            ]

            if len(code_phase_range) > 0:
                second_peak_size = np.max(results[frequency_bin_index, code_phase_range])
                acq_results['peakMetric'][prn - 1] = peak_size / max(second_peak_size, 1e-12)

                if (peak_size / max(second_peak_size, 1e-12)) > acq_threshold:
                    # Fine frequency search
                    prn_primary = self.generate_primary_code(prn)
                    code_value_index = np.floor((ts * np.arange(1, 10 * samples_per_code + 1)) /
                                                (1.0 / self.code_freq_basis))
                    long_ca_code = prn_primary[(code_value_index.astype(int) % self.primary_code_length_chips)]

                    x_carrier = (signal0_dc[int(code_phase):(int(code_phase) + 10 * samples_per_code)] *
                                 long_ca_code)

                    fft_num_pts = 8 * (2 ** int(np.log2(len(x_carrier)) + 1))
                    fft_xc = np.abs(np.fft.fft(x_carrier, fft_num_pts))

                    uniq_fft_pts = int(np.ceil((fft_num_pts + 1) / 2))
                    fft_max_index = int(np.argmax(fft_xc))
                    fft_freq_bins = np.arange(uniq_fft_pts) * self.sampling_freq / fft_num_pts

                    if fft_max_index > uniq_fft_pts:
                        if fft_num_pts % 2 == 0:
                            fft_freq_bins_rev = -fft_freq_bins[(uniq_fft_pts - 1):0:-1]
                            fft_max_index = int(np.argmax(fft_xc[(uniq_fft_pts):]))
                            acq_results['carrFreq'][prn - 1] = -fft_freq_bins_rev[fft_max_index]
                        else:
                            fft_freq_bins_rev = -fft_freq_bins[(uniq_fft_pts):0:-1]
                            fft_max_index = int(np.argmax(fft_xc[(uniq_fft_pts):]))
                            acq_results['carrFreq'][prn - 1] = fft_freq_bins_rev[fft_max_index]
                    else:
                        acq_results['carrFreq'][prn - 1] = fft_freq_bins[fft_max_index]

                    acq_results['codePhase'][prn - 1] = code_phase

        return acq_results

    def _find_peak(self, data):
        max_idx = np.unravel_index(np.argmax(data), data.shape)
        return data[max_idx], max_idx[0]

    def process_samples(self, samples):
        samples = np.asarray(samples, dtype=np.complex128).flatten()
        acq_results = self.acquisition(samples, self.acq_threshold)
        sat_peaks = acq_results['peakMetric']

        detected_prns = []
        for i, peak in enumerate(sat_peaks):
            if peak > self.acq_threshold:
                detected_prns.append(i + 1)

        return sat_peaks, detected_prns


