import numpy as np


class BeidouB1CAcquisition:
    """
    BeiDou B1C (pilot, B1CP) acquisition engine.
    Notes:
    - B1C primary code length: 10230 chips at 1.023 Mcps (10 ms)
    - We use the official Weil-code parameters and truncation points from
      PocketSDR to ensure PRN-perfect primary codes.
    - For coarse acquisition we correlate with the primary sequence only
      (ignoring BOC/QMBOC subcarrier shaping), similar to the E1C engine.
    """

    def __init__(self,
                 sampling_freq=3.2e6,
                 IF=0.0,
                 acq_satellite_list=None,
                 acq_search_band=14.0,
                 acq_threshold=2.0,
                 channel='pilot'):
        self.sampling_freq = float(sampling_freq)
        self.IF = float(IF)
        self.acq_threshold = float(acq_threshold)
        self.acq_search_band = float(acq_search_band)  # kHz, around IF
        # channel: 'pilot' (B1CP), 'data' (B1CD), or 'both'
        self.channel = channel.lower()

        # BeiDou PRN range for B1C is 1..63; search all by default
        if acq_satellite_list is None:
            #  self.acq_satellite_list = list(range(19, 59))
            self.acq_satellite_list = list(range(1, 64))
            
        else:
            self.acq_satellite_list = list(acq_satellite_list)

        # B1C primary code constants
        self.code_freq_basis = 1.023e6
        self.primary_code_length_chips = 10230  # 10 ms

        # Precompute primary code table (digitized at sampling_freq)
        self.ca_codes_table = self._make_primary_table()

    # -------------- PocketSDR-derived B1C primary codes ------------------------
    # Arrays copied from PocketSDR (sdr_code.py): B1CD/B1CP *_ph_diff, *_trunc_pnt
    # and the generic Legendre-derived Weil sequence helpers.

    B1CD_ph_diff = (
        2678, 4802,  958,  859, 3843, 2232,  124, 4352, 1816, 1126, 1860, 4800,
        2267,  424, 4192, 4333, 2656, 4148,  243, 1330, 1593, 1470,  882, 3202,
        5095, 2546, 1733, 4795, 4577, 1627, 3638, 2553, 3646, 1087, 1843,  216,
        2245,  726, 1966,  670, 4130,   53, 4830,  182, 2181, 2006, 1080, 2288,
        2027,  271,  915,  497,  139, 3693, 2054, 4342, 3342, 2592, 1007,  310,
        4203,  455, 4318
    )

    B1CD_trunc_pnt = (
         699,  694, 7318, 2127,  715, 6682, 7850, 5495, 1162, 7682, 6792, 9973,
        6596, 2092,   19,10151, 6297, 5766, 2359, 7136, 1706, 2128, 6827,  693,
        9729, 1620, 6805,  534,  712, 1929, 5355, 6139, 6339, 1470, 6867, 7851,
        1162, 7659, 1156, 2672, 6043, 2862,  180, 2663, 6940, 1645, 1582,  951,
        6878, 7701, 1823, 2391, 2606,  822, 6403,  239,  442, 6769, 2560, 2502,
        5072, 7268,  341
    )

    B1CP_ph_diff = (
         796,  156, 4198, 3941, 1374, 1338, 1833, 2521, 3175,  168, 2715, 4408,
        3160, 2796,  459, 3594, 4813,  586, 1428, 2371, 2285, 3377, 4965, 3779,
        4547, 1646, 1430,  607, 2118, 4709, 1149, 3283, 2473, 1006, 3670, 1817,
         771, 2173,  740, 1433, 2458, 3459, 2155, 1205,  413,  874, 2463, 1106,
        1590, 3873, 4026, 4272, 3556,  128, 1200,  130, 4494, 1871, 3073, 4386,
        4098, 1923, 1176
    )

    B1CP_trunc_pnt = (
        7575, 2369, 5688,  539, 2270, 7306, 6457, 6254, 5644, 7119, 1402, 5557,
        5764, 1073, 7001, 5910,10060, 2710, 1546, 6887, 1883, 5613, 5062, 1038,
       10170, 6484, 1718, 2535, 1158,  526, 7331, 5844, 6423, 6968, 1280, 1838,
        1989, 6468, 2091, 1581, 1453, 6252, 7122, 7711, 7216, 2113, 1095, 1628,
        1713, 6102, 6123, 6070, 1115, 8047, 6795, 2575,   53, 1729, 6388,  682,
        5565, 7160, 2277
    )

    def _gen_legendre_seq(self, N: int) -> np.ndarray:
        L = np.full(N, 1, dtype=np.int8)
        for i in range(1, N):
            L[(i * i) % N] = -1
        return L

    def _b1c_weil_code(self, k: int, w: int, L_seq: np.ndarray) -> int:
        return int(L_seq[k] * L_seq[(k + w) % len(L_seq)])

    def _generate_primary_code_cp(self, prn: int) -> np.ndarray:
        if prn < 1 or prn > 63:
            raise ValueError("PRN out of range for B1C (1..63)")
        # Use pilot channel (B1CP) primary code as per PocketSDR (no subcarrier)
        N = self.primary_code_length_chips
        L10243 = self._gen_legendre_seq(10243)
        code = np.zeros(N, dtype=np.int8)
        w = self.B1CP_ph_diff[prn - 1]
        tp = self.B1CP_trunc_pnt[prn - 1]
        for i in range(N):
            j = (i + tp - 1) % 10243
            code[i] = self._b1c_weil_code(j, w, L10243)
        return code.astype(np.float64)

    def _generate_primary_code_cd(self, prn: int) -> np.ndarray:
        if prn < 1 or prn > 63:
            raise ValueError("PRN out of range for B1C (1..63)")
        # Data channel (B1CD)
        N = self.primary_code_length_chips
        L10243 = self._gen_legendre_seq(10243)
        code = np.zeros(N, dtype=np.int8)
        w = self.B1CD_ph_diff[prn - 1]
        tp = self.B1CD_trunc_pnt[prn - 1]
        for i in range(N):
            j = (i + tp - 1) % 10243
            code[i] = self._b1c_weil_code(j, w, L10243)
        return code.astype(np.float64)

    def _make_primary_table(self):
        samples_per_code = int(round(self.sampling_freq / (self.code_freq_basis / self.primary_code_length_chips)))
        table_cp = np.zeros((64, samples_per_code), dtype=np.float64)
        table_cd = np.zeros((64, samples_per_code), dtype=np.float64)

        ts = 1.0 / self.sampling_freq
        tc = 1.0 / self.code_freq_basis

        for prn in self.acq_satellite_list:
            code_value_index = np.ceil((ts * np.arange(1, samples_per_code + 1)) / tc) - 1
            code_value_index = code_value_index.astype(int)
            code_value_index[code_value_index >= self.primary_code_length_chips] = self.primary_code_length_chips - 1

            seq_cp = self._generate_primary_code_cp(prn)
            table_cp[prn - 1, :] = seq_cp[code_value_index]

            seq_cd = self._generate_primary_code_cd(prn)
            table_cd[prn - 1, :] = seq_cd[code_value_index]

        return {'cp': table_cp, 'cd': table_cd}

    # ----------------------------- Acquisition ---------------------------------
    def acquisition(self, long_signal, acq_threshold):
        samples_per_code = int(round(self.sampling_freq / (self.code_freq_basis / self.primary_code_length_chips)))

        # Use two code periods for robustness
        signal1 = long_signal[0:samples_per_code]
        signal2 = long_signal[samples_per_code:2 * samples_per_code]

        signal0_dc = long_signal - np.mean(long_signal)

        ts = 1.0 / self.sampling_freq
        phase_points = np.arange(samples_per_code) * 2.0 * np.pi * ts

        number_of_frq_bins = int(round(self.acq_search_band * 2.0)) + 1

        acq_results = {
            'carrFreq': np.zeros(64),
            'codePhase': np.zeros(64),
            'peakMetric': np.zeros(64)
        }

        for prn in self.acq_satellite_list:
            # Prepare FFTs for selected channel(s)
            use_cp = self.channel in ('pilot', 'both')
            use_cd = self.channel in ('data', 'both')
            cp_fft = None
            cd_fft = None
            if use_cp:
                cp_fft = np.conj(np.fft.fft(self.ca_codes_table['cp'][prn - 1, :]))
            if use_cd:
                cd_fft = np.conj(np.fft.fft(self.ca_codes_table['cd'][prn - 1, :]))

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

                # Correlate with selected channel(s) and take max power
                chan_powers = []
                if use_cp:
                    conv1 = iq_freq_dom1 * cp_fft
                    conv2 = iq_freq_dom2 * cp_fft
                    acq1 = np.abs(np.fft.ifft(conv1)) ** 2
                    acq2 = np.abs(np.fft.ifft(conv2)) ** 2
                    chan_powers.append(acq1 if np.max(acq1) > np.max(acq2) else acq2)
                if use_cd:
                    conv1 = iq_freq_dom1 * cd_fft
                    conv2 = iq_freq_dom2 * cd_fft
                    acq1 = np.abs(np.fft.ifft(conv1)) ** 2
                    acq2 = np.abs(np.fft.ifft(conv2)) ** 2
                    chan_powers.append(acq1 if np.max(acq1) > np.max(acq2) else acq2)

                if len(chan_powers) == 1:
                    results[frq_bin_index, :] = chan_powers[0]
                else:
                    # both channels: pick elementwise max
                    results[frq_bin_index, :] = np.maximum(chan_powers[0], chan_powers[1])

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
                    # Use the same channel(s) as above for fine frequency; default to CP
                    prn_primary = self._generate_primary_code_cp(prn) if use_cp else self._generate_primary_code_cd(prn)
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


