import numpy as np

class GPSAcquisition:
    """
    GPS Acquisition - Core functionality for GPS signal acquisition
    """
    
    def __init__(self, 
                 sampling_freq=2.4e6,
                 IF=0e6,
                 acq_satellite_list=None,
                 acq_search_band=14,
                 acq_threshold=2.0):
        """
        Initialize GPS Acquisition
        
        Args:
            sampling_freq (float): Sampling frequency in Hz
            IF (float): Intermediate frequency in Hz
            acq_satellite_list (list): List of satellites to search for
            acq_search_band (float): Search band around IF in kHz
            acq_threshold (float): Threshold for signal presence decision
        """
        
        self.sampling_freq = sampling_freq
        self.IF = IF
        
        # Acquisition settings
        if acq_satellite_list is None:
            self.acq_satellite_list = list(range(1, 30)) + [31, 32]  # PRN 1-29, 31, 32
        else:
            self.acq_satellite_list = acq_satellite_list
        self.acq_search_band = acq_search_band
        self.acq_threshold = acq_threshold
        
        # GPS constants
        self.code_freq_basis = 1.023e6
        self.code_length = 1023
        
        # Initialize CA codes table
        self.ca_codes_table = self.make_ca_table()
    
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
    
    def acquisition(self, long_signal, acq_threshold):
        """Perform acquisition on signal"""
        samples_per_code = int(round(self.sampling_freq / (self.code_freq_basis / self.code_length)))
        
        # Create two 1msec vectors
        signal1 = long_signal[0:samples_per_code]
        signal2 = long_signal[samples_per_code:2*samples_per_code]
        
        signal0_dc = long_signal - np.mean(long_signal)
        
        ts = 1 / self.sampling_freq
        phase_points = np.arange(samples_per_code) * 2 * np.pi * ts
        
        number_of_frq_bins = int(round(self.acq_search_band * 2)) + 1
        
        acq_results = {
            'carrFreq': np.zeros(32),
            'codePhase': np.zeros(32),
            'peakMetric': np.zeros(32)
        }
        
        for prn in self.acq_satellite_list:
            ca_code_freq_dom = np.conj(np.fft.fft(self.ca_codes_table[prn - 1, :]))
            
            results = np.zeros((number_of_frq_bins, samples_per_code))
            frq_bins = np.zeros(number_of_frq_bins)
            
            for frq_bin_index in range(number_of_frq_bins):
                frq_bins[frq_bin_index] = (self.IF - 
                                          (self.acq_search_band/2) * 1000 + 
                                          0.5e3 * frq_bin_index)
                
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
            
            # Find peaks
            peak_size, frequency_bin_index = self._find_peak(results)
            _, code_phase = self._find_peak(results.T)
            
            # Calculate peak metric
            samples_per_code_chip = int(round(self.sampling_freq / self.code_freq_basis))
            exclude_range_index1 = code_phase - samples_per_code_chip
            exclude_range_index2 = code_phase + samples_per_code_chip
            
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
                acq_results['peakMetric'][prn - 1] = peak_size / second_peak_size
                
                if (peak_size / second_peak_size) > acq_threshold:
                    # Fine frequency search
                    ca_code = self.generate_ca_code(prn)
                    
                    code_value_index = np.floor((ts * np.arange(1, 10*samples_per_code + 1)) / 
                                               (1/self.code_freq_basis))
                    long_ca_code = ca_code[(code_value_index.astype(int) % 1023)]
                    
                    x_carrier = (signal0_dc[code_phase:(code_phase + 10*samples_per_code)] * 
                                long_ca_code)
                    
                    fft_num_pts = 8 * (2 ** int(np.log2(len(x_carrier)) + 1))
                    fft_xc = np.abs(np.fft.fft(x_carrier, fft_num_pts))
                    
                    uniq_fft_pts = int(np.ceil((fft_num_pts + 1) / 2))
                    fft_max_index = np.argmax(fft_xc)
                    fft_freq_bins = np.arange(uniq_fft_pts) * self.sampling_freq / fft_num_pts
                    
                    if fft_max_index > uniq_fft_pts:
                        if fft_num_pts % 2 == 0:
                            fft_freq_bins_rev = -fft_freq_bins[(uniq_fft_pts-1):0:-1]
                            fft_max_index = np.argmax(fft_xc[(uniq_fft_pts):])
                            acq_results['carrFreq'][prn - 1] = -fft_freq_bins_rev[fft_max_index]
                        else:
                            fft_freq_bins_rev = -fft_freq_bins[(uniq_fft_pts):0:-1]
                            fft_max_index = np.argmax(fft_xc[(uniq_fft_pts):])
                            acq_results['carrFreq'][prn - 1] = fft_freq_bins_rev[fft_max_index]
                    else:
                        acq_results['carrFreq'][prn - 1] = fft_freq_bins[fft_max_index]
                    
                    acq_results['codePhase'][prn - 1] = code_phase
        
        return acq_results
    
    def _find_peak(self, data):
        """Helper function to find peak in 2D array"""
        max_idx = np.unravel_index(np.argmax(data), data.shape)
        return data[max_idx], max_idx[0]
    
    def process_samples(self, samples):
        """Main processing function"""
        samples = np.asarray(samples, dtype=np.complex128)
        samples = samples.flatten()
        
        acq_results = self.acquisition(samples, self.acq_threshold)
        sat_peaks = acq_results['peakMetric']
        
        # Find detected satellites
        detected_prns = []
        for i, peak in enumerate(sat_peaks):
            if peak > self.acq_threshold:
                detected_prns.append(i + 1)  # PRN is 1-indexed
        
        return sat_peaks, detected_prns
