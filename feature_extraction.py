import numpy as np
from scipy.signal import find_peaks
from filteringNoise import X_train_wavelet, X_test_wavelet

class ECGFeatureExtractor:
    """
    Extracts features from ECG signals, including R-peaks, T-peaks, RR intervals,
    QRS durations, heart rate, and amplitude statistics.
    """

    def __init__(self, signals, sampling_rate=1000):
        """
        Parameters:
        -----------
        signals : np.ndarray
            2D array of shape (n_signals, n_samples)
        sampling_rate : float
            Sampling rate of the signals (default 1000 Hz)
        """
        self.signals = np.array(signals)
        self.sampling_rate = sampling_rate

    def extract_features_single(self, signal):
        """Extract features from a single ECG signal."""
        # 1. Find R-peaks
        r_peaks = find_peaks(signal)[0]

        r_amplitudes = []
        t_amplitudes = []

        # 2. Extract R and T peak amplitudes
        for r_peak in r_peaks:
            t_peak = r_peak + np.argmin(signal[r_peak:r_peak+200])
            r_amplitudes.append(signal[r_peak])
            t_amplitudes.append(signal[t_peak])

        # 3. Metrics for R-peaks
        std_r, mean_r, median_r, sum_r = np.std(r_amplitudes), np.mean(r_amplitudes), np.median(r_amplitudes), np.sum(r_amplitudes)
        # 4. Metrics for T-peaks
        std_t, mean_t, median_t, sum_t = np.std(t_amplitudes), np.mean(t_amplitudes), np.median(t_amplitudes), np.sum(t_amplitudes)

        # 5. RR intervals
        rr_intervals = np.diff(r_peaks)
        std_rr, mean_rr, median_rr, sum_rr = np.std(rr_intervals), np.mean(rr_intervals), np.median(rr_intervals), np.sum(rr_intervals)

        # 6. QRS durations (differences between consecutive R-peaks)
        qrs_durations = np.diff(np.insert(r_peaks, 0, r_peaks[0]))  # insert first element to avoid empty diff
        std_qrs, mean_qrs, median_qrs, sum_qrs = np.std(qrs_durations), np.mean(qrs_durations), np.median(qrs_durations), np.sum(qrs_durations)

        # 7. Overall statistics
        std_signal, mean_signal = np.std(signal), np.mean(signal)

        # 8. Heart rate
        duration_sec = len(signal) / self.sampling_rate
        heart_rate = (len(r_peaks) / duration_sec) * 60

        # 9. Combine all features
        features = [
            mean_signal, std_signal,
            std_qrs, mean_qrs, median_qrs, sum_qrs,
            std_r, mean_r, median_r, sum_r,
            std_t, mean_t, median_t, sum_t,
            sum_rr, std_rr, mean_rr, median_rr,
            heart_rate
        ]
        return features

    def extract_features(self):
        """Extract features for all signals in the dataset."""
        all_features = [self.extract_features_single(sig) for sig in self.signals]
        return np.array(all_features)


# training set
feature_extractor_train = ECGFeatureExtractor(X_train_wavelet, sampling_rate=360)
X_train_fe = feature_extractor_train.extract_features()

# test set
feature_extractor_test = ECGFeatureExtractor(X_test_wavelet, sampling_rate=360)
X_test_fe = feature_extractor_test.extract_features()

print("Feature array shape (train):", X_train_fe.shape)
print("Feature test array shape (test):", X_test_fe.shape)

