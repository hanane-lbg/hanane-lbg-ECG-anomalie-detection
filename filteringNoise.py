import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.signal import medfilt, butter, filtfilt
import pywt

path = ''
df= pd.read_csv(path)

mapping = {'Normal': 1, 'Anomaly': 0}
df['target_numeric'] = df['target'].map(mapping)

labels=df['target_numeric']
ecg_data = df.drop(['Unnamed: 0', 'target', 'target_numeric'], axis=1)

scaler = MinMaxScaler(feature_range=(-1, 1))
ecg_scaled = scaler.fit_transform(ecg_data) 

#Split the data 
X_train, X_test, y_train, y_test = train_test_split(
    ecg_scaled, labels, 
    test_size=0.2, 
    stratify=labels, 
    random_state=42
)

class ECGFilter:
    """
    Class for ECG signal filtering and evaluation.
    Supports median, bandpass, and wavelet filtering.
    """

    def __init__(self, ecg_data, fs=360.0):
        """
        Parameters:
        -----------
        ecg_data : numpy.ndarray
            ECG signals (shape: n_signals x n_samples)
        fs : float
            Sampling frequency of the ECG signals
        """
        self.ecg_data = np.array(ecg_data)  
        self.fs = fs
        self.filtered = {}  # store filtered signals

    # FILTERING METHODS
    def median_filter(self, kernel_size=3):
        """Apply median filtering to all ECG signals."""
        self.filtered['median'] = np.array([medfilt(sig, kernel_size) for sig in self.ecg_data])
        return self.filtered['median']

    def bandpass_filter(self, lowcut=0.05, highcut=20.0, order=4):
        """Apply bandpass filtering to all ECG signals."""
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        self.filtered['bandpass'] = np.array([filtfilt(b, a, sig) for sig in self.ecg_data])
        return self.filtered['bandpass']

    def wavelet_filter(self, wavelet='db4', level=1):
        """Apply wavelet denoising to all ECG signals."""
        filtered_signals = []
        for sig in self.ecg_data:
            coeffs = pywt.wavedec(sig, wavelet, level=level)
            threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(sig)))
            coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
            filtered_sig = pywt.waverec(coeffs, wavelet)
            filtered_signals.append(filtered_sig[:len(sig)])  # ensure same length
        self.filtered['wavelet'] = np.array(filtered_signals)
        return self.filtered['wavelet']


    # MSE EVALUATION
    @staticmethod
    def compute_mse(original, filtered):
        """Compute mean squared error between original and filtered signals."""
        original = np.array(original)
        filtered = np.array(filtered)

        # pad or trim if needed
        diff = original.shape[1] - filtered.shape[1]
        if diff > 0:
            padding = np.zeros((filtered.shape[0], diff))
            filtered = np.concatenate((filtered, padding), axis=1)
        elif diff < 0:
            filtered = filtered[:, :original.shape[1]]

        return np.mean((original - filtered) ** 2)

    def evaluate(self):
        """Compute MSE for all applied filters."""
        mse_results = {}
        for key, filt in self.filtered.items():
            mse_results[key] = self.compute_mse(self.ecg_data, filt)
        return mse_results

# Note: Initialize filter object on TRAIN data only to avoid data leakage
ecg_filter_train = ECGFilter(X_train, fs=360)

# Apply filters on training data
X_train_med = ecg_filter_train.median_filter(kernel_size=3)
X_train_band = ecg_filter_train.bandpass_filter(lowcut=0.05, highcut=20.0)
X_train_wavelet = ecg_filter_train.wavelet_filter(wavelet='db4', level=1)

# Apply the same filters to test data
ecg_filter_test = ECGFilter(X_test, fs=360)
X_test_med = ecg_filter_test.median_filter(kernel_size=3)
X_test_band = ecg_filter_test.bandpass_filter(lowcut=0.05, highcut=20.0)
X_test_wavelet = ecg_filter_test.wavelet_filter(wavelet='db4', level=1)


