import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import welch

path=''
df = pd.read_csv(path)
target_column = 'target'  
y_labels_text = df[target_column].values
numerical_columns = [col for col in df.columns if col != target_column]
X_signals = df[numerical_columns].values 

print(f"\nX_signals shape: {X_signals.shape}")
print(f"Target column values: {np.unique(y_labels_text)}")



unique_classes = np.unique(y_labels_text)
print(f"\nUnique classes: {unique_classes}")

class_to_numeric = {unique_classes[0]: 0, unique_classes[1]: 1}
print(f"Class mapping: {class_to_numeric}")

y_labels = np.array([class_to_numeric[label] for label in y_labels_text])

print(f"Numeric labels: {np.unique(y_labels)}")
print(f"Normal (0) count: {np.sum(y_labels == 0)}")
print(f"Abnormal (1) count: {np.sum(y_labels == 1)}")




def create_overlay_with_mean(X_signals, y_labels, n_samples=10):
    """
    Create overlay plot with mean line
    
    Parameters:
    -----------
    X_signals : numpy array
        Shape: (n_samples, signal_length)
        All ECG signals
    
    y_labels : numpy array
        Shape: (n_samples,)
        Numeric labels: 0 = Normal, 1 = Abnormal
    
    n_samples : int
        Number of signals to plot from each class
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    normal_indices = np.where(y_labels == 0)[0][:n_samples]
    
    abnormal_indices = np.where(y_labels == 1)[0][:n_samples]
    

    # Normal ECG Signals 
    # Step 1: Plot each normal signal (transparent)
    for idx in normal_indices:
        signal = X_signals[idx]
        axes[0].plot(signal, color='green', alpha=0.4, linewidth=1)
    
    # Step 2: Calculate average of ALL normal signals
    all_normal_signals = X_signals[y_labels == 0]
    normal_mean = np.mean(all_normal_signals, axis=0)
    
    # Step 3: Plot the average (thick line)
    axes[0].plot(normal_mean, color='darkgreen', linewidth=3, label='Mean')
    
    # Step 4: Add labels and formatting
    axes[0].set_title('Normal ECG Signals', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Amplitude', fontsize=12)
    axes[0].set_xlabel('Sample Index', fontsize=12)
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=11)
    
    
  
    # Abnormal ECG Signals  
    # Step 1: Plot each abnormal signal (transparent)
    for idx in abnormal_indices:
        signal = X_signals[idx]
        axes[1].plot(signal, color='red', alpha=0.4, linewidth=1)
    
    # Step 2: Calculate average of ALL abnormal signals
    all_abnormal_signals = X_signals[y_labels == 1]
    abnormal_mean = np.mean(all_abnormal_signals, axis=0)
    
    # Step 3: Plot the average (thick line)
    axes[1].plot(abnormal_mean, color='darkred', linewidth=3, label='Mean')
    
    # Step 4: Add labels and formatting
    axes[1].set_title('Abnormal ECG Signals', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Amplitude', fontsize=12)
    axes[1].set_xlabel('Sample Index', fontsize=12)
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=11)
    
    

    plt.tight_layout()
    plt.savefig('03_overlay_with_mean.png', dpi=300, bbox_inches='tight')
    print("\n  '03_overlay_with_mean.png'")
    
    plt.show()
    
    return fig, axes



def plot_fft_comparison(signals, labels, sampling_rate=125):
    """
    Compare one normal and one abnormal ECG signal in both 
    time domain and frequency domain.
    """
    print("Creating FFT comparison plot...")

    normal_index = np.where(labels == 0)[0][0]
    abnormal_index = np.where(labels == 1)[0][0]

    normal_signal = signals[normal_index]
    abnormal_signal = signals[abnormal_index]

    # Compute the FFT (magnitude)
    fft_normal = np.abs(fft(normal_signal))
    fft_abnormal = np.abs(fft(abnormal_signal))

    # Compute the corresponding frequencies
    freqs = fftfreq(len(normal_signal), d=1/sampling_rate)
    freqs_pos = freqs[:len(normal_signal)//2]

    fft_normal_pos = fft_normal[:len(normal_signal)//2]
    fft_abnormal_pos = fft_abnormal[:len(abnormal_signal)//2]

    fig, ax = plt.subplots(2, 2, figsize=(16, 10))
    # normal signal in time domain
    ax[0, 0].plot(normal_signal, color='green')
    ax[0, 0].set_title('Normal Signal - Time Domain')
    ax[0, 0].set_ylabel('Amplitude')
    ax[0, 0].grid(alpha=0.3)

    # Abnormal signal in time domain
    ax[0, 1].plot(abnormal_signal, color='red')
    ax[0, 1].set_title('Abnormal Signal - Time Domain')
    ax[0, 1].set_ylabel('Amplitude')
    ax[0, 1].grid(alpha=0.3)

    # Normal signal in frequency domain
    ax[1, 0].semilogy(freqs_pos, fft_normal_pos, color='green')
    ax[1, 0].set_title('Normal Signal - Frequency Domain (FFT)')
    ax[1, 0].set_xlabel('Frequency (Hz)')
    ax[1, 0].set_ylabel('Magnitude (log scale)')
    ax[1, 0].set_xlim(0, 50)
    ax[1, 0].grid(alpha=0.3, which='both')

    # Abnormal signal in frequency domain
    ax[1, 1].semilogy(freqs_pos, fft_abnormal_pos, color='red')
    ax[1, 1].set_title('Abnormal Signal - Frequency Domain (FFT)')
    ax[1, 1].set_xlabel('Frequency (Hz)')
    ax[1, 1].set_ylabel('Magnitude (log scale)')
    ax[1, 1].set_xlim(0, 50)
    ax[1, 1].grid(alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('fft_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Plot saved as 'fft_comparison.png'.")



def plot_psd_comparison(signals, labels, sampling_rate=125):
    """
    Compare the Power Spectral Density (PSD) of one normal 
    and one abnormal signal using Welch's method.
    Saves the figure as 'psd_comparison.png'.
    """
    print("Creating PSD comparison plot...")

    # Select one example from each class
    normal_idx = np.where(labels == 0)[0][0]
    abnormal_idx = np.where(labels == 1)[0][0]

    normal_signal = signals[normal_idx]
    abnormal_signal = signals[abnormal_idx]

    # Compute PSD using Welch's method
    freqs_normal, psd_normal = welch(normal_signal, fs=sampling_rate, nperseg=256)
    freqs_abnormal, psd_abnormal = welch(abnormal_signal, fs=sampling_rate, nperseg=256)

    # Create figure with two subplots
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    # Normal signal PSD
    ax[0].semilogy(freqs_normal, psd_normal, color='green', linewidth=2)
    ax[0].fill_between(freqs_normal, psd_normal, alpha=0.3, color='green')
    ax[0].set_title('Normal - Power Spectral Density (Welch)')
    ax[0].set_xlabel('Frequency (Hz)')
    ax[0].set_ylabel('Power (log scale)')
    ax[0].set_xlim(0, 50)
    ax[0].grid(alpha=0.3, which='both')

    # Abnormal signal PSD
    ax[1].semilogy(freqs_abnormal, psd_abnormal, color='red', linewidth=2)
    ax[1].fill_between(freqs_abnormal, psd_abnormal, alpha=0.3, color='red')
    ax[1].set_title('Abnormal - Power Spectral Density (Welch)')
    ax[1].set_xlabel('Frequency (Hz)')
    ax[1].set_ylabel('Power (log scale)')
    ax[1].set_xlim(0, 50)
    ax[1].grid(alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('psd_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Plot saved as 'psd_comparison.png'.")


def plot_statistical_comparison(signals, labels):
    """
    Create multiple statistical comparisons between normal and abnormal signals:
    - Mean ± Standard Deviation
    - Min-Max range
    - Summary metrics bar chart
    - Amplitude distribution histogram
    .
    """
    print("Creating statistical comparison plot...")

    normal_signals = signals[labels == 0]
    abnormal_signals = signals[labels == 1]

    # Compute statistics
    normal_mean = np.mean(normal_signals, axis=0)
    normal_std = np.std(normal_signals, axis=0)
    abnormal_mean = np.mean(abnormal_signals, axis=0)
    abnormal_std = np.std(abnormal_signals, axis=0)

    x = np.arange(len(normal_mean))
    fig, ax = plt.subplots(2, 2, figsize=(16, 10))
    ax[0, 0].plot(x, normal_mean, color='green', label='Normal')
    ax[0, 0].fill_between(x, normal_mean - normal_std, normal_mean + normal_std, alpha=0.3, color='green')
    ax[0, 0].plot(x, abnormal_mean, color='red', label='Abnormal')
    ax[0, 0].fill_between(x, abnormal_mean - abnormal_std, abnormal_mean + abnormal_std, alpha=0.3, color='red')
    ax[0, 0].set_title('Mean ± Std Dev')
    ax[0, 0].set_ylabel('Amplitude')
    ax[0, 0].legend()
    ax[0, 0].grid(alpha=0.3)

    # Min-Max Range
    normal_min, normal_max = np.min(normal_signals, axis=0), np.max(normal_signals, axis=0)
    abnormal_min, abnormal_max = np.min(abnormal_signals, axis=0), np.max(abnormal_signals, axis=0)
    ax[0, 1].fill_between(x, normal_min, normal_max, alpha=0.4, color='green', label='Normal')
    ax[0, 1].fill_between(x, abnormal_min, abnormal_max, alpha=0.4, color='red', label='Abnormal')
    ax[0, 1].set_title('Min-Max Range')
    ax[0, 1].set_ylabel('Amplitude')
    ax[0, 1].legend()
    ax[0, 1].grid(alpha=0.3)

    # Statistical Metrics Bar Chart
    stats_normal = [np.mean(normal_signals), np.std(normal_signals), np.min(normal_signals), np.max(normal_signals)]
    stats_abnormal = [np.mean(abnormal_signals), np.std(abnormal_signals), np.min(abnormal_signals), np.max(abnormal_signals)]
    metrics = ['Mean', 'Std Dev', 'Min', 'Max']
    x_pos = np.arange(len(metrics))
    width = 0.35
    ax[1, 0].bar(x_pos - width/2, stats_normal, width, label='Normal', color='green', alpha=0.7)
    ax[1, 0].bar(x_pos + width/2, stats_abnormal, width, label='Abnormal', color='red', alpha=0.7)
    ax[1, 0].set_xticks(x_pos)
    ax[1, 0].set_xticklabels(metrics)
    ax[1, 0].set_xlabel('Metrics')
    ax[1, 0].set_ylabel('Value')
    ax[1, 0].set_title('Statistical Metrics Comparison')
    ax[1, 0].legend()
    ax[1, 0].grid(axis='y', alpha=0.3)

    # Amplitude Distribution Histogram
    ax[1, 1].hist(normal_signals.flatten(), bins=50, alpha=0.6, color='green', label='Normal', density=True)
    ax[1, 1].hist(abnormal_signals.flatten(), bins=50, alpha=0.6, color='red', label='Abnormal', density=True)
    ax[1, 1].set_xlabel('Amplitude')
    ax[1, 1].set_ylabel('Density')
    ax[1, 1].set_title('Amplitude Distribution')
    ax[1, 1].legend()
    ax[1, 1].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('statistical_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Plot saved as 'statistical_comparison.png'.")


import numpy as np
import matplotlib.pyplot as plt

def plot_boxplot_comparison(signals, labels):
    """
    Create box plots comparing Normal and Abnormal signals
    for mean amplitude and signal energy.
    Saves the figure as 'boxplot_comparison.png'.
    """
    print("Creating boxplot comparison...")

    normal_signals = signals[labels == 0]
    abnormal_signals = signals[labels == 1]

    # Feature extraction: mean and energy of each signal
    normal_means = [np.mean(s) for s in normal_signals]
    normal_energy = [np.sum(s**2) for s in normal_signals]

    abnormal_means = [np.mean(s) for s in abnormal_signals]
    abnormal_energy = [np.sum(s**2) for s in abnormal_signals]

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Boxplot: Mean values
    bp1 = ax[0].boxplot([normal_means, abnormal_means], labels=['Normal', 'Abnormal'], patch_artist=True)
    for patch, color in zip(bp1['boxes'], ['green', 'red']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax[0].set_title('Mean Values Distribution')
    ax[0].set_ylabel('Mean Amplitude')
    ax[0].grid(axis='y', alpha=0.3)

    # Boxplot: Signal energy
    bp2 = ax[1].boxplot([normal_energy, abnormal_energy], labels=['Normal', 'Abnormal'], patch_artist=True)
    for patch, color in zip(bp2['boxes'], ['green', 'red']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax[1].set_title('Signal Energy Distribution')
    ax[1].set_ylabel('Energy (sum of squares)')
    ax[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('boxplot_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Plot saved as 'boxplot_comparison.png'.")


def plot_heatmap_signals(signals, labels, n_samples=20):
    """
    Create heatmaps of multiple normal and abnormal signals.
    Shows signal amplitude across samples.
    """
    print(f"Creating heatmaps for {n_samples} signals each class...")

    normal_data = signals[labels == 0][:n_samples]
    abnormal_data = signals[labels == 1][:n_samples]

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    # Heatmap: Normal signals
    im1 = ax[0].imshow(normal_data, aspect='auto', cmap='RdYlGn_r', interpolation='nearest')
    ax[0].set_title(f'Normal ECG Signals (n={n_samples})')
    ax[0].set_xlabel('Sample Index')
    ax[0].set_ylabel('Signal ID')
    plt.colorbar(im1, ax=ax[0], label='Amplitude')

    # Heatmap: Abnormal signals
    im2 = ax[1].imshow(abnormal_data, aspect='auto', cmap='RdYlGn_r', interpolation='nearest')
    ax[1].set_title(f'Abnormal ECG Signals (n={n_samples})')
    ax[1].set_xlabel('Sample Index')
    ax[1].set_ylabel('Signal ID')
    plt.colorbar(im2, ax=ax[1], label='Amplitude')

    plt.tight_layout()
    plt.savefig('heatmap_signals.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Plot saved as 'heatmap_signals.png'.")


def plot_differences(signals, labels):
    """
    Visualize differences between average normal and abnormal signals:
    - Top: Overlay of mean signals with shaded difference
    - Bottom: Bar chart of differences (Abnormal - Normal)
    """
    print("Creating differences plot...")

    normal_mean = np.mean(signals[labels == 0], axis=0)
    abnormal_mean = np.mean(signals[labels == 1], axis=0)

    fig, ax = plt.subplots(2, 1, figsize=(14, 8))

    # Overlay of mean signals
    ax[0].plot(normal_mean, color='green', linewidth=2, label='Normal Mean', alpha=0.8)
    ax[0].plot(abnormal_mean, color='red', linewidth=2, label='Abnormal Mean', alpha=0.8)
    ax[0].fill_between(np.arange(len(normal_mean)), normal_mean, abnormal_mean, alpha=0.2, color='blue', label='Difference')
    ax[0].set_title('Normal vs Abnormal Mean Signals Overlay')
    ax[0].set_ylabel('Amplitude')
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    # Bar chart: Differences
    difference = abnormal_mean - normal_mean
    colors = ['red' if x > 0 else 'blue' for x in difference]
    ax[1].bar(np.arange(len(difference)), difference, color=colors, alpha=0.7)
    ax[1].axhline(0, color='black', linewidth=0.8)
    ax[1].set_title('Difference: Abnormal - Normal')
    ax[1].set_xlabel('Sample Index')
    ax[1].set_ylabel('Amplitude Difference')
    ax[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('differences.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Plot saved as 'differences.png'.")

plot_fft_comparison(X_signals, y_labels, sampling_rate=125)
plot_statistical_comparison(X_signals, y_labels)
plot_boxplot_comparison(X_signals, y_labels)
plot_heatmap_signals(X_signals, y_labels, n_samples=20)
plot_differences(X_signals, y_labels)

