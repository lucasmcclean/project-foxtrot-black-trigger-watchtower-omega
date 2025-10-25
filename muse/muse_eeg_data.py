import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pylsl import StreamInlet, resolve_byprop
from scipy import signal
import csv
import os
from collections import deque

# --- Configuration ---
WINDOW_LENGTH = 5  # Display window in seconds
EEG_OFFSET = 100
ACC_Y_RANGE = [-1.5, 1.5]

# --- Data Collection Configuration ---
SAMPLES_PER_WINDOW = 128  # 0.5 seconds at 256Hz (was 12 = 47ms)
OVERLAP_SAMPLES = 0      # 25% overlap for smoother collection
LABEL_MODE = 0            # Change to 1 when collecting command data
COUNT = 0

class FeatureExtractor:
    """Extract frequency domain features from EEG data"""
    
    def __init__(self, sfreq=256):
        self.sfreq = sfreq
        
    def extract_features(self, data):
        """
        Extract features from EEG window
        data: (n_samples, n_channels) array
        returns: flattened feature vector
        """
        n_samples, n_channels = data.shape
        features = []
        
        for ch in range(n_channels):
            channel_data = data[:, ch]
            
            # Time domain features
            features.append(np.mean(channel_data))
            features.append(np.std(channel_data))
            features.append(np.max(channel_data) - np.min(channel_data))
            
            # Frequency domain features
            freqs, psd = signal.welch(channel_data, fs=self.sfreq, nperseg=min(64, n_samples))
            
            # Band powers
            delta = self._band_power(freqs, psd, 1, 4)
            theta = self._band_power(freqs, psd, 4, 8)
            alpha = self._band_power(freqs, psd, 8, 13)
            beta = self._band_power(freqs, psd, 13, 30)
            gamma = self._band_power(freqs, psd, 30, 50)
            
            features.extend([delta, theta, alpha, beta, gamma])
            
            # Ratios (important for mental state detection)
            features.append(beta / alpha if alpha > 0 else 0)
            features.append(theta / alpha if alpha > 0 else 0)
            
        return np.array(features)
    
    def _band_power(self, freqs, psd, fmin, fmax):
        """Calculate power in a frequency band"""
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        return np.trapz(psd[idx], freqs[idx])


def setup_lsl_inlet(stream_type):
    """Resolve and return an LSL stream inlet."""
    print(f"Looking for a {stream_type} stream...")
    streams = resolve_byprop('type', stream_type, 1, 1.0)
    if not streams:
        raise RuntimeError(f"Unable to find {stream_type} stream. Make sure muselsl is streaming.")
    
    inlet = StreamInlet(streams[0], max_buflen=WINDOW_LENGTH)
    print(f"{stream_type} stream found!")
    return inlet


def main():
    """
    Connects to Muse LSL streams, and plots EEG and Accelerometer data in real-time.
    Collects improved feature set for ML training.
    """
    # Setup LSL Streams
    eeg_inlet = setup_lsl_inlet('EEG')
    acc_inlet = setup_lsl_inlet('ACC')

    eeg_info = eeg_inlet.info()
    acc_info = acc_inlet.info()

    eeg_sfreq = int(eeg_info.nominal_srate())
    acc_sfreq = int(acc_info.nominal_srate())

    n_channels_eeg = eeg_info.channel_count()
    n_channels_acc = acc_info.channel_count()

    # Initialize feature extractor
    extractor = FeatureExtractor(sfreq=eeg_sfreq)

    # Initialize Data Buffers
    eeg_buffer_size = int(eeg_sfreq * WINDOW_LENGTH)
    acc_buffer_size = int(acc_sfreq * WINDOW_LENGTH)

    eeg_data = np.zeros((eeg_buffer_size, n_channels_eeg))
    acc_data = np.zeros((acc_buffer_size, n_channels_acc))
    
    # Rolling buffer for feature extraction
    eeg_window_buffer = deque(maxlen=SAMPLES_PER_WINDOW)
    samples_since_last_save = 0
    
    eeg_timestamps = np.linspace(-WINDOW_LENGTH, 0, eeg_buffer_size)
    acc_timestamps = np.linspace(-WINDOW_LENGTH, 0, acc_buffer_size)

    # Setup Matplotlib Plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # EEG Plot
    eeg_lines = []
    for i in range(n_channels_eeg):
        line, = ax1.plot(eeg_timestamps, eeg_data[:, i] + (i * EEG_OFFSET), lw=1)
        eeg_lines.append(line)
    
    ax1.set_title(f'EEG Channels - Collecting Label: {LABEL_MODE}')
    ax1.set_ylabel('Voltage (uV) + Offset')
    ax1.set_ylim(-EEG_OFFSET, n_channels_eeg * EEG_OFFSET)
    ax1.set_yticks([i * EEG_OFFSET for i in range(n_channels_eeg)])
    
    # Get channel names
    ch_names = []
    channels = eeg_info.desc().child("channels")
    ch = channels.child("channel")
    for i in range(n_channels_eeg):
        ch_names.append(ch.child_value("label"))
        ch = ch.next_sibling("channel")
    ax1.set_yticklabels(ch_names)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.invert_yaxis()

    # Accelerometer Plot
    acc_lines = [ax2.plot(acc_timestamps, acc_data[:, i], lw=1)[0] for i in range(n_channels_acc)]
    ax2.set_title('Accelerometer')
    ax2.set_ylabel('g')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylim(ACC_Y_RANGE)
    ax2.legend(['X', 'Y', 'Z'], loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    # Real-time Update Function
    def update(frame):
        nonlocal samples_since_last_save
        global COUNT
        
        # Update EEG Data
        eeg_samples, _ = eeg_inlet.pull_chunk(timeout=0.0, max_samples=eeg_buffer_size)
        
        if eeg_samples:
            eeg_samples_arr = np.array(eeg_samples)
            
            # Add samples to rolling buffer
            for sample in eeg_samples_arr:
                eeg_window_buffer.append(sample)
                samples_since_last_save += 1
            
            # Save features when we have enough samples and reached overlap threshold
            if len(eeg_window_buffer) == SAMPLES_PER_WINDOW and samples_since_last_save >= OVERLAP_SAMPLES:
                window_data = np.array(list(eeg_window_buffer))
                
                # Extract features
                features = extractor.extract_features(window_data)
                
                # Add label
                labeled_data = np.append(features, LABEL_MODE)
                
                # Append to CSV
                with open('eeg_features.csv', 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(labeled_data)
                    COUNT += 1

                samples_since_last_save = 0
                print(f"Saved window - Label: {LABEL_MODE}, Features: {len(features)}, Count: {COUNT}")
            
            # Update plot
            new_samples_count = len(eeg_samples)
            eeg_data[:] = np.roll(eeg_data, -new_samples_count, axis=0)
            eeg_data[-new_samples_count:, :] = eeg_samples
            
            for i in range(n_channels_eeg):
                eeg_lines[i].set_ydata(eeg_data[:, i] + (i * EEG_OFFSET))

        # Update ACC Data
        acc_samples, _ = acc_inlet.pull_chunk(timeout=0.0, max_samples=acc_buffer_size)
        if acc_samples:
            new_samples_count = len(acc_samples)
            acc_data[:] = np.roll(acc_data, -new_samples_count, axis=0)
            acc_data[-new_samples_count:, :] = acc_samples

            for i in range(n_channels_acc):
                acc_lines[i].set_ydata(acc_data[:, i])
        
        return eeg_lines + acc_lines

    # Create CSV with header
    filename = 'eeg_features.csv'
    if not os.path.exists(filename):
        # Generate header for features
        feature_names = []
        for ch in range(n_channels_eeg):
            feature_names.extend([
                f'ch{ch}_mean', f'ch{ch}_std', f'ch{ch}_range',
                f'ch{ch}_delta', f'ch{ch}_theta', f'ch{ch}_alpha',
                f'ch{ch}_beta', f'ch{ch}_gamma',
                f'ch{ch}_beta_alpha_ratio', f'ch{ch}_theta_alpha_ratio'
            ])
        feature_names.append('label')
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(feature_names)
        
        print(f"Created {filename} with {len(feature_names)-1} features")

    ani = animation.FuncAnimation(fig, update, blit=True, interval=30, cache_frame_data=False)
    plt.show()


if __name__ == '__main__':
    try:
        main()
    except RuntimeError as e:
        print(e)
