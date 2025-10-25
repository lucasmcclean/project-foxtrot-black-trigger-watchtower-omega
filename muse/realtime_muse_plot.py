import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pylsl import StreamInlet, resolve_byprop
import csv
import os

# --- Plotting Configuration ---
# Length of the display window in seconds
WINDOW_LENGTH = 5
# Vertical offset for EEG channels to make them visible
EEG_OFFSET = 100
# Y-axis limits for the plots
ACC_Y_RANGE = [-1.5, 1.5]


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
    """
    # ---------------------------------------------------------
    # 1. Setup LSL Streams
    eeg_inlet = setup_lsl_inlet('EEG')
    acc_inlet = setup_lsl_inlet('ACC')

    # Get stream info
    eeg_info = eeg_inlet.info()
    acc_info = acc_inlet.info()

    eeg_sfreq = int(eeg_info.nominal_srate())
    acc_sfreq = int(acc_info.nominal_srate())

    n_channels_eeg = eeg_info.channel_count()
    n_channels_acc = acc_info.channel_count()

    # ---------------------------------------------------------
    # 2. Initialize Data Buffers
    eeg_buffer_size = int(eeg_sfreq * WINDOW_LENGTH)
    acc_buffer_size = int(acc_sfreq * WINDOW_LENGTH)

    eeg_data = np.zeros((eeg_buffer_size, n_channels_eeg))
    acc_data = np.zeros((acc_buffer_size, n_channels_acc))
    
    # Timestamps for x-axis
    eeg_timestamps = np.linspace(-WINDOW_LENGTH, 0, eeg_buffer_size)
    acc_timestamps = np.linspace(-WINDOW_LENGTH, 0, acc_buffer_size)

    # ---------------------------------------------------------
    # 3. Setup Matplotlib Plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # --- EEG Plot ---
    # Create lines for each EEG channel with vertical offset
    eeg_lines = []
    for i in range(n_channels_eeg):
        line, = ax1.plot(eeg_timestamps, eeg_data[:, i] + (i * EEG_OFFSET), lw=1)
        eeg_lines.append(line)
    
    ax1.set_title('EEG Channels')
    ax1.set_ylabel('Voltage (uV) + Offset')
    ax1.set_ylim(-EEG_OFFSET, n_channels_eeg * EEG_OFFSET)
    ax1.set_yticks([i * EEG_OFFSET for i in range(n_channels_eeg)])
    # Get channel names from LSL stream info
    ch_names = []
    channels = eeg_info.desc().child("channels")
    ch = channels.child("channel")
    for i in range(n_channels_eeg):
        ch_names.append(ch.child_value("label"))
        ch = ch.next_sibling("channel")
    ax1.set_yticklabels(ch_names)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.invert_yaxis()

    # --- Accelerometer Plot ---
    acc_lines = [ax2.plot(acc_timestamps, acc_data[:, i], lw=1)[0] for i in range(n_channels_acc)]
    ax2.set_title('Accelerometer')
    ax2.set_ylabel('g')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylim(ACC_Y_RANGE)
    ax2.legend(['X', 'Y', 'Z'], loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    # ---------------------------------------------------------
    # 4. Real-time Update Function
    def update(frame):
        # --- Update EEG Data ---
        eeg_samples, _ = eeg_inlet.pull_chunk(timeout=0.0, max_samples=eeg_buffer_size)
        if eeg_samples:
            # --- CSV Logging ---
            eeg_samples_arr = np.array(eeg_samples)
            n_new_samples, n_channels = eeg_samples_arr.shape
            
            if n_new_samples > 0:
                # Pad or truncate to 12 samples
                if n_new_samples < 12:
                    padding = np.zeros((12 - n_new_samples, n_channels))
                    processed_samples = np.vstack((eeg_samples_arr, padding))
                else:
                    processed_samples = eeg_samples_arr[:12, :]

                # Flatten the data and add label
                flat_data = processed_samples.flatten()
                labeled_data = np.append(flat_data, 0)

                # Append to CSV
                with open('eeg_data.csv', 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(labeled_data)

            # --- Update Plot Data ---
            new_samples_count = len(eeg_samples)
            eeg_data[:] = np.roll(eeg_data, -new_samples_count, axis=0)
            eeg_data[-new_samples_count:, :] = eeg_samples
            print(eeg_data.shape)
            
            # Update plot lines
            for i in range(n_channels_eeg):
                eeg_lines[i].set_ydata(eeg_data[:, i] + (i * EEG_OFFSET))

        # --- Update ACC Data ---
        acc_samples, _ = acc_inlet.pull_chunk(timeout=0.0, max_samples=acc_buffer_size)
        if acc_samples:
            new_samples_count = len(acc_samples)
            acc_data[:] = np.roll(acc_data, -new_samples_count, axis=0)
            acc_data[-new_samples_count:, :] = acc_samples

            # Update plot lines
            for i in range(n_channels_acc):
                acc_lines[i].set_ydata(acc_data[:, i])
        
        return eeg_lines + acc_lines

    # ---------------------------------------------------------
    # 5. Create and Run Animation
    
    # --- CSV Header ---
    filename = 'eeg_data.csv'
    if not os.path.exists(filename):
        header = [f'eeg_{i}' for i in range(12 * n_channels_eeg)] + ['label']
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    ani = animation.FuncAnimation(fig, update, blit=True, interval=30, cache_frame_data=False)
    plt.show()


if __name__ == '__main__':
    try:
        main()
    except RuntimeError as e:
        print(e)
