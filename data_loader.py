import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
warnings.filterwarnings('ignore')

# Set style for defense-focused visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Defense Sensor Anomaly Detection: Data Loader for UCI Sonar and NASA IMS Bearing Datasets
# This module provides preprocessing pipelines for cross-domain anomaly detection in defense sensor networks.
# Sonar data represents underwater acoustic signals (normal: rocks 'R', anomalous: mines 'M')
# IMS data represents bearing vibration signals (normal: early-cycle, anomalous: late-cycle degradation)

def set_random_seed(seed=42):
    """Set random seed for reproducibility in defense sensor data analysis."""
    np.random.seed(seed)
    random.seed(seed)

def load_sonar_data(data_path='Datasets/UCI/sonar.all-data'):
    """
    Load and preprocess UCI Sonar dataset for anomaly detection.

    Parameters:
    data_path (str): Path to sonar.all-data file

    Returns:
    tuple: (X, y) where X is normalized features, y is binary labels (0=normal, 1=anomaly)
    """
    # Defense Application: Sonar signals critical for naval underwater threat detection
    # Mines (M) represent anomalous threats, Rocks (R) represent normal seabed clutter

    # Load data
    df = pd.read_csv(data_path, header=None)

    # Extract features (first 60 columns) and labels (last column)
    X = df.iloc[:, :-1].values  # 60 sonar signal features
    y_raw = df.iloc[:, -1].values  # Labels: 'R' or 'M'

    # Convert labels: R=normal (0), M=anomaly (1)
    y = np.where(y_raw == 'M', 1, 0)

    # Normalize features using Min-Max scaling for consistent sensor signal ranges
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)

    print(f"Loaded sonar data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Normal samples: {np.sum(y == 0)}, Anomalous samples: {np.sum(y == 1)}")

    return X_normalized, y

def load_ims_data(data_path='Datasets/IMS', window_size=1024):
    """
    Load and preprocess NASA IMS Bearing vibration data for anomaly detection.

    Parameters:
    data_path (str): Path to IMS dataset directory containing ASCII files
    window_size (int): Window size for signal segmentation (1024 samples)

    Returns:
    tuple: (X_windows, y_labels) where X is windowed vibration signals,
            y is binary labels (0=normal early-cycle, 1=anomaly late-cycle)
    """
    # Defense Application: Bearing vibration monitoring critical for aircraft/engine failure prevention
    # Early-cycle represents normal operation, late-cycle represents degradation/anomaly
    # Windowing enables temporal analysis of vibration patterns in defense sensor networks

    # IMS Data Structure:
    # - Each file: 20,480 points (1 second @ 20 kHz sampling)
    # - Set 1: 8 channels (Bearing 1-2: Ch1-2, Bearing 3-4: Ch3-4 per bearing)
    # - Set 2: 4 channels (1 channel per bearing)
    # - ASCII format, space/tab delimited
    # - Files named with timestamps indicating collection time

    try:
        # Find vibration data files (ASCII format, timestamp-named files in subdirectories)
        vibration_files = []

        # Recursively find all files in subdirectories
        for root, dirs, files in os.walk(data_path):
            for file in files:
                # Include vibration data files (exclude archives and documentation)
                if not file.endswith(('.rar', '.pdf')):
                    vibration_files.append(os.path.join(root, file))

        # Sort files chronologically by filename
        vibration_files.sort()

        # Process files in parallel batches for faster loading
        batch_size = 50  # Process 50 files at a time
        file_batches = [vibration_files[i:i + batch_size] for i in range(0, len(vibration_files), batch_size)]

        all_windows = []
        all_labels = []
        file_info = []

        print(f"Processing {len(vibration_files)} IMS files in {len(file_batches)} batches...")

        # Create logs directory
        os.makedirs('results/logs', exist_ok=True)
        log_file = open('results/logs/ims_loading_log.txt', 'w')
        log_file.write(f"IMS Data Loading Log - {pd.Timestamp.now()}\n")
        log_file.write("="*50 + "\n")
        log_file.write(f"Total files to process: {len(vibration_files)}\n")
        log_file.write(f"Batch size: {batch_size}\n")
        log_file.write(f"Number of batches: {len(file_batches)}\n\n")

        with ProcessPoolExecutor(max_workers=4) as executor:  # Use 4 parallel workers
            future_to_batch = {executor.submit(process_file_batch, batch, data_path): batch for batch in file_batches}

            batch_count = 0
            for future in as_completed(future_to_batch):
                batch_count += 1
                batch_windows, batch_labels, batch_info = future.result()
                all_windows.extend(batch_windows)
                all_labels.extend(batch_labels)
                file_info.extend(batch_info)

                # Log batch completion
                batch_files = future_to_batch[future]
                log_file.write(f"Batch {batch_count}/{len(file_batches)} completed: {len(batch_files)} files, {len(batch_windows)} windows generated\n")
                print(f"Completed batch {batch_count}/{len(file_batches)}: {len(batch_windows)} windows from {len(batch_files)} files")

        log_file.close()

        # Update labels based on chronological order
        total_windows = len(all_windows)
        for i in range(total_windows):
            # Simple heuristic: first 70% of windows = normal, last 30% = anomalous
            is_anomaly = i >= int(0.7 * total_windows)
            all_labels[i] = 1 if is_anomaly else 0

        # Update file info anomaly status
        total_files = len(file_info)
        for i in range(total_files):
            file_info[i]['is_anomaly'] = i >= int(0.7 * total_files)

        if all_windows:
            # Convert to numpy arrays with consistent shapes
            try:
                X_windows = np.array(all_windows, dtype=np.float32)
                y_labels = np.array(all_labels, dtype=np.int32)
            except ValueError:
                # Handle inhomogeneous shapes by padding or truncating
                max_channels = max(window.shape[1] for window in all_windows)
                padded_windows = []
                for window in all_windows:
                    if window.shape[1] < max_channels:
                        # Pad with zeros
                        padded = np.zeros((window.shape[0], max_channels), dtype=np.float32)
                        padded[:, :window.shape[1]] = window
                        padded_windows.append(padded)
                    else:
                        padded_windows.append(window)
                X_windows = np.array(padded_windows, dtype=np.float32)
                y_labels = np.array(all_labels, dtype=np.int32)

            print(f"Loaded IMS data: {X_windows.shape[0]} windows, {X_windows.shape[1]} samples per window, {X_windows.shape[2]} channels")
            print(f"Normal windows: {np.sum(y_labels == 0)} ({np.sum(y_labels == 0)/len(y_labels)*100:.1f}%)")
            print(f"Anomalous windows: {np.sum(y_labels == 1)} ({np.sum(y_labels == 1)/len(y_labels)*100:.1f}%)")

            # Save file information for reference (defense sensor data traceability)
            os.makedirs(os.path.join('results', 'preprocessing', 'ims'), exist_ok=True)
            with open(os.path.join('results', 'preprocessing', 'ims', 'file_info.txt'), 'w') as f:
                f.write("IMS Dataset File Information (Windowed):\n")
                f.write("="*50 + "\n")
                for info in file_info:
                    f.write(f"File: {info['filename']}, Channels: {info['channels']}, Samples: {info['samples']}, Windows: {info['windows']}, Label: {'Anomaly' if info['is_anomaly'] else 'Normal'}\n")

            return X_windows, y_labels
        else:
            print("Warning: No valid IMS data files found. Returning empty arrays.")
            return np.array([]), np.array([])

    except FileNotFoundError:
        print(f"IMS data path {data_path} not found. Please ensure NASA IMS dataset is available.")
        return np.array([]), np.array([])

# Defense-specific preprocessing notes:
# - Normalization ensures consistent feature scales across different sensor modalities
# - Binary labeling enables supervised anomaly detection training
# - Windowing captures temporal dependencies in vibration signals
# - Reproducibility with seeds ensures consistent model evaluation in defense scenarios

def process_file_batch(file_batch, data_path, expected_samples=20480, window_size=1024, stride=512):
    """
    Process a batch of IMS files in parallel for faster loading.

    Parameters:
    file_batch (list): List of file paths to process
    data_path (str): Base data path
    expected_samples (int): Expected samples per file
    window_size (int): Window size for segmentation
    stride (int): Stride for overlapping windows

    Returns:
    tuple: (windows_batch, labels_batch, file_info_batch)
    """
    windows_batch = []
    labels_batch = []
    file_info_batch = []
    file_index = 0  # This will be set externally for anomaly labeling

    for file_path in file_batch:
        try:
            # Load ASCII vibration data
            data = np.loadtxt(file_path)

            if data.ndim == 1:
                if len(data) == expected_samples:
                    channels = data.reshape(-1, 1)
                else:
                    continue
            elif data.ndim == 2:
                channels = data
                if channels.shape[0] != expected_samples:
                    continue
            else:
                continue

            # Normalize per channel
            channels_normalized = np.zeros_like(channels, dtype=np.float32)
            for ch in range(channels.shape[1]):
                scaler = MinMaxScaler()
                channels_normalized[:, ch] = scaler.fit_transform(channels[:, ch].reshape(-1, 1)).flatten()

            # Window the signals
            n_windows = (expected_samples - window_size) // stride + 1
            for start in range(0, expected_samples - window_size + 1, stride):
                end = start + window_size
                window = channels_normalized[start:end, :]
                windows_batch.append(window)
                labels_batch.append(0)  # Placeholder, will be updated

            file_info_batch.append({
                'filename': os.path.basename(file_path),
                'channels': channels.shape[1],
                'samples': channels.shape[0],
                'windows': n_windows,
                'is_anomaly': False  # Placeholder
            })

        except Exception as e:
            continue

    return windows_batch, labels_batch, file_info_batch

def window_signals(signals, window_size=1024, stride=None):
    """
    Window time-series signals for temporal analysis in anomaly detection.

    Parameters:
    signals (np.ndarray): Input signals of shape (n_samples, n_features) or (n_samples,)
    window_size (int): Size of each window
    stride (int): Step size for window sliding (default: window_size//2 for 50% overlap)

    Returns:
    np.ndarray: Windowed signals of shape (n_windows, window_size, n_features)
    """
    if stride is None:
        stride = window_size // 2

    if signals.ndim == 1:
        signals = signals.reshape(-1, 1)

    n_samples, n_features = signals.shape
    n_windows = (n_samples - window_size) // stride + 1

    windows = []
    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        window = signals[start:end, :]
        windows.append(window)

    return np.array(windows)


if __name__ == "__main__":
    # Set seed for reproducible data loading (critical for defense sensor evaluation consistency)
    set_random_seed(42)

    # Load sonar data (defense application: underwater threat detection)
    print("Loading UCI Sonar dataset...")
    X_sonar, y_sonar = load_sonar_data()

    # Load IMS data (defense application: bearing vibration monitoring in critical systems)
    print("\nLoading NASA IMS Bearing dataset...")
    X_ims, y_ims = load_ims_data()

    print("\n" + "="*60)
    print("DATA LOADING COMPLETED FOR DEFENSE SENSOR ANOMALY DETECTION")
    print("="*60)
    print(f"UCI Sonar: {X_sonar.shape[0]} samples, {X_sonar.shape[1]} features")
    if X_ims.size > 0:
        print(f"NASA IMS: {X_ims.shape[0]} windows, {X_ims.shape[1]} samples/window, {X_ims.shape[2]} channels")

        # Save basic statistics for reference
        os.makedirs('results/preprocessing/ims', exist_ok=True)
        with open('results/preprocessing/ims/dataset_stats.txt', 'w') as f:
            f.write("NASA IMS Bearing Dataset Statistics (Windowed)\n")
            f.write("="*50 + "\n")
            f.write(f"Total windows: {X_ims.shape[0]}\n")
            f.write(f"Samples per window: {X_ims.shape[1]}\n")
            f.write(f"Channels per window: {X_ims.shape[2]}\n")
            f.write(f"Normal windows: {np.sum(y_ims == 0)} ({np.sum(y_ims == 0)/len(y_ims)*100:.1f}%)\n")
            f.write(f"Anomalous windows: {np.sum(y_ims == 1)} ({np.sum(y_ims == 1)/len(y_ims)*100:.1f}%)\n")
            f.write("Window size: 1024 samples\n")
            f.write("Overlap: 512 samples (50%)\n")
            f.write("Defense Application: Bearing vibration monitoring\n")
    else:
        print("NASA IMS: No data loaded (dataset not available)")

    # Save UCI Sonar statistics
    os.makedirs('results/preprocessing/sonar', exist_ok=True)
    with open('results/preprocessing/sonar/dataset_stats.txt', 'w') as f:
        f.write("UCI Sonar Dataset Statistics\n")
        f.write("="*30 + "\n")
        f.write(f"Total samples: {X_sonar.shape[0]}\n")
        f.write(f"Features per sample: {X_sonar.shape[1]}\n")
        f.write(f"Normal samples (Rocks): {np.sum(y_sonar == 0)} ({np.sum(y_sonar == 0)/len(y_sonar)*100:.1f}%)\n")
        f.write(f"Anomalous samples (Mines): {np.sum(y_sonar == 1)} ({np.sum(y_sonar == 1)/len(y_sonar)*100:.1f}%)\n")
        f.write("Defense Application: Underwater acoustic threat detection\n")

    # Create research-specific visualizations for anomaly detection
    print("\nGenerating research visualizations...")

    # Sonar dataset visualizations for research
    os.makedirs('results/visualizations/sonar', exist_ok=True)

    # 1. Feature correlation heatmap for sonar
    plt.figure(figsize=(12, 10))
    corr_matrix = np.corrcoef(X_sonar.T)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdBu_r', center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .8})
    plt.title('Sonar Feature Correlation Matrix (Research Analysis)')
    plt.savefig('results/visualizations/sonar/feature_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. PCA projection for dimensionality analysis
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_sonar)
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red']
    labels = ['Normal (Rocks)', 'Anomaly (Mines)']
    for i, (color, label) in enumerate(zip(colors, labels)):
        mask = y_sonar == i
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=label, alpha=0.7, s=60, edgecolors='black')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    plt.title('PCA Projection: Sonar Anomaly Detection Manifold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/visualizations/sonar/pca_projection.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Feature importance via variance
    feature_variances = np.var(X_sonar, axis=0)
    plt.figure(figsize=(15, 6))
    plt.plot(feature_variances, 'b-o', linewidth=2, markersize=4)
    plt.title('Sonar Feature Variances (Anomaly Detection Relevance)')
    plt.xlabel('Feature Index')
    plt.ylabel('Variance')
    plt.grid(True, alpha=0.3)
    plt.savefig('results/visualizations/sonar/feature_variances.png', dpi=300, bbox_inches='tight')
    plt.close()

    # IMS dataset research visualizations
    if X_ims.size > 0:
        os.makedirs('results/visualizations/ims', exist_ok=True)

        # 1. Time-frequency analysis sample (FFT of sample windows)
        plt.figure(figsize=(16, 10))
        n_channels = min(4, X_ims.shape[2])  # Show first 4 channels

        for ch in range(n_channels):
            # Normal bearing signal
            plt.subplot(n_channels, 2, 2*ch + 1)
            sample_normal = X_ims[y_ims == 0, :, ch][0]  # First normal window
            plt.plot(sample_normal, 'b-', linewidth=1.5)
            plt.title(f'Normal Bearing - Channel {ch+1} (Time Domain)')
            plt.xlabel('Sample Index')
            plt.ylabel('Normalized Amplitude')
            plt.grid(True, alpha=0.3)

            # Frequency domain analysis
            plt.subplot(n_channels, 2, 2*ch + 2)
            fft_normal = np.abs(np.fft.fft(sample_normal))[:len(sample_normal)//2]
            freqs = np.fft.fftfreq(len(sample_normal))[:len(sample_normal)//2]
            plt.plot(freqs, fft_normal, 'b-', linewidth=1.5)
            plt.title(f'Normal Bearing - Channel {ch+1} (Frequency Domain)')
            plt.xlabel('Frequency (normalized)')
            plt.ylabel('Magnitude')
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('results/visualizations/ims/time_frequency_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. RMS evolution over time (degradation indicator)
        rms_values = np.sqrt(np.mean(X_ims**2, axis=(1, 2)))  # RMS across all channels per window
        plt.figure(figsize=(14, 8))

        # Plot RMS trend with anomaly highlighting
        window_indices = np.arange(len(rms_values))
        normal_mask = y_ims == 0
        anomaly_mask = y_ims == 1

        plt.scatter(window_indices[normal_mask], rms_values[normal_mask],
                   c='blue', label='Normal Operation', alpha=0.6, s=30)
        plt.scatter(window_indices[anomaly_mask], rms_values[anomaly_mask],
                   c='red', label='Anomalous Degradation', alpha=0.8, s=30)

        # Add trend line
        plt.plot(window_indices, rms_values, 'k-', alpha=0.3, linewidth=2, label='RMS Trend')
        plt.xlabel('Window Index (Chronological Order)')
        plt.ylabel('RMS Value (Vibration Intensity)')
        plt.title('Bearing Degradation Analysis: RMS Evolution Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('results/visualizations/ims/rms_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Channel-wise statistical comparison
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        stat_names = ['Mean', 'Std', 'RMS', 'Peak']

        for ch in range(min(8, X_ims.shape[2])):
            ch_data_normal = X_ims[y_ims == 0, :, ch]
            ch_data_anomaly = X_ims[y_ims == 1, :, ch]

            # Calculate statistics
            normal_stats = [
                np.mean(ch_data_normal),
                np.std(ch_data_normal),
                np.sqrt(np.mean(ch_data_normal**2)),
                np.max(np.abs(ch_data_normal))
            ]
            anomaly_stats = [
                np.mean(ch_data_anomaly),
                np.std(ch_data_anomaly),
                np.sqrt(np.mean(ch_data_anomaly**2)),
                np.max(np.abs(ch_data_anomaly))
            ]

            ax = axes[ch//4, ch%4]
            x = np.arange(len(stat_names))
            width = 0.35

            ax.bar(x - width/2, normal_stats, width, label='Normal', alpha=0.8, color='blue')
            ax.bar(x + width/2, anomaly_stats, width, label='Anomaly', alpha=0.8, color='red')
            ax.set_title(f'Channel {ch+1} Statistics')
            ax.set_xticks(x)
            ax.set_xticklabels(stat_names)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('results/visualizations/ims/channel_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()

    print("Research visualizations saved to results/visualizations/")