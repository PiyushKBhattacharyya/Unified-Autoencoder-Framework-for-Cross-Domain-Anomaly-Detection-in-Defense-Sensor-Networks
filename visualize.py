import torch
import torch.nn as nn
import numpy as np
import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fft import fft
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Import custom modules for defense sensor anomaly detection
from data_loader import load_sonar_data, load_ims_data, set_random_seed
from models import create_dae_sonar, create_vae_ims
from evaluate import load_trained_model, compute_reconstruction_errors

# Set style for defense-focused visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Defense Sensor Network Anomaly Detection Visualization Framework
# This module implements specialized visualization capabilities for autoencoder-based
# anomaly detection in cross-domain defense sensor networks (sonar acoustics and bearing vibrations)
# Critical for operational threat detection and research analysis in naval and aerospace systems


def load_and_prepare_data(model_type='dae'):
    """
    Load test data for visualization based on model type.

    Defense Context: Data preparation ensures reliable visualization of sensor anomalies
    in operational defense environments where data quality is paramount.
    """
    if model_type == 'dae':
        X_data, y_data = load_sonar_data()
        X_tensor = torch.FloatTensor(X_data)
        y_tensor = torch.LongTensor(y_data)
        batch_size = 32
    elif model_type == 'vae':
        X_data, y_data = load_ims_data()
        if X_data.size == 0:
            return None, None, None
        X_tensor = torch.FloatTensor(X_data)
        y_tensor = torch.LongTensor(y_data)
        batch_size = 16
    else:
        raise ValueError("Invalid model_type. Must be 'dae' or 'vae'")

    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(X_tensor, y_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return data_loader, X_data, y_data


def denoise_signal(model, noisy_input, model_type='dae', device='cpu'):
    """
    Apply denoising to input signal using trained autoencoder.

    Defense Application: Signal denoising critical for improving signal-to-noise ratio
    in defense sensor networks, enabling clearer detection of subtle threats.
    """
    model.eval()
    with torch.no_grad():
        input_tensor = torch.FloatTensor(noisy_input).to(device)
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)

        if model_type == 'dae':
            denoised, _ = model(input_tensor)
        elif model_type == 'vae':
            denoised, _, _ = model(input_tensor)

        return denoised.cpu().numpy()


def create_error_heatmap_sonar(reconstruction_errors, sample_indices, save_path):
    """
    Create reconstruction error heatmap for sonar signals (frequency-wise analysis).

    Defense Context: Frequency-domain error visualization aids in identifying
    specific acoustic signatures of underwater threats like mines or submarines.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(16, 10))

    # Heatmap of reconstruction errors across samples and frequencies
    sns.heatmap(reconstruction_errors.T, cmap='RdYlBu_r', cbar_kws={'label': 'Reconstruction Error'},
                xticklabels=sample_indices, yticklabels=False)
    plt.xlabel('Sample Index')
    plt.ylabel('Frequency Bin')
    plt.title('Sonar Reconstruction Error Heatmap (Frequency-Wise) - Defense Acoustic Anomaly Detection')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"💾 Saved sonar error heatmap to {save_path}")


def create_time_domain_error_plot_ims(reconstruction_errors, time_indices, save_path):
    """
    Create time-domain reconstruction error plot for seismic/bearing signals.

    Defense Context: Time-domain analysis reveals temporal patterns of bearing degradation
    critical for predicting failure in aerospace systems before catastrophic incidents.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(16, 10))

    # Plot reconstruction error over time
    plt.plot(time_indices, reconstruction_errors, 'r-', linewidth=2, alpha=0.8, label='Reconstruction Error')
    plt.fill_between(time_indices, reconstruction_errors, alpha=0.3, color='red')

    # Add statistical annotations
    mean_error = np.mean(reconstruction_errors)
    std_error = np.std(reconstruction_errors)
    plt.axhline(y=mean_error, color='blue', linestyle='--', alpha=0.7, label=f'Mean Error: {mean_error:.6f}')
    plt.axhline(y=mean_error + 2*std_error, color='orange', linestyle=':', alpha=0.7,
                label=f'Mean + 2σ: {(mean_error + 2*std_error):.6f}')

    plt.xlabel('Time Sample Index')
    plt.ylabel('Reconstruction MSE')
    plt.title('IMS Bearing Time-Domain Reconstruction Error - Aerospace Vibration Anomaly Detection')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"💾 Saved IMS time-domain error plot to {save_path}")


def create_error_distribution_plots(reconstruction_errors, labels, model_type, save_path):
    """
    Create research-focused error distribution plots comparing normal vs anomalous samples.

    Defense Context: Error distribution analysis provides statistical insights for
    optimizing anomaly detection thresholds in critical defense applications.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(16, 12))

    # Separate normal and anomalous errors
    normal_errors = reconstruction_errors[labels == 0]
    anomaly_errors = reconstruction_errors[labels == 1]

    # Error distribution histogram
    plt.subplot(2, 2, 1)
    if len(normal_errors) > 0:
        plt.hist(normal_errors, bins=50, alpha=0.7, color='blue', label='Normal Operation', density=True)
    if len(anomaly_errors) > 0:
        plt.hist(anomaly_errors, bins=50, alpha=0.7, color='red', label='Anomalous Degradation', density=True)
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Density')
    plt.title(f'{model_type.upper()} Reconstruction Error Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Cumulative distribution
    plt.subplot(2, 2, 2)
    if len(normal_errors) > 0:
        plt.hist(normal_errors, bins=50, alpha=0.7, color='blue', label='Normal', density=True, cumulative=True, histtype='step', linewidth=2)
    if len(anomaly_errors) > 0:
        plt.hist(anomaly_errors, bins=50, alpha=0.7, color='red', label='Anomaly', density=True, cumulative=True, histtype='step', linewidth=2)
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Cumulative Density')
    plt.title(f'{model_type.upper()} Cumulative Error Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Box plot comparison
    plt.subplot(2, 2, 3)
    data_to_plot = []
    labels_plot = []
    if len(normal_errors) > 0:
        data_to_plot.append(normal_errors)
        labels_plot.append('Normal')
    if len(anomaly_errors) > 0:
        data_to_plot.append(anomaly_errors)
        labels_plot.append('Anomaly')

    if data_to_plot:
        plt.boxplot(data_to_plot, labels=labels_plot, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2))
        plt.ylabel('Reconstruction Error')
        plt.title(f'{model_type.upper()} Error Distribution Box Plot')
        plt.grid(True, alpha=0.3)

    # Q-Q plot (Quantile-Quantile) for normality check
    plt.subplot(2, 2, 4)
    if len(normal_errors) > 0 and len(anomaly_errors) > 0:
        # Sort errors for Q-Q plot
        normal_sorted = np.sort(normal_errors)
        anomaly_sorted = np.sort(anomaly_errors)

        # Take minimum length for comparison
        min_len = min(len(normal_sorted), len(anomaly_sorted))
        plt.scatter(normal_sorted[:min_len], anomaly_sorted[:min_len], alpha=0.6, color='purple')
        plt.plot([normal_sorted.min(), normal_sorted.max()],
                [anomaly_sorted.min(), anomaly_sorted.max()], 'r--', alpha=0.7)

        plt.xlabel('Normal Operation Quantiles')
        plt.ylabel('Anomalous Degradation Quantiles')
        plt.title(f'{model_type.upper()} Q-Q Plot: Normal vs Anomaly')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"💾 Saved error distribution plots to {save_path}")


def create_denoised_comparison_plots(original_samples, denoised_samples, noisy_samples, model_type, save_path):
    """
    Create research-focused plots comparing original, noisy, and denoised signals.

    Defense Context: Denoising comparison demonstrates autoencoder effectiveness in
    signal restoration, critical for maintaining sensor signal integrity in noisy environments.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    n_samples = min(5, len(original_samples))  # Show up to 5 samples

    if model_type == 'dae':  # Sonar signals
        fig, axes = plt.subplots(n_samples, 1, figsize=(16, 4*n_samples))

        for i in range(n_samples):
            if n_samples == 1:
                ax = axes
            else:
                ax = axes[i]

            ax.plot(original_samples[i], 'b-', linewidth=2, label='Original', alpha=0.8)
            if noisy_samples is not None:
                ax.plot(noisy_samples[i], 'g-', linewidth=1, label='Noisy Input', alpha=0.6)
            ax.plot(denoised_samples[i], 'r-', linewidth=2, label='Denoised', alpha=0.8)

            ax.set_xlabel('Feature Index')
            ax.set_ylabel('Normalized Amplitude')
            ax.set_title(f'Sonar Signal {i+1}: Denoising Performance')
            ax.legend()
            ax.grid(True, alpha=0.3)

    else:  # VAE - IMS signals (time series)
        fig, axes = plt.subplots(n_samples, 1, figsize=(16, 4*n_samples))

        for i in range(n_samples):
            if n_samples == 1:
                ax = axes
            else:
                ax = axes[i]

            # Show first channel only for clarity
            ax.plot(original_samples[i, 0, :], 'b-', linewidth=2, label='Original', alpha=0.8)
            if noisy_samples is not None:
                ax.plot(noisy_samples[i, 0, :], 'g-', linewidth=1, label='Noisy Input', alpha=0.6)
            ax.plot(denoised_samples[i, 0, :], 'r-', linewidth=2, label='Denoised', alpha=0.8)

            ax.set_xlabel('Time Sample')
            ax.set_ylabel('Normalized Amplitude')
            ax.set_title(f'IMS Bearing Signal {i+1}: Denoising Performance (Channel 1)')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"💾 Saved denoising comparison plots to {save_path}")


def generate_sonar_frequency_error_analysis(model, data_loader, device, save_path):
    """
    Generate detailed frequency-domain error analysis for sonar signals.

    Defense Context: Frequency analysis reveals resonant characteristics of
    underwater anomalies, enabling classification of threat types.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    model.eval()
    all_freq_errors = []
    sample_count = 0

    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            reconstructed, _ = model(inputs)

            # Convert to numpy
            orig_np = inputs.cpu().numpy()
            recon_np = reconstructed.cpu().numpy()

            # Compute FFT and frequency errors
            for i in range(len(orig_np)):
                orig_fft = np.abs(fft(orig_np[i]))
                recon_fft = np.abs(fft(recon_np[i]))
                freq_error = np.abs(orig_fft - recon_fft)
                all_freq_errors.append(freq_error)

                sample_count += 1
                if sample_count >= 50:  # Limit to 50 samples for visualization
                    break
            if sample_count >= 50:
                break

    if all_freq_errors:
        # Stack errors for analysis
        freq_errors_array = np.array(all_freq_errors)
        mean_freq_errors = np.mean(freq_errors_array, axis=0)

        # Create frequency bands
        n_freqs = len(mean_freq_errors)
        freq_bands = np.linspace(0, n_freqs//2, 20)  # 20 frequency bands

        band_errors = []
        band_labels = []

        for i in range(len(freq_bands)-1):
            start_idx = int(freq_bands[i])
            end_idx = int(freq_bands[i+1])
            error = np.mean(mean_freq_errors[start_idx:end_idx])
            band_errors.append(error)
            band_labels.append(f'{start_idx}-{end_idx}')

        # Plot frequency band errors
        plt.figure(figsize=(14, 8))
        bars = plt.bar(band_labels, band_errors, color='steelblue', alpha=0.8, edgecolor='black')
        plt.xlabel('Frequency Band (FFT Bins)')
        plt.ylabel('Mean Reconstruction Error')
        plt.title('Sonar Frequency-Band Reconstruction Errors - Underwater Threat Detection')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    '.4f', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"💾 Saved sonar frequency error analysis to {save_path}")


def visualize_sonar_anomalies(model, data_loader, device):
    """
    Generate comprehensive visualizations for sonar anomaly detection.

    Defense Application: Multi-faceted visualization enables operators to
    identify acoustic anomalies in real-time defense sensor monitoring.
    """
    print("\n🚢 GENERATING SONAR ANOMALY VISUALIZATIONS")

    # Compute reconstruction errors
    errors, originals, labels, _ = compute_reconstruction_errors(model, data_loader, 'dae', device)

    # 1. Reconstruction Error Heatmap (Frequency-wise)
    sample_indices = np.arange(min(50, len(errors)))  # Show first 50 samples
    error_matrix = np.array([errors[i] for i in sample_indices])[:, np.newaxis]  # Dummy 2D for heatmap
    # Actually, for sonar, errors is per sample, so we need frequency-wise errors
    # Let's get frequency errors
    model.eval()
    freq_errors = []
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            reconstructed, _ = model(inputs)
            orig_np = inputs.cpu().numpy()
            recon_np = reconstructed.cpu().numpy()

            for i in range(len(orig_np)):
                orig_fft = np.abs(fft(orig_np[i]))
                recon_fft = np.abs(fft(recon_np[i]))
                freq_error = np.abs(orig_fft - recon_fft)
                freq_errors.append(freq_error[:30])  # Take first 30 frequency bins
                if len(freq_errors) >= 50:
                    break
            if len(freq_errors) >= 50:
                break

    freq_error_matrix = np.array(freq_errors[:50]).T  # Shape: (freq_bins, samples)
    create_error_heatmap_sonar(freq_error_matrix, sample_indices,
                              'results/visualizations/sonar/reconstruction_error_heatmap.png')

    # 2. Error Distribution Analysis
    create_error_distribution_plots(errors, labels, 'dae',
                                   'results/visualizations/sonar/error_distributions.png')

    # 3. Frequency Error Analysis
    generate_sonar_frequency_error_analysis(model, data_loader, device,
                                           'results/visualizations/sonar/frequency_error_analysis.png')

    # 4. Denoising Comparison (add noise to some samples)
    np.random.seed(42)
    noisy_samples = []
    denoised_samples = []
    original_clean = []

    model.eval()
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)

            # Add noise for denoising demonstration
            noise = torch.randn_like(inputs) * 0.2
            noisy_inputs = inputs + noise

            # Denoise
            reconstructed, _ = model(noisy_inputs)

            # Store first few samples
            orig_np = inputs.cpu().numpy()
            noisy_np = noisy_inputs.cpu().numpy()
            denoise_np = reconstructed.cpu().numpy()

            for i in range(min(5, len(orig_np))):
                original_clean.append(orig_np[i])
                noisy_samples.append(noisy_np[i])
                denoised_samples.append(denoise_np[i])

            if len(original_clean) >= 5:
                break

    create_denoised_comparison_plots(np.array(original_clean), np.array(denoised_samples),
                                    np.array(noisy_samples), 'dae',
                                    'results/visualizations/sonar/denoising_comparison.png')


def visualize_ims_anomalies(model, data_loader, device):
    """
    Generate comprehensive visualizations for IMS bearing anomaly detection.

    Defense Application: Temporal vibration analysis detects bearing wear patterns
    enabling preventive maintenance in critical aerospace components.
    """
    print("\n✈️ GENERATING IMS BEARING ANOMALY VISUALIZATIONS")

    # Compute reconstruction errors
    errors, originals, labels, _ = compute_reconstruction_errors(model, data_loader, 'vae', device)

    # For time-domain analysis, we need per-sample, per-time-step errors
    model.eval()
    time_errors_list = []

    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            reconstructed, _, _ = model(inputs)

            orig_np = inputs.cpu().numpy()
            recon_np = reconstructed.cpu().numpy()

            # Compute MSE per time step across all channels and batch
            time_errors = np.mean((orig_np - recon_np)**2, axis=(0, 1))  # Shape: (window_size,)
            time_errors_list.append(time_errors)

            if len(time_errors_list) >= 10:  # Aggregate from 10 batches
                break

    if time_errors_list:
        # Average time errors across batches
        avg_time_errors = np.mean(np.array(time_errors_list), axis=0)
        time_indices = np.arange(len(avg_time_errors))

        # 1. Time-Domain Error Plot
        create_time_domain_error_plot_ims(avg_time_errors, time_indices,
                                         'results/visualizations/ims/time_domain_errors.png')

        # 2. Error Distribution Analysis
        create_error_distribution_plots(errors, labels, 'vae',
                                       'results/visualizations/ims/error_distributions.png')

        # 3. Denoising Comparison
        np.random.seed(42)
        noisy_samples = []
        denoised_samples = []
        original_clean = []

        with torch.no_grad():
            for inputs, _ in data_loader:
                inputs = inputs.to(device)

                # Add noise for denoising demonstration
                noise = torch.randn_like(inputs) * 0.2
                noisy_inputs = inputs + noise

                # Denoise
                reconstructed, _, _ = model(noisy_inputs)

                # Store first few samples
                orig_np = inputs.cpu().numpy()
                noisy_np = noisy_inputs.cpu().numpy()
                denoise_np = reconstructed.cpu().numpy()

                for i in range(min(5, len(orig_np))):
                    original_clean.append(orig_np[i])
                    noisy_samples.append(noisy_np[i])
                    denoised_samples.append(denoise_np[i])

                if len(original_clean) >= 5:
                    break

        create_denoised_comparison_plots(np.array(original_clean), np.array(denoised_samples),
                                        np.array(noisy_samples), 'vae',
                                        'results/visualizations/ims/denoising_comparison.png')


def setup_visualization_logging():
    """Setup comprehensive logging for defense sensor visualization monitoring."""
    os.makedirs('results/logs', exist_ok=True)
    logging.basicConfig(
        filename='results/logs/visualization.log',
        level=logging.INFO,
        format='%(asctime)s - VISUALIZATION - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger('VISUALIZATION')
    return logger


def main():
    """Main visualization function for cross-domain autoencoder anomaly detection."""
    # Set seeds for reproducible visualization
    set_random_seed(42)

    # Setup logging
    logger = setup_visualization_logging()
    logger.info("Starting Defense Sensor Anomaly Detection Visualization")
    logger.info("="*70)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Visualization on device: {device}")

    print("🛡️ DEFENSE SENSOR ANOMALY DETECTION VISUALIZATION FRAMEWORK")
    print("="*70)
    print(f"Device: {device}")
    print("Generating research-quality visualizations...")

    # Visualize Sonar Anomalies (DAE)
    try:
        print("\n🚢 LOADING DAE MODEL FOR SONAR VISUALIZATION")
        dae_model = load_trained_model('checkpoints/dae_sonar_best.pth', 'dae', device)
        sonar_loader, _, _ = load_and_prepare_data('dae')
        visualize_sonar_anomalies(dae_model, sonar_loader, device)
        logger.info("Sonar anomaly visualizations completed successfully")
    except Exception as e:
        logger.error(f"Sonar visualization failed: {str(e)}")
        print(f"⚠️ Sonar visualization failed: {str(e)}")

    # Visualize IMS Bearing Anomalies (VAE)
    try:
        print("\n✈️ LOADING VAE MODEL FOR IMS VISUALIZATION")
        vae_model = load_trained_model('checkpoints/vae_ims_best.pth', 'vae', device)
        ims_loader, X_ims, _ = load_and_prepare_data('vae')
        if X_ims is not None and X_ims.size > 0:
            visualize_ims_anomalies(vae_model, ims_loader, device)
            logger.info("IMS bearing anomaly visualizations completed successfully")
        else:
            print("❌ IMS dataset not available. Skipping VAE visualization.")
            logger.info("IMS dataset not available - VAE visualization skipped")
    except Exception as e:
        logger.error(f"IMS visualization failed: {str(e)}")
        print(f"⚠️ IMS visualization failed: {str(e)}")

    # Final summary
    print("\n🎯 VISUALIZATION COMPLETED SUCCESSFULLY")
    print("="*70)
    print("📊 Visualization Results:")
    print("   📈 Sonar plots: results/visualizations/sonar/")
    print("   📈 IMS plots: results/visualizations/ims/")
    print("   📝 Logs: results/logs/visualization.log")
    print("🛡️ Research-ready visualizations generated for defense sensor analysis")

    logger.info("Visualization pipeline completed successfully")
    logger.info("="*70)


if __name__ == "__main__":
    main()