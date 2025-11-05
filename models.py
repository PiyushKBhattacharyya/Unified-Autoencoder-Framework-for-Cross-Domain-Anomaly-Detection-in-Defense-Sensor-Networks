import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import logging
from sklearn.manifold import TSNE
from scipy.fft import fft

# Set style for defense-focused visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Defense Sensor Network Anomaly Detection Models
# This module defines autoencoder architectures for cross-domain anomaly detection
# DAE for Sonar acoustic signals, VAE for IMS bearing seismic/vibration signals
# Critical for early threat detection in naval and aerospace defense systems

class DenoisingAutoencoder(nn.Module):
    """
    Deep Denoising Autoencoder for UCI Sonar dataset anomaly detection with false negative penalization.

    Defense Application: Acoustic signal reconstruction for underwater mine detection
    - Input: 60-dimensional sonar signal features
    - Architecture: Ultra-deep encoder-decoder with asymmetric loss for anomaly sensitivity
    - Loss: Weighted MSE reconstruction loss (heavily penalizes false negatives)
    - Purpose: Learn normal seabed patterns (rocks), detect anomalous mine signatures
    """

    def __init__(self, input_dim=60, hidden_dims=[256, 128, 64]):
        super(DenoisingAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        # Balanced anomaly weighting
        self.anomaly_weight = 3.0

        # Optimized encoder layers with balanced capacity and regularization
        encoder_layers = []

        # Input layer
        encoder_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        encoder_layers.append(nn.BatchNorm1d(hidden_dims[0]))
        encoder_layers.append(nn.ReLU())
        encoder_layers.append(nn.Dropout(0.1))

        # Hidden layers with progressive reduction
        for i in range(len(hidden_dims)-1):
            in_features = hidden_dims[i]
            out_features = hidden_dims[i+1]

            encoder_layers.append(nn.Linear(in_features, out_features))
            encoder_layers.append(nn.BatchNorm1d(out_features))
            encoder_layers.append(nn.ReLU())
            # Reduced dropout for better learning
            if i < len(hidden_dims)-2:
                encoder_layers.append(nn.Dropout(0.05))

        self.encoder = nn.Sequential(*encoder_layers)

        # Optimized decoder layers with symmetric architecture
        decoder_layers = []

        # Hidden layers with progressive expansion
        for i in reversed(range(len(hidden_dims)-1)):
            in_features = hidden_dims[i+1]
            out_features = hidden_dims[i]

            decoder_layers.append(nn.Linear(in_features, out_features))
            decoder_layers.append(nn.BatchNorm1d(out_features))
            decoder_layers.append(nn.ReLU())
            if i > 0:
                decoder_layers.append(nn.Dropout(0.05))

        # Output layer
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        decoder_layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*decoder_layers)

        # Setup logging for research and defense sensor monitoring
        self.setup_logging()

    def setup_logging(self):
        """Initialize logging for model training and anomaly detection research."""
        os.makedirs('results/logs', exist_ok=True)
        logging.basicConfig(
            filename='results/logs/dae_sonar_training.log',
            level=logging.INFO,
            format='%(asctime)s - DAE_Sonar - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('DAE_Sonar')

    def forward(self, x):
        """Forward pass through encoder and decoder."""
        # Encode to latent space
        latent = self.encoder(x)

        # Decode back to original space
        reconstructed = self.decoder(latent)

        return reconstructed, latent

    def loss_function(self, reconstructed, original, labels=None):
        """
        Weighted MSE reconstruction loss that heavily penalizes false negatives.

        Defense Application: Anomalies (mines) must not be missed, so we heavily weight
        reconstruction errors for anomalous samples to improve anomaly detection sensitivity.
        """
        mse_loss = F.mse_loss(reconstructed, original, reduction='none')  # Per-sample loss

        if labels is not None:
            # Apply heavy weighting to anomalous samples (false negative penalization)
            # Normal samples (rocks): weight = 1.0
            # Anomalous samples (mines): weight = self.anomaly_weight (10.0)
            sample_weights = torch.where(labels == 1,
                                       torch.full_like(labels, self.anomaly_weight, dtype=torch.float),
                                       torch.ones_like(labels, dtype=torch.float))
            weighted_loss = mse_loss * sample_weights.unsqueeze(-1)  # Broadcast to feature dimension
            return torch.mean(weighted_loss)
        else:
            # Default MSE loss when labels not available (during inference)
            return torch.mean(mse_loss)

    def add_noise(self, x, noise_factor=0.1):
        """Add Gaussian noise for denoising training (defense sensor robustness)."""
        noise = torch.randn_like(x) * noise_factor
        return x + noise

    def visualize_reconstruction(self, original, reconstructed, epoch, batch_idx):
        """Generate research visualizations for reconstruction quality assessment."""
        os.makedirs('results/visualizations/sonar', exist_ok=True)

        # Convert to numpy for plotting
        orig_np = original.detach().cpu().numpy()
        recon_np = reconstructed.detach().cpu().numpy()

        # Sample reconstruction comparison
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))

        for i in range(5):
            # Original signal
            axes[0, i].plot(orig_np[i], 'b-', linewidth=2, label='Original')
            axes[0, i].set_title(f'Sample {i+1} - Original Sonar Signal')
            axes[0, i].set_xlabel('Feature Index')
            axes[0, i].set_ylabel('Normalized Amplitude')
            axes[0, i].grid(True, alpha=0.3)

            # Reconstructed signal
            axes[1, i].plot(recon_np[i], 'r-', linewidth=2, label='Reconstructed')
            axes[1, i].plot(orig_np[i], 'b--', linewidth=1, alpha=0.7, label='Original')
            axes[1, i].set_title(f'Sample {i+1} - Reconstruction')
            axes[1, i].set_xlabel('Feature Index')
            axes[1, i].set_ylabel('Normalized Amplitude')
            axes[1, i].legend()
            axes[1, i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'results/visualizations/sonar/dae_reconstruction_epoch_{epoch}_batch_{batch_idx}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        # Advanced visualization: Reconstruction error per frequency band
        if epoch % 10 == 0:  # Generate advanced plots every 10 epochs
            self.visualize_frequency_analysis(orig_np, recon_np, epoch)

        # Log reconstruction quality
        reconstruction_error = np.mean((orig_np - recon_np)**2, axis=1)
        self.logger.info(f'Epoch {epoch}, Batch {batch_idx} - Mean Reconstruction Error: {np.mean(reconstruction_error):.6f}')

    def visualize_frequency_analysis(self, original, reconstructed, epoch):
        """Advanced sonar visualization: Frequency domain analysis for anomaly detection."""
        os.makedirs('results/visualizations/sonar', exist_ok=True)

        # Compute FFT for frequency analysis
        fft_orig = np.abs(fft(original, axis=1))
        fft_recon = np.abs(fft(reconstructed, axis=1))

        # Define frequency bands (sonar features are in specific ranges)
        n_freqs = original.shape[1]
        freq_bands = np.linspace(0, n_freqs//2, 10)  # 10 frequency bands

        # Reconstruction error per frequency band
        band_errors = []
        band_labels = []

        for i in range(len(freq_bands)-1):
            start_idx = int(freq_bands[i])
            end_idx = int(freq_bands[i+1])

            orig_band = fft_orig[:, start_idx:end_idx]
            recon_band = fft_recon[:, start_idx:end_idx]
            error = np.mean((orig_band - recon_band)**2, axis=(0, 1))
            band_errors.append(error)
            band_labels.append(f'{start_idx}-{end_idx}')

        # Bar plot of reconstruction error per frequency band
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        bars = ax.bar(band_labels, band_errors, color='steelblue', alpha=0.8, edgecolor='black')
        ax.set_xlabel('Frequency Band (FFT Bins)')
        ax.set_ylabel('Mean Squared Reconstruction Error')
        ax.set_title(f'Sonar Reconstruction Error by Frequency Band - Epoch {epoch}')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(f'results/visualizations/sonar/dae_frequency_error_epoch_{epoch}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def log_training_step(self, epoch, loss, lr):
        """Log training progress for research and defense monitoring."""
        self.logger.info(f'Epoch {epoch} - Training Loss: {loss:.6f}, Learning Rate: {lr:.6f}')


class VariationalAutoencoder(nn.Module):
    """
    Deep Variational Autoencoder for NASA IMS Bearing dataset anomaly detection with false negative penalization.

    Defense Application: Vibration signal generation for bearing failure prediction
    - Input: Time-series vibration windows (1024 samples x n_channels)
    - Architecture: Ultra-deep convolutional encoder-decoder with asymmetric loss
    - Loss: Weighted MSE reconstruction + KL divergence (heavily penalizes false negatives)
    - Purpose: Model normal bearing operation, detect degradation anomalies
    """

    def __init__(self, window_size=1024, n_channels=4, latent_dim=128, conv_filters=[128, 256, 512]):
        super(VariationalAutoencoder, self).__init__()
        self.window_size = window_size
        self.n_channels = n_channels
        self.latent_dim = latent_dim
        self.conv_filters = conv_filters
        # Balanced anomaly weighting for stable training
        self.anomaly_weight = 5.0

        # Optimized encoder: Balanced Conv1D layers for vibration signal processing
        encoder_layers = []

        # First conv block
        encoder_layers.append(nn.Conv1d(n_channels, conv_filters[0], kernel_size=11, stride=2, padding=5))
        encoder_layers.append(nn.BatchNorm1d(conv_filters[0]))
        encoder_layers.append(nn.ReLU())
        encoder_layers.append(nn.Dropout(0.1))

        # Second conv block
        encoder_layers.append(nn.Conv1d(conv_filters[0], conv_filters[1], kernel_size=7, stride=2, padding=3))
        encoder_layers.append(nn.BatchNorm1d(conv_filters[1]))
        encoder_layers.append(nn.ReLU())
        encoder_layers.append(nn.Dropout(0.1))

        # Third conv block
        encoder_layers.append(nn.Conv1d(conv_filters[1], conv_filters[2], kernel_size=5, stride=2, padding=2))
        encoder_layers.append(nn.BatchNorm1d(conv_filters[2]))
        encoder_layers.append(nn.ReLU())
        encoder_layers.append(nn.Dropout(0.05))

        self.encoder_conv = nn.Sequential(*encoder_layers)

        # Calculate flattened dimension after conv layers
        self.flattened_dim = self._get_flattened_dim()

        # Latent space: Mean and log-variance with higher capacity
        self.fc_mu = nn.Linear(self.flattened_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_dim, latent_dim)

        # Decoder: Symmetric ConvTranspose1D layers
        self.decoder_fc = nn.Linear(latent_dim, self.flattened_dim)

        decoder_layers = []

        # First deconv block
        decoder_layers.append(nn.ConvTranspose1d(conv_filters[2], conv_filters[1], kernel_size=5, stride=2, padding=2, output_padding=1))
        decoder_layers.append(nn.BatchNorm1d(conv_filters[1]))
        decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Dropout(0.05))

        # Second deconv block
        decoder_layers.append(nn.ConvTranspose1d(conv_filters[1], conv_filters[0], kernel_size=7, stride=2, padding=3, output_padding=1))
        decoder_layers.append(nn.BatchNorm1d(conv_filters[0]))
        decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Dropout(0.1))

        # Third deconv block
        decoder_layers.append(nn.ConvTranspose1d(conv_filters[0], n_channels, kernel_size=11, stride=2, padding=5, output_padding=1))
        decoder_layers.append(nn.Sigmoid())

        self.decoder_conv = nn.Sequential(*decoder_layers)

        # Setup logging for research and defense sensor monitoring
        self.setup_logging()

    def _get_flattened_dim(self):
        """Calculate dimension after convolutional encoding."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.n_channels, self.window_size)
            conv_output = self.encoder_conv(dummy_input)
            return conv_output.numel()

    def setup_logging(self):
        """Initialize logging for model training and anomaly detection research."""
        os.makedirs('results/logs', exist_ok=True)
        logging.basicConfig(
            filename='results/logs/vae_ims_training.log',
            level=logging.INFO,
            format='%(asctime)s - VAE_IMS - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('VAE_IMS')

    def encode(self, x):
        """Encode input to latent distribution parameters."""
        # x shape: (batch, channels, window_size)
        conv_features = self.encoder_conv(x)
        flattened = conv_features.view(conv_features.size(0), -1)

        mu = self.fc_mu(flattened)
        logvar = self.fc_logvar(flattened)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick for latent sampling."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode latent vector to reconstructed signal."""
        # z shape: (batch, latent_dim)
        fc_output = self.decoder_fc(z)
        # Reshape to match the last conv layer output shape
        conv_input = fc_output.view(fc_output.size(0), self.conv_filters[3], -1)
        reconstructed = self.decoder_conv(conv_input)
        return reconstructed

    def forward(self, x):
        """Forward pass through VAE."""
        # Handle input shape flexibility: (batch, window_size, channels) or (batch, channels, window_size)
        if x.dim() == 3 and x.shape[1] == self.window_size:
            # Transpose from (batch, window_size, channels) to (batch, channels, window_size)
            x = x.permute(0, 2, 1)

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar

    def loss_function(self, reconstructed, original, mu, logvar, labels=None):
        """
        Weighted VAE loss: Weighted MSE reconstruction + KL divergence with false negative penalization.

        Defense Application: Bearing failures in aerospace systems are critical - false negatives
        (missing degradation) are catastrophic, so we heavily weight anomalous samples.
        """
        # Weighted MSE reconstruction loss
        mse_loss_per_sample = F.mse_loss(reconstructed, original, reduction='none').mean(dim=[1, 2])  # Per-sample MSE

        if labels is not None:
            # Apply heavy weighting to anomalous samples (false negative penalization)
            # Normal samples: weight = 1.0
            # Anomalous samples: weight = self.anomaly_weight (15.0)
            sample_weights = torch.where(labels == 1,
                                       torch.full_like(labels, self.anomaly_weight, dtype=torch.float),
                                       torch.ones_like(labels, dtype=torch.float))
            weighted_mse_loss = (mse_loss_per_sample * sample_weights).mean()
        else:
            weighted_mse_loss = mse_loss_per_sample.mean()

        # KL divergence loss (unchanged)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / original.numel()  # Normalize by input size

        total_loss = weighted_mse_loss + kl_loss
        return total_loss, weighted_mse_loss, kl_loss

    def visualize_latent_space(self, mu, logvar, labels, epoch):
        """Generate research visualizations for latent space analysis."""
        os.makedirs('results/visualizations/ims', exist_ok=True)

        # Convert to numpy
        mu_np = mu.detach().cpu().numpy()
        logvar_np = logvar.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        # Advanced latent space visualization with t-SNE
        if mu_np.shape[0] > 50:  # Only if we have enough samples
            self.visualize_latent_tsne(mu_np, labels_np, epoch)

        # Latent space scatter plot (first 2 dimensions)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Color by anomaly labels
        colors = ['blue' if label == 0 else 'red' for label in labels_np]

        # Mu scatter
        axes[0].scatter(mu_np[:, 0], mu_np[:, 1], c=colors, alpha=0.7, s=50, edgecolors='black')
        axes[0].set_xlabel('Latent Dimension 1 (μ)')
        axes[0].set_ylabel('Latent Dimension 2 (μ)')
        axes[0].set_title('Latent Space Distribution (Mean Vectors)')
        axes[0].grid(True, alpha=0.3)
        # Add legend
        axes[0].scatter([], [], c='blue', label='Normal Operation', alpha=0.7, s=50, edgecolors='black')
        axes[0].scatter([], [], c='red', label='Anomalous Degradation', alpha=0.7, s=50, edgecolors='black')
        axes[0].legend()

        # Log-variance distribution
        axes[1].hist(logvar_np.flatten(), bins=50, alpha=0.7, color='purple', edgecolor='black')
        axes[1].set_xlabel('Log-Variance Values')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Latent Space Uncertainty Distribution')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'results/visualizations/ims/vae_latent_space_epoch_{epoch}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        # Log latent space statistics
        self.logger.info(f'Epoch {epoch} - Latent μ mean: {np.mean(mu_np):.4f}, std: {np.std(mu_np):.4f}')
        self.logger.info(f'Epoch {epoch} - Latent logvar mean: {np.mean(logvar_np):.4f}, std: {np.std(logvar_np):.4f}')

    def visualize_latent_tsne(self, mu, labels, epoch):
        """t-SNE visualization for better latent space separation analysis."""
        os.makedirs('results/visualizations/ims', exist_ok=True)

        # Apply t-SNE to latent space
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(mu)-1))
        latent_tsne = tsne.fit_transform(mu)

        # Plot t-SNE projection
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        colors = ['blue' if label == 0 else 'red' for label in labels]

        scatter = ax.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=colors, alpha=0.7, s=60, edgecolors='black')
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.set_title(f'VAE Latent Space t-SNE Projection - Epoch {epoch}')
        ax.grid(True, alpha=0.3)

        # Add legend
        ax.scatter([], [], c='blue', label='Normal Operation', alpha=0.7, s=60, edgecolors='black')
        ax.scatter([], [], c='red', label='Anomalous Degradation', alpha=0.7, s=60, edgecolors='black')
        ax.legend()

        plt.tight_layout()
        plt.savefig(f'results/visualizations/ims/vae_latent_tsne_epoch_{epoch}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def visualize_reconstruction(self, original, reconstructed, epoch, batch_idx):
        """Generate research visualizations for signal reconstruction."""
        os.makedirs('results/visualizations/ims', exist_ok=True)

        # Convert to numpy
        orig_np = original.detach().cpu().numpy()
        recon_np = reconstructed.detach().cpu().numpy()

        # Time-domain reconstruction comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        for ch in range(min(2, orig_np.shape[1])):  # Show first 2 channels
            # Original signal
            axes[0, ch].plot(orig_np[0, ch, :], 'b-', linewidth=2, label='Original')
            axes[0, ch].set_title(f'Channel {ch+1} - Original Vibration Signal')
            axes[0, ch].set_xlabel('Time Sample')
            axes[0, ch].set_ylabel('Normalized Amplitude')
            axes[0, ch].grid(True, alpha=0.3)

            # Reconstructed signal
            axes[1, ch].plot(recon_np[0, ch, :], 'r-', linewidth=2, label='Reconstructed')
            axes[1, ch].plot(orig_np[0, ch, :], 'b--', linewidth=1, alpha=0.7, label='Original')
            axes[1, ch].set_title(f'Channel {ch+1} - VAE Reconstruction')
            axes[1, ch].set_xlabel('Time Sample')
            axes[1, ch].set_ylabel('Normalized Amplitude')
            axes[1, ch].legend()
            axes[1, ch].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'results/visualizations/ims/vae_reconstruction_epoch_{epoch}_batch_{batch_idx}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        # Advanced visualization: Time-domain reconstruction error analysis
        if batch_idx % 50 == 0:  # Generate advanced plots periodically
            self.visualize_reconstruction_error(orig_np, recon_np, epoch, batch_idx)

        # Log reconstruction quality
        reconstruction_error = np.mean((orig_np - recon_np)**2)
        self.logger.info(f'Epoch {epoch}, Batch {batch_idx} - Reconstruction MSE: {reconstruction_error:.6f}')

    def visualize_reconstruction_error(self, original, reconstructed, epoch, batch_idx):
        """Advanced seismic visualization: Time-domain reconstruction error analysis."""
        os.makedirs('results/visualizations/ims', exist_ok=True)

        # Compute reconstruction error across time
        error = (original - reconstructed) ** 2  # MSE per sample

        # Aggregate error across channels and batch
        time_error = np.mean(error, axis=(0, 1))  # Mean error per time step

        # Plot time-domain reconstruction error
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        ax.plot(time_error, 'r-', linewidth=2, label='Reconstruction MSE')
        ax.fill_between(range(len(time_error)), time_error, alpha=0.3, color='red')
        ax.set_xlabel('Time Sample Index')
        ax.set_ylabel('Mean Squared Reconstruction Error')
        ax.set_title(f'VAE Time-Domain Reconstruction Error - Epoch {epoch}, Batch {batch_idx}')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Highlight potential anomaly regions (high error spikes)
        threshold = np.mean(time_error) + 2 * np.std(time_error)
        anomaly_indices = time_error > threshold
        if np.any(anomaly_indices):
            ax.scatter(np.where(anomaly_indices)[0], time_error[anomaly_indices],
                      c='darkred', s=50, marker='x', label='Potential Anomalies', zorder=5)
            ax.legend()

        plt.tight_layout()
        plt.savefig(f'results/visualizations/ims/vae_error_analysis_epoch_{epoch}_batch_{batch_idx}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def log_training_step(self, epoch, total_loss, mse_loss, kl_loss, lr):
        """Log training progress for research and defense monitoring."""
        self.logger.info(f'Epoch {epoch} - Total Loss: {total_loss:.6f}, MSE: {mse_loss:.6f}, KL: {kl_loss:.6f}, LR: {lr:.6f}')


# Model factory functions for defense sensor integration
def create_dae_sonar(input_dim=60):
    """
    Factory function for Sonar DAE model instantiation.

    Defense Context: Acoustic anomaly detection for naval underwater surveillance
    """
    model = DenoisingAutoencoder(input_dim=input_dim)
    print(f"Created DAE for Sonar: Input dim {input_dim}, Latent dim 8")
    return model

def create_vae_ims(window_size=1024, n_channels=8, latent_dim=128):
    """
    Factory function for IMS VAE model instantiation.

    Defense Context: Vibration anomaly detection for aerospace bearing monitoring
    Note: n_channels=8 to match the actual IMS data (8 channels)
    """
    model = VariationalAutoencoder(
        window_size=window_size,
        n_channels=n_channels,
        latent_dim=latent_dim
    )
    print(f"Created VAE for IMS: Window {window_size}, Channels {n_channels}, Latent dim {latent_dim}")
    return model


if __name__ == "__main__":
    # Model instantiation and validation tests
    print("Defense Sensor Autoencoder Models - Validation Test")
    print("="*60)

    # Test DAE for Sonar
    dae = create_dae_sonar()
    dummy_sonar = torch.randn(10, 60)  # Batch of 10 sonar samples
    recon_sonar, latent_sonar = dae(dummy_sonar)
    loss_sonar = dae.loss_function(recon_sonar, dummy_sonar)
    print(f"DAE Sonar Test - Input: {dummy_sonar.shape}, Output: {recon_sonar.shape}, Latent: {latent_sonar.shape}, Loss: {loss_sonar:.4f}")

    # Test VAE for IMS
    vae = create_vae_ims()
    dummy_ims = torch.randn(10, 4, 1024)  # Batch of 10 vibration windows
    recon_ims, mu_ims, logvar_ims = vae(dummy_ims)
    total_loss, mse_loss, kl_loss = vae.loss_function(recon_ims, dummy_ims, mu_ims, logvar_ims)
    print(f"VAE IMS Test - Input: {dummy_ims.shape}, Output: {recon_ims.shape}, Latent μ: {mu_ims.shape}, Loss: {total_loss:.4f} (MSE: {mse_loss:.4f}, KL: {kl_loss:.4f})")

    print("\nModel architectures validated for defense sensor anomaly detection.")
    print("Ready for integration with data_loader.py and future train/test pipeline.")