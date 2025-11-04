import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import logging

# Set style for defense-focused visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Defense Sensor Network Anomaly Detection Models
# This module defines autoencoder architectures for cross-domain anomaly detection
# DAE for Sonar acoustic signals, VAE for IMS bearing seismic/vibration signals
# Critical for early threat detection in naval and aerospace defense systems

class DenoisingAutoencoder(nn.Module):
    """
    Denoising Autoencoder for UCI Sonar dataset anomaly detection.

    Defense Application: Acoustic signal reconstruction for underwater mine detection
    - Input: 60-dimensional sonar signal features
    - Architecture: Symmetric encoder-decoder with bottleneck compression
    - Loss: MSE reconstruction loss
    - Purpose: Learn normal seabed patterns (rocks), detect anomalous mine signatures
    """

    def __init__(self, input_dim=60, hidden_dims=[32, 16, 8]):
        super(DenoisingAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[2])
        )

        # Decoder layers (symmetric to encoder)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dims[2], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], input_dim),
            nn.Sigmoid()
        )

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

    def loss_function(self, reconstructed, original):
        """MSE reconstruction loss for denoising autoencoder."""
        mse_loss = F.mse_loss(reconstructed, original, reduction='mean')
        return mse_loss

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

        # Log reconstruction quality
        reconstruction_error = np.mean((orig_np - recon_np)**2, axis=1)
        self.logger.info(f'Epoch {epoch}, Batch {batch_idx} - Mean Reconstruction Error: {np.mean(reconstruction_error):.6f}')

    def log_training_step(self, epoch, loss, lr):
        """Log training progress for research and defense monitoring."""
        self.logger.info(f'Epoch {epoch} - Training Loss: {loss:.6f}, Learning Rate: {lr:.6f}')


class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder for NASA IMS Bearing dataset anomaly detection.

    Defense Application: Vibration signal generation for bearing failure prediction
    - Input: Time-series vibration windows (1024 samples x n_channels)
    - Architecture: Convolutional encoder, dense latent space, transposed decoder
    - Loss: MSE reconstruction + KL divergence regularization
    - Purpose: Model normal bearing operation, detect degradation anomalies
    """

    def __init__(self, window_size=1024, n_channels=4, latent_dim=64, conv_filters=[32, 64]):
        super(VariationalAutoencoder, self).__init__()
        self.window_size = window_size
        self.n_channels = n_channels
        self.latent_dim = latent_dim
        self.conv_filters = conv_filters

        # Encoder: Conv1D layers for temporal feature extraction
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(n_channels, conv_filters[0], kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(conv_filters[0], conv_filters[1], kernel_size=5, stride=2, padding=2),
            nn.ReLU()
        )

        # Calculate flattened dimension after conv layers
        self.flattened_dim = self._get_flattened_dim()

        # Latent space: Mean and log-variance
        self.fc_mu = nn.Linear(self.flattened_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_dim, latent_dim)

        # Decoder: Dense to ConvTranspose1D
        self.decoder_fc = nn.Linear(latent_dim, self.flattened_dim)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(conv_filters[1], conv_filters[0], kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(conv_filters[0], n_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid()
        )

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
        conv_input = fc_output.view(fc_output.size(0), self.conv_filters[1], -1)
        reconstructed = self.decoder_conv(conv_input)
        return reconstructed

    def forward(self, x):
        """Forward pass through VAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar

    def loss_function(self, reconstructed, original, mu, logvar):
        """VAE loss: MSE reconstruction + KL divergence."""
        # MSE reconstruction loss
        mse_loss = F.mse_loss(reconstructed, original, reduction='mean')

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / original.numel()  # Normalize by input size

        total_loss = mse_loss + kl_loss
        return total_loss, mse_loss, kl_loss

    def visualize_latent_space(self, mu, logvar, labels, epoch):
        """Generate research visualizations for latent space analysis."""
        os.makedirs('results/visualizations/ims', exist_ok=True)

        # Convert to numpy
        mu_np = mu.detach().cpu().numpy()
        logvar_np = logvar.detach().cpu().numpy()

        # Latent space scatter plot (first 2 dimensions)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Color by anomaly labels
        colors = ['blue' if label == 0 else 'red' for label in labels]

        # Mu scatter
        axes[0].scatter(mu_np[:, 0], mu_np[:, 1], c=colors, alpha=0.7, s=50, edgecolors='black')
        axes[0].set_xlabel('Latent Dimension 1 (μ)')
        axes[0].set_ylabel('Latent Dimension 2 (μ)')
        axes[0].set_title('Latent Space Distribution (Mean Vectors)')
        axes[0].grid(True, alpha=0.3)

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

        # Log reconstruction quality
        reconstruction_error = np.mean((orig_np - recon_np)**2)
        self.logger.info(f'Epoch {epoch}, Batch {batch_idx} - Reconstruction MSE: {reconstruction_error:.6f}')

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

def create_vae_ims(window_size=1024, n_channels=4, latent_dim=64):
    """
    Factory function for IMS VAE model instantiation.

    Defense Context: Vibration anomaly detection for aerospace bearing monitoring
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