import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random

# Import custom modules for defense sensor anomaly detection
from data_loader import load_sonar_data, load_ims_data, set_random_seed
from models import create_dae_sonar, create_vae_ims

# Set style for defense-focused visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Defense Sensor Network Anomaly Detection Training Framework
# This module implements training pipelines for autoencoders in cross-domain anomaly detection
# Critical for early threat detection in naval and aerospace defense systems

def set_training_seeds(seed=42):
    """
    Set comprehensive random seeds for reproducible training in defense scenarios.

    Defense Context: Consistent model training critical for reliable sensor anomaly detection
    across different deployment environments and hardware configurations.
    """
    set_random_seed(seed)  # Custom seed setter from data_loader
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_validation_split(X, y, val_split=0.1, seed=42):
    """
    Create validation subset from training data for early stopping.

    Parameters:
    X (np.ndarray): Feature data
    y (np.ndarray): Labels
    val_split (float): Fraction for validation
    seed (int): Random seed

    Returns:
    tuple: (X_train, X_val, y_train, y_val)
    """
    np.random.seed(seed)
    n_samples = X.shape[0]
    val_size = int(n_samples * val_split)
    indices = np.random.permutation(n_samples)

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    X_train = X[train_indices]
    X_val = X[val_indices]
    y_train = y[train_indices]
    y_val = y[val_indices]

    return X_train, X_val, y_train, y_val

def train_dae(model, train_loader, val_loader, device, epochs=100, lr=1e-3, patience=10):
    """
    Train Denoising Autoencoder for sonar anomaly detection with parallel processing acceleration.

    Defense Application: Acoustic signal reconstruction for underwater mine detection
    - Early stopping prevents overfitting in limited defense sensor data
    - Validation monitoring ensures model generalization to unseen threats
    - Parallel processing enables faster training for real-time defense deployment
    """
    # Enable parallel processing if multiple GPUs available
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"🚀 Using DataParallel with {torch.cuda.device_count()} GPUs")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Early stopping parameters
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_path = 'checkpoints/dae_sonar_best.pth'

    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score

    # Training history for visualization and comprehensive metrics tracking
    train_losses = []
    val_losses = []
    train_aucs = []
    val_aucs = []
    train_f1s = []
    val_f1s = []
    train_precisions = []
    val_precisions = []
    train_recalls = []
    val_recalls = []
    train_accuracies = []
    val_accuracies = []

    model.to(device)
    model.train()

    print(f"\n🚢 Training DAE for Sonar Anomaly Detection on {device}")
    print("="*60)

    # Use CUDA streams for asynchronous data loading if on GPU
    if device.type == 'cuda':
        stream = torch.cuda.current_stream(device)

    for epoch in range(epochs):
        # Training phase with batch processing optimization and metrics calculation
        epoch_train_loss = 0.0
        train_errors = []
        train_labels = []

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]', leave=False)

        for batch_idx, (inputs, labels_batch) in enumerate(train_pbar):
            inputs = inputs.to(device)
            labels_batch = labels_batch.to(device)

            # Asynchronous noise addition and forward pass
            if device.type == 'cuda':
                with torch.cuda.stream(stream):
                    noisy_inputs = model.add_noise(inputs)
                    reconstructed, latent = model(noisy_inputs)
            else:
                noisy_inputs = model.add_noise(inputs)
                reconstructed, latent = model(noisy_inputs)

            optimizer.zero_grad()
            loss = model.loss_function(reconstructed, inputs)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

            # Collect reconstruction errors and labels for metrics (per-sample MSE)
            # Compute MSE per sample (mean over features for each sample in batch)
            per_sample_errors = torch.mean((reconstructed - inputs) ** 2, dim=1).detach().cpu().numpy()
            train_errors.extend(per_sample_errors)
            train_labels.extend(labels_batch.cpu().numpy())

            # Periodic visualization for research monitoring (reduced frequency for speed)
            if batch_idx % 100 == 0 and batch_idx > 0:
                model.visualize_reconstruction(inputs, reconstructed, epoch, batch_idx)

            train_pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Calculate training metrics
        train_errors = np.array(train_errors)
        train_labels = np.array(train_labels)

        # Use 95th percentile as anomaly threshold for training metrics
        train_threshold = np.percentile(train_errors, 95)
        train_predictions = (train_errors > train_threshold).astype(int)

        train_precision = precision_score(train_labels, train_predictions, zero_division=0)
        train_recall = recall_score(train_labels, train_predictions, zero_division=0)
        train_f1 = f1_score(train_labels, train_predictions, zero_division=0)
        train_accuracy = accuracy_score(train_labels, train_predictions)

        if len(np.unique(train_labels)) > 1:
            train_auc = roc_auc_score(train_labels, train_errors)
        else:
            train_auc = np.nan

        train_aucs.append(train_auc)
        train_f1s.append(train_f1)
        train_precisions.append(train_precision)
        train_recalls.append(train_recall)
        train_accuracies.append(train_accuracy)

        # Validation phase with optimized batch processing and metrics
        model.eval()
        val_loss = 0.0
        val_errors = []
        val_labels = []

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]', leave=False)
            for inputs, labels_batch in val_pbar:
                inputs = inputs.to(device)
                labels_batch = labels_batch.to(device)

                if device.type == 'cuda':
                    with torch.cuda.stream(stream):
                        reconstructed, _ = model(inputs)
                else:
                    reconstructed, _ = model(inputs)

                loss = model.loss_function(reconstructed, inputs)
                val_loss += loss.item()

                # Collect errors and labels for validation metrics (per-sample MSE)
                per_sample_val_errors = torch.mean((reconstructed - inputs) ** 2, dim=1).detach().cpu().numpy()
                val_errors.extend(per_sample_val_errors)
                val_labels.extend(labels_batch.cpu().numpy())

                val_pbar.set_postfix({'val_loss': f'{loss.item():.6f}'})

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Calculate validation metrics
        val_errors = np.array(val_errors)
        val_labels = np.array(val_labels)

        val_threshold = np.percentile(val_errors, 95)
        val_predictions = (val_errors > val_threshold).astype(int)

        val_precision = precision_score(val_labels, val_predictions, zero_division=0)
        val_recall = recall_score(val_labels, val_predictions, zero_division=0)
        val_f1 = f1_score(val_labels, val_predictions, zero_division=0)
        val_accuracy = accuracy_score(val_labels, val_predictions)

        if len(np.unique(val_labels)) > 1:
            val_auc = roc_auc_score(val_labels, val_errors)
        else:
            val_auc = np.nan

        val_aucs.append(val_auc)
        val_f1s.append(val_f1)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_accuracies.append(val_accuracy)

        scheduler.step(avg_val_loss)

        # Log training progress with comprehensive metrics
        current_lr = optimizer.param_groups[0]['lr']
        model.log_training_step(epoch, avg_train_loss, current_lr)

        print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {avg_train_loss:.6f} F1: {train_f1:.3f} AUC: {train_auc:.3f} | Val Loss: {avg_val_loss:.6f} F1: {val_f1:.3f} AUC: {val_auc:.3f} | LR: {current_lr:.6f}")

        # Log additional defense metrics for operational monitoring
        logger.info(f"Defense Metrics - Train Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, Accuracy: {train_accuracy:.4f}")
        logger.info(f"Defense Metrics - Val Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, Accuracy: {val_accuracy:.4f}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Handle DataParallel model state saving
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            best_model_state = state_dict

            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'lr': current_lr
            }, checkpoint_path)

            print(f"💾 Best model saved to {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹️ Early stopping triggered after {epoch+1} epochs (patience: {patience})")
                break

        model.train()

    # Load best model (handle DataParallel)
    if best_model_state is not None:
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(best_model_state)
        else:
            model.load_state_dict(best_model_state)

    # Defense Application: Comprehensive metrics tracking enables detailed performance analysis
    # for operational anomaly detection deployment in defense sensor networks
    return model, train_losses, val_losses, train_aucs, val_aucs, train_f1s, val_f1s, train_precisions, val_precisions, train_recalls, val_recalls, train_accuracies, val_accuracies

def train_vae(model, train_loader, val_loader, device, epochs=100, lr=1e-3, patience=10):
    """
    Train Variational Autoencoder for IMS bearing anomaly detection with parallel processing acceleration.

    Defense Application: Vibration signal generation for bearing failure prediction
    - KL divergence regularization prevents mode collapse in aerospace monitoring
    - Latent space visualization aids in anomaly pattern recognition
    - Parallel processing accelerates training for mission-critical deployment timelines
    """
    # Enable parallel processing if multiple GPUs available
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"🚀 Using DataParallel with {torch.cuda.device_count()} GPUs")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Early stopping parameters
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_path = 'checkpoints/vae_ims_best.pth'

    # Training history for visualization
    train_total_losses = []
    train_mse_losses = []
    train_kl_losses = []
    val_total_losses = []
    val_mse_losses = []
    val_kl_losses = []

    # Store latent space data for visualization (limited for memory efficiency)
    val_mu_list = []
    val_logvar_list = []
    val_labels_list = []

    model.to(device)
    model.train()

    print(f"\n✈️ Training VAE for IMS Bearing Anomaly Detection on {device}")
    print("="*60)

    # Use CUDA streams for asynchronous data loading if on GPU
    if device.type == 'cuda':
        stream = torch.cuda.current_stream(device)

    for epoch in range(epochs):
        # Training phase with optimized batch processing
        epoch_train_total = 0.0
        epoch_train_mse = 0.0
        epoch_train_kl = 0.0

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]', leave=False)

        for batch_idx, (inputs, labels) in enumerate(train_pbar):
            inputs = inputs.to(device)

            # Asynchronous forward pass for acceleration
            if device.type == 'cuda':
                with torch.cuda.stream(stream):
                    reconstructed, mu, logvar = model(inputs)
            else:
                reconstructed, mu, logvar = model(inputs)

            optimizer.zero_grad()
            total_loss, mse_loss, kl_loss = model.loss_function(reconstructed, inputs, mu, logvar)
            total_loss.backward()
            optimizer.step()

            epoch_train_total += total_loss.item()
            epoch_train_mse += mse_loss.item()
            epoch_train_kl += kl_loss.item()

            # Periodic visualization for research monitoring (reduced frequency for speed)
            if batch_idx % 100 == 0 and batch_idx > 0:
                model.visualize_reconstruction(inputs, reconstructed, epoch, batch_idx)

            train_pbar.set_postfix({
                'total': f'{total_loss.item():.6f}',
                'mse': f'{mse_loss.item():.6f}',
                'kl': f'{kl_loss.item():.6f}'
            })

        avg_train_total = epoch_train_total / len(train_loader)
        avg_train_mse = epoch_train_mse / len(train_loader)
        avg_train_kl = epoch_train_kl / len(train_loader)

        train_total_losses.append(avg_train_total)
        train_mse_losses.append(avg_train_mse)
        train_kl_losses.append(avg_train_kl)

        # Validation phase with optimized batch processing
        model.eval()
        val_total = 0.0
        val_mse = 0.0
        val_kl = 0.0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]', leave=False)
            for inputs, labels in val_pbar:
                inputs = inputs.to(device)

                if device.type == 'cuda':
                    with torch.cuda.stream(stream):
                        reconstructed, mu, logvar = model(inputs)
                else:
                    reconstructed, mu, logvar = model(inputs)

                total_loss, mse_loss, kl_loss = model.loss_function(reconstructed, inputs, mu, logvar)

                val_total += total_loss.item()
                val_mse += mse_loss.item()
                val_kl += kl_loss.item()

                # Collect latent space data for visualization (limit memory usage)
                if len(val_mu_list) < 10:  # Limit to 10 batches for memory efficiency
                    val_mu_list.append(mu.cpu())
                    val_logvar_list.append(logvar.cpu())
                    val_labels_list.append(labels.cpu())

                val_pbar.set_postfix({
                    'total': f'{total_loss.item():.6f}',
                    'mse': f'{mse_loss.item():.6f}',
                    'kl': f'{kl_loss.item():.6f}'
                })

        avg_val_total = val_total / len(val_loader)
        avg_val_mse = val_mse / len(val_loader)
        avg_val_kl = val_kl / len(val_loader)

        val_total_losses.append(avg_val_total)
        val_mse_losses.append(avg_val_mse)
        val_kl_losses.append(avg_val_kl)

        scheduler.step(avg_val_total)

        # Log training progress
        current_lr = optimizer.param_groups[0]['lr']
        model.log_training_step(epoch, avg_train_total, avg_train_mse, avg_train_kl, current_lr)

        print(f"Epoch {epoch+1:3d}/{epochs} | Train Total: {avg_train_total:.6f} MSE: {avg_train_mse:.6f} KL: {avg_train_kl:.6f} | Val Total: {avg_val_total:.6f} MSE: {avg_val_mse:.6f} KL: {avg_val_kl:.6f} | LR: {current_lr:.6f}")

        # Early stopping check
        if avg_val_total < best_val_loss:
            best_val_loss = avg_val_total
            patience_counter = 0
            # Handle DataParallel model state saving
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            best_model_state = state_dict

            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_total,
                'val_loss': avg_val_total,
                'lr': current_lr
            }, checkpoint_path)

            print(f"💾 Best model saved to {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹️ Early stopping triggered after {epoch+1} epochs (patience: {patience})")
                break

        model.train()

    # Load best model (handle DataParallel)
    if best_model_state is not None:
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(best_model_state)
        else:
            model.load_state_dict(best_model_state)

    # Generate final latent space visualization (limit data for memory efficiency)
    if val_mu_list and val_logvar_list and val_labels_list:
        mu_all = torch.cat(val_mu_list[:5], dim=0)  # Limit to first 5 batches
        logvar_all = torch.cat(val_logvar_list[:5], dim=0)
        labels_all = torch.cat(val_labels_list[:5], dim=0)
        if isinstance(model, nn.DataParallel):
            model.module.visualize_latent_space(mu_all, logvar_all, labels_all, epoch=len(train_total_losses)-1)
        else:
            model.visualize_latent_space(mu_all, logvar_all, labels_all, epoch=len(train_total_losses)-1)

    return model, train_total_losses, train_mse_losses, train_kl_losses, val_total_losses, val_mse_losses, val_kl_losses

def plot_training_curves(losses_dict, model_name, save_path, aucs=None, f1s=None):
    """
    Generate comprehensive research-quality training curve visualizations including metrics.

    Defense Context: Multi-metric visualization enables assessment of model performance
    across different evaluation criteria critical for defense sensor deployment decisions.
    """
    os.makedirs('results/visualizations', exist_ok=True)

    epochs = range(1, len(losses_dict[list(losses_dict.keys())[0]]) + 1)

    if model_name == 'DAE':
        plt.figure(figsize=(20, 12))

        # Loss curves
        plt.subplot(2, 3, 1)
        plt.plot(epochs, losses_dict['train'], 'b-', linewidth=2, label='Training Loss', marker='o', markersize=3)
        plt.plot(epochs, losses_dict['val'], 'r-', linewidth=2, label='Validation Loss', marker='s', markersize=3)
        plt.xlabel('Epoch')
        plt.ylabel('MSE Reconstruction Loss')
        plt.title(f'{model_name} Loss Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # AUC curves if available
        if aucs and len(aucs[0]) > 0:
            plt.subplot(2, 3, 2)
            plt.plot(epochs, aucs[0], 'g-', linewidth=2, label='Train AUC', marker='o', markersize=3)
            plt.plot(epochs, aucs[1], 'm-', linewidth=2, label='Val AUC', marker='s', markersize=3)
            plt.xlabel('Epoch')
            plt.ylabel('AUC Score')
            plt.title(f'{model_name} AUC Curves')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim([0, 1])

        # F1 curves if available
        if f1s and len(f1s[0]) > 0:
            plt.subplot(2, 3, 3)
            plt.plot(epochs, f1s[0], 'c-', linewidth=2, label='Train F1', marker='o', markersize=3)
            plt.plot(epochs, f1s[1], 'y-', linewidth=2, label='Val F1', marker='s', markersize=3)
            plt.xlabel('Epoch')
            plt.ylabel('F1 Score')
            plt.title(f'{model_name} F1 Curves')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim([0, 1])

        # Precision/Recall subplot
        if aucs and f1s and len(aucs[0]) > 0 and len(f1s[0]) > 0:
            plt.subplot(2, 3, 4)
            plt.plot(epochs, aucs[0], 'g-', linewidth=2, label='Train AUC')
            plt.plot(epochs, aucs[1], 'm-', linewidth=2, label='Val AUC')
            plt.plot(epochs, f1s[0], 'c-', linewidth=2, label='Train F1')
            plt.plot(epochs, f1s[1], 'y-', linewidth=2, label='Val F1')
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.title(f'{model_name} Performance Metrics')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim([0, 1])

        # Loss trend analysis
        plt.subplot(2, 3, 5)
        train_trend = np.polyfit(range(len(losses_dict['train'])), losses_dict['train'], 1)[0]
        val_trend = np.polyfit(range(len(losses_dict['val'])), losses_dict['val'], 1)[0]
        plt.plot(epochs, losses_dict['train'], 'b-', linewidth=2, label='.4f')
        plt.plot(epochs, losses_dict['val'], 'r-', linewidth=2, label='.4f')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{model_name} Loss Trends')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Performance summary
        plt.subplot(2, 3, 6)
        if aucs and f1s and len(aucs[0]) > 0 and len(f1s[0]) > 0:
            final_train_auc = aucs[0][-1] if not np.isnan(aucs[0][-1]) else 0
            final_val_auc = aucs[1][-1] if not np.isnan(aucs[1][-1]) else 0
            final_train_f1 = f1s[0][-1] if not np.isnan(f1s[0][-1]) else 0
            final_val_f1 = f1s[1][-1] if not np.isnan(f1s[1][-1]) else 0

            metrics = ['Final Train AUC', 'Final Val AUC', 'Final Train F1', 'Final Val F1']
            values = [final_train_auc, final_val_auc, final_train_f1, final_val_f1]
            colors = ['green', 'magenta', 'cyan', 'yellow']

            bars = plt.bar(metrics, values, color=colors, alpha=0.7)
            plt.ylabel('Score')
            plt.title(f'{model_name} Final Performance')
            plt.xticks(rotation=45, ha='right')
            plt.ylim([0, 1])
            plt.grid(True, alpha=0.3, axis='y')

            # Add value labels
            for bar, value in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        '.3f', ha='center', va='bottom', fontsize=10)

    else:
        # VAE plots remain similar but can be extended
        plt.figure(figsize=(16, 10))

        plt.subplot(2, 2, 1)
        plt.plot(epochs, losses_dict['train_total'], 'k-', linewidth=2, label='Train Total', marker='o', markersize=3)
        plt.plot(epochs, losses_dict['val_total'], 'k--', linewidth=2, label='Val Total', marker='s', markersize=3)
        plt.xlabel('Epoch')
        plt.ylabel('Total Loss')
        plt.title(f'{model_name} Total Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 2)
        plt.plot(epochs, losses_dict['train_mse'], 'b-', linewidth=2, label='Train MSE', marker='o', markersize=3)
        plt.plot(epochs, losses_dict['val_mse'], 'b--', linewidth=2, label='Val MSE', marker='s', markersize=3)
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title(f'{model_name} MSE Reconstruction Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 3)
        plt.plot(epochs, losses_dict['train_kl'], 'r-', linewidth=2, label='Train KL', marker='o', markersize=3)
        plt.plot(epochs, losses_dict['val_kl'], 'r--', linewidth=2, label='Val KL', marker='s', markersize=3)
        plt.xlabel('Epoch')
        plt.ylabel('KL Divergence Loss')
        plt.title(f'{model_name} KL Divergence Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 4)
        plt.plot(epochs, losses_dict['train_kl'], 'r-', linewidth=2, label='KL Loss', marker='o', markersize=3)
        plt.plot(epochs, losses_dict['train_mse'], 'b-', linewidth=2, label='MSE Loss', marker='^', markersize=3)
        plt.xlabel('Epoch')
        plt.ylabel('Loss Components')
        plt.title(f'{model_name} Loss Component Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📊 Enhanced training curves saved to {save_path}")

def setup_training_logging():
    """Setup comprehensive logging for defense sensor training monitoring."""
    os.makedirs('results/logs', exist_ok=True)
    logging.basicConfig(
        filename='results/logs/train.log',
        level=logging.INFO,
        format='%(asctime)s - TRAIN - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger('TRAIN')
    return logger

def main():
    """Main training function for cross-domain autoencoder anomaly detection."""
    # Set seeds for reproducible defense sensor training
    set_training_seeds(42)

    # Setup logging
    logger = setup_training_logging()
    logger.info("Starting Unified Autoencoder Training for Defense Sensor Networks")
    logger.info("="*70)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on device: {device}")

    # Hyperparameters optimized for faster training with batch processing
    batch_size_sonar = 64  # Increased for better GPU utilization
    batch_size_ims = 32    # Increased for better GPU utilization
    epochs = 100
    learning_rate = 1e-3
    patience = 15
    val_split = 0.1
    num_workers = 4  # Parallel data loading for faster training

    print("🛡️ UNIFIED AUTOENCODER FRAMEWORK FOR CROSS-DOMAIN ANOMALY DETECTION")
    print("="*70)
    print(f"Device: {device}")
    print(f"Seeds set for reproducibility")
    print(f"Validation split: {val_split}")
    print(f"Early stopping patience: {patience}")
    print(f"Parallel data loading: {num_workers} workers")
    logger.info(f"Batch sizes - Sonar: {batch_size_sonar}, IMS: {batch_size_ims}")
    logger.info(f"Epochs: {epochs}, Learning rate: {learning_rate}")
    logger.info(f"Data loading workers: {num_workers}")

    # Load and prepare sonar data for DAE training
    print("\n🔊 Loading UCI Sonar Dataset...")
    logger.info("Loading sonar data for DAE training")
    X_sonar, y_sonar = load_sonar_data()
    X_sonar_train, X_sonar_val, y_sonar_train, y_sonar_val = create_validation_split(X_sonar, y_sonar, val_split)

    # Convert to tensors
    X_sonar_train = torch.FloatTensor(X_sonar_train)
    X_sonar_val = torch.FloatTensor(X_sonar_val)
    y_sonar_train = torch.LongTensor(y_sonar_train)
    y_sonar_val = torch.LongTensor(y_sonar_val)

    # Create data loaders with parallel processing for faster training
    # Use persistent workers for reduced overhead in iterative training
    sonar_train_dataset = TensorDataset(X_sonar_train, y_sonar_train)
    sonar_val_dataset = TensorDataset(X_sonar_val, y_sonar_val)
    sonar_train_loader = DataLoader(sonar_train_dataset, batch_size=batch_size_sonar, shuffle=True,
                                   num_workers=num_workers, pin_memory=True, persistent_workers=True,
                                   prefetch_factor=2)
    sonar_val_loader = DataLoader(sonar_val_dataset, batch_size=batch_size_sonar, shuffle=False,
                                 num_workers=num_workers, pin_memory=True, persistent_workers=True,
                                 prefetch_factor=2)

    print(f"Sonar data: {X_sonar_train.shape[0]} train, {X_sonar_val.shape[0]} val samples")

    # Load and prepare IMS data for VAE training
    print("\n⚙️ Loading NASA IMS Bearing Dataset...")
    logger.info("Loading IMS data for VAE training")
    X_ims, y_ims = load_ims_data()
    if X_ims.size == 0:
        print("❌ IMS dataset not available. Skipping VAE training.")
        logger.error("IMS dataset not found - VAE training skipped")
        X_ims_train, X_ims_val, y_ims_train, y_ims_val = None, None, None, None
    else:
        X_ims_train, X_ims_val, y_ims_train, y_ims_val = create_validation_split(X_ims, y_ims, val_split)

        # Convert to tensors
        X_ims_train = torch.FloatTensor(X_ims_train)
        X_ims_val = torch.FloatTensor(X_ims_val)
        y_ims_train = torch.LongTensor(y_ims_train)
        y_ims_val = torch.LongTensor(y_ims_val)

        # Create data loaders with parallel processing for faster training
        # Use persistent workers for reduced overhead in iterative training
        ims_train_dataset = TensorDataset(X_ims_train, y_ims_train)
        ims_val_dataset = TensorDataset(X_ims_val, y_ims_val)
        ims_train_loader = DataLoader(ims_train_dataset, batch_size=batch_size_ims, shuffle=True,
                                     num_workers=num_workers, pin_memory=True, persistent_workers=True,
                                     prefetch_factor=2)
        ims_val_loader = DataLoader(ims_val_dataset, batch_size=batch_size_ims, shuffle=False,
                                   num_workers=num_workers, pin_memory=True, persistent_workers=True,
                                   prefetch_factor=2)

        print(f"IMS data: {X_ims_train.shape[0]} train, {X_ims_val.shape[0]} val windows")

    # Train DAE for Sonar anomaly detection
    print("\n🚢 PHASE 1: Training DAE for Sonar Acoustic Anomaly Detection")
    logger.info("Starting DAE training for sonar anomaly detection")
    dae_model = create_dae_sonar()
    trained_dae, train_losses_dae, val_losses_dae, train_aucs_dae, val_aucs_dae, train_f1s_dae, val_f1s_dae, train_precisions_dae, val_precisions_dae, train_recalls_dae, val_recalls_dae, train_accuracies_dae, val_accuracies_dae = train_dae(
        dae_model, sonar_train_loader, sonar_val_loader, device,
        epochs=epochs, lr=learning_rate, patience=patience
    )

    # Plot DAE training curves with comprehensive metrics
    losses_dict_dae = {'train': train_losses_dae, 'val': val_losses_dae}
    plot_training_curves(losses_dict_dae, 'DAE', 'results/visualizations/dae_training_curves.png',
                        aucs=(train_aucs_dae, val_aucs_dae), f1s=(train_f1s_dae, val_f1s_dae))

    # Log final DAE metrics
    if train_aucs_dae and val_aucs_dae:
        logger.info(f"DAE Final Metrics - Train AUC: {train_aucs_dae[-1]:.4f}, Val AUC: {val_aucs_dae[-1]:.4f}")
        logger.info(f"DAE Final Metrics - Train F1: {train_f1s_dae[-1]:.4f}, Val F1: {val_f1s_dae[-1]:.4f}")
        logger.info(f"DAE Final Metrics - Train Precision: {train_precisions_dae[-1]:.4f}, Val Precision: {val_precisions_dae[-1]:.4f}")
        logger.info(f"DAE Final Metrics - Train Recall: {train_recalls_dae[-1]:.4f}, Val Recall: {val_recalls_dae[-1]:.4f}")
        logger.info(f"DAE Final Metrics - Train Accuracy: {train_accuracies_dae[-1]:.4f}, Val Accuracy: {val_accuracies_dae[-1]:.4f}")

    logger.info("DAE training completed and curves plotted")

    # Train VAE for IMS bearing anomaly detection (if data available)
    if X_ims is not None and X_ims.size > 0:
        print("\n✈️ PHASE 2: Training VAE for IMS Bearing Vibration Anomaly Detection")
        logger.info("Starting VAE training for IMS bearing anomaly detection")
        vae_model = create_vae_ims()
        trained_vae, train_total_vae, train_mse_vae, train_kl_vae, val_total_vae, val_mse_vae, val_kl_vae = train_vae(
            vae_model, ims_train_loader, ims_val_loader, device,
            epochs=epochs, lr=learning_rate, patience=patience
        )

        # Plot VAE training curves
        losses_dict_vae = {
            'train_total': train_total_vae, 'train_mse': train_mse_vae, 'train_kl': train_kl_vae,
            'val_total': val_total_vae, 'val_mse': val_mse_vae, 'val_kl': val_kl_vae
        }
        plot_training_curves(losses_dict_vae, 'VAE', 'results/visualizations/vae_training_curves.png')
        logger.info("VAE training completed and curves plotted")
    else:
        trained_vae = None
        logger.info("VAE training skipped due to missing IMS data")

    # Final summary
    print("\n🎯 TRAINING COMPLETED SUCCESSFULLY")
    print("="*70)
    print("📊 Results Summary:")
    print(f"   DAE (Sonar): Trained with {len(train_losses_dae)} epochs")
    print(f"   Final Train Loss: {train_losses_dae[-1]:.6f}")
    print(f"   Final Val Loss: {val_losses_dae[-1]:.6f}")
    if train_aucs_dae and val_aucs_dae:
        print(f"   Final Train F1: {train_f1s_dae[-1]:.4f}, AUC: {train_aucs_dae[-1]:.4f}")
        print(f"   Final Val F1: {val_f1s_dae[-1]:.4f}, AUC: {val_aucs_dae[-1]:.4f}")
    print(f"   Model saved to: checkpoints/dae_sonar_best.pth")

    if trained_vae is not None:
        print(f"   VAE (IMS): Trained with {len(train_total_vae)} epochs")
        print(f"   Final Train Loss: {train_total_vae[-1]:.6f}")
        print(f"   Final Val Loss: {val_total_vae[-1]:.6f}")
        print(f"   Model saved to: checkpoints/vae_ims_best.pth")

    print("📈 Visualizations saved to: results/visualizations/")
    print("📝 Training logs saved to: results/logs/train.log")
    print("🛡️ Ready for anomaly detection deployment in defense sensor networks")

    logger.info("Training pipeline completed successfully")
    logger.info("="*70)

if __name__ == "__main__":
    main()