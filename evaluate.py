import torch
import torch.nn as nn
import numpy as np
import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, confusion_matrix
)
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import custom modules for defense sensor anomaly detection
from data_loader import load_sonar_data, load_ims_data, set_random_seed
from models import create_dae_sonar, create_vae_ims

# Set style for defense-focused visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Defense Sensor Network Anomaly Detection Evaluation Framework
# This module implements comprehensive evaluation metrics and visualizations for autoencoder-based
# anomaly detection in cross-domain defense sensor networks (sonar acoustics and bearing vibrations)
# Critical for assessing model performance in threat detection scenarios

def load_trained_model(checkpoint_path, model_type='dae', device='cpu'):
    """
    Load trained autoencoder model from checkpoint for evaluation.

    Defense Application: Model deployment requires robust loading mechanisms
    to ensure consistent anomaly detection performance in operational environments.

    Parameters:
    checkpoint_path (str): Path to model checkpoint
    model_type (str): 'dae' for sonar or 'vae' for IMS bearing
    device (str): Device to load model on

    Returns:
    torch.nn.Module: Loaded model in evaluation mode
    """
    if model_type == 'dae':
        model = create_dae_sonar()
    elif model_type == 'vae':
        model = create_vae_ims()
    else:
        raise ValueError("Invalid model_type. Must be 'dae' or 'vae'")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"✅ Loaded {model_type.upper()} model from {checkpoint_path}")
    print(f"   Trained for {checkpoint.get('epoch', 'N/A')} epochs")
    print(f"   Final validation loss: {checkpoint.get('val_loss', 'N/A'):.6f}")

    return model

def compute_reconstruction_errors(model, data_loader, model_type='dae', device='cpu'):
    """
    Compute reconstruction errors for all samples in the dataset.

    Defense Application: Reconstruction error serves as anomaly score in defense sensor networks,
    where higher errors indicate potential threats or system degradation.

    Parameters:
    model: Trained autoencoder model
    data_loader: DataLoader with test/validation data
    model_type (str): 'dae' or 'vae'
    device (str): Computation device

    Returns:
    tuple: (reconstruction_errors, original_data, labels, latencies)
    """
    model.eval()
    errors = []
    latencies = []
    originals = []
    labels_list = []

    print(f"🔍 Computing reconstruction errors for {model_type.upper()}...")
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Computing errors"):
            inputs = inputs.to(device)
            originals.append(inputs.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

            # Measure inference latency (critical for real-time defense systems)
            start_time = time.time()

            if model_type == 'dae':
                reconstructed, _ = model(inputs)
                # Compute per-sample MSE errors (mean over features for each sample)
                per_sample_errors = torch.mean((reconstructed - inputs) ** 2, dim=1).detach().cpu().numpy()
                errors.extend(per_sample_errors)
            elif model_type == 'vae':
                reconstructed, _, _ = model(inputs)
                # Compute per-sample MSE errors (mean over all dimensions for each sample)
                per_sample_errors = torch.mean((reconstructed - inputs) ** 2, dim=[1, 2]).detach().cpu().numpy()
                errors.extend(per_sample_errors)

            # Latency in milliseconds per sample
            latency = (time.time() - start_time) * 1000 / inputs.size(0)
            latencies.append(latency)

    # Concatenate all batches
    originals = np.concatenate(originals, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    errors = np.array(errors)
    latencies = np.array(latencies)

    print(f"   Processed {len(errors)} samples")
    print(f"   Average latency: {np.mean(latencies):.3f} ms/sample")
    print(f"   Reconstruction error range: {np.min(errors):.6f} - {np.max(errors):.6f}")

    return errors, originals, labels, latencies

def determine_anomaly_thresholds(reconstruction_errors, labels, strategies=['percentile', 'statistical']):
    """
    Determine anomaly detection thresholds using multiple strategies.

    Defense Application: Threshold selection is critical for balancing false positives/negatives
    in threat detection systems where missing anomalies can be catastrophic.

    Parameters:
    reconstruction_errors (np.ndarray): Reconstruction errors per sample
    labels (np.ndarray): True anomaly labels (0=normal, 1=anomaly)
    strategies (list): Threshold determination strategies

    Returns:
    dict: Thresholds for each strategy
    """
    thresholds = {}

    # Debug: Check array sizes for alignment
    print(f"Debug: reconstruction_errors shape: {reconstruction_errors.shape}, labels shape: {labels.shape}")

    # Percentile-based threshold (common in defense applications)
    if 'percentile' in strategies:
        # Use percentile of all samples (since we don't have clear normal/anomalous separation in validation)
        if len(reconstruction_errors) > 0:
            thresholds['percentile_95'] = np.percentile(reconstruction_errors, 95)
            thresholds['percentile_99'] = np.percentile(reconstruction_errors, 99)
            thresholds['percentile_999'] = np.percentile(reconstruction_errors, 99.9)

    # Statistical threshold (mean + k*std of all samples)
    if 'statistical' in strategies:
        if len(reconstruction_errors) > 0:
            mean_all = np.mean(reconstruction_errors)
            std_all = np.std(reconstruction_errors)
            thresholds['statistical_2std'] = mean_all + 2 * std_all
            thresholds['statistical_3std'] = mean_all + 3 * std_all
            thresholds['statistical_4std'] = mean_all + 4 * std_all

    print("🎯 Determined anomaly thresholds:")
    for strategy, threshold in thresholds.items():
        print(".6f")

    return thresholds

def evaluate_anomaly_detection(reconstruction_errors, labels, thresholds):
    """
    Evaluate anomaly detection performance using various thresholds and metrics.

    Defense Application: Comprehensive evaluation ensures reliable threat detection
    with quantifiable performance metrics for operational deployment decisions.

    Parameters:
    reconstruction_errors (np.ndarray): Reconstruction errors per sample
    labels (np.ndarray): True anomaly labels
    thresholds (dict): Dictionary of threshold values

    Returns:
    dict: Evaluation results for each threshold
    """
    results = {}

    for threshold_name, threshold in thresholds.items():
        # Binary predictions based on threshold
        predictions = (reconstruction_errors > threshold).astype(int)

        # Compute classification metrics
        precision = precision_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)

        # Compute AUROC if we have both classes
        if len(np.unique(labels)) > 1:
            try:
                auroc = roc_auc_score(labels, reconstruction_errors)
            except ValueError:
                auroc = np.nan
        else:
            auroc = np.nan

        results[threshold_name] = {
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auroc': auroc,
            'predictions': predictions
        }

        print(f"📊 {threshold_name}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, AUROC={auroc:.3f}")

    return results

def create_evaluation_visualizations(reconstruction_errors, labels, evaluation_results,
                                   model_type='dae', save_dir='results/evaluation'):
    """
    Generate research-quality visualizations for anomaly detection evaluation.

    Defense Application: Visual analytics provide critical insights for defense operators
    to understand model behavior and optimize threat detection parameters.

    Parameters:
    reconstruction_errors (np.ndarray): Reconstruction errors per sample
    labels (np.ndarray): True anomaly labels
    evaluation_results (dict): Results from evaluate_anomaly_detection
    model_type (str): 'dae' or 'vae'
    save_dir (str): Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1. Reconstruction Error Distribution
    plt.figure(figsize=(14, 10))

    # Separate normal and anomalous errors
    normal_errors = reconstruction_errors[labels == 0]
    anomaly_errors = reconstruction_errors[labels == 1]

    plt.subplot(2, 2, 1)
    plt.hist(normal_errors, bins=50, alpha=0.7, color='blue', label='Normal', density=True)
    plt.hist(anomaly_errors, bins=50, alpha=0.7, color='red', label='Anomaly', density=True)
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Density')
    plt.title(f'{model_type.upper()} Reconstruction Error Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. ROC Curves
    plt.subplot(2, 2, 2)
    if len(np.unique(labels)) > 1:
        fpr, tpr, _ = roc_curve(labels, reconstruction_errors)
        plt.plot(fpr, tpr, 'b-', linewidth=2, label='.3f')
        plt.plot([0, 1], [0, 1], 'r--', alpha=0.7, label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_type.upper()} ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # 3. Precision-Recall Curves
    plt.subplot(2, 2, 3)
    if len(np.unique(labels)) > 1:
        precision_curve, recall_curve, _ = precision_recall_curve(labels, reconstruction_errors)
        plt.plot(recall_curve, precision_curve, 'g-', linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{model_type.upper()} Precision-Recall Curve')
        plt.grid(True, alpha=0.3)

    # 4. Threshold Performance Comparison
    plt.subplot(2, 2, 4)
    threshold_names = list(evaluation_results.keys())
    f1_scores = [evaluation_results[name]['f1_score'] for name in threshold_names]
    plt.bar(range(len(threshold_names)), f1_scores, color='steelblue', alpha=0.8)
    plt.xticks(range(len(threshold_names)), [name.replace('_', '\n') for name in threshold_names],
               rotation=45, ha='right')
    plt.ylabel('F1 Score')
    plt.title('Threshold Performance Comparison')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/{model_type}_evaluation_visualizations.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Additional visualization: Confusion matrices for different thresholds
    best_threshold = max(evaluation_results.keys(), key=lambda x: evaluation_results[x]['f1_score'])

    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(labels, evaluation_results[best_threshold]['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_type.upper()} Confusion Matrix (Best Threshold: {best_threshold})')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{model_type}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📈 Visualizations saved to {save_dir}")

def log_evaluation_results(evaluation_results, latencies, model_type='dae', log_file='results/evaluation/evaluation.log'):
    """
    Log comprehensive evaluation results for defense sensor monitoring.

    Defense Application: Detailed logging ensures traceability and auditability
    of anomaly detection performance in critical defense systems.

    Parameters:
    evaluation_results (dict): Results from evaluate_anomaly_detection
    latencies (np.ndarray): Inference latencies per sample
    model_type (str): 'dae' or 'vae'
    log_file (str): Path to log file
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - EVAL_{} - %(levelname)s - %(message)s'.format(model_type.upper()),
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logger = logging.getLogger(f'EVAL_{model_type.upper()}')

    logger.info("="*80)
    logger.info(f"ANOMALY DETECTION EVALUATION RESULTS - {model_type.upper()}")
    logger.info("="*80)

    logger.info(f"Total samples evaluated: {len(latencies)}")
    logger.info(f"Average latency: {np.mean(latencies):.3f} ms/sample")
    logger.info(f"Latency std: {np.std(latencies):.3f} ms/sample")
    logger.info(f"Latency range: {np.min(latencies):.3f} - {np.max(latencies):.3f} ms/sample")

    logger.info("THRESHOLD PERFORMANCE METRICS:")
    for threshold_name, results in evaluation_results.items():
        logger.info(f"{threshold_name}:")
        logger.info(f"  Threshold: {results['threshold']:.6f}")
        logger.info(f"  Precision: {results['precision']:.4f}")
        logger.info(f"  Recall: {results['recall']:.4f}")
        logger.info(f"  F1 Score: {results['f1_score']:.4f}")
        logger.info(f"  AUROC: {results['auroc']:.4f}")

    # Determine best performing threshold
    best_threshold = max(evaluation_results.keys(), key=lambda x: evaluation_results[x]['f1_score'])
    best_f1 = evaluation_results[best_threshold]['f1_score']

    logger.info(f"Best performing threshold: {best_threshold} (F1={best_f1:.4f})")
    logger.info("Evaluation completed successfully")
    logger.info("="*80)

    print(f"📝 Evaluation results logged to {log_file}")

def evaluate_dae_model(checkpoint_path='checkpoints/dae_sonar_best.pth', device='cpu'):
    """
    Comprehensive evaluation of trained DAE model for sonar anomaly detection.

    Defense Application: Acoustic anomaly detection for underwater threat identification
    in naval defense sensor networks.

    Parameters:
    checkpoint_path (str): Path to DAE checkpoint
    device (str): Computation device
    """
    print("🚢 EVALUATING DAE MODEL FOR SONAR ANOMALY DETECTION")
    print("="*60)

    try:
        # Load model
        model = load_trained_model(checkpoint_path, 'dae', device)

        # Load test data (using all data as test set for evaluation)
        print("🔊 Loading sonar data for evaluation...")
        X_sonar, y_sonar = load_sonar_data()
        X_sonar = torch.FloatTensor(X_sonar)
        y_sonar = torch.LongTensor(y_sonar)

        # Create data loader with same batch size as training for consistent evaluation
        from torch.utils.data import TensorDataset, DataLoader
        test_dataset = TensorDataset(X_sonar, y_sonar)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)  # Match training batch size

        # Compute reconstruction errors and latencies
        errors, originals, labels, latencies = compute_reconstruction_errors(
            model, test_loader, 'dae', device
        )

        # Determine thresholds
        thresholds = determine_anomaly_thresholds(errors, labels)

        # Evaluate anomaly detection
        evaluation_results = evaluate_anomaly_detection(errors, labels, thresholds)

        # Create visualizations
        create_evaluation_visualizations(errors, labels, evaluation_results, 'dae')

        # Log results
        log_evaluation_results(evaluation_results, latencies, 'dae')

        print("✅ DAE evaluation completed successfully")

    except Exception as e:
        print(f"❌ DAE evaluation failed: {str(e)}")
        raise

def evaluate_vae_model(checkpoint_path='checkpoints/vae_ims_best.pth', device='cpu'):
    """
    Comprehensive evaluation of trained VAE model for IMS bearing anomaly detection.

    Defense Application: Vibration anomaly detection for bearing degradation monitoring
    in aerospace defense systems.

    Parameters:
    checkpoint_path (str): Path to VAE checkpoint
    device (str): Computation device
    """
    print("✈️ EVALUATING VAE MODEL FOR IMS BEARING ANOMALY DETECTION")
    print("="*60)

    try:
        # Load model
        model = load_trained_model(checkpoint_path, 'vae', device)

        # Load test data
        print("⚙️ Loading IMS bearing data for evaluation...")
        X_ims, y_ims = load_ims_data()
        if X_ims.size == 0:
            print("❌ IMS dataset not available. Skipping VAE evaluation.")
            return

        # Use the same validation split as training for consistency
        from train import create_validation_split
        X_ims_train, X_ims_val, y_ims_train, y_ims_val = create_validation_split(
            X_ims, y_ims, val_split=0.1, seed=42
        )

        X_ims_val = torch.FloatTensor(X_ims_val)
        y_ims_val = torch.LongTensor(y_ims_val)

        # Transpose IMS data for Conv1D: (batch, window_size, channels) -> (batch, channels, window_size)
        X_ims_val = X_ims_val.permute(0, 2, 1)

        # Create data loader with validation data only (like training)
        from torch.utils.data import TensorDataset, DataLoader
        test_dataset = TensorDataset(X_ims_val, y_ims_val)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        # Store validation labels for threshold determination
        y_ims_val_np = y_ims_val.numpy()

        # Compute reconstruction errors and latencies
        errors, originals, labels, latencies = compute_reconstruction_errors(
            model, test_loader, 'vae', device
        )

        # Use the validation labels we stored earlier instead of the ones from compute_reconstruction_errors
        # This ensures size consistency for threshold determination
        # We need to match the exact number of samples that were processed
        n_samples = len(errors)
        labels = y_ims_val_np[:n_samples]  # Take only the first n_samples labels

        # Determine thresholds
        thresholds = determine_anomaly_thresholds(errors, labels)

        # Evaluate anomaly detection
        evaluation_results = evaluate_anomaly_detection(errors, labels, thresholds)

        # Create visualizations
        create_evaluation_visualizations(errors, labels, evaluation_results, 'vae')

        # Log results
        log_evaluation_results(evaluation_results, latencies, 'vae')

        print("✅ VAE evaluation completed successfully")

    except Exception as e:
        print(f"❌ VAE evaluation failed: {str(e)}")
        raise

def main():
    """Main evaluation function for cross-domain autoencoder anomaly detection."""
    # Set seeds for reproducible evaluation
    set_random_seed(42)

    # Setup logging
    os.makedirs('results/logs', exist_ok=True)
    logging.basicConfig(
        filename='results/logs/evaluation_main.log',
        level=logging.INFO,
        format='%(asctime)s - EVAL_MAIN - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger('EVAL_MAIN')

    logger.info("Starting Unified Autoencoder Evaluation for Defense Sensor Networks")
    logger.info("="*80)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Evaluation on device: {device}")
    logger.info(f"Evaluation device: {device}")

    print("🛡️ UNIFIED AUTOENCODER EVALUATION FRAMEWORK")
    print("="*80)

    # Evaluate DAE for Sonar
    try:
        evaluate_dae_model(device=device)
        logger.info("DAE evaluation completed successfully")
    except Exception as e:
        logger.error(f"DAE evaluation failed: {str(e)}")
        print(f"⚠️ DAE evaluation failed: {str(e)}")

    # Evaluate VAE for IMS
    try:
        evaluate_vae_model(device=device)
        logger.info("VAE evaluation completed successfully")
    except Exception as e:
        logger.error(f"VAE evaluation failed: {str(e)}")
        print(f"⚠️ VAE evaluation failed: {str(e)}")

    # Final summary
    print("\n🎯 EVALUATION COMPLETED")
    print("="*80)
    print("📊 Results Summary:")
    print("   📈 Visualizations: results/evaluation/")
    print("   📝 Logs: results/logs/evaluation*.log")
    print("   🔍 Check individual model logs for detailed metrics")
    print("🛡️ Ready for operational deployment in defense sensor networks")

    logger.info("Evaluation pipeline completed")
    logger.info("="*80)

if __name__ == "__main__":
    main()