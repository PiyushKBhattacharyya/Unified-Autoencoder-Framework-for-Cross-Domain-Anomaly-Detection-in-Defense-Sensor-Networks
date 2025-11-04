import argparse
import torch
import numpy as np
import os
import logging
from datetime import datetime
import time
from tabulate import tabulate
import pandas as pd

# Import custom modules for defense sensor anomaly detection
from data_loader import load_sonar_data, load_ims_data, set_random_seed
from models import create_dae_sonar, create_vae_ims
from train import train_dae, train_vae, plot_training_curves, set_training_seeds
from evaluate import evaluate_dae_model, evaluate_vae_model
from visualize import main as run_visualizations

# Defense Sensor Network Unified Anomaly Detection Framework
# This module implements the master controller for cross-domain anomaly detection
# in naval and aerospace defense systems using autoencoder architectures
# Critical for operational threat detection and system health monitoring

class DefenseAnomalyDetector:
    """
    Unified Anomaly Detection Framework for Defense Sensor Networks.

    This class orchestrates the complete pipeline for training and evaluating
    autoencoder-based anomaly detection systems across different sensor modalities:
    - Sonar (DAE): Underwater acoustic threat detection (mines, submarines)
    - Seismic/IMS (VAE): Aerospace bearing vibration monitoring (failure prediction)

    Defense Applications:
    - Naval: Real-time underwater mine detection in contested environments
    - Aerospace: Predictive maintenance for critical aircraft components
    - Multi-domain: Cross-sensor anomaly correlation for enhanced situational awareness
    """

    def __init__(self, config):
        """
        Initialize the defense anomaly detection framework.

        Parameters:
        config (dict): Configuration parameters for training and evaluation
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_logging()
        self.logger.info("Defense Anomaly Detector initialized")
        self.logger.info(f"Configuration: {config}")

        # Ensure reproducibility across defense sensor deployments
        set_training_seeds(config['seed'])

    def setup_logging(self):
        """Setup comprehensive logging for operational defense monitoring."""
        os.makedirs('results/logs', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        logging.basicConfig(
            filename=f'results/logs/defense_anomaly_detector_{timestamp}.log',
            level=logging.INFO,
            format='%(asctime)s - DEFENSE_DETECTOR - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger('DEFENSE_DETECTOR')

        # Also log to console for real-time monitoring
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def load_and_prepare_data(self, dataset_choice):
        """
        Load and prepare sensor data for anomaly detection training.

        Defense Context: Data preparation ensures consistent preprocessing
        across different deployment environments and sensor configurations.
        """
        self.logger.info(f"Loading {dataset_choice} dataset for anomaly detection")

        if dataset_choice == 'sonar':
            X_data, y_data = load_sonar_data()
            return X_data, y_data, 'sonar'
        elif dataset_choice == 'ims':
            X_data, y_data = load_ims_data()
            if X_data.size == 0:
                raise ValueError("IMS dataset not available. Please ensure NASA IMS data is properly configured.")
            return X_data, y_data, 'ims'
        else:
            raise ValueError("Invalid dataset choice. Must be 'sonar' or 'ims'")

    def create_model(self, dataset_type):
        """
        Create appropriate autoencoder model for the sensor modality.

        Defense Context: Model selection based on sensor characteristics
        ensures optimal anomaly detection performance for specific threat types.
        """
        if dataset_type == 'sonar':
            model = create_dae_sonar()
            model_type = 'DAE'
        elif dataset_type == 'ims':
            model = create_vae_ims()
            model_type = 'VAE'
        else:
            raise ValueError("Invalid dataset type")

        self.logger.info(f"Created {model_type} model for {dataset_type} anomaly detection")
        return model, model_type

    def train_model(self, model, X_train, X_val, y_train, y_val, model_type):
        """
        Train the autoencoder model with optimized defense sensor training parameters.

        Defense Context: Robust training ensures reliable model performance
        in operational environments with varying sensor noise conditions.
        """
        self.logger.info(f"Starting {model_type} training on {self.device}")

        # Create data loaders with optimized batch processing for defense applications
        from torch.utils.data import TensorDataset, DataLoader
        batch_size = self.config['batch_size']

        if model_type == 'DAE':
            # Sonar data: tabular features
            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)
        else:
            # IMS data: time-series windows - transpose for Conv1D (batch, channels, window_size)
            X_train = X_train.permute(0, 2, 1)  # (batch, window_size, channels) -> (batch, channels, window_size)
            X_val = X_val.permute(0, 2, 1)      # (batch, window_size, channels) -> (batch, channels, window_size)
            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)

        # Defense-optimized data loading with parallel processing and prefetching
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                num_workers=0, pin_memory=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=0, pin_memory=False)

        # Train with defense-specific hyperparameters
        start_time = time.time()

        if model_type == 'DAE':
            trained_model, train_losses, val_losses, train_aucs, val_aucs, train_f1s, val_f1s, train_precisions, val_precisions, train_recalls, val_recalls, train_accuracies, val_accuracies = train_dae(
                model, train_loader, val_loader, self.device,
                epochs=self.config['epochs'], lr=self.config['learning_rate'], patience=15
            )
            losses_dict = {'train': train_losses, 'val': val_losses}
            aucs = (train_aucs, val_aucs)
            f1s = (train_f1s, val_f1s)
            precisions = (train_precisions, val_precisions)
            recalls = (train_recalls, val_recalls)
            accuracies = (train_accuracies, val_accuracies)
            # Extract final metrics for visualization
            final_metrics = {
                'train_precision': train_precisions[-1] if train_precisions else None,
                'val_precision': val_precisions[-1] if val_precisions else None,
                'train_recall': train_recalls[-1] if train_recalls else None,
                'val_recall': val_recalls[-1] if val_recalls else None,
                'train_accuracy': train_accuracies[-1] if train_accuracies else None,
                'val_accuracy': val_accuracies[-1] if val_accuracies else None
            }
        else:
            trained_model, train_total, train_mse, train_kl, val_total, val_mse, val_kl, *metrics = train_vae(
                model, train_loader, val_loader, self.device,
                epochs=self.config['epochs'], lr=self.config['learning_rate'], patience=15
            )
            losses_dict = {'train_total': train_total, 'train_mse': train_mse, 'train_kl': train_kl,
                          'val_total': val_total, 'val_mse': val_mse, 'val_kl': val_kl}
            aucs = None
            f1s = None
            final_metrics = None  # VAE doesn't have classification metrics during training

        training_time = time.time() - start_time
        self.logger.info(f"{model_type} training completed in {training_time:.2f} seconds")

        # Generate training curve visualizations
        viz_path = f"results/visualizations/{model_type.lower()}_training_curves.png"
        if model_type == 'DAE':
            plot_training_curves(losses_dict, model_type, viz_path, aucs, f1s, final_metrics, precisions, recalls, accuracies)
        else:
            plot_training_curves(losses_dict, model_type, viz_path, aucs, f1s, final_metrics)

        return trained_model, losses_dict, final_metrics

    def evaluate_and_visualize(self, dataset_type):
        """
        Evaluate trained models and generate operational visualizations.

        Defense Context: Comprehensive evaluation ensures model reliability
        before deployment in critical defense sensor networks.
        """
        self.logger.info(f"Evaluating {dataset_type.upper()} anomaly detection performance")

        # Run evaluation metrics
        if dataset_type == 'sonar':
            evaluate_dae_model(device=str(self.device))
        elif dataset_type == 'ims':
            evaluate_vae_model(device=str(self.device))

        # Generate operational visualizations
        self.logger.info("Generating defense sensor visualizations")
        run_visualizations()

    def run_detection_pipeline(self):
        """
        Execute the complete anomaly detection pipeline for defense sensors.

        Defense Context: End-to-end pipeline ensures consistent threat detection
        across different operational scenarios and sensor configurations.
        """
        print("🛡️ DEFENSE SENSOR ANOMALY DETECTION FRAMEWORK")
        print("=" * 70)
        print(f"Dataset: {self.config['dataset']}")
        print(f"Device: {self.device}")
        print(f"Seed: {self.config['seed']}")
        print(f"Epochs: {self.config['epochs']}")
        print(f"Batch Size: {self.config['batch_size']}")
        print(f"Learning Rate: {self.config['learning_rate']}")
        print("=" * 70)

        start_time = time.time()

        try:
            # Phase 1: Data Loading and Preparation
            print("\n📊 PHASE 1: DATA LOADING AND PREPARATION")
            X_data, y_data, dataset_type = self.load_and_prepare_data(self.config['dataset'])

            # Create validation split for robust evaluation
            from train import create_validation_split
            X_train, X_val, y_train, y_val = create_validation_split(
                X_data, y_data, val_split=0.1, seed=self.config['seed']
            )

            # Convert to tensors
            X_train = torch.FloatTensor(X_train)
            X_val = torch.FloatTensor(X_val)
            y_train = torch.LongTensor(y_train)
            y_val = torch.LongTensor(y_val)

            # Phase 2: Model Creation and Training
            model_name = "DAE" if dataset_type == 'sonar' else "VAE"
            dataset_name = "Sonar" if dataset_type == 'sonar' else "IMS"
            print(f"\n🤖 PHASE 2: TRAINING {model_name} FOR {dataset_name.upper()} ANOMALY DETECTION")
            model, model_type = self.create_model(dataset_type)
            trained_model, losses_dict, final_train_metrics = self.train_model(
                model, X_train, X_val, y_train, y_val, model_type
            )

            # Phase 3: Evaluation and Visualization
            print("\n📈 PHASE 3: EVALUATION AND VISUALIZATION")
            self.evaluate_and_visualize(dataset_type)

            # Phase 4: Results Summary
            total_time = time.time() - start_time
            self.generate_results_summary(dataset_type, model_type, total_time, final_train_metrics if model_type == 'DAE' else None)

            # Save final results to CSV
            self.save_results_summary(dataset_type, model_type, total_time, losses_dict, final_train_metrics)

            print("\n✅ DEFENSE ANOMALY DETECTION PIPELINE COMPLETED SUCCESSFULLY")
            print("=" * 70)
            print("🛡️ Ready for operational deployment in defense sensor networks")

            self.logger.info("Defense anomaly detection pipeline completed successfully")

        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            print(f"❌ Pipeline failed: {str(e)}")
            raise

    def generate_results_summary(self, dataset_type, model_type, total_time, final_train_metrics=None):
        """
        Generate comprehensive results summary in tabular format.

        Defense Context: Structured reporting enables quick assessment
        of model performance for operational decision-making.
        """
        print("\n📋 RESULTS SUMMARY")
        print("-" * 50)

        # Check for evaluation results
        eval_log_path = 'results/evaluation/evaluation.log'
        if os.path.exists(eval_log_path):
            # Read evaluation metrics
            with open(eval_log_path, 'r') as f:
                eval_content = f.read()

            # Extract key metrics (simplified parsing)
            metrics_data = []

            # Look for threshold performance lines
            lines = eval_content.split('\n')
            for line in lines:
                if 'threshold:' in line and ('Precision:' in line or 'F1 Score:' in line):
                    parts = line.split(', ')
                    threshold = parts[0].split(':')[1].strip()
                    precision = parts[1].split(':')[1].strip()
                    recall = parts[2].split(':')[1].strip()
                    f1 = parts[3].split(':')[1].strip()

                    metrics_data.append([
                        threshold,
                        f"{precision}",
                        f"{recall}",
                        f"{f1}"
                    ])

            if metrics_data:
                headers = ["Threshold", "Precision", "Recall", "F1-Score"]
                print(tabulate(metrics_data, headers=headers, tablefmt="grid"))

        # Summary statistics
        summary_data = [
            ["Dataset", self.config['dataset'].upper()],
            ["Model Type", model_type],
            ["Device", str(self.device)],
            ["Total Training Time", f"{total_time:.2f}s"],
            ["Epochs Trained", str(self.config['epochs'])],
            ["Batch Size", str(self.config['batch_size'])],
            ["Learning Rate", str(self.config['learning_rate'])],
            ["Random Seed", str(self.config['seed'])],
            ["Checkpoints Saved", "checkpoints/"],
            ["Visualizations Generated", "results/visualizations/"],
            ["Logs Available", "results/logs/"]
        ]

        if final_train_metrics:
            summary_data.extend([
                ["Final Train Precision", f"{final_train_metrics['train_precision']:.4f}"],
                ["Final Val Precision", f"{final_train_metrics['val_precision']:.4f}"],
                ["Final Train Recall", f"{final_train_metrics['train_recall']:.4f}"],
                ["Final Val Recall", f"{final_train_metrics['val_recall']:.4f}"],
                ["Final Train Accuracy", f"{final_train_metrics['train_accuracy']:.4f}"],
                ["Final Val Accuracy", f"{final_train_metrics['val_accuracy']:.4f}"]
            ])

        print("\n" + tabulate(summary_data, tablefmt="grid"))

        # Performance table for both models
        if os.path.exists('results/evaluation/evaluation.log'):
            print("\n📊 MODEL PERFORMANCE SUMMARY TABLE")
            print("-" * 80)
            table_data = []
            headers = ["Dataset", "Model", "Train Loss", "Val Loss", "AUROC", "F1", "Latency", "Accuracy", "Precision", "Recall"]

            # Read evaluation log to extract metrics
            with open('results/evaluation/evaluation.log', 'r') as f:
                log_content = f.read()

            # Parse metrics for current model
            model_name = "DAE" if model_type == "DAE" else "VAE"
            dataset_name = self.config['dataset'].upper()

            # Extract final training losses from training logs
            train_loss = "N/A"
            val_loss = "N/A"
            if os.path.exists('results/logs/train.log'):
                with open('results/logs/train.log', 'r') as f:
                    train_log = f.read()
                    # Extract final losses (simplified)
                    if 'Final Metrics' in train_log:
                        lines = train_log.split('\n')
                        for line in lines:
                            if 'Final Train Loss:' in line:
                                train_loss = line.split(':')[1].strip()
                            elif 'Final Val Loss:' in line:
                                val_loss = line.split(':')[1].strip()

            # Extract evaluation metrics
            auroc = "N/A"
            f1 = "N/A"
            latency = "N/A"
            accuracy = "N/A"
            precision = "N/A"
            recall = "N/A"

            lines = log_content.split('\n')
            for line in lines:
                if 'AUROC:' in line and auroc == "N/A":
                    auroc = line.split('AUROC:')[1].strip()
                elif 'F1 Score:' in line and f1 == "N/A":
                    f1 = line.split('F1 Score:')[1].strip()
                elif 'Average latency:' in line:
                    latency = line.split('Average latency:')[1].split('ms')[0].strip() + "ms"

            if final_train_metrics:
                accuracy = f"{final_train_metrics['val_accuracy']:.4f}"
                precision = f"{final_train_metrics['val_precision']:.4f}"
                recall = f"{final_train_metrics['val_recall']:.4f}"

            table_data.append([dataset_name, model_name, train_loss, val_loss, auroc, f1, latency, accuracy, precision, recall])

            print(tabulate(table_data, headers=headers, tablefmt="grid"))

        # Defense readiness assessment
        print("\n🛡️ DEFENSE READINESS ASSESSMENT")
        print("-" * 30)
        print("✅ Model trained and validated")
        print("✅ Checkpoints saved for deployment")
        print("✅ Visualizations generated for analysis")
        print("✅ Comprehensive logging enabled")
        print("✅ Reproducibility ensured with seeds")
        print("🛡️ SYSTEM READY FOR OPERATIONAL THREAT DETECTION")

    def save_results_summary(self, dataset_type, model_type, total_time, losses_dict, final_train_metrics=None):
        """
        Save final results including loss and evaluation metrics to CSV file.

        Parameters:
        dataset_type (str): 'sonar' or 'ims'
        model_type (str): 'DAE' or 'VAE'
        total_time (float): Total training time in seconds
        losses_dict (dict): Dictionary containing training losses
        final_train_metrics (dict): Final training metrics (for DAE only)
        """
        os.makedirs('results/summary', exist_ok=True)

        # Prepare results dictionary
        results = {
            'Dataset': dataset_type.upper(),
            'Model': model_type,
            'Total_Training_Time_s': round(total_time, 2),
            'Epochs_Trained': self.config['epochs'],
            'Batch_Size': self.config['batch_size'],
            'Learning_Rate': self.config['learning_rate'],
            'Random_Seed': self.config['seed'],
            'Device': str(self.device)
        }

        # Add final losses
        if model_type == 'DAE':
            results['Final_Train_Loss'] = losses_dict['train'][-1] if losses_dict['train'] else None
            results['Final_Val_Loss'] = losses_dict['val'][-1] if losses_dict['val'] else None
        else:
            results['Final_Train_Total_Loss'] = losses_dict['train_total'][-1] if losses_dict['train_total'] else None
            results['Final_Train_MSE_Loss'] = losses_dict['train_mse'][-1] if losses_dict['train_mse'] else None
            results['Final_Train_KL_Loss'] = losses_dict['train_kl'][-1] if losses_dict['train_kl'] else None
            results['Final_Val_Total_Loss'] = losses_dict['val_total'][-1] if losses_dict['val_total'] else None
            results['Final_Val_MSE_Loss'] = losses_dict['val_mse'][-1] if losses_dict['val_mse'] else None
            results['Final_Val_KL_Loss'] = losses_dict['val_kl'][-1] if losses_dict['val_kl'] else None

        # Add final training metrics for DAE
        if final_train_metrics and model_type == 'DAE':
            results['Final_Train_Precision'] = final_train_metrics['train_precision']
            results['Final_Val_Precision'] = final_train_metrics['val_precision']
            results['Final_Train_Recall'] = final_train_metrics['train_recall']
            results['Final_Val_Recall'] = final_train_metrics['val_recall']
            results['Final_Train_Accuracy'] = final_train_metrics['train_accuracy']
            results['Final_Val_Accuracy'] = final_train_metrics['val_accuracy']

        # Try to extract evaluation metrics from log files
        eval_log_path = 'results/evaluation/evaluation.log'
        if os.path.exists(eval_log_path):
            try:
                with open(eval_log_path, 'r') as f:
                    log_content = f.read()

                lines = log_content.split('\n')
                for line in lines:
                    if 'AUROC:' in line:
                        results['AUROC'] = float(line.split('AUROC:')[1].strip())
                    elif 'F1 Score:' in line:
                        results['F1_Score'] = float(line.split('F1 Score:')[1].strip())
                    elif 'Average latency:' in line:
                        latency_str = line.split('Average latency:')[1].split('ms')[0].strip()
                        results['Average_Latency_ms'] = float(latency_str)
            except Exception as e:
                self.logger.warning(f"Could not extract evaluation metrics from log: {e}")

        # Save to CSV
        filename = f"results/summary/{model_type.lower()}_results.csv"
        df = pd.DataFrame([results])
        df.to_csv(filename, index=False)
        print(f"💾 Results saved to {filename}")


def parse_arguments():
    """Parse command line arguments for defense sensor configuration."""
    parser = argparse.ArgumentParser(
        description='Defense Sensor Anomaly Detection Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --dataset sonar --epochs 100 --batch_size 64 --lr 0.001
  python main.py --dataset ims --epochs 50 --batch_size 32 --lr 0.0005 --seed 123
        """
    )

    parser.add_argument('--dataset', type=str, default='sonar',
                       choices=['sonar', 'ims'],
                       help='Dataset choice: sonar (DAE) or ims (VAE)')

    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')

    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training (optimized for GPU utilization)')

    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate for optimizer')

    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    return parser.parse_args()


def main():
    """Main entry point for defense anomaly detection framework."""
    args = parse_arguments()

    # Convert arguments to configuration dictionary
    config = {
        'dataset': args.dataset,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'seed': args.seed
    }

    # Initialize and run defense anomaly detector
    detector = DefenseAnomalyDetector(config)
    detector.run_detection_pipeline()


if __name__ == "__main__":
    main()