# API Reference for Unified Autoencoder Framework

## Core Classes

### DefenseAnomalyDetector

Master controller class for the anomaly detection pipeline.

#### Initialization
```python
detector = DefenseAnomalyDetector(config)
```

**Parameters:**
- `config` (dict): Configuration dictionary with keys:
  - `'dataset'`: `'sonar'` or `'ims'`
  - `'epochs'`: Number of training epochs (int)
  - `'batch_size'`: Training batch size (int)
  - `'learning_rate'`: Learning rate (float)
  - `'seed'`: Random seed (int)
  - `'grid_search'`: Enable hyperparameter optimization (bool)

#### Methods

##### `run_detection_pipeline()`
Executes the complete anomaly detection workflow.

**Returns:** None

**Side Effects:**
- Creates results directories
- Saves model checkpoints
- Generates visualizations
- Exports results to CSV

## Module Functions

### data_loader.py

#### `load_sonar_data(data_path='Datasets/UCI/sonar.all-data')`
Load and preprocess UCI Sonar dataset.

**Parameters:**
- `data_path` (str): Path to sonar dataset file

**Returns:**
- `X_normalized` (np.ndarray): Normalized feature matrix (208, 60)
- `y_binary` (np.ndarray): Binary labels (0=normal, 1=anomaly)

#### `load_ims_data(data_path='Datasets/IMS', window_size=1024)`
Load and preprocess NASA IMS bearing dataset.

**Parameters:**
- `data_path` (str): Path to IMS dataset directory
- `window_size` (int): Size of sliding windows

**Returns:**
- `X_windows` (np.ndarray): Windowed vibration signals
- `y_labels` (np.ndarray): Anomaly labels (0=normal, 1=anomaly)

### models.py

#### `create_dae_sonar(input_dim=60)`
Factory function for sonar DAE model.

**Parameters:**
- `input_dim` (int): Input feature dimension

**Returns:**
- `DenoisingAutoencoder`: Initialized DAE model

#### `create_vae_ims(window_size=1024, n_channels=8, latent_dim=128)`
Factory function for IMS VAE model.

**Parameters:**
- `window_size` (int): Time series window size
- `n_channels` (int): Number of vibration channels
- `latent_dim` (int): Latent space dimension

**Returns:**
- `VariationalAutoencoder`: Initialized VAE model

### train.py

#### `train_dae(model, train_loader, val_loader, device, epochs=100, lr=1e-3, patience=10)`
Train DAE model with comprehensive metrics tracking.

**Parameters:**
- `model`: DAE model instance
- `train_loader`: Training data loader
- `val_loader`: Validation data loader
- `device`: PyTorch device
- `epochs` (int): Maximum training epochs
- `lr` (float): Learning rate
- `patience` (int): Early stopping patience

**Returns:**
- `trained_model`: Trained model
- `train_losses` (list): Training loss history
- `val_losses` (list): Validation loss history
- `train_aucs` (list): Training AUROC history
- `val_aucs` (list): Validation AUROC history
- `train_f1s` (list): Training F1 history
- `val_f1s` (list): Validation F1 history
- `train_precisions` (list): Training precision history
- `val_precisions` (list): Validation precision history
- `train_recalls` (list): Training recall history
- `val_recalls` (list): Validation recall history
- `train_accuracies` (list): Training accuracy history
- `val_accuracies` (list): Validation accuracy history

#### `train_vae(model, train_loader, val_loader, device, epochs=100, lr=1e-3, patience=10)`
Train VAE model with KL divergence regularization.

**Parameters:** Similar to `train_dae()`

**Returns:** Similar to `train_dae()` with additional VAE-specific losses

### evaluate.py

#### `evaluate_dae_model(checkpoint_path='checkpoints/dae_sonar_best.pth', device='cpu')`
Complete evaluation pipeline for DAE model.

**Parameters:**
- `checkpoint_path` (str): Path to trained model checkpoint
- `device` (str): Computation device

**Returns:** None (saves results to files)

#### `evaluate_vae_model(checkpoint_path='checkpoints/vae_ims_best.pth', device='cpu')`
Complete evaluation pipeline for VAE model.

**Parameters:** Similar to `evaluate_dae_model()`

**Returns:** None

#### `determine_anomaly_thresholds(reconstruction_errors, labels, strategies=['percentile', 'statistical', 'optimized'])`
Determine optimal anomaly detection thresholds.

**Parameters:**
- `reconstruction_errors` (np.ndarray): Per-sample reconstruction errors
- `labels` (np.ndarray): True anomaly labels
- `strategies` (list): Threshold determination strategies

**Returns:**
- `thresholds` (dict): Dictionary of threshold values

#### `evaluate_anomaly_detection(reconstruction_errors, labels, thresholds, latencies=None)`
Evaluate anomaly detection performance across thresholds.

**Parameters:**
- `reconstruction_errors` (np.ndarray): Reconstruction errors
- `labels` (np.ndarray): True labels
- `thresholds` (dict): Threshold dictionary
- `latencies` (np.ndarray, optional): Inference latencies

**Returns:**
- `results` (dict): Evaluation results per threshold

### visualize.py

#### Visualization Functions

All visualization functions save plots to the `results/visualizations/` directory.

##### `visualize_sonar_anomalies(model, data_loader, device)`
Generate comprehensive sonar anomaly visualizations.

##### `visualize_ims_anomalies(model, data_loader, device)`
Generate comprehensive IMS bearing anomaly visualizations.

## Configuration Parameters

### Training Configuration

| Parameter | DAE Default | VAE Default | Description |
|-----------|-------------|-------------|-------------|
| `learning_rate` | 1e-3 | 1e-3 | Optimizer learning rate |
| `batch_size` | 64 | 32 | Training batch size |
| `epochs` | 100 | 100 | Maximum training epochs |
| `patience` | 50 | 50 | Early stopping patience |
| `seed` | 42 | 42 | Random seed |

### Model Architecture Parameters

#### DAE (DenoisingAutoencoder)
| Parameter | Value | Description |
|-----------|-------|-------------|
| `input_dim` | 60 | Sonar feature dimension |
| `hidden_dims` | [256, 128, 64] | Encoder hidden dimensions |
| `anomaly_weight` | 3.0 | False negative penalization |

#### VAE (VariationalAutoencoder)
| Parameter | Value | Description |
|-----------|-------|-------------|
| `window_size` | 1024 | Time series window size |
| `n_channels` | 8 | Vibration signal channels |
| `latent_dim` | 128 | Latent space dimension |
| `conv_filters` | [128, 256, 512] | Convolutional filter sizes |
| `anomaly_weight` | 5.0 | False negative penalization |

## Output Formats

### CSV Results Format

Results are saved to `results/summary/{model}_results.csv` with columns:

```csv
Dataset,Model,Total_Training_Time_s,Epochs_Trained,Batch_Size,Learning_Rate,Random_Seed,Device,Final_Train_Loss,Final_Val_Loss,Precision,Recall,F1_Score,Accuracy,AUROC,Best_Threshold,Average_Latency_ms
```

### Log File Format

Logs are saved to `results/logs/` with format:
```
%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

### Checkpoint Format

PyTorch checkpoints contain:
```python
{
    'epoch': int,
    'model_state_dict': dict,
    'optimizer_state_dict': dict,
    'train_loss': float,
    'val_loss': float,
    'lr': float
}
```

## Error Handling

### Custom Exceptions

- `ValueError`: Invalid dataset choice, model type, or configuration
- `FileNotFoundError`: Missing dataset files or checkpoints
- `RuntimeError`: CUDA/GPU memory issues

### Recovery Mechanisms

- **Early Stopping**: Automatic convergence detection
- **Checkpoint Saving**: Regular model state preservation
- **Fallback Parameters**: Default values when grid search fails

## Performance Considerations

### Memory Usage

- **DAE**: ~500MB GPU memory for batch_size=64
- **VAE**: ~1.2GB GPU memory for batch_size=32
- **Data Loading**: Additional 2-4GB RAM for large datasets

### Inference Latency

- **DAE**: <1ms per sample on GPU
- **VAE**: <2ms per sample on GPU
- **Batch Processing**: Sub-linear scaling with batch size

### Scalability

- **Multi-GPU**: Automatic DataParallel support
- **Data Parallelism**: Parallel file processing for IMS data
- **Memory Optimization**: Gradient checkpointing available

## Integration Examples

### Real-time Anomaly Detection

```python
from models import create_dae_sonar
from evaluate import load_trained_model

class RealTimeDetector:
    def __init__(self, checkpoint_path, threshold=0.95):
        self.model = load_trained_model(checkpoint_path, 'dae')
        self.model.eval()
        self.threshold = threshold

    def detect(self, sensor_data):
        with torch.no_grad():
            reconstructed, _ = self.model(sensor_data)
            errors = torch.mean((reconstructed - sensor_data)**2, dim=1)
            return (errors > self.threshold).cpu().numpy()
```

### Batch Evaluation

```python
from evaluate import evaluate_dae_model

# Evaluate trained model
evaluate_dae_model(
    checkpoint_path='checkpoints/dae_sonar_best.pth',
    device='cuda:0'
)

# Results saved to results/evaluation/ and results/summary/
```

This API reference provides comprehensive documentation for programmatic use of the framework.