# Usage Guide for Unified Autoencoder Framework

## Quick Start Commands

### Basic Training

```bash
# Train DAE on sonar data (default parameters)
python main.py --dataset sonar

# Train VAE on IMS bearing data
python main.py --dataset ims

# Custom hyperparameters
python main.py --dataset sonar --epochs 200 --batch_size 32 --lr 0.0005
```

### Advanced Training with Grid Search

```bash
# Enable hyperparameter optimization
python main.py --dataset sonar --grid_search --epochs 300 --batch_size 16 --lr 0.001

# IMS with grid search
python main.py --dataset ims --grid_search --epochs 400 --batch_size 8 --lr 0.0001
```

## Command Line Arguments

### Core Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset` | str | `'sonar'` | Dataset choice: `'sonar'` or `'ims'` |
| `--epochs` | int | `100` | Number of training epochs |
| `--batch_size` | int | `64` | Training batch size |
| `--lr` | float | `1e-3` | Learning rate |
| `--seed` | int | `42` | Random seed for reproducibility |
| `--grid_search` | flag | `False` | Enable hyperparameter grid search |

### Grid Search Details

When `--grid_search` is enabled, the framework automatically optimizes:

#### Learning Rate Exploration
- Range: `[1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 2e-4, 1e-3]`
- Purpose: Find optimal convergence rate for different architectures

#### Batch Size Optimization
- Range: `[4, 8, 12, 16]`
- Purpose: Balance GPU memory utilization and training stability

#### Epoch Range
- Range: `[300, 400, 500]`
- Purpose: Ensure sufficient convergence without overfitting

### Scoring Function
```python
combined_score = 0.7 * reconstruction_quality + 0.3 * anomaly_detection_f1
```

## Data Requirements

### UCI Sonar Dataset

**Location**: `Datasets/UCI/sonar.all-data`

**Format**:
- CSV format with tab/space separation
- 208 samples × 60 features
- Last column: labels ('R' = normal rocks, 'M' = anomalous mines)

**Preprocessing Applied**:
- MinMax scaling to [0,1]
- Binary label conversion (0 = normal, 1 = anomaly)

### NASA IMS Bearing Dataset

**Location**: `Datasets/IMS/`

**Format**:
- ASCII files in timestamped subdirectories
- 8-channel vibration signals (4 bearings × 2 channels each)
- 20,480 samples per file at 20kHz

**Preprocessing Applied**:
- 1024-sample windows with 512-sample overlap (50%)
- Per-channel MinMax normalization
- Chronological anomaly labeling (70% normal, 30% anomalous)

## Module-by-Module Guide

### main.py - Master Controller

**Purpose**: Orchestrates the complete anomaly detection pipeline

**Key Functions**:
- `DefenseAnomalyDetector.run_detection_pipeline()`: Main execution flow
- Data loading → Model creation → Training → Evaluation → Visualization

**Configuration Flow**:
```python
config = {
    'dataset': args.dataset,
    'epochs': args.epochs,
    'batch_size': args.batch_size,
    'learning_rate': args.lr,
    'seed': args.seed,
    'grid_search': args.grid_search
}
```

### models.py - Autoencoder Architectures

#### DenoisingAutoencoder (DAE)
**Input**: 60-dimensional sonar features
**Architecture**:
- Encoder: [60 → 256 → 128 → 64 → 32 → 16 → 8]
- Decoder: Symmetric expansion
- Loss: MSE + false negative penalization (weight = 3.0)

#### VariationalAutoencoder (VAE)
**Input**: 1024×8 vibration windows
**Architecture**:
- Encoder: 4 Conv1D blocks [128 → 256 → 512 → 1024]
- Latent: 256-dimensional (μ, logvar)
- Decoder: Symmetric ConvTranspose1D
- Loss: MSE + KL divergence + false negative penalization (weight = 5.0)

### train.py - Training Framework

**Key Functions**:
- `train_dae()`: DAE training with early stopping
- `train_vae()`: VAE training with KL annealing
- `plot_training_curves()`: Individual metric visualizations

**Training Features**:
- Multi-GPU support with DataParallel
- Learning rate scheduling (ReduceLROnPlateau)
- Early stopping (patience = 50)
- Comprehensive metrics tracking

### evaluate.py - Evaluation Pipeline

**Key Functions**:
- `determine_anomaly_thresholds()`: Multiple threshold strategies
- `evaluate_anomaly_detection()`: Comprehensive metrics calculation

**Threshold Strategies**:
1. **Percentile-based**: Conservative approach (95th, 99th percentiles)
2. **Statistical**: Mean + k×std (k = 2, 3, 4)
3. **Optimized**: F1-maximization across percentile range

**Metrics Computed**:
- Precision, Recall, F1-Score, Accuracy
- AUROC, Average Precision
- Confusion Matrix, Latency Analysis

### visualize.py - Analytics Framework

**Visualization Types**:
- Reconstruction error distributions (normal vs anomalous)
- ROC and Precision-Recall curves
- Training loss evolution (individual metrics)
- Latent space t-SNE projections (VAE only)
- Frequency-domain error analysis (sonar)
- Time-domain error analysis (IMS)

## Expected Outputs

### Directory Structure After Execution

```
results/
├── checkpoints/
│   ├── dae_sonar_best.pth
│   └── vae_ims_best.pth
├── logs/
│   ├── defense_anomaly_detector_[timestamp].log
│   ├── train.log
│   └── evaluation.log
├── visualizations/
│   ├── sonar/
│   │   ├── dae_reconstruction_epoch_*.png
│   │   ├── error_distributions.png
│   │   └── roc_curves.png
│   └── ims/
│       ├── vae_latent_tsne_epoch_*.png
│       └── time_domain_errors.png
├── evaluation/
│   ├── dae_evaluation_visualizations.png
│   └── vae_confusion_matrix_*.png
└── summary/
    ├── dae_results.csv
    └── vae_results.csv
```

### Results Interpretation

#### Performance Metrics

**F1-Score**: Primary metric balancing precision and recall
- **> 0.9**: Excellent anomaly detection
- **0.8-0.9**: Good performance
- **0.7-0.8**: Acceptable for defense applications
- **< 0.7**: Requires parameter tuning

**Precision vs Recall Trade-off**:
- **High Precision**: Few false positives (conservative detection)
- **High Recall**: Few false negatives (critical for defense)

#### Threshold Selection

**Best Practices**:
1. Use **optimized_recall_weighted** for defense-critical applications
2. Validate threshold stability across different data splits
3. Monitor operational performance in deployment

## Advanced Usage

### Custom Hyperparameter Ranges

Modify `perform_grid_search()` in `main.py`:

```python
param_grid = {
    'learning_rate': [1e-5, 1e-4, 1e-3],  # Custom range
    'batch_size': [16, 32, 64],           # Larger batches
    'epochs': [200, 300, 400]             # Shorter training
}
```

### Multi-GPU Training

The framework automatically detects and utilizes multiple GPUs:

```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    print(f"🚀 Using {torch.cuda.device_count()} GPUs")
```

### Custom Loss Functions

Modify anomaly weighting in `models.py`:

```python
# Increase false negative penalization
self.anomaly_weight = 5.0  # Default: 3.0 for DAE, 5.0 for VAE
```

### Memory Optimization

For large datasets or limited GPU memory:

```bash
# Reduce batch size
python main.py --batch_size 8

# Use CPU training
CUDA_VISIBLE_DEVICES="" python main.py

# Enable memory optimization in train.py
torch.cuda.empty_cache()
```

## Monitoring and Debugging

### Training Monitoring

**Real-time Metrics**:
- Loss curves (training/validation)
- F1-score evolution
- Learning rate scheduling
- Early stopping triggers

**Log Files**:
- `results/logs/train.log`: Training progress
- `results/logs/defense_anomaly_detector_[timestamp].log`: Pipeline execution
- `results/logs/evaluation.log`: Performance metrics

### Performance Troubleshooting

**Slow Training**:
- Check GPU utilization: `nvidia-smi`
- Reduce batch size or disable visualizations
- Enable multi-worker data loading

**Poor Performance**:
- Enable grid search: `--grid_search`
- Adjust anomaly weights in models
- Verify data preprocessing

**Memory Issues**:
- Reduce model complexity (latent dimensions)
- Use CPU training for testing
- Implement gradient checkpointing

## Integration Examples

### Real-time Inference

```python
from models import create_dae_sonar
from evaluate import load_trained_model

# Load trained model
model = load_trained_model('checkpoints/dae_sonar_best.pth', 'dae')

# Real-time prediction
def predict_anomaly(sensor_data, threshold=0.95):
    model.eval()
    with torch.no_grad():
        reconstructed, _ = model(sensor_data)
        error = torch.mean((reconstructed - sensor_data)**2, dim=1)
        return (error > threshold).float()
```

### Batch Processing

```python
# Process multiple sensor readings
batch_data = torch.FloatTensor(sensor_readings)
predictions = predict_anomaly(batch_data)
anomalous_indices = torch.where(predictions == 1)[0]
```

This comprehensive guide covers all aspects of using the Unified Autoencoder Framework for cross-domain anomaly detection in defense sensor networks.