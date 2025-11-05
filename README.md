# Unified Autoencoder Framework for Cross-Domain Anomaly Detection in Defense Sensor Networks

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

A comprehensive framework implementing cross-domain anomaly detection in defense sensor networks using specialized autoencoder architectures. This research framework addresses critical challenges in naval underwater threat detection and aerospace bearing failure prediction through unified machine learning approaches.

## 🎯 Overview

This project implements a unified anomaly detection framework that leverages:
- **Denoising Autoencoders (DAE)** for sonar acoustic signal analysis (underwater mine detection)
- **Variational Autoencoders (VAE)** for bearing vibration monitoring (aerospace failure prediction)

The framework provides end-to-end capabilities for data preprocessing, model training, hyperparameter optimization, evaluation, and visualization, specifically designed for defense sensor network applications.

## 🏗️ Architecture

### Core Components

- **main.py**: Master controller orchestrating the complete anomaly detection pipeline
- **models.py**: Autoencoder architectures (DAE for sonar, VAE for IMS bearing data)
- **train.py**: Training pipelines with grid search hyperparameter optimization
- **evaluate.py**: Comprehensive evaluation metrics and threshold selection
- **visualize.py**: Research-quality visualization and analytics
- **data_loader.py**: Cross-domain data preprocessing and loading utilities

### Defense Applications

- **Naval Operations**: Real-time underwater mine detection using acoustic signatures
- **Aerospace Systems**: Predictive maintenance for critical aircraft components
- **Multi-Domain Integration**: Cross-sensor anomaly correlation for enhanced situational awareness

## 🚀 Quick Start

### Prerequisites

```bash
# Required Python packages
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib seaborn scikit-learn
pip install tqdm tabulate
```

### Installation

```bash
# Clone the repository
# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Train DAE for sonar anomaly detection
python main.py --dataset sonar --epochs 100 --batch_size 64 --lr 0.001

# Train VAE for IMS bearing anomaly detection
python main.py --dataset ims --epochs 50 --batch_size 32 --lr 0.0005

# Enable hyperparameter grid search
python main.py --dataset sonar --grid_search
```

## 📊 Datasets

### UCI Sonar Dataset
- **Source**: UCI Machine Learning Repository
- **Features**: 60-dimensional sonar signals
- **Classes**: Rocks (normal), Mines (anomalous)
- **Purpose**: Underwater acoustic threat detection

### NASA IMS Bearing Dataset
- **Source**: NASA Ames Prognostics Data Repository
- **Features**: Multi-channel vibration time-series (8 channels)
- **Structure**: 1024-sample windows with 50% overlap
- **Purpose**: Bearing degradation monitoring in aerospace systems

## 🔧 Configuration

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `sonar` | Dataset choice: `sonar` or `ims` |
| `--epochs` | `100` | Number of training epochs |
| `--batch_size` | `64` | Batch size for training |
| `--lr` | `1e-3` | Learning rate |
| `--seed` | `42` | Random seed for reproducibility |
| `--grid_search` | `False` | Enable hyperparameter grid search |

### Hyperparameter Optimization

When `--grid_search` is enabled, the framework performs comprehensive optimization:

- **Learning Rate**: [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 2e-4, 1e-3]
- **Batch Size**: [4, 8, 12, 16]
- **Epochs**: [300, 400, 500]
- **Scoring**: Combined reconstruction quality + anomaly detection F1

## 🎯 Key Features

### Advanced Training Pipeline
- **Multi-GPU Support**: DataParallel training acceleration
- **Early Stopping**: Validation-based convergence monitoring
- **Learning Rate Scheduling**: Adaptive optimization
- **Comprehensive Logging**: Real-time performance tracking

### Robust Evaluation Framework
- **Multiple Threshold Strategies**: Percentile, statistical, and optimized approaches
- **Comprehensive Metrics**: F1, AUROC, precision, recall, accuracy
- **Latency Monitoring**: Real-time inference performance analysis

### Research-Quality Visualization
- **Training Curves**: Loss, F1, AUROC evolution
- **Error Distributions**: Normal vs anomalous reconstruction analysis
- **ROC/PR Curves**: Probabilistic performance assessment
- **Latent Space Analysis**: t-SNE projections (VAE)

## 📈 Results and Outputs

### Directory Structure
```
results/
├── checkpoints/          # Trained model checkpoints
├── logs/                # Comprehensive training logs
├── visualizations/      # Research-quality plots
│   ├── sonar/          # Acoustic anomaly visualizations
│   └── ims/            # Vibration anomaly visualizations
├── evaluation/         # Performance metrics and analyses
└── summary/            # CSV exports and final results
```

### Performance Metrics

The framework provides detailed evaluation across multiple dimensions:

- **Classification Performance**: Precision, recall, F1-score, accuracy
- **Probabilistic Metrics**: AUROC, average precision
- **Operational Metrics**: Inference latency, memory usage
- **Threshold Analysis**: Multiple strategies with performance comparison

## 🔍 Module Details

### main.py - Master Controller
Executes the complete anomaly detection pipeline:
- Data loading and preprocessing
- Model training with optional grid search
- Evaluation and visualization
- Results summary and export

### models.py - Autoencoder Architectures
- **DenoisingAutoencoder**: Optimized for sonar signals with asymmetric loss
- **VariationalAutoencoder**: Convolutional VAE for vibration time-series
- Specialized loss functions with false negative penalization

### train.py - Training Framework
- Multi-GPU accelerated training
- Grid search hyperparameter optimization
- Early stopping and learning rate scheduling
- Comprehensive metrics tracking

### evaluate.py - Evaluation Pipeline
- Reconstruction error computation
- Multiple threshold determination strategies
- Performance metrics calculation
- Automated reporting and CSV export

### visualize.py - Analytics Framework
- Error distribution analysis
- Latent space visualization
- Denoising performance comparison
- Research-quality plotting utilities

## 🛡️ Defense Applications

### Naval Underwater Surveillance
- **Threat Detection**: Real-time mine identification in sonar data
- **Signal Processing**: Acoustic pattern recognition in noisy environments
- **Operational Integration**: Seamless integration with existing sonar systems

### Aerospace Structural Health
- **Predictive Maintenance**: Bearing failure prediction before catastrophic events
- **Vibration Analysis**: Time-series degradation monitoring
- **Mission Critical**: Early warning systems for aircraft safety

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size: `--batch_size 8`
   - Enable gradient checkpointing in models.py

2. **Dataset Not Found**
   - Ensure UCI sonar data is in `Datasets/UCI/sonar.all-data`
   - Verify IMS data structure in `Datasets/IMS/`

3. **Training Instability**
   - Adjust learning rate: `--lr 1e-4`
   - Enable grid search for optimal parameters

### Performance Optimization

- Use CUDA-compatible GPU for accelerated training
- Adjust batch size based on available memory
- Enable multi-GPU training with `torch.nn.DataParallel`

## 📚 Research Context

This framework addresses critical research challenges in defense sensor networks:

- **Cross-Domain Integration**: Unified approach for heterogeneous sensor modalities
- **Real-Time Performance**: Optimized inference for operational deployment
- **Robust Anomaly Detection**: False negative penalization for defense-critical applications
- **Scalable Architecture**: Modular design for integration with existing systems

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests
4. Update documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

For questions or collaboration opportunities:
- **Email**: defense-research@organization.org
- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: See `Research_Paper/` directory for detailed methodology

## 🔬 Citation

If you use this framework in your research, please cite:

```bibtex
@article{defense_sensor_anomaly_2024,
  title={Unified Autoencoder Framework for Cross-Domain Anomaly Detection in Defense Sensor Networks},
  author={Defense Research Team},
  journal={Defense Technology Journal},
  year={2024},
  publisher={Defense Research Organization}
}
```

---

**⚠️ Defense Classification Notice**: This software is developed for research purposes in defense sensor anomaly detection. Users are responsible for compliance with applicable export control regulations and usage policies.