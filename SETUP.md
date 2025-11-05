# Setup Guide for Unified Autoencoder Framework

## Prerequisites

- **Python 3.8+**: Required for PyTorch 2.0+ compatibility
- **CUDA 11.8+** (optional): For GPU acceleration
- **Git**: For cloning the repository
- **~2GB disk space**: For datasets and model checkpoints

## System Requirements

### Minimum Requirements
- **RAM**: 8GB
- **Storage**: 10GB free space
- **OS**: Windows 10/11, Ubuntu 18.04+, macOS 10.15+
- **GPU**: NVIDIA GPU with CUDA support (recommended)

### Recommended Requirements
- **RAM**: 16GB+
- **Storage**: 50GB+ SSD
- **GPU**: NVIDIA RTX 30-series or higher with 8GB+ VRAM
- **CPU**: Multi-core processor for parallel data processing

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/defense-sensor-anomaly-detection.git
cd defense-sensor-anomaly-detection
```

### 2. Create Virtual Environment

#### Using conda (recommended):
```bash
conda create -n defense_anomaly python=3.9
conda activate defense_anomaly
```

#### Using venv:
```bash
python -m venv defense_anomaly_env
# On Windows:
defense_anomaly_env\Scripts\activate
# On Linux/macOS:
source defense_anomaly_env/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### Alternative: Install with conda
```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia
conda install numpy pandas matplotlib seaborn scikit-learn tqdm tabulate scipy -c conda-forge
```

### 4. Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch version: 2.0.1+cu118
CUDA available: True
```

## Dataset Setup

### UCI Sonar Dataset

1. **Download** from UCI Machine Learning Repository:
   ```bash
   # Download link: https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data
   wget https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data
   ```

2. **Create directory structure**:
   ```bash
   mkdir -p Datasets/UCI
   mv sonar.all-data Datasets/UCI/
   ```

3. **Verify dataset**:
   ```bash
   wc -l Datasets/UCI/sonar.all-data  # Should show 208 lines
   head -1 Datasets/UCI/sonar.all-data  # Should show 60 features + label
   ```

### NASA IMS Bearing Dataset

1. **Download** from NASA Prognostics Repository:
   - Visit: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
   - Download: IMS Bearing Dataset
   - Extract to: `Datasets/IMS/`

2. **Expected structure**:
   ```
   Datasets/IMS/
   ├── 1st_test/
   │   ├── 2003.10.22.12.06.24
   │   ├── 2003.10.22.12.09.13
   │   └── ...
   ├── 2nd_test/
   └── 3rd_test/
   ```

3. **Verify data loading**:
   ```python
   from data_loader import load_ims_data
   X_ims, y_ims = load_ims_data()
   print(f"IMS data shape: {X_ims.shape if X_ims.size > 0 else 'Not found'}")
   ```

## Configuration

### Environment Variables

```bash
# Set Python path (optional)
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Configure PyTorch
export TORCH_USE_CUDA_DSA=1  # Enable CUDA device-side assertions
```

### GPU Configuration

For multi-GPU systems:
```python
# In your training script
import torch
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
```

## Testing the Installation

### Basic Functionality Test

```bash
# Test data loading
python -c "from data_loader import load_sonar_data; X, y = load_sonar_data(); print(f'Sonar: {X.shape[0]} samples')"

# Test model creation
python -c "from models import create_dae_sonar; model = create_dae_sonar(); print('DAE created successfully')"
```

### Full Pipeline Test

```bash
# Quick test run (reduced parameters for testing)
python main.py --dataset sonar --epochs 5 --batch_size 32 --lr 0.01
```

Expected output includes:
- Model initialization messages
- Training progress bars
- Evaluation metrics
- Results saved to `results/` directory

## Troubleshooting

### Common Issues

#### 1. CUDA Not Available
```
Error: CUDA is not available
```
**Solution**:
- Install CUDA-compatible PyTorch: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
- Verify GPU drivers: `nvidia-smi`

#### 2. Dataset Not Found
```
FileNotFoundError: Datasets/UCI/sonar.all-data
```
**Solution**:
- Check file paths in `data_loader.py`
- Ensure datasets are in correct directories
- Run: `find . -name "*sonar*" -o -name "*IMS*"`

#### 3. Memory Issues
```
CUDA out of memory
```
**Solutions**:
- Reduce batch size: `--batch_size 8`
- Use CPU: `CUDA_VISIBLE_DEVICES="" python main.py`
- Enable gradient checkpointing in models

#### 4. Import Errors
```
ModuleNotFoundError: No module named 'torch'
```
**Solution**:
- Activate virtual environment
- Reinstall requirements: `pip install -r requirements.txt --force-reinstall`

### Performance Optimization

#### GPU Memory Management
```python
# Add to training scripts
torch.cuda.empty_cache()
torch.cuda.synchronize()
```

#### Data Loading Optimization
- Use pinned memory: `pin_memory=True`
- Enable parallel workers: `num_workers=4`
- Use prefetch: `prefetch_factor=2`

## Verification Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment activated
- [ ] All packages installed (`pip list`)
- [ ] CUDA available (if GPU present)
- [ ] Datasets downloaded and accessible
- [ ] Basic import test passes
- [ ] Full pipeline test completes
- [ ] Results directory created
- [ ] Logs generated successfully

## Next Steps

Once setup is complete:
1. **Explore the code**: Review `main.py` for pipeline understanding
2. **Run experiments**: Use different hyperparameters and datasets
3. **Analyze results**: Check `results/` directory for outputs
4. **Customize models**: Modify architectures in `models.py`

For detailed usage instructions, see `README.md`.