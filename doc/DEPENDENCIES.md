# Project Dependencies Documentation

## Python Environment

- **Python Version**: 3.11.13 (required: 3.11+)
- **Virtual Environment**: Located at `.venv/`

## Installation Instructions

### Quick Setup
```bash
# Load Python module (on HPC systems)
module load python3

# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

## Core Dependencies

### Deep Learning Framework
- **torch** (2.5.1+cu124) - PyTorch with CUDA 12.4 support
- **torchvision** (0.20.1+cu124) - Computer vision library for PyTorch
- **torchaudio** (2.5.1+cu124) - Audio processing for PyTorch
- **triton** (3.1.0) - GPU kernel compiler for PyTorch

### Data Science & ML
- **numpy** (2.2.6) - Numerical computing
- **pandas** (2.3.1) - Data manipulation and analysis
- **scikit-learn** (1.6.2) - Machine learning algorithms
- **scipy** (1.15.2) - Scientific computing

### Computer Vision
- **opencv-python** (4.12.0.88) - Computer vision library
- **Pillow** (11.1.0) - Image processing

### Visualization
- **matplotlib** (3.10.3) - Plotting library
- **seaborn** (0.13.3) - Statistical data visualization

### Utilities
- **tqdm** (4.67.3) - Progress bars
- **joblib** (1.5.1) - Parallel computing
- **filelock** (3.18.0) - File locking mechanism

## CUDA Dependencies (GPU Support)

The project includes NVIDIA CUDA dependencies for GPU acceleration:
- CUDA Runtime 12.6.77
- cuDNN 9.5.1.17
- CUBLAS, cuFFT, cuRAND, cuSOLVER, cuSPARSE libraries
- NCCL 2.26.2 for multi-GPU support

**Note**: These will be installed automatically with PyTorch, but the system must have:
- NVIDIA GPU with compute capability 3.5+
- NVIDIA driver version 525.60.13 or newer
- CUDA 12.4 compatible system

## System Requirements

### Minimum Hardware
- **RAM**: 16GB (32GB recommended for training)
- **GPU**: NVIDIA GPU with 8GB VRAM (for training)
- **Storage**: 10GB free space for models and data
- **CPU**: 4+ cores recommended

### HPC Environment (Agate/MSI)
```bash
# Load required modules
module load python3

# For GPU jobs
#SBATCH -p a100-4,a100-8
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=180g
```

## Complete Package List

All 47 packages are listed in `requirements.txt`. Key packages include:

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.5.1+cu124 | Deep learning framework |
| numpy | 2.2.6 | Numerical arrays |
| pandas | 2.3.1 | Data manipulation |
| opencv-python | 4.12.0.88 | Computer vision |
| matplotlib | 3.10.3 | Plotting |
| scikit-learn | 1.6.2 | ML algorithms |
| Pillow | 11.1.0 | Image processing |
| tqdm | 4.67.3 | Progress bars |

## Troubleshooting

### Issue: CUDA out of memory
```bash
# Reduce batch size in config file or use CPU
export CUDA_VISIBLE_DEVICES=""  # Force CPU usage
```

### Issue: Module not found errors
```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Verify with
which python  # Should show .venv/bin/python
```

### Issue: Permission denied on HPC
```bash
# Use user install
pip install --user -r requirements.txt
```

### Issue: Slow installation
```bash
# Use cached packages
pip install --no-deps -r requirements.txt
```

## Version Compatibility

- Python 3.11+ required (tested with 3.11.13)
- PyTorch 2.5.1 with CUDA 12.4
- All dependencies frozen at working versions in requirements.txt

## Development vs Production

For development, you may also want:
```bash
pip install jupyter ipython  # Interactive development
pip install pytest black flake8  # Testing and linting
```

## Updating Dependencies

To update dependencies while maintaining compatibility:
```bash
# Update specific package
pip install --upgrade package_name

# Regenerate requirements
pip freeze > requirements.txt

# Test thoroughly before committing
```

## Contact for Issues

If dependency issues arise:
1. Check the requirements.txt file for exact versions
2. Verify Python version (must be 3.11+)
3. Ensure CUDA compatibility for GPU usage
4. Contact: Paul Reiners (paul.reiners@gmail.com)