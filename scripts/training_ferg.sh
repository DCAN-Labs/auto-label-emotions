#!/bin/sh
#SBATCH --job-name=facial-expression-training # job name
#SBATCH --mem=180g        
#SBATCH --time=1:00:00          
#SBATCH -p a100-4,a100-8
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=6      
#SBATCH --mail-user=reine097@umn.edu
#SBATCH -e facial-expression-training-%j.err
#SBATCH -o facial-expression-training-%j.out

# Diagnostic information - print job details
echo "======= JOB DIAGNOSTIC INFO ======="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Start time: $(date)"
echo "Allocated resources:"
echo "  CPUs: $SLURM_NTASKS"
echo "  Memory: 180GB"
echo "  GPUs: 2 x NVIDIA A100"

# Check GPU availability before starting
echo "======= GPU INFORMATION ======="
nvidia-smi

# Check available disk space
echo "======= DISK SPACE ======="
df -h /users/9/reine097/projects/auto-label-emotions

# Directory navigation with error checking
cd /users/9/reine097/projects/auto-label-emotions || { echo "ERROR: Failed to change directory"; exit 1; }
echo "Current directory: $(pwd)"

# Check if data directory exists
echo "======= DATA CHECK ======="
if [ -d "data/tvt/pixar" ]; then
    echo "Dataset exists. File count: $(find data/tvt/pixar -type f | wc -l)"
else
    echo "ERROR: Dataset directory not found: data/tvt/pixar"
    exit 1
fi

# Check Python environment
echo "======= PYTHON ENVIRONMENT ======="
echo "Python version: $(/users/9/reine097/projects/auto-label-emotions/.venv/bin/python --version)"
echo "PyTorch version: $(/users/9/reine097/projects/auto-label-emotions/.venv/bin/python -c 'import torch; print(f"PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}, CUDA version: {torch.version.cuda if torch.cuda.is_available() else None}")')"

# Export Python path
export PYTHONPATH=PYTHONPATH:"/users/9/reine097/projects/auto-label-emotions/src"
echo "PYTHONPATH: $PYTHONPATH"

# Record start time for duration calculation
START_TIME=$(date +%s)

echo "======= STARTING MODEL TRAINING ======="
# Run the script with time command to track execution time
time /users/9/reine097/projects/auto-label-emotions/.venv/bin/python \
  /users/9/reine097/projects/auto-label-emotions/src/cdni/deep_learning/pytorch_image_classifier_png_data.py \
  "data/tvt/FERG" "/home/feczk001/shared/data/auto_label_emotions/models/ferg00.pth"

# Calculate duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo "======= JOB COMPLETED ======="
echo "End time: $(date)"
echo "Total duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "==============================="
