# Emotion Detection Pipeline - Handover Documentation

## EXECUTIVE SUMMARY
- 22 trained models, 96.9% average accuracy
- Production-ready system for video emotion detection
- Known working configuration and commands

## QUICK START - COPY/PASTE THESE COMMANDS

For now, use my existing Python environment.  You should later create your own Python environment with the necessary dependencies.

# Environment setup
    cd /users/9/reine097/projects/auto-label-emotions/
    module load python3
    source .venv/bin/activate

# Test prediction
    export PYTHONPATH="/users/9/reine097/projects/auto-label-emotions/src:$PYTHONPATH"
    python src/enhanced_pipeline/predict.py \
        --models data/my_results/comprehensive_pipeline_results.json \
        --video data/clip01/in/clip1_MLP.mp4 \
        --output test_handover.csv

# Train new models (if needed)
    ssh -Y agate
    cd /users/9/reine097/projects/auto-label-emotions/scripts
    sbatch run_pipeline.sh

## CRITICAL FILES LOCATIONS
- Trained models: data/my_results/*.pth
- Results summary: data/my_results/comprehensive_pipeline_results.json
- Training data: data/clip*/
- Main scripts: 
  * Prediction: [src/enhanced_pipeline/predict.py](./src/enhanced_pipeline/predict.py)
  * Training: [scripts/run_pipeline.sh](./scripts/run_pipeline.sh)

## KNOWN ISSUES & FIXES

Issue: Config file required error
Fix: Add --default flag

## PERFORMANCE BENCHMARKS
- Training time: ~3.6 minutes per model
- Prediction speed: ~12-15 seconds for a 14.6 MB video file
  * Model loading: ~10-12 seconds for all 22 models
  * Processing: Varies based on video length and resolution
- Disk space needed: ~500 MB for trained models, ~2 GB for training data

## EMERGENCY CONTACTS
Your name: [email/phone for first 2 weeks]
