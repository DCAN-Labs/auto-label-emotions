# Data Files Inventory and Purpose

## Directory Structure Overview

```
data/
├── clip01-08/          # Training video clips and annotations
├── combined/           # Combined dataset from all clips
├── face_detection_retrain/  # Face detection retraining data
├── my_results/         # Trained models and analysis outputs
└── config_file_example.json  # Example configuration file
```

## Training Data Directories (clip01-08)

Each clip directory contains:
- `in/` subdirectory with:
  - `clip#_MLP.mp4` - The source video file
  - `clip#_codes_MLP.csv` - Manual emotion annotations for training
- `out/` subdirectory with:
  - Processed frames and intermediate outputs

**Purpose**: These are the annotated video clips used to train the 22 emotion classifiers. Each clip has been manually labeled with various emotion codes.

## Combined Dataset
- **Location**: `data/combined/`
- **Purpose**: Aggregated dataset combining all individual clips for comprehensive training
- **Contents**: Merged annotations and frames from all clips

## Face Detection Retraining Data
- **Location**: `data/face_detection_retrain/`
- **Purpose**: Additional data for improving face detection model performance
- **Use Case**: Fine-tuning face detection for cartoon/animation content

## Model Results Directory (`data/my_results/`)

### Trained Model Files (.pth)
All 22 trained classifier models, each ~13-47MB:
- `closeup_classifier.pth` - Detects closeup shots
- `collective_classifier.pth` - Detects group/collective scenes
- `has_body_classifier.pth` - Detects visible body language
- `has_faces_classifier.pth` - Detects presence of faces
- `has_words_classifier.pth` - Detects text/words on screen
- `time_of_day_classifier.pth` - Classifies time of day in scene
- `on_screen_classifier.pth` - Detects on-screen elements
- `char_valence_positive_classifier.pth` - Character positive emotion

#### Character Emotion Models (Body, Face, Verbal channels)
- `c_happy_[body|face|verbal]_classifier.pth` - Happiness detection
- `c_excite_[body|face|verbal]_classifier.pth` - Excitement detection
- `c_fear_[body|face|verbal]_classifier.pth` - Fear detection
- `c_sad_[body|face|verbal]_classifier.pth` - Sadness detection
- `c_anger_[body|face|verbal]_classifier.pth` - Anger detection
- `c_disgust_[body|face|verbal]_classifier.pth` - Disgust detection

### Dataset Directories
For each model, there's a corresponding `_dataset` directory containing:
- Training/validation split information
- Preprocessed features
- Data augmentation configurations

### Analysis Outputs
- `comprehensive_pipeline_results.json` - Main results file with all model paths and performance metrics
- `analysis_dashboard_page_*.png` - Performance visualization dashboards (7 pages)
- Various `.json` files with individual model configurations and results

### Performance Tracking
- Training logs and metrics for each model
- Validation accuracy scores
- Confusion matrices and performance charts

## Configuration Files

### config_file_example.json
- **Purpose**: Template configuration for running the pipeline
- **Contents**:
  - Data paths
  - Model hyperparameters
  - Training settings
  - Output specifications

## File Size Summary
- **Video clips**: ~15MB each
- **Trained models**: 13-47MB each (total ~500MB for all 22 models)
- **Complete dataset**: ~2GB including all frames and annotations
- **Analysis outputs**: ~3MB for dashboards and reports

## Data Flow
1. **Input**: Raw video files (clip#_MLP.mp4) + manual annotations (clip#_codes_MLP.csv)
2. **Processing**: Frame extraction, feature engineering, data augmentation
3. **Training**: Creates model files (.pth) and dataset directories
4. **Output**: Trained classifiers + performance metrics (comprehensive_pipeline_results.json)

## Critical Files for Production
1. `comprehensive_pipeline_results.json` - Contains all model paths and configurations
2. All `.pth` files in `my_results/` - The actual trained models
3. `config_file_example.json` - Template for new predictions

## Notes
- All models were trained on 8 video clips with manual annotations
- The system achieved 96.9% average accuracy across all 22 models
- Models are saved in PyTorch format (.pth files)
- Each model specializes in detecting specific visual or emotional features