# Trained Models Inventory

## Overview
This document provides a comprehensive inventory of all trained models in the auto-label-emotions project, including their purpose, location, size, and performance metrics.

Some sample pre-trained models are located here:

* `/home/feczk001/shared/data/score-conners-3/trained_models.zip`

## Core Models

### 1. Face Detection Models

#### cartoon_face_detector_pytorch.pth
- **Location:** `/cartoon_face_detector_pytorch.pth`
- **Size:** 11MB
- **Purpose:** Cartoon face detection in video frames
- **Architecture:** PyTorch-based CNN
- **Last Modified:** July 16, 2024

#### face_detector_model.pth
- **Location:** `/data/face_detection_retrain/face_detector_model.pth`
- **Size:** 13MB
- **Purpose:** Retrained face detection model
- **Architecture:** Custom CNN for cartoon faces
- **Last Modified:** July 18, 2024

### 2. Best Model

#### best_model.pth
- **Location:** `/best_model.pth`
- **Size:** 13MB
- **Purpose:** Primary emotion classification model
- **Performance:** 96.9% accuracy (as reported)
- **Last Modified:** July 16, 2024

## Emotion Classification Models

### Character Emotion Models (Body Language)
All models use transfer learning with MobileNet backbone, trained for 50 epochs.

| Model | File | Size | Last Modified |
|-------|------|------|--------------|
| Excitement (Body) | `c_excite_body_classifier.pth` | 20MB | Sep 18, 2024 |
| Fear (Body) | `c_fear_body_classifier.pth` | 13MB | Sep 18, 2024 |
| Happiness (Body) | `c_happy_body_classifier.pth` | 45MB | Sep 18, 2024 |
| Sadness (Body) | `c_sad_body_classifier.pth` | 13MB | Sep 18, 2024 |

### Character Emotion Models (Facial Expression)

| Model | File | Size | Last Modified |
|-------|------|------|--------------|
| Excitement (Face) | `c_excite_face_classifier.pth` | 20MB | Sep 18, 2024 |
| Fear (Face) | `c_fear_face_classifier.pth` | 13MB | Sep 18, 2024 |
| Happiness (Face) | `c_happy_face_classifier.pth` | 45MB | Sep 18, 2024 |
| Sadness (Face) | `c_sad_face_classifier.pth` | 13MB | Sep 18, 2024 |

### Character Emotion Models (Verbal/Audio)

| Model | File | Size | Last Modified |
|-------|------|------|--------------|
| Excitement (Verbal) | `c_excite_verbal_classifier.pth` | 20MB | Sep 18, 2024 |
| Fear (Verbal) | `c_fear_verbal_classifier.pth` | 13MB | Sep 18, 2024 |
| Happiness (Verbal) | `c_happy_verbal_classifier.pth` | 45MB | Sep 18, 2024 |
| Sadness (Verbal) | `c_sad_verbal_classifier.pth` | 13MB | Sep 18, 2024 |

## Scene Analysis Models

### Valence Classification

| Model | File | Size | Last Modified |
|-------|------|------|--------------|
| Negative Valence | `char_valence_negative_classifier.pth` | 13MB | Sep 18, 2024 |
| Positive Valence | `char_valence_positive_classifier.pth` | 13MB | Sep 18, 2024 |

### Scene Composition Models

| Model | File | Size | Last Modified | Purpose |
|-------|------|------|--------------|---------|
| Closeup Detection | `closeup_classifier.pth` | 30MB | Sep 18, 2024 | Detect closeup shots |
| Collective Scene | `collective_classifier.pth` | 30MB | Sep 18, 2024 | Detect group/collective scenes |
| Body Presence | `has_body_classifier.pth` | 30MB | Sep 18, 2024 | Detect body presence in frame |
| Face Presence | `has_faces_classifier.pth` | 13MB | Sep 18, 2024 | Detect faces in frame |
| Text/Words | `has_words_classifier.pth` | 30MB | Sep 18, 2024 | Detect text/words in frame |
| Character Count | `num_chars_classifier.pth` | 13MB | Sep 18, 2024 | Count number of characters |

## Model Architecture Details

### Common Configuration
Based on `comprehensive_pipeline_results.json`:

- **Model Type:** Transfer learning
- **Backbone:** MobileNet (lightweight, efficient)
- **Pretrained:** Yes (ImageNet weights)
- **Input Size:** 224x224 pixels
- **Training Configuration:**
  - Epochs: 50
  - Learning Rate: 0.001
  - Batch Size: 32
  - Early Stopping Patience: 10

### Model Categories

1. **Face Detection Models (11-13MB)**
   - Specialized for cartoon face detection
   - Custom CNN architectures

2. **Small Models (13MB)**
   - Binary classifiers for specific attributes
   - Valence detection, face presence, character counting

3. **Medium Models (20MB)**
   - Excitement and related emotion classifiers
   - More complex feature extraction

4. **Large Models (30-45MB)**
   - Scene composition analysis
   - Multi-class or complex binary classification
   - Happy emotion classifiers (largest at 45MB)

## Model Storage Organization

```
/
├── best_model.pth (main production model)
├── cartoon_face_detector_pytorch.pth (face detection)
└── data/
    ├── face_detection_retrain/
    │   └── face_detector_model.pth (retrained face detector)
    └── my_results/
        ├── c_*_classifier.pth (emotion classifiers)
        ├── char_valence_*_classifier.pth (valence classifiers)
        └── *_classifier.pth (scene analysis models)
```

## Usage Notes

1. **Model Loading:** Models are loaded using PyTorch's `torch.load()` function
2. **Compatibility:** All models require PyTorch and are trained for inference on CPU or GPU
3. **Input Format:** Models expect preprocessed video frames (224x224 RGB images)
4. **Output:** Binary classification (0 or 1) for presence/absence of features

## Performance Metrics

- **Primary Model Accuracy:** 96.9% (best_model.pth)
- **Class Balance:** Most models trained on imbalanced datasets (see comprehensive_pipeline_results.json)
- **Training Data:** 3,692 samples across 8 video clips (MLP and AHKJ series)

## Maintenance Notes

- **Last Training Date:** September 18, 2024
- **Framework Version:** PyTorch (check requirements.txt for specific version)
- **Storage Requirements:** ~350MB total for all models
- **Backup Location:** Ensure models are backed up before modifications

---

*Last Updated: September 19, 2024*