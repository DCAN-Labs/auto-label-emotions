# Enhanced Pipeline Main Usage Guide

## Overview

The `enhanced_pipeline/main.py` file demonstrates how to use the Enhanced Multi-Column Classification Pipeline with both simple and advanced usage patterns. This modular pipeline processes video files and trains multiple binary classifiers for emotion and scene detection.

## Quick Start

### Basic Usage

```bash
python main.py
```

This will run the complete pipeline with default settings using the sample video clips.

## Pipeline Components

The main pipeline integrates several modular components:

### Core Components
- **`EnhancedMultiColumnPipeline`**: Main orchestrator
- **`ColumnAnalyzer`**: Analyzes CSV data and column statistics
- **`DashboardGenerator`**: Creates visualization dashboards
- **`ModelTrainer`**: Handles model training operations
- **`ConfigurationManager`**: Manages pipeline configurations
- **`ResultsManager`**: Handles result storage and retrieval

## Configuration

### Video Input Setup

```python
video_list = [
    "data/clip01/in/clip1_MLP.mp4",
    "data/clip02/in/clip2_AHKJ.mp4", 
    "data/clip03/in/clip3_MLP.mp4",
    "data/clip04/in/clip4_MLP.mp4"
]

clip_mapping = {
    'clip1_MLP': 'data/clip01/in/clip1_codes_MLP.csv',
    'clip2_AHKJ': 'data/clip02/in/clip2_codes_AHKJ.csv',
    'clip3_MLP': 'data/clip03/in/clip3_codes_MLP.csv', 
    'clip4_MLP': 'data/clip04/in/clip4_codes_MLP.csv'
}
```

### Pipeline Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video_list` | List[str] | Required | List of video file paths |
| `clip_csv_mapping` | Dict[str, str] | Required | Maps video names to CSV annotation files |
| `base_output_dir` | str | 'data/my_results' | Directory for outputs |
| `debug` | bool | False | Enable debug mode for verbose output |

## Usage Patterns

### 1. Complete Pipeline Execution

```python
from enhanced_pipeline.core_pipeline import EnhancedMultiColumnPipeline

pipeline = EnhancedMultiColumnPipeline(
    video_list=video_list,
    clip_csv_mapping=clip_mapping,
    base_output_dir='data/my_results',
    debug=False
)

results = pipeline.run_comprehensive_pipeline(
    skip_extraction=False,      # Set to True if frames already extracted
    create_visualizations=True, # Generate analysis dashboards
    verbose=True               # Show detailed progress
)
```

### 2. Advanced Component Usage

```python
from enhanced_pipeline import (
    ColumnAnalyzer, 
    DashboardGenerator,
    ConfigurationManager
)

# Analyze data only
analyzer = ColumnAnalyzer(clip_csv_mapping=clip_mapping, debug=True)
analysis_results = analyzer.analyze_all_columns(verbose=True)

# Create custom visualizations
dashboard_gen = DashboardGenerator(debug=True)
dashboard_gen.create_dashboard(
    column_stats=analysis_results['column_stats'],
    save_path='custom_dashboard.png'
)

# Manage configurations
config_manager = ConfigurationManager(debug=True)
config = config_manager.create_default_config(columns)
config_manager.save_config_to_file(config, 'my_config.json')
```

## Pipeline Execution Flow

### Phase 1: Data Analysis
- Loads and analyzes CSV annotation files
- Calculates column statistics and distributions
- Identifies suitable columns for classification

### Phase 2: Frame Extraction
- Extracts frames from video files at specified intervals
- Organizes frames by video and timestamp
- Applies preprocessing (black frame detection, etc.)

### Phase 3: Model Training
- Trains binary classifiers for each viable column
- Uses transfer learning with pre-trained backbones
- Implements proper train/validation splits

### Phase 4: Results Generation
- Saves trained models and configurations
- Generates performance metrics and visualizations
- Creates comprehensive results dashboard

## Expected Output Structure

```
data/my_results/
├── frames/                          # Extracted video frames
│   ├── clip1_MLP/
│   ├── clip2_AHKJ/
│   └── ...
├── models/                          # Trained model files
│   ├── has_faces_classifier.pth
│   ├── c_happy_face_classifier.pth
│   └── ...
├── comprehensive_pipeline_results.json  # Complete results summary
├── column_analysis_dashboard.png        # Analysis visualization
└── training_dashboard.png              # Training results visualization
```

## Result Interpretation

### Success Metrics
- **Models Trained**: Number of successful classifiers
- **Column Coverage**: Percentage of viable columns processed
- **Training Accuracy**: Performance metrics per model

### Common Output Columns
- `has_faces`: Face detection in frames
- `c_happy_face`: Happy facial expressions
- `c_sad_face`: Sad facial expressions
- `collective`: Group scenes
- `closeup`: Close-up shots
- `has_body`: Body visibility
- `num_chars`: Number of characters

## Troubleshooting

### Common Issues

**1. Video Files Not Found**
```
FileNotFoundError: Video file not found
```
- Check video file paths in `video_list`
- Ensure files exist and are readable

**2. CSV Mapping Errors**
```
KeyError: Video name not found in mapping
```
- Verify `clip_mapping` keys match video filenames
- Check CSV file paths are correct

**3. Insufficient Training Data**
```
Warning: Column has too few positive examples
```
- Some columns may be skipped due to class imbalance
- This is normal behavior for rare events

### Debug Mode

Enable debug mode for detailed logging:

```python
pipeline = EnhancedMultiColumnPipeline(
    video_list=video_list,
    clip_csv_mapping=clip_mapping,
    debug=True  # Enable debug output
)
```

## Performance Considerations

### Memory Usage
- Frame extraction can use significant disk space
- Consider using `skip_extraction=True` for subsequent runs
- Monitor available storage space

### Training Time
- Model training time depends on:
  - Number of frames extracted
  - Number of viable columns
  - Hardware specifications (GPU recommended)

### Optimization Tips
- Use GPU acceleration when available
- Adjust frame extraction intervals for faster processing
- Consider subset of videos for initial testing

## Advanced Configuration

### Custom Frame Extraction
```python
# Modify frame extraction parameters
pipeline.frame_extractor.extract_frames_from_videos(
    video_list,
    output_dir="custom_frames",
    interval_ms=500,  # Extract frames every 500ms
    skip_black_frames=True
)
```

### Custom Model Training
```python
# Access individual trainer for custom settings
trainer = pipeline.model_trainer
trainer.train_classifier(
    column_name="custom_column",
    train_data=train_dataset,
    val_data=val_dataset,
    model_type="transfer",  # or "custom"
    backbone="mobilenet",   # or "resnet18", "efficientnet"
    epochs=10
)
```

## Integration with Prediction Pipeline

After training, use the prediction pipeline to process new videos:

```python
from enhanced_pipeline.prediction import ModelLoader, VideoPredictor

# Load trained models
model_paths = {
    'has_faces': 'data/my_results/has_faces_classifier.pth',
    'c_happy_face': 'data/my_results/c_happy_face_classifier.pth'
}

loader = ModelLoader(model_paths)
load_results = loader.load_models()

# Predict on new video
predictor = VideoPredictor(loader.get_loaded_models())
result = predictor.predict_video(
    video_path="new_video.mp4",
    output_csv="predictions.csv"
)
```

## Further Reading

- See `predict.py` for standalone prediction usage
- Check individual component documentation for advanced features
- Review `core_pipeline.py` for pipeline internals