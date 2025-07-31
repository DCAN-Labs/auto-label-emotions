# Prediction Module - Enhanced Multi-Column Classification Pipeline

A standalone prediction module that loads trained models and generates CSV predictions for videos.

## 🚀 Quick Start

### Step 1: Create Model Paths Configuration

```bash
python predict.py --create_template
```

This creates `model_paths_template.json`. Edit it with your actual model paths:

```json
{
  "model_paths": {
    "has_faces": "data/my_results/has_faces_classifier.pth",
    "c_happy_face": "data/my_results/c_happy_face_classifier.pth",
    "c_sad_face": "data/my_results/c_sad_face_classifier.pth",
    "collective": "data/my_results/collective_classifier.pth",
    "num_chars": "data/my_results/num_chars_classifier.pth"
  },
  "config": {
    "frame_interval_ms": 100,
    "threshold": 0.5,
    "include_probabilities": true
  }
}
```

### Step 2: Run Predictions

**Single video:**
```bash
python predict.py --models model_paths.json --video clip1.mp4 --output predictions.csv
```

**Batch processing:**
```bash
python predict.py --models model_paths.json --batch video_list.txt --output_dir results/
```

## 📋 Usage Options

### Single Video Prediction
```bash
python predict.py \
    --models model_paths.json \
    --video path/to/video.mp4 \
    --output predictions.csv \
    --interval 100 \
    --threshold 0.5 \
    --probabilities
```

### Batch Video Prediction

**From video list file:**
```bash
# Create video_list.txt with one video path per line:
# data/clip01/in/clip1_MLP.mp4
# data/clip02/in/clip2_AHKJ.mp4
# data/clip03/in/clip3_MLP.mp4

python predict.py --models model_paths.json --batch video_list.txt --output_dir results/
```

**From directory:**
```bash
python predict.py --models model_paths.json --batch videos/ --output_dir results/
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--models` | JSON file with model paths | Required |
| `--video` | Single video file | - |
| `--batch` | Video list file or directory | - |
| `--output` | Output CSV (single video) | Required for single |
| `--output_dir` | Output directory (batch) | Required for batch |
| `--interval` | Frame interval (ms) | 100 |
| `--threshold` | Classification threshold | 0.5 |
| `--probabilities` | Include probabilities | True |
| `--no_probabilities` | Exclude probabilities | - |
| `--verbose` | Verbose output | True |
| `--quiet` | Quiet mode | - |

## 📊 Output Format

The CSV output contains:
- `onset_milliseconds`: Timestamp for each frame
- `frame_path`: Frame filename
- `[column_name]`: Binary prediction (0/1) for each model
- `[column_name]_prob`: Probability (0.0-1.0) if `--probabilities` enabled

Example output:
```csv
onset_milliseconds,frame_path,has_faces,has_faces_prob,c_happy_face,c_happy_face_prob
0,frame_0000.jpg,1,0.89,0,0.23
100,frame_0001.jpg,1,0.92,1,0.78
200,frame_0002.jpg,0,0.12,0,0.15
```

## 🐍 Python API Usage

### Basic Usage

```python
from enhanced_pipeline.prediction import ModelLoader, VideoPredictor

# Define model paths
model_paths = {
    'has_faces': 'data/my_results/has_faces_classifier.pth',
    'c_happy_face': 'data/my_results/c_happy_face_classifier.pth'
}

# Load models
loader = ModelLoader(model_paths, verbose=True)
load_results = loader.load_models()

# Create predictor
predictor = VideoPredictor(loader.get_loaded_models(), verbose=True)

# Run prediction
result = predictor.predict_video(
    video_path='video.mp4',
    output_csv='predictions.csv',
    frame_interval_ms=100,
    threshold=0.5,
    include_probabilities=True
)

print(f"Success: {result['success']}")
print(f"Total frames: {result['total_frames']}")
```

### Batch Processing

```python
from enhanced_pipeline.prediction import ModelLoader, BatchPredictor

# Load models
loader = ModelLoader(model_paths)
loader.load_models()

# Create batch predictor
batch_predictor = BatchPredictor(loader, verbose=True)

# Process multiple videos
video_list = ['video1.mp4', 'video2.mp4', 'video3.mp4']

batch_results = batch_predictor.predict_multiple_videos(
    video_paths=video_list,
    output_dir='batch_results/',
    frame_interval_ms=100,
    threshold=0.5,
    include_probabilities=True
)

print(f"Processed {len(batch_results['successful'])} videos successfully")
```

## ⚙️ Advanced Configuration

### Custom Model Configuration

You can include model configurations in your JSON file:

```json
{
  "model_paths": {
    "has_faces": "models/face_detector.pth",
    "c_happy_face": "models/happy_classifier.pth"
  },
  "model_configs": {
    "has_faces": {
      "model_type": "transfer",
      "backbone": "mobilenet",
      "img_size": 224,
      "freeze_features": true
    },
    "c_happy_face": {
      "model_type": "transfer", 
      "backbone": "resnet18",
      "img_size": 224,
      "freeze_features": false
    }
  }
}
```

### Loading from Pipeline Results

You can also use the results JSON from the training pipeline:

```bash
python predict.py --models data/my_results/comprehensive_pipeline_results.json --video clip.mp4 --output pred.csv
```

The module will automatically extract model paths from the training results.

## 🔧 Troubleshooting

### Common Issues

**CUDA Errors:**
- The module automatically forces CPU usage to avoid CUDA/cuDNN issues
- All predictions run on CPU for maximum compatibility

**Model Loading Errors:**
- Ensure all model files exist and are accessible
- Check that model architectures match the original training configuration
- Use the same Python environment as training

**Memory Issues:**
- For long videos, consider increasing `frame_interval_ms` to reduce frame count
- Use `--no_probabilities` to reduce memory usage

**Performance:**
- CPU prediction is slower than GPU but more reliable
- Processing time is approximately 1-2 seconds per frame depending on model complexity
- Use batch processing for multiple videos to maximize efficiency

## 📈 Performance Examples

Based on your excellent training results:

| Model | Expected Accuracy | Use Case |
|-------|------------------|----------|
| `collective` | 99.7% | Group vs individual detection |
| `c_sad_verbal` | 99.3% | Sad verbal expressions |
| `time_of_day` | 99.4% | Temporal classification |
| `has_faces` | 95.4% | Face detection |
| `c_happy_face` | 93.3% | Happy facial expressions |

## 📝 Tips

1. **Start with a small test video** to verify everything works
2. **Use appropriate thresholds** - 0.5 is good for balanced data, adjust based on your needs
3. **Include probabilities** for more nuanced analysis
4. **Use batch processing** for multiple videos to save time
5. **Monitor output file sizes** - probabilities double the file size

## 🆘 Getting Help

If you encounter issues:
1. Check that all model files exist
2. Verify video files are readable
3. Ensure sufficient disk space for output
4. Use `--verbose` mode to see detailed processing information

Example successful output:
```
📦 Found 22 model paths
📦 Loading model for has_faces...
   ✅ Successfully loaded has_faces
📦 Loading model for c_happy_face...
   ✅ Successfully loaded c_happy_face
...
📊 Loading Summary: 22/22 models loaded successfully
🎬 Processing video: clip1.mp4
📽️ Extracting frames...
📊 Processing 1,200 frames...
💾 Predictions saved to: predictions.csv
✅ Prediction complete!
```