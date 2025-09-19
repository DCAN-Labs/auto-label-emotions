# Training Configuration Guide

## Quick Start: Creating a Training Config File

This guide explains how to create configuration files for training emotion classification models.

## Minimum Required Config

### Step 1: Create a Simple Config File

Create a new file called `my_training_config.json`:

```json
{
  "video_list": [
    "data/clip01/in/clip1_MLP.mp4",
    "data/clip02/in/clip2_AHKJ.mp4"
  ],
  "clip_mapping": {
    "clip1_MLP": "data/clip01/in/clip1_codes_MLP.csv",
    "clip2_AHKJ": "data/clip02/in/clip2_codes_AHKJ.csv"
  }
}
```

This minimal config will:
- Process 2 videos
- Use their corresponding annotation CSV files
- Apply default settings for everything else

### Step 2: Run Training

```bash
# Adjust file paths as needed for you MSI folders.
cd /users/9/reine097/projects/auto-label-emotions/ || exit
export PYTHONPATH=PYTHONPATH:"/users/9/reine097/projects/auto-label-emotions/src"

python src/enhanced_pipeline/main.py --config \
    data/configuration/my_training_config.json
```

## Full Config with All Options

For more control, include pipeline settings:

```json
{
  "video_list": [
    "path/to/video1.mp4",
    "path/to/video2.mp4"
  ],
  "clip_mapping": {
    "video1_name": "path/to/annotations1.csv",
    "video2_name": "path/to/annotations2.csv"
  },
  "pipeline_settings": {
    "base_output_dir": "data/my_results",
    "debug": false,
    "frame_interval_ms": 100,
    "skip_extraction": false,
    "create_visualizations": true,
    "verbose": true
  }
}
```

## Key Components Explained

### 1. video_list
- **Purpose:** List of video files to process
- **Format:** Array of file paths
- **Example:** `["video1.mp4", "video2.mp4"]`
- **Note:** Videos must exist or pipeline will warn you

### 2. clip_mapping
- **Purpose:** Maps video names to their annotation CSV files
- **Format:** Object with video_name: csv_path pairs
- **Example:** `{"clip1_MLP": "clip1_codes_MLP.csv"}`
- **Note:** CSV files contain ground truth labels for training

### 3. pipeline_settings (Optional)

| Setting | Default | Description |
|---------|---------|-------------|
| `base_output_dir` | `"data/my_results"` | Where to save trained models |
| `debug` | `false` | Enable debug mode (smaller datasets, fewer epochs) |
| `frame_interval_ms` | `100` | Extract frames every N milliseconds |
| `skip_extraction` | `false` | Skip frame extraction if already done |
| `create_visualizations` | `true` | Generate performance charts |
| `verbose` | `true` | Show detailed progress |

## CSV Annotation Format

Your annotation CSV files must have:
- `timestamp_ms`: Frame timestamp in milliseconds
- Binary columns (0 or 1) for each feature to train:
  - `has_faces`
  - `closeup`
  - `c_happy_face`
  - etc.

Example CSV structure:
```csv
timestamp_ms,has_faces,closeup,c_happy_face
0,1,0,0
100,1,1,1
200,0,0,0
```

## Common Scenarios

### Scenario 1: Quick Test with Debug Mode
```json
{
  "video_list": ["test_video.mp4"],
  "clip_mapping": {"test_video": "test_annotations.csv"},
  "pipeline_settings": {
    "debug": true,
    "verbose": true
  }
}
```

### Scenario 2: Batch Processing Multiple Videos
```json
{
  "video_list": [
    "data/batch1/video1.mp4",
    "data/batch1/video2.mp4",
    "data/batch1/video3.mp4"
  ],
  "clip_mapping": {
    "video1": "data/batch1/labels1.csv",
    "video2": "data/batch1/labels2.csv",
    "video3": "data/batch1/labels3.csv"
  },
  "pipeline_settings": {
    "base_output_dir": "models/batch1",
    "frame_interval_ms": 50
  }
}
```

### Scenario 3: Rerun Training (Skip Frame Extraction)
```json
{
  "video_list": ["video.mp4"],
  "clip_mapping": {"video": "labels.csv"},
  "pipeline_settings": {
    "skip_extraction": true,
    "base_output_dir": "models/retrain"
  }
}
```

## Creating Your First Config

1. **Copy the template:**
   ```bash
   cp data/config_file_example.json my_config.json
   ```

2. **Edit paths to your data:**
   - Replace video paths with your video files
   - Replace CSV paths with your annotation files

3. **Adjust settings as needed:**
   - Set `debug: true` for testing
   - Change `base_output_dir` for organization
   - Modify `frame_interval_ms` based on video content

4. **Run training:**
   ```bash
   python src/enhanced_pipeline/main.py --config my_config.json
   ```

## Tips for Success

1. **Start Small:** Test with 1-2 videos first
2. **Use Debug Mode:** Set `"debug": true` for faster iterations
3. **Check Paths:** Ensure all video and CSV paths are correct
4. **Match Names:** Video names in `clip_mapping` should match actual filenames
5. **Frame Interval:** 100ms (10 fps) is usually sufficient for cartoon content

## Output Structure

After training with your config, you'll find:
```
data/my_results/
├── comprehensive_pipeline_results.json  # Training metrics
├── closeup_classifier.pth              # Trained models
├── has_faces_classifier.pth
├── training_history_closeup.png        # Performance graphs
└── ...
```

## Troubleshooting

**Missing files warning:**
- Pipeline will warn but let you continue
- Fix paths or type 'y' to proceed anyway

**Out of memory:**
- Set `"debug": true` to use smaller batch sizes
- Increase `frame_interval_ms` to extract fewer frames

**Poor accuracy:**
- Ensure annotation CSVs are correctly labeled
- Check class balance in your data
- Try more training videos

---

*For prediction after training, see the Prediction Guide*