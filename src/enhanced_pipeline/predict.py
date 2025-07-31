#!/usr/bin/env python3
"""
Prediction CLI - Command Line Interface for Model Prediction

Usage:
    python predict.py --models models.json --video video.mp4 --output predictions.csv
    python predict.py --models models.json --batch video_list.txt --output_dir batch_results/
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

# Force CPU usage to avoid CUDA issues
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from enhanced_pipeline.prediction import ModelLoader, VideoPredictor, BatchPredictor


def load_model_paths_from_json(json_file: str) -> Dict[str, str]:
    """Load model paths from JSON file"""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Support different JSON formats
        if 'model_paths' in data:
            return data['model_paths']
        elif 'training_results' in data:
            # Extract from pipeline results
            model_paths = {}
            for col, result in data['training_results'].items():
                if result.get('success', False) and 'model_path' in result:
                    model_paths[col] = result['model_path']
            return model_paths
        else:
            # Assume the JSON itself is the model paths dictionary
            return data
            
    except Exception as e:
        print(f"\u274c Error loading model paths from {json_file}: {e}")
        return {}


def load_video_list_from_file(video_list_file: str) -> List[str]:
    """Load video paths from text file (one per line)"""
    try:
        with open(video_list_file, 'r') as f:
            videos = [line.strip() for line in f.readlines() if line.strip()]
        return videos
    except Exception as e:
        print(f"\u274c Error loading video list from {video_list_file}: {e}")
        return []


def create_model_paths_template(output_file: str = "model_paths_template.json"):
    """Create a template JSON file for model paths"""
    template = {
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
    
    with open(output_file, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"\U0001f4dd Template created: {output_file}")
    print("   Edit this file with your actual model paths")


def predict_single_video(args):
    """Handle single video prediction"""
    print(f"\U0001f3ac SINGLE VIDEO PREDICTION")
    print(f"   Video: {args.video}")
    print(f"   Models: {args.models}")
    print(f"   Output: {args.output}")
    print("="*50)
    
    # Load model paths
    model_paths = load_model_paths_from_json(args.models)
    if not model_paths:
        print("\u274c No model paths loaded!")
        return False
    
    print(f"\U0001f4e6 Found {len(model_paths)} model paths")
    
    # Load models
    loader = ModelLoader(model_paths, verbose=args.verbose)
    load_results = loader.load_models()
    
    if not load_results['loaded']:
        print("\u274c No models loaded successfully!")
        return False
    
    # Run prediction
    predictor = VideoPredictor(loader.get_loaded_models(), verbose=args.verbose)
    
    result = predictor.predict_video(
        video_path=args.video,
        output_csv=args.output,
        frame_interval_ms=args.interval,
        threshold=args.threshold,
        include_probabilities=args.probabilities
    )
    
    if result['success']:
        print(f"\n\u2705 Prediction complete!")
        print(f"\U0001f4ca Results: {result['predictions_shape'][0]:,} frames, {result['predictions_shape'][1]} columns")
        print(f"\U0001f4be Saved to: {result['output_csv']}")
        return True
    else:
        print(f"\n\u274c Prediction failed: {result['error']}")
        return False


def predict_batch_videos(args):
    """Handle batch video prediction"""
    print(f"\U0001f3ac BATCH VIDEO PREDICTION")
    print(f"   Video list: {args.batch}")
    print(f"   Models: {args.models}")
    print(f"   Output dir: {args.output_dir}")
    print("="*50)
    
    # Load video list
    if args.batch.endswith('.txt'):
        video_paths = load_video_list_from_file(args.batch)
    else:
        # Assume it's a directory with videos
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_paths = []
        for ext in video_extensions:
            video_paths.extend(Path(args.batch).glob(f"*{ext}"))
        video_paths = [str(p) for p in video_paths]
    
    if not video_paths:
        print(f"\u274c No videos found in {args.batch}")
        return False
    
    print(f"\U0001f3a5 Found {len(video_paths)} videos")
    
    # Load model paths
    model_paths = load_model_paths_from_json(args.models)
    if not model_paths:
        print("\u274c No model paths loaded!")
        return False
    
    # Load models
    loader = ModelLoader(model_paths, verbose=args.verbose)
    load_results = loader.load_models()
    
    if not load_results['loaded']:
        print("\u274c No models loaded successfully!")
        return False
    
    # Run batch prediction
    batch_predictor = BatchPredictor(loader, verbose=args.verbose)
    
    batch_results = batch_predictor.predict_multiple_videos(
        video_paths=video_paths,
        output_dir=args.output_dir,
        frame_interval_ms=args.interval,
        threshold=args.threshold,
        include_probabilities=args.probabilities
    )
    
    success_count = len(batch_results['successful'])
    total_count = batch_results['total']
    
    if success_count > 0:
        print(f"\n\u2705 Batch prediction complete!")
        print(f"\U0001f4ca Success rate: {success_count}/{total_count} videos")
        if 'statistics' in batch_results:
            stats = batch_results['statistics']
            print(f"\U0001f39e\ufe0f Total frames: {stats['total_frames']:,}")
            print(f"\u23f1\ufe0f Total duration: {stats['total_duration_seconds']:.1f} seconds")
        print(f"\U0001f4be Results saved to: {args.output_dir}")
        return True
    else:
        print(f"\n\u274c Batch prediction failed!")
        return False


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Prediction CLI for Enhanced Multi-Column Classification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single video prediction
  python predict.py --models model_paths.json --video clip1.mp4 --output predictions.csv
  
  # Batch prediction from video list file
  python predict.py --models model_paths.json --batch video_list.txt --output_dir results/
  
  # Batch prediction from directory
  python predict.py --models model_paths.json --batch videos/ --output_dir results/
  
  # Create template JSON file
  python predict.py --create_template
  
  # Custom settings
  python predict.py --models models.json --video clip.mp4 --output pred.csv \\
                    --interval 50 --threshold 0.7 --no_probabilities
        """
    )
    
    # Main action arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--video', type=str, help='Single video file to process')
    group.add_argument('--batch', type=str, help='Video list file (.txt) or directory with videos')
    group.add_argument('--create_template', action='store_true', help='Create model paths template JSON')
    
    # Required arguments (except for template creation)
    parser.add_argument('--models', type=str, help='JSON file containing model paths')
    parser.add_argument('--output', type=str, help='Output CSV file (for single video)')
    parser.add_argument('--output_dir', type=str, help='Output directory (for batch processing)')
    
    # Optional arguments
    parser.add_argument('--interval', type=int, default=100, 
                       help='Frame extraction interval in milliseconds (default: 100)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold (default: 0.5)')
    parser.add_argument('--probabilities', action='store_true', default=True,
                       help='Include probability columns (default: True)')
    parser.add_argument('--no_probabilities', action='store_false', dest='probabilities',
                       help='Exclude probability columns')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Enable verbose output (default: True)')
    parser.add_argument('--quiet', action='store_false', dest='verbose',
                       help='Disable verbose output')
    
    args = parser.parse_args()
    
    # Handle template creation
    if args.create_template:
        create_model_paths_template()
        return
    
    # Validate required arguments
    if not args.models:
        print("\u274c --models argument is required (except for --create_template)")
        parser.print_help()
        sys.exit(1)
    
    if args.video and not args.output:
        print("\u274c --output argument is required for single video prediction")
        parser.print_help()
        sys.exit(1)
    
    if args.batch and not args.output_dir:
        print("\u274c --output_dir argument is required for batch prediction")
        parser.print_help()
        sys.exit(1)
    
    # Check if model file exists
    if not os.path.exists(args.models):
        print(f"\u274c Model file not found: {args.models}")
        print("\U0001f4a1 Use --create_template to create a template JSON file")
        sys.exit(1)
    
    # Run appropriate prediction mode
    try:
        if args.video:
            # Check if video exists
            if not os.path.exists(args.video):
                print(f"\u274c Video file not found: {args.video}")
                sys.exit(1)
            
            success = predict_single_video(args)
        else:
            # Check if batch input exists
            if not os.path.exists(args.batch):
                print(f"\u274c Batch input not found: {args.batch}")
                sys.exit(1)
            
            success = predict_batch_videos(args)
        
        if success:
            print(f"\n\U0001f389 All done! Check your output files.")
            sys.exit(0)
        else:
            print(f"\n\U0001f4a5 Process completed with errors.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n\U0001f6d1 Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\U0001f4a5 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()