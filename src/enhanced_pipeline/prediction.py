#!/usr/bin/env python3
"""
Prediction Module

This module handles prediction-only operations using trained models.
Loads models from disk and generates CSV predictions for videos or image directories.
"""

import os
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import json
import time
from datetime import datetime

from pytorch_cartoon_face_detector import BinaryClassifier, FaceDetector
from mp4_frame_extractor import extract_frames_from_videos


class ModelLoader:
    """Loads and manages trained models for prediction"""
    
    def __init__(self, model_paths: Dict[str, str], verbose: bool = True):
        """
        Initialize model loader with paths to trained models
        
        Args:
            model_paths: Dictionary mapping column names to model file paths
            verbose: Enable verbose output
        """
        self.model_paths = model_paths
        self.verbose = verbose
        self.loaded_models = {}
        self.model_configs = {}
        self.face_column = 'has_faces'
        
    def load_models(self, config_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Load all models from disk
        
        Args:
            config_file: Optional path to configuration file with model details
            
        Returns:
            Dictionary with loading results
        """
        results = {'loaded': [], 'failed': [], 'total': len(self.model_paths)}
        
        # Load configuration if provided
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
                self.model_configs = config.get('model_configs', {})
        
        for column, model_path in self.model_paths.items():
            if self.verbose:
                print(f"\U0001f4e6 Loading model for {column}...")
            
            try:
                model = self._load_single_model(column, model_path)
                if model is not None:
                    self.loaded_models[column] = model
                    results['loaded'].append(column)
                    if self.verbose:
                        print(f"   \u2705 Successfully loaded {column}")
                else:
                    results['failed'].append(column)
                    if self.verbose:
                        print(f"   \u274c Failed to load {column}")
                        
            except Exception as e:
                results['failed'].append(column)
                if self.verbose:
                    print(f"   \u274c Error loading {column}: {e}")
        
        if self.verbose:
            print(f"\n\U0001f4ca Loading Summary: {len(results['loaded'])}/{results['total']} models loaded successfully")
        
        return results
    
    def _load_single_model(self, column: str, model_path: str) -> Optional[Any]:
        """Load a single model from disk"""
        if not os.path.exists(model_path):
            return None
        
        # Get model configuration
        config = self.model_configs.get(column, self._get_default_config(column))
        
        # Create appropriate classifier
        if column == self.face_column:
            classifier = FaceDetector(
                model_type=config.get('model_type', 'transfer'),
                backbone=config.get('backbone', 'mobilenet'),
                img_size=config.get('img_size', 224),
                device='cpu'  # Force CPU for prediction to avoid CUDA issues
            )
        else:
            # Determine class names
            class_names = self._get_class_names(column)
            classifier = BinaryClassifier(
                task_name=f"{column}_classification",
                class_names=class_names,
                model_type=config.get('model_type', 'transfer'),
                backbone=config.get('backbone', 'mobilenet'),
                img_size=config.get('img_size', 224),
                device='cpu'  # Force CPU for prediction
            )
        
        # Create model structure
        classifier.create_model(
            pretrained=False,  # We'll load weights
            freeze_features=config.get('freeze_features', True)
        )
        
        # Load trained weights
        classifier.load_model(model_path)
        
        return classifier
    
    def _get_default_config(self, column: str) -> Dict[str, Any]:
        """Get default configuration for a column"""
        if column == self.face_column:
            return {
                'model_type': 'transfer',
                'backbone': 'mobilenet',
                'img_size': 224,
                'freeze_features': True
            }
        else:
            return {
                'model_type': 'transfer',
                'backbone': 'mobilenet',
                'img_size': 224,
                'freeze_features': True
            }
    
    def _get_class_names(self, column: str) -> Dict[int, str]:
        """Get class names for a column"""
        if column == self.face_column:
            return {0: 'no_face', 1: 'face'}
        elif 'happy' in column.lower():
            return {0: 'not_happy', 1: 'happy'}
        elif 'excite' in column.lower():
            return {0: 'calm', 1: 'excited'}
        elif 'sad' in column.lower():
            return {0: 'not_sad', 1: 'sad'}
        elif 'fear' in column.lower():
            return {0: 'not_fearful', 1: 'fearful'}
        elif 'anger' in column.lower():
            return {0: 'not_angry', 1: 'angry'}
        else:
            return {0: f'not_{column}', 1: column}
    
    def get_loaded_models(self) -> Dict[str, Any]:
        """Get all loaded models"""
        return self.loaded_models.copy()


class VideoPredictor:
    """Handles video prediction using loaded models"""
    
    def __init__(self, loaded_models: Dict[str, Any], verbose: bool = True):
        """
        Initialize video predictor
        
        Args:
            loaded_models: Dictionary of loaded model objects
            verbose: Enable verbose output
        """
        self.loaded_models = loaded_models
        self.verbose = verbose
    
    def predict_video(self, 
                     video_path: str,
                     output_csv: str,
                     frame_interval_ms: int = 100,
                     threshold: float = 0.5,
                     include_probabilities: bool = True,
                     temp_frames_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Predict on a video and save results to CSV
        
        Args:
            video_path: Path to input video
            output_csv: Path to output CSV file
            frame_interval_ms: Interval between frames in milliseconds
            threshold: Classification threshold
            include_probabilities: Whether to include probability columns
            temp_frames_dir: Temporary directory for frames (will be cleaned up)
            
        Returns:
            Dictionary with prediction results and statistics
        """
        if self.verbose:
            print(f"\U0001f3ac Processing video: {video_path}")
            print(f"   Frame interval: {frame_interval_ms}ms")
            print(f"   Models loaded: {len(self.loaded_models)}")
        
        # Create temporary frames directory
        if temp_frames_dir is None:
            temp_frames_dir = f"temp_frames_{int(time.time())}"
        
        try:
            # Extract frames from video
            if self.verbose:
                print("\U0001f4fd\ufe0f Extracting frames...")
            
            frame_results = extract_frames_from_videos(
                [video_path],
                output_dir=temp_frames_dir,
                interval_ms=frame_interval_ms,
                include_video_name=True,
                skip_black_frames=True,
                verbose=self.verbose
            )
            
            if not frame_results or frame_results[0].get('extracted_count', 0) == 0:
                raise ValueError("No frames extracted from video")
            
            # Get frame files
            frame_files = []
            for root, dirs, files in os.walk(temp_frames_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        frame_files.append(os.path.join(root, file))
            
            frame_files.sort()  # Ensure chronological order
            
            if self.verbose:
                print(f"\U0001f4ca Processing {len(frame_files)} frames...")
            
            # Run predictions on all frames
            predictions = self._predict_frames(frame_files, threshold, include_probabilities)
            
            # Create timestamps based on frame interval
            timestamps = [i * frame_interval_ms for i in range(len(predictions))]
            
            # Create DataFrame
            df = pd.DataFrame(predictions)
            df.insert(0, 'onset_milliseconds', timestamps)
            df.insert(1, 'frame_path', [os.path.basename(f) for f in frame_files])
            
            # Save to CSV
            Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_csv, index=False)
            
            # Calculate statistics
            stats = self._calculate_prediction_stats(df, include_probabilities)
            
            if self.verbose:
                print(f"\U0001f4be Predictions saved to: {output_csv}")
                self._print_prediction_stats(stats)
            
            return {
                'success': True,
                'output_csv': output_csv,
                'total_frames': len(frame_files),
                'predictions_shape': df.shape,
                'statistics': stats
            }
            
        except Exception as e:
            if self.verbose:
                print(f"\u274c Error processing video: {e}")
            return {
                'success': False,
                'error': str(e)
            }
        
        finally:
            # Clean up temporary frames
            if os.path.exists(temp_frames_dir):
                import shutil
                shutil.rmtree(temp_frames_dir)
                if self.verbose:
                    print(f"\U0001f9f9 Cleaned up temporary frames: {temp_frames_dir}")
    
    def _predict_frames(self, 
                    frame_files: List[str], 
                    threshold: float,
                    include_probabilities: bool) -> List[Dict[str, Any]]:
        """Run predictions on a list of frame files"""
        predictions = []
        
        for i, frame_path in enumerate(frame_files):
            if self.verbose and (i + 1) % 100 == 0:
                print(f"   Processing frame {i + 1}/{len(frame_files)}")
            
            frame_predictions = {}
            
            for column, model in self.loaded_models.items():
                try:
                    if hasattr(model, 'predict_image'):
                        result = model.predict_image(frame_path, threshold=threshold)
                        
                        # Handle the specific result format from your models
                        if isinstance(result, dict):
                            if 'confidence' in result:
                                # Format: {'predicted_class': 1, 'class_name': 'closeup', 'confidence': 0.9988, 'is_positive': True}
                                prediction = result.get('predicted_class', 0)
                                probability = result.get('confidence', 0.0)
                                result = {'prediction': prediction, 'probability': probability}
                            # If it already has 'prediction' and 'probability', use as-is
                        
                    elif hasattr(model, 'predict_frame'):
                        result = model.predict_frame(frame_path, threshold=threshold)
                    else:
                        # Fallback to manual prediction (this shouldn't be needed now)
                        result = self._manual_prediction_direct(model, frame_path, threshold)                    
                    # Extract prediction and probability
                    if isinstance(result, dict):
                        prediction = result.get('prediction', 0)
                        probability = result.get('probability', 0.0)
                    elif isinstance(result, tuple) and len(result) >= 2:
                        prediction = int(result[0])
                        probability = float(result[1])
                    elif isinstance(result, (int, bool)):
                        prediction = int(result)
                        probability = 1.0 if prediction else 0.0
                    elif isinstance(result, float):
                        probability = result
                        prediction = 1 if probability > threshold else 0
                    else:
                        prediction = 0
                        probability = 0.0
                    
                    # Store prediction
                    frame_predictions[column] = prediction
                    
                    # Store probability if requested
                    if include_probabilities:
                        frame_predictions[f'{column}_prob'] = probability
                
                except Exception as e:
                    if self.verbose and i < 3:  # Only show errors for first few frames
                        print(f"\u26a0\ufe0f Error predicting {column} for frame {i}: {e}")
                    
                    frame_predictions[column] = 0
                    if include_probabilities:
                        frame_predictions[f'{column}_prob'] = 0.0
            
            predictions.append(frame_predictions)
        
        return predictions
    
    def _manual_prediction_direct(self, model, frame_path: str, threshold: float):
        """Direct manual prediction using the model structure we discovered"""
        try:
            import torch
            from PIL import Image
            import torchvision.transforms as transforms
            
            # Load and preprocess image
            image = Image.open(frame_path).convert('RGB')
            img_size = getattr(model, 'img_size', 224)
            
            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            input_tensor = transform(image).unsqueeze(0)
            
            # Use the model we know exists
            net = model.model
            net.eval()
            
            with torch.no_grad():
                outputs = net(input_tensor)
                
                # Based on our inspection, it's single output with sigmoid
                if outputs.shape[-1] == 1:
                    probability = float(torch.sigmoid(outputs)[0])
                elif outputs.shape[-1] == 2:
                    # Two outputs - use softmax
                    probs = torch.softmax(outputs, dim=-1)
                    probability = float(probs[0, 1])
                else:
                    probability = 0.0
                
                prediction = 1 if probability > threshold else 0
                
                return {
                    'prediction': prediction,
                    'probability': probability
                }
        
        except Exception as e:
            return {'prediction': 0, 'probability': 0.0}
    
    
    def _calculate_prediction_stats(self, df: pd.DataFrame, include_probabilities: bool) -> Dict[str, Any]:
        """Calculate statistics from predictions"""
        stats = {
            'total_frames': len(df),
            'columns': {},
            'summary': {}
        }
        
        # Analyze each prediction column
        prediction_columns = [col for col in df.columns 
                            if col not in ['onset_milliseconds', 'frame_path'] 
                            and not col.endswith('_prob')]
        
        total_positive = 0
        for col in prediction_columns:
            positive_count = df[col].sum()
            positive_ratio = positive_count / len(df)
            
            stats['columns'][col] = {
                'positive_frames': int(positive_count),
                'negative_frames': int(len(df) - positive_count),
                'positive_ratio': float(positive_ratio)
            }
            
            if include_probabilities and f'{col}_prob' in df.columns:
                avg_prob = df[f'{col}_prob'].mean()
                stats['columns'][col]['average_probability'] = float(avg_prob)
            
            total_positive += positive_count
        
        # Overall summary
        stats['summary'] = {
            'total_prediction_columns': len(prediction_columns),
            'avg_positive_ratio': float(total_positive / (len(df) * len(prediction_columns))) if prediction_columns else 0,
            'duration_seconds': float(df['onset_milliseconds'].max() / 1000) if len(df) > 0 else 0
        }
        
        return stats
    
    def _print_prediction_stats(self, stats: Dict[str, Any]):
        """Print prediction statistics"""
        print(f"\n\U0001f4ca PREDICTION STATISTICS:")
        print(f"   Total frames: {stats['total_frames']:,}")
        print(f"   Duration: {stats['summary']['duration_seconds']:.1f} seconds")
        print(f"   Columns predicted: {stats['summary']['total_prediction_columns']}")
        print(f"   Average positive ratio: {stats['summary']['avg_positive_ratio']:.1%}")
        
        # Show top positive columns
        sorted_cols = sorted(stats['columns'].items(), 
                           key=lambda x: x[1]['positive_ratio'], reverse=True)
        
        print(f"\n\U0001f51d TOP ACTIVE COLUMNS:")
        for col, col_stats in sorted_cols[:5]:
            emoji = "\U0001f464" if col == 'has_faces' else "\U0001f3af"
            print(f"   {emoji} {col}: {col_stats['positive_ratio']:.1%} ({col_stats['positive_frames']:,} frames)")


class BatchPredictor:
    """Handles batch prediction on multiple videos"""
    
    def __init__(self, model_loader: ModelLoader, verbose: bool = True):
        """
        Initialize batch predictor
        
        Args:
            model_loader: Loaded ModelLoader instance
            verbose: Enable verbose output
        """
        self.model_loader = model_loader
        self.loaded_models = model_loader.get_loaded_models()
        self.verbose = verbose
        self.video_predictor = VideoPredictor(self.loaded_models, verbose)
    
    def predict_multiple_videos(self,
                               video_paths: List[str],
                               output_dir: str,
                               frame_interval_ms: int = 100,
                               threshold: float = 0.5,
                               include_probabilities: bool = True) -> Dict[str, Any]:
        """
        Predict on multiple videos and save individual CSV files
        
        Args:
            video_paths: List of video file paths
            output_dir: Directory to save CSV files
            frame_interval_ms: Frame extraction interval
            threshold: Classification threshold
            include_probabilities: Include probability columns
            
        Returns:
            Dictionary with batch results
        """
        if self.verbose:
            print(f"\U0001f3ac BATCH PREDICTION ON {len(video_paths)} VIDEOS")
            print(f"\U0001f4c1 Output directory: {output_dir}")
            print("="*60)
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        batch_results = {
            'successful': [],
            'failed': [],
            'total': len(video_paths),
            'statistics': {}
        }
        
        for i, video_path in enumerate(video_paths, 1):
            video_name = Path(video_path).stem
            output_csv = os.path.join(output_dir, f"{video_name}_predictions.csv")
            
            if self.verbose:
                print(f"\n\U0001f4f9 Processing video {i}/{len(video_paths)}: {video_name}")
            
            result = self.video_predictor.predict_video(
                video_path=video_path,
                output_csv=output_csv,
                frame_interval_ms=frame_interval_ms,
                threshold=threshold,
                include_probabilities=include_probabilities
            )
            
            if result['success']:
                batch_results['successful'].append({
                    'video': video_name,
                    'path': video_path,
                    'output_csv': output_csv,
                    'frames': result['total_frames'],
                    'statistics': result['statistics']
                })
            else:
                batch_results['failed'].append({
                    'video': video_name,
                    'path': video_path,
                    'error': result['error']
                })
        
        # Calculate batch statistics
        if batch_results['successful']:
            batch_results['statistics'] = self._calculate_batch_stats(batch_results['successful'])
        
        if self.verbose:
            self._print_batch_summary(batch_results)
        
        return batch_results
    
    def _calculate_batch_stats(self, successful_results: List[Dict]) -> Dict[str, Any]:
        """Calculate statistics across all successful predictions"""
        total_frames = sum(r['frames'] for r in successful_results)
        total_duration = sum(r['statistics']['summary']['duration_seconds'] for r in successful_results)
        
        # Aggregate column statistics
        all_columns = set()
        for result in successful_results:
            all_columns.update(result['statistics']['columns'].keys())
        
        aggregated_stats = {}
        for col in all_columns:
            total_positive = sum(r['statistics']['columns'].get(col, {}).get('positive_frames', 0) 
                               for r in successful_results)
            total_frames_for_col = sum(r['frames'] for r in successful_results 
                                     if col in r['statistics']['columns'])
            
            if total_frames_for_col > 0:
                aggregated_stats[col] = {
                    'total_positive_frames': total_positive,
                    'total_frames': total_frames_for_col,
                    'overall_positive_ratio': total_positive / total_frames_for_col
                }
        
        return {
            'total_videos': len(successful_results),
            'total_frames': total_frames,
            'total_duration_seconds': total_duration,
            'avg_frames_per_video': total_frames / len(successful_results),
            'column_statistics': aggregated_stats
        }
    
    def _print_batch_summary(self, batch_results: Dict[str, Any]):
        """Print batch processing summary"""
        print(f"\n{'='*60}")
        print(f"\U0001f4ca BATCH PREDICTION SUMMARY")
        print(f"{'='*60}")
        
        successful = len(batch_results['successful'])
        total = batch_results['total']
        
        print(f"\u2705 Successful: {successful}/{total} videos")
        
        if batch_results['failed']:
            print(f"\u274c Failed: {len(batch_results['failed'])} videos")
            for failed in batch_results['failed']:
                print(f"   \u2022 {failed['video']}: {failed['error']}")
        
        if 'statistics' in batch_results and batch_results['statistics']:
            stats = batch_results['statistics']
            print(f"\n\U0001f4c8 OVERALL STATISTICS:")
            print(f"   Total frames processed: {stats['total_frames']:,}")
            print(f"   Total duration: {stats['total_duration_seconds']:.1f} seconds")
            print(f"   Average frames per video: {stats['avg_frames_per_video']:.0f}")
            
            # Show most active columns across all videos
            if 'column_statistics' in stats:
                sorted_cols = sorted(stats['column_statistics'].items(),
                                   key=lambda x: x[1]['overall_positive_ratio'], reverse=True)
                
                print(f"\n\U0001f51d MOST ACTIVE COLUMNS ACROSS ALL VIDEOS:")
                for col, col_stats in sorted_cols[:10]:
                    emoji = "\U0001f464" if col == 'has_faces' else "\U0001f3af"
                    ratio = col_stats['overall_positive_ratio']
                    frames = col_stats['total_positive_frames']
                    print(f"   {emoji} {col}: {ratio:.1%} ({frames:,} total positive frames)")


def main():
    """Example usage of the prediction module"""
    
    # Example model paths (update these with your actual model paths)
    model_paths = {
        'has_faces': 'data/my_results/has_faces_classifier.pth',
        'c_happy_face': 'data/my_results/c_happy_face_classifier.pth',
        'c_sad_face': 'data/my_results/c_sad_face_classifier.pth',
        'collective': 'data/my_results/collective_classifier.pth'
    }
    
    # Load models
    print("\U0001f680 PREDICTION MODULE DEMO")
    print("="*50)
    
    loader = ModelLoader(model_paths, verbose=True)
    load_results = loader.load_models()
    
    if not load_results['loaded']:
        print("\u274c No models loaded successfully!")
        return
    
    # Single video prediction
    video_path = "data/clip01/in/clip1_MLP.mp4"  # Update with your video path
    output_csv = "predictions_output.csv"
    
    predictor = VideoPredictor(loader.get_loaded_models(), verbose=True)
    
    result = predictor.predict_video(
        video_path=video_path,
        output_csv=output_csv,
        frame_interval_ms=100,
        threshold=0.5,
        include_probabilities=True
    )
    
    if result['success']:
        print(f"\n\u2705 Single video prediction complete!")
        print(f"\U0001f4ca Results saved to: {result['output_csv']}")
    else:
        print(f"\n\u274c Single video prediction failed: {result['error']}")
    
    # Batch prediction example
    video_list = [
        "data/clip01/in/clip1_MLP.mp4",
        "data/clip02/in/clip2_AHKJ.mp4"
    ]  # Update with your video paths
    
    batch_predictor = BatchPredictor(loader, verbose=True)
    
    batch_results = batch_predictor.predict_multiple_videos(
        video_paths=video_list,
        output_dir="batch_predictions",
        frame_interval_ms=100,
        threshold=0.5,
        include_probabilities=True
    )
    
    print(f"\n\u2705 Batch prediction complete!")
    print(f"\U0001f4ca Processed {len(batch_results['successful'])} videos successfully")


if __name__ == "__main__":
    main()