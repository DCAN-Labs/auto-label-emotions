import cv2
import numpy as np
import os
import json
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import colorsys

from pytorch_cartoon_face_detector import BinaryClassifier

class VideoAnnotationSystem:
    def __init__(self, 
                 classifiers_dir: str,
                 output_dir: str = "annotated_videos",
                 font_scale: float = 0.6,
                 thickness: int = 2):
        """
        Initialize video annotation system
        
        Args:
            classifiers_dir: Directory containing trained classifier models
            output_dir: Directory to save annotated videos
            font_scale: Text font scale
            thickness: Text thickness
        """
        self.classifiers_dir = classifiers_dir
        self.output_dir = output_dir
        self.font_scale = font_scale
        self.thickness = thickness
        self.classifiers = {}
        self.classifier_info = {}
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Font and colors
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.colors = self._generate_distinct_colors()
        
        print(f"Video annotation system initialized")
        print(f"Output directory: {output_dir}")
    
    def _generate_distinct_colors(self, n_colors: int = 20) -> List[Tuple[int, int, int]]:
        """Generate visually distinct colors for different classifiers"""
        colors = []
        for i in range(n_colors):
            hue = i / n_colors
            saturation = 0.8
            value = 0.9
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            # Convert to BGR for OpenCV
            bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            colors.append(bgr)
        return colors
    
    def load_classifiers(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Load all trained classifiers from directory
        
        Args:
            verbose: Print loading progress
        
        Returns:
            Dictionary with loading results
        """
        print("\U0001f504 LOADING TRAINED CLASSIFIERS")
        print("-" * 40)
        
        model_files = list(Path(self.classifiers_dir).glob("*_classifier.pth"))
        
        if not model_files:
            print(f"\u274c No classifier files found in {self.classifiers_dir}")
            return {'error': 'No classifier files found'}
        
        loading_results = {}
        successful_loads = 0
        
        for model_file in model_files:
            classifier_name = model_file.stem.replace('_classifier', '')
            
            if verbose:
                print(f"Loading {classifier_name}...")
            
            try:
                # Create a generic binary classifier
                classifier = BinaryClassifier(
                    task_name=f"{classifier_name}_classification",
                    class_names={0: f'not_{classifier_name}', 1: classifier_name}
                )
                
                # Load the model
                classifier.load_model(str(model_file))
                
                # Store classifier and info
                self.classifiers[classifier_name] = classifier
                self.classifier_info[classifier_name] = {
                    'model_path': str(model_file),
                    'class_names': classifier.class_names,
                    'task_name': classifier.task_name
                }
                
                successful_loads += 1
                
                if verbose:
                    print(f"  \u2705 {classifier_name}: {classifier.class_names}")
                
                loading_results[classifier_name] = {
                    'success': True,
                    'class_names': classifier.class_names
                }
                
            except Exception as e:
                if verbose:
                    print(f"  \u274c Error loading {classifier_name}: {e}")
                loading_results[classifier_name] = {
                    'success': False,
                    'error': str(e)
                }
        
        print(f"\n\u2705 Successfully loaded {successful_loads}/{len(model_files)} classifiers")
        return loading_results
    
    def predict_frame_all_classifiers(self, frame: np.ndarray, 
                                    threshold: float = 0.5) -> Dict[str, Any]:
        """
        Run all classifiers on a single frame
        
        Args:
            frame: Input frame (BGR format)
            threshold: Confidence threshold for predictions
        
        Returns:
            Dictionary with all classifier predictions
        """
        predictions = {}
        
        for classifier_name, classifier in self.classifiers.items():
            try:
                result = classifier.predict_frame(frame, threshold)
                predictions[classifier_name] = {
                    'class': result['class_name'],
                    'confidence': result['confidence'],
                    'is_positive': result['is_positive'],
                    'predicted_class': result['predicted_class']
                }
            except Exception as e:
                predictions[classifier_name] = {
                    'error': str(e),
                    'class': 'error',
                    'confidence': 0.0,
                    'is_positive': False
                }
        
        return predictions
    
    def create_annotation_layout(self, frame_shape: Tuple[int, int], 
                               n_classifiers: int) -> Dict[str, Any]:
        """
        Create layout for annotations on video frame
        
        Args:
            frame_shape: (height, width) of video frame
            n_classifiers: Number of classifiers to display
        
        Returns:
            Layout configuration
        """
        height, width = frame_shape[:2]
        
        # Configuration
        margin = 10
        line_height = 25
        col_width = 300
        
        # Determine layout
        if width >= 1200:  # Wide frame - use two columns
            n_cols = 2
            items_per_col = (n_classifiers + 1) // 2
        else:  # Narrow frame - use one column
            n_cols = 1
            items_per_col = n_classifiers
        
        # Calculate positions
        layout = {
            'n_cols': n_cols,
            'items_per_col': items_per_col,
            'line_height': line_height,
            'margin': margin,
            'col_width': col_width,
            'positions': []
        }
        
        # Generate text positions
        for i, classifier_name in enumerate(self.classifiers.keys()):
            col = i // items_per_col
            row = i % items_per_col
            
            x = margin + col * col_width
            y = margin + (row + 1) * line_height
            
            layout['positions'].append({
                'classifier': classifier_name,
                'x': x,
                'y': y,
                'color_index': i % len(self.colors)
            })
        
        return layout
    
    def draw_predictions_on_frame(self, frame: np.ndarray, 
                                predictions: Dict[str, Any],
                                layout: Dict[str, Any],
                                show_confidence: bool = True,
                                show_only_positive: bool = False) -> np.ndarray:
        """
        Draw prediction annotations on frame
        
        Args:
            frame: Input frame
            predictions: Classifier predictions
            layout: Annotation layout
            show_confidence: Show confidence scores
            show_only_positive: Only show positive predictions
        
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        # Add semi-transparent background for text
        overlay = frame.copy()
        
        for pos_info in layout['positions']:
            classifier_name = pos_info['classifier']
            
            if classifier_name not in predictions:
                continue
            
            pred = predictions[classifier_name]
            
            # Skip if showing only positive and this is negative
            if show_only_positive and not pred.get('is_positive', False):
                continue
            
            # Skip if there was an error
            if 'error' in pred:
                continue
            
            # Prepare text
            class_name = pred['class']
            confidence = pred['confidence']
            
            if show_confidence:
                text = f"{classifier_name}: {class_name} ({confidence:.2f})"
            else:
                text = f"{classifier_name}: {class_name}"
            
            # Color coding
            color = self.colors[pos_info['color_index']]
            if pred.get('is_positive', False):
                # Brighter color for positive predictions
                color = tuple(min(255, int(c * 1.2)) for c in color)
            else:
                # Dimmer color for negative predictions
                color = tuple(int(c * 0.7) for c in color)
            
            # Draw text with background
            x, y = pos_info['x'], pos_info['y']
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                text, self.font, self.font_scale, self.thickness
            )
            
            # Draw background rectangle
            cv2.rectangle(
                overlay,
                (x - 2, y - text_height - 2),
                (x + text_width + 2, y + baseline + 2),
                (0, 0, 0),
                -1
            )
            
            # Draw text
            cv2.putText(
                annotated_frame,
                text,
                (x, y),
                self.font,
                self.font_scale,
                color,
                self.thickness,
                cv2.LINE_AA
            )
        
        # Blend overlay for semi-transparent background
        annotated_frame = cv2.addWeighted(annotated_frame, 0.8, overlay, 0.2, 0)
        
        # Add title
        title = f"Multi-Classifier Analysis ({len(predictions)} classifiers)"
        cv2.putText(
            annotated_frame,
            title,
            (10, 25),
            self.font,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        
        return annotated_frame
    
    def annotate_video(self, 
                      video_path: str,
                      output_filename: Optional[str] = None,
                      confidence_threshold: float = 0.5,
                      show_confidence: bool = True,
                      show_only_positive: bool = False,
                      frame_skip: int = 1,
                      verbose: bool = True) -> Dict[str, Any]:
        """
        Annotate video with classifier predictions
        
        Args:
            video_path: Path to input video
            output_filename: Output filename (auto-generated if None)
            confidence_threshold: Threshold for predictions
            show_confidence: Show confidence scores
            show_only_positive: Only show positive predictions
            frame_skip: Process every nth frame (for speed)
            verbose: Show progress
        
        Returns:
            Processing results
        """
        if not self.classifiers:
            return {'error': 'No classifiers loaded. Call load_classifiers() first.'}
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'error': f'Could not open video: {video_path}'}
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Generate output filename
        if output_filename is None:
            video_name = Path(video_path).stem
            output_filename = f"{video_name}_annotated.mp4"
        
        output_path = os.path.join(self.output_dir, output_filename)
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Create layout
        layout = self.create_annotation_layout((height, width), len(self.classifiers))
        
        if verbose:
            print(f"\n\U0001f3ac ANNOTATING VIDEO: {Path(video_path).name}")
            print(f"   Resolution: {width}x{height}")
            print(f"   FPS: {fps}")
            print(f"   Total frames: {total_frames}")
            print(f"   Classifiers: {len(self.classifiers)}")
            print(f"   Output: {output_filename}")
        
        # Process video
        frame_count = 0
        processed_frames = 0
        all_predictions = []
        
        with tqdm(total=total_frames, desc="Annotating video", disable=not verbose) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_skip == 0:
                    # Get predictions for this frame
                    predictions = self.predict_frame_all_classifiers(
                        frame, confidence_threshold
                    )
                    
                    # Annotate frame
                    annotated_frame = self.draw_predictions_on_frame(
                        frame, predictions, layout, 
                        show_confidence, show_only_positive
                    )
                    
                    # Store predictions with timestamp
                    timestamp_ms = (frame_count / fps) * 1000
                    frame_data = {
                        'frame_number': frame_count,
                        'timestamp_ms': timestamp_ms,
                        'predictions': predictions
                    }
                    all_predictions.append(frame_data)
                    
                    processed_frames += 1
                else:
                    # Use original frame if skipping
                    annotated_frame = frame
                
                # Write frame
                out.write(annotated_frame)
                frame_count += 1
                pbar.update(1)
        
        # Cleanup
        cap.release()
        out.release()
        
        # Generate statistics
        stats = self._generate_video_statistics(all_predictions)
        
        results = {
            'input_video': video_path,
            'output_video': output_path,
            'total_frames': frame_count,
            'processed_frames': processed_frames,
            'frame_skip': frame_skip,
            'fps': fps,
            'duration_seconds': frame_count / fps,
            'classifiers_used': list(self.classifiers.keys()),
            'statistics': stats,
            'success': True
        }
        
        # Save detailed results
        results_file = output_path.replace('.mp4', '_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        if verbose:
            print(f"\u2705 Video annotation complete!")
            print(f"   Output video: {output_path}")
            print(f"   Results saved: {results_file}")
            self._print_video_statistics(stats)
        
        return results
    
    def _generate_video_statistics(self, all_predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate statistics from video predictions"""
        stats = {}
        
        for classifier_name in self.classifiers.keys():
            positive_count = 0
            total_count = 0
            confidence_sum = 0.0
            confidence_values = []
            
            for frame_data in all_predictions:
                if classifier_name in frame_data['predictions']:
                    pred = frame_data['predictions'][classifier_name]
                    if 'error' not in pred:
                        total_count += 1
                        confidence_sum += pred['confidence']
                        confidence_values.append(pred['confidence'])
                        
                        if pred.get('is_positive', False):
                            positive_count += 1
            
            if total_count > 0:
                stats[classifier_name] = {
                    'positive_frames': positive_count,
                    'total_frames': total_count,
                    'positive_ratio': positive_count / total_count,
                    'average_confidence': confidence_sum / total_count,
                    'min_confidence': min(confidence_values),
                    'max_confidence': max(confidence_values)
                }
            else:
                stats[classifier_name] = {
                    'error': 'No valid predictions'
                }
        
        return stats
    
    def _print_video_statistics(self, stats: Dict[str, Any]):
        """Print video analysis statistics"""
        print("\n\U0001f4ca VIDEO ANALYSIS STATISTICS:")
        print("-" * 50)
        
        for classifier_name, classifier_stats in stats.items():
            if 'error' in classifier_stats:
                print(f"\u274c {classifier_name}: {classifier_stats['error']}")
                continue
            
            positive_ratio = classifier_stats['positive_ratio']
            avg_confidence = classifier_stats['average_confidence']
            
            print(f"\U0001f3af {classifier_name.upper()}:")
            print(f"   Positive frames: {classifier_stats['positive_frames']}/{classifier_stats['total_frames']} ({positive_ratio:.1%})")
            print(f"   Avg confidence: {avg_confidence:.3f}")
            print(f"   Confidence range: {classifier_stats['min_confidence']:.3f} - {classifier_stats['max_confidence']:.3f}")
    
    def create_summary_video(self, 
                           video_results: List[Dict[str, Any]],
                           output_filename: str = "summary_analysis.mp4") -> str:
        """
        Create a summary video showing statistics across multiple videos
        
        Args:
            video_results: List of video analysis results
            output_filename: Output filename for summary video
        
        Returns:
            Path to created summary video
        """
        # This would create a summary visualization
        # Implementation would depend on specific requirements
        pass
    
    def batch_annotate_videos(self, 
                            video_list: List[str],
                            **annotation_kwargs) -> List[Dict[str, Any]]:
        """
        Annotate multiple videos in batch
        
        Args:
            video_list: List of video file paths
            **annotation_kwargs: Arguments passed to annotate_video
        
        Returns:
            List of annotation results
        """
        results = []
        
        print(f"\U0001f3ac BATCH ANNOTATION: {len(video_list)} videos")
        print("=" * 50)
        
        for i, video_path in enumerate(video_list):
            print(f"\n[{i+1}/{len(video_list)}] Processing: {Path(video_path).name}")
            
            try:
                result = self.annotate_video(video_path, **annotation_kwargs)
                results.append(result)
                
                if result.get('success', False):
                    print(f"\u2705 Success: {result['output_video']}")
                else:
                    print(f"\u274c Failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"\u274c Error processing {video_path}: {e}")
                results.append({
                    'input_video': video_path,
                    'error': str(e),
                    'success': False
                })
        
        successful = sum(1 for r in results if r.get('success', False))
        print(f"\n\U0001f4ca BATCH SUMMARY: {successful}/{len(video_list)} videos processed successfully")
        
        return results

def main():
    """Example usage of the video annotation system"""
    
    # Initialize annotation system
    annotator = VideoAnnotationSystem(
        classifiers_dir="data/multi_column_results",
        output_dir="annotated_videos",
        font_scale=0.6,
        thickness=2
    )
    
    # Load all trained classifiers
    loading_results = annotator.load_classifiers(verbose=True)
    
    if not annotator.classifiers:
        print("\u274c No classifiers loaded successfully!")
        return
    
    # Annotate a single video
    video_path = "data/clip01/in/clip1_MLP.mp4"
    
    # Option 1: Standard annotation with all predictions
    # result = annotator.annotate_video(
    #     video_path=video_path,
    #     output_filename="clip1_full_analysis.mp4",
    #     confidence_threshold=0.5,
    #     show_confidence=True,
    #     show_only_positive=False,
    #     frame_skip=1,  # Process every frame
    #     verbose=True
    # )
    
    # Option 2: Show only positive predictions (cleaner display)
    result2 = annotator.annotate_video(
        video_path=video_path,
        output_filename="clip1_positive_only.mp4",
        confidence_threshold=0.7,
        show_confidence=True,
        show_only_positive=True,  # Only show positive detections
        frame_skip=2,  # Process every other frame for speed
        verbose=True
    )
    
    # Option 3: Batch process multiple videos
    # video_list = [
    #     "data/clip01/in/clip1_MLP.mp4",
    #     "data/clip02/in/clip2_AHKJ.mp4",
    #     "data/clip03/in/clip3_MLP.mp4"
    # ]
    
    # batch_results = annotator.batch_annotate_videos(
    #     video_list=video_list,
    #     confidence_threshold=0.6,
    #     show_only_positive=True,
    #     frame_skip=3,  # Every 3rd frame for faster processing
    #     verbose=True
    # )
    
    print("\n\U0001f389 Video annotation complete!")
    print(f"Check the 'annotated_videos' directory for results.")

if __name__ == "__main__":
    main()