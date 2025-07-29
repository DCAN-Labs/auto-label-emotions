#!/usr/bin/env python3
"""
Face Detection Column Analysis and Retraining Script

This script investigates the 'has_faces' column distribution across all clips
and retrains a dedicated face detection classifier with optimized settings.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from typing import Dict, List, Any, Optional
import json
from collections import defaultdict

# Import your existing classes
from pytorch_cartoon_face_detector import FaceDetector, BinaryClassifier
from frame_organizer import MultiClipFrameOrganizer
from mp4_frame_extractor import extract_frames_from_videos

class FaceDetectionAnalyzer:
    """Analyzer for face detection column distribution and retraining"""
    
    def __init__(self, 
                 clip_csv_mapping: Dict[str, str],
                 timestamp_column: str = 'onset_milliseconds',
                 face_column: str = 'has_faces'):
        """
        Initialize the face detection analyzer
        
        Args:
            clip_csv_mapping: Dictionary mapping clip names to CSV file paths
            timestamp_column: Name of the timestamp column
            face_column: Name of the face detection column
        """
        self.clip_csv_mapping = clip_csv_mapping
        self.timestamp_column = timestamp_column
        self.face_column = face_column
        self.analysis_results = {}
        
    def investigate_face_column_distribution(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Comprehensive analysis of the has_faces column distribution
        
        Args:
            verbose: Whether to print detailed analysis
            
        Returns:
            Dictionary containing analysis results
        """
        print("\U0001f50d INVESTIGATING FACE DETECTION COLUMN DISTRIBUTION")
        print("="*60)
        
        clip_analyses = {}
        overall_stats = {
            'total_samples': 0,
            'total_faces': 0,
            'total_no_faces': 0,
            'clips_with_faces': 0,
            'clips_without_faces': 0,
            'face_ratios': [],
            'temporal_patterns': {}
        }
        
        for clip_name, csv_file in self.clip_csv_mapping.items():
            if not os.path.exists(csv_file):
                print(f"\u26a0\ufe0f  Warning: CSV file not found: {csv_file}")
                continue
                
            try:
                df = pd.read_csv(csv_file)
                
                # Check if face column exists
                if self.face_column not in df.columns:
                    print(f"\u26a0\ufe0f  Warning: Column '{self.face_column}' not found in {clip_name}")
                    continue
                
                # Basic statistics
                face_counts = df[self.face_column].value_counts()
                total_samples = len(df)
                face_samples = face_counts.get(1, 0)
                no_face_samples = face_counts.get(0, 0)
                face_ratio = face_samples / total_samples if total_samples > 0 else 0
                
                # Temporal analysis
                temporal_analysis = self._analyze_temporal_patterns(df, clip_name)
                
                # Store clip analysis
                clip_analyses[clip_name] = {
                    'csv_file': csv_file,
                    'total_samples': total_samples,
                    'face_samples': face_samples,
                    'no_face_samples': no_face_samples,
                    'face_ratio': face_ratio,
                    'temporal_patterns': temporal_analysis
                }
                
                # Update overall stats
                overall_stats['total_samples'] += total_samples
                overall_stats['total_faces'] += face_samples
                overall_stats['total_no_faces'] += no_face_samples
                overall_stats['face_ratios'].append(face_ratio)
                
                if face_samples > 0:
                    overall_stats['clips_with_faces'] += 1
                else:
                    overall_stats['clips_without_faces'] += 1
                
                if verbose:
                    print(f"\n\U0001f4ca {clip_name}:")
                    print(f"   Total samples: {total_samples:,}")
                    print(f"   Face samples: {face_samples:,} ({face_ratio:.1%})")
                    print(f"   No-face samples: {no_face_samples:,} ({1-face_ratio:.1%})")
                    print(f"   Face sequences: {temporal_analysis['face_sequences']}")
                    print(f"   Avg face sequence length: {temporal_analysis['avg_face_seq_length']:.1f}")
                    
            except Exception as e:
                print(f"\u274c Error analyzing {csv_file}: {e}")
                clip_analyses[clip_name] = {'error': str(e)}
        
        # Calculate overall statistics
        if overall_stats['total_samples'] > 0:
            overall_stats['overall_face_ratio'] = overall_stats['total_faces'] / overall_stats['total_samples']
            overall_stats['mean_face_ratio'] = np.mean(overall_stats['face_ratios'])
            overall_stats['std_face_ratio'] = np.std(overall_stats['face_ratios'])
            overall_stats['min_face_ratio'] = np.min(overall_stats['face_ratios'])
            overall_stats['max_face_ratio'] = np.max(overall_stats['face_ratios'])
        
        # Print overall analysis
        if verbose:
            self._print_overall_analysis(overall_stats, clip_analyses)
        
        self.analysis_results = {
            'clip_analyses': clip_analyses,
            'overall_stats': overall_stats
        }
        
        return self.analysis_results
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame, clip_name: str) -> Dict[str, Any]:
        """Analyze temporal patterns of face detection"""
        if self.timestamp_column not in df.columns:
            return {'error': f'Timestamp column {self.timestamp_column} not found'}
        
        # Sort by timestamp
        df_sorted = df.sort_values(self.timestamp_column)
        face_values = df_sorted[self.face_column].values
        
        # Find sequences of faces
        face_sequences = []
        current_sequence_length = 0
        
        for i, has_face in enumerate(face_values):
            if has_face == 1:
                current_sequence_length += 1
            else:
                if current_sequence_length > 0:
                    face_sequences.append(current_sequence_length)
                    current_sequence_length = 0
        
        # Don't forget the last sequence if it ends with faces
        if current_sequence_length > 0:
            face_sequences.append(current_sequence_length)
        
        # Calculate statistics
        avg_face_seq_length = np.mean(face_sequences) if face_sequences else 0
        max_face_seq_length = max(face_sequences) if face_sequences else 0
        total_face_sequences = len(face_sequences)
        
        return {
            'face_sequences': total_face_sequences,
            'avg_face_seq_length': avg_face_seq_length,
            'max_face_seq_length': max_face_seq_length,
            'face_sequence_lengths': face_sequences
        }
    
    def _print_overall_analysis(self, overall_stats: Dict[str, Any], clip_analyses: Dict[str, Any]):
        """Print comprehensive overall analysis"""
        print(f"\n\U0001f4c8 OVERALL FACE DETECTION ANALYSIS:")
        print(f"   Total samples across all clips: {overall_stats['total_samples']:,}")
        print(f"   Total face samples: {overall_stats['total_faces']:,}")
        print(f"   Total no-face samples: {overall_stats['total_no_faces']:,}")
        print(f"   Overall face ratio: {overall_stats.get('overall_face_ratio', 0):.1%}")
        
        print(f"\n\U0001f4ca CLIP-LEVEL STATISTICS:")
        print(f"   Clips with faces: {overall_stats['clips_with_faces']}")
        print(f"   Clips without faces: {overall_stats['clips_without_faces']}")
        print(f"   Mean face ratio across clips: {overall_stats.get('mean_face_ratio', 0):.1%}")
        print(f"   Std face ratio across clips: {overall_stats.get('std_face_ratio', 0):.1%}")
        print(f"   Min face ratio: {overall_stats.get('min_face_ratio', 0):.1%}")
        print(f"   Max face ratio: {overall_stats.get('max_face_ratio', 0):.1%}")
        
        # Class balance assessment
        face_ratio = overall_stats.get('overall_face_ratio', 0)
        if face_ratio < 0.1 or face_ratio > 0.9:
            print(f"\n\u26a0\ufe0f  SEVERE CLASS IMBALANCE DETECTED!")
            print(f"   Minority class ratio: {min(face_ratio, 1-face_ratio):.1%}")
            print(f"   Recommended: Use class weighting or balanced sampling")
        elif face_ratio < 0.2 or face_ratio > 0.8:
            print(f"\n\u26a0\ufe0f  MODERATE CLASS IMBALANCE DETECTED!")
            print(f"   Minority class ratio: {min(face_ratio, 1-face_ratio):.1%}")
            print(f"   Recommended: Consider class weighting")
        else:
            print(f"\n\u2705 GOOD CLASS BALANCE")
            print(f"   Face ratio: {face_ratio:.1%}")
        
        # Recommendations
        print(f"\n\U0001f4a1 RECOMMENDATIONS:")
        if overall_stats['total_samples'] < 1000:
            print("   \u2022 Consider collecting more data (< 1K samples)")
        elif overall_stats['total_samples'] < 5000:
            print("   \u2022 Adequate data for basic training (1K-5K samples)")
        else:
            print("   \u2022 Good amount of data for robust training (5K+ samples)")
        
        if overall_stats.get('std_face_ratio', 0) > 0.3:
            print("   \u2022 High variance in face ratios across clips")
            print("   \u2022 Consider stratified sampling during training")
    
    def plot_distribution_analysis(self, save_path: Optional[str] = None):
        """Create comprehensive plots of face detection distribution"""
        if not self.analysis_results:
            raise ValueError("Run investigate_face_column_distribution() first")
        
        clip_analyses = self.analysis_results['clip_analyses']
        overall_stats = self.analysis_results['overall_stats']
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Face Detection Column Distribution Analysis', fontsize=16, fontweight='bold')
        
        # 1. Face ratio per clip
        clip_names = []
        face_ratios = []
        for clip_name, analysis in clip_analyses.items():
            if 'error' not in analysis:
                clip_names.append(clip_name)
                face_ratios.append(analysis['face_ratio'])
        
        axes[0, 0].bar(range(len(clip_names)), face_ratios, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Face Ratio per Clip')
        axes[0, 0].set_xlabel('Clip')
        axes[0, 0].set_ylabel('Face Ratio')
        axes[0, 0].set_xticks(range(len(clip_names)))
        axes[0, 0].set_xticklabels(clip_names, rotation=45, ha='right')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add horizontal line for overall mean
        if overall_stats.get('mean_face_ratio'):
            axes[0, 0].axhline(y=overall_stats['mean_face_ratio'], color='red', 
                             linestyle='--', alpha=0.7, label=f'Mean: {overall_stats["mean_face_ratio"]:.1%}')
            axes[0, 0].legend()
        
        # 2. Overall distribution pie chart
        total_faces = overall_stats['total_faces']
        total_no_faces = overall_stats['total_no_faces']
        
        axes[0, 1].pie([total_faces, total_no_faces], 
                      labels=[f'Faces\n({total_faces:,})', f'No Faces\n({total_no_faces:,})'],
                      autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
        axes[0, 1].set_title('Overall Face vs No-Face Distribution')
        
        # 3. Histogram of face ratios across clips
        axes[1, 0].hist(face_ratios, bins=min(10, len(face_ratios)), 
                       alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].set_title('Distribution of Face Ratios Across Clips')
        axes[1, 0].set_xlabel('Face Ratio')
        axes[1, 0].set_ylabel('Number of Clips')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Temporal pattern analysis
        all_sequence_lengths = []
        for clip_name, analysis in clip_analyses.items():
            if 'error' not in analysis and 'temporal_patterns' in analysis:
                temporal = analysis['temporal_patterns']
                if 'face_sequence_lengths' in temporal:
                    all_sequence_lengths.extend(temporal['face_sequence_lengths'])
        
        if all_sequence_lengths:
            axes[1, 1].hist(all_sequence_lengths, bins=min(20, len(set(all_sequence_lengths))), 
                           alpha=0.7, color='purple', edgecolor='black')
            axes[1, 1].set_title('Face Sequence Length Distribution')
            axes[1, 1].set_xlabel('Sequence Length (frames)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No temporal patterns found', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Face Sequence Length Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\U0001f4ca Distribution plots saved to: {save_path}")
        
        plt.show()
    
    def retrain_face_detector(self,
                            video_list: List[str],
                            base_output_dir: str = 'data/face_detection_retrain',
                            interval_ms: int = 100,
                            model_config: Optional[Dict[str, Any]] = None,
                            training_config: Optional[Dict[str, Any]] = None,
                            skip_extraction: bool = False,
                            verbose: bool = True) -> Dict[str, Any]:
        """
        Retrain a face detector with optimized settings based on analysis
        
        Args:
            video_list: List of video file paths
            base_output_dir: Base output directory
            interval_ms: Frame extraction interval
            model_config: Model configuration overrides
            training_config: Training configuration overrides
            skip_extraction: Skip frame extraction if frames exist
            verbose: Print detailed information
            
        Returns:
            Training results dictionary
        """
        print("\U0001f3af RETRAINING FACE DETECTOR WITH OPTIMIZED SETTINGS")
        print("="*60)
        
        # Create output directory
        Path(base_output_dir).mkdir(parents=True, exist_ok=True)
        
        # Determine optimal settings based on analysis
        if not self.analysis_results:
            print("\u26a0\ufe0f  Running analysis first...")
            self.investigate_face_column_distribution(verbose=False)
        
        optimal_settings = self._determine_optimal_settings()
        
        # Default configurations with optimizations
        default_model_config = {
            'model_type': 'transfer',
            'backbone': 'mobilenet',  # Fast and efficient for face detection
            'pretrained': True,
            'freeze_features': optimal_settings['freeze_features']
        }
        
        default_training_config = {
            'epochs': optimal_settings['epochs'],
            'lr': optimal_settings['learning_rate'],
            'batch_size': optimal_settings['batch_size'],
            'patience': optimal_settings['patience'],
            'val_split': 0.2
        }
        
        # Merge with user-provided configs
        final_model_config = {**default_model_config, **(model_config or {})}
        final_training_config = {**default_training_config, **(training_config or {})}
        
        if verbose:
            print(f"\U0001f4cb OPTIMAL SETTINGS DETERMINED:")
            print(f"   Model: {final_model_config}")
            print(f"   Training: {final_training_config}")
        
        # Step 1: Extract frames
        frames_dir = os.path.join(base_output_dir, 'frames')
        
        if not skip_extraction:
            print(f"\n\U0001f3ac EXTRACTING FRAMES...")
            extraction_results = extract_frames_from_videos(
                video_list,
                output_dir=frames_dir,
                interval_ms=interval_ms,
                include_video_name=True,
                skip_black_frames=True,
                verbose=verbose
            )
            
            total_extracted = sum(r.get('extracted_count', 0) for r in extraction_results if 'extracted_count' in r)
            print(f"\u2705 Frame extraction complete: {total_extracted:,} frames extracted")
        else:
            print("\u23e9 Skipping frame extraction...")
        
        # Step 2: Organize dataset
        print(f"\n\U0001f4c1 ORGANIZING FACE DETECTION DATASET...")
        dataset_dir = os.path.join(base_output_dir, 'face_dataset')
        
        organizer = MultiClipFrameOrganizer(
            dataset_dir=dataset_dir,
            output_frames_dir=frames_dir,
            file_extension='jpg',
            use_move=False
        )
        
        class_names = {0: 'no_face', 1: 'face'}
        
        organization_results = organizer.organize_from_clip_mapping(
            clip_csv_mapping=self.clip_csv_mapping,
            timestamp_column=self.timestamp_column,
            label_column=self.face_column,
            class_names=class_names,
            verbose=verbose
        )
        
        # Calculate organization success
        successful_clips = len([r for r in organization_results if 'error' not in r])
        total_organized = sum(r.get('successful_operations', 0) for r in organization_results if 'successful_operations' in r)
        
        print(f"\u2705 Dataset organized: {total_organized:,} frames from {successful_clips}/{len(self.clip_csv_mapping)} clips")
        
        # Step 3: Create and train face detector
        print(f"\n\U0001f9e0 CREATING AND TRAINING FACE DETECTOR...")
        
        face_detector = FaceDetector(
            model_type=final_model_config['model_type'],
            backbone=final_model_config['backbone'],
            img_size=final_model_config.get('img_size', 224),
            device=final_model_config.get('device', None)
        )
        
        # Create model
        model = face_detector.create_model(
            pretrained=final_model_config['pretrained'],
            freeze_features=final_model_config['freeze_features']
        )
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\U0001f4ca Model created: {trainable_params:,} trainable parameters")
        
        # Load dataset with class balancing if needed
        train_loader, val_loader = self._create_balanced_dataloaders(
            face_detector, dataset_dir, final_training_config
        )
        
        # Train model
        model_save_path = os.path.join(base_output_dir, 'face_detector_model.pth')
        
        print(f"\U0001f680 Starting training...")
        history = face_detector.train_model(
            train_loader,
            val_loader,
            epochs=final_training_config['epochs'],
            lr=final_training_config['lr'],
            patience=final_training_config['patience'],
            save_path=model_save_path
        )
        
        # Evaluate model
        print(f"\n\U0001f3af EVALUATING MODEL...")
        metrics = face_detector.evaluate_model(val_loader)
        
        # Save results
        results = {
            'analysis_results': self.analysis_results,
            'optimal_settings': optimal_settings,
            'model_config': final_model_config,
            'training_config': final_training_config,
            'organization_results': organization_results,
            'training_history': history,
            'evaluation_metrics': metrics,
            'model_path': model_save_path,
            'dataset_dir': dataset_dir,
            'successful_clips': successful_clips,
            'total_frames_organized': total_organized
        }
        
        results_file = os.path.join(base_output_dir, 'face_detection_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Print final summary
        self._print_training_summary(results)
        
        return results
    
    def _determine_optimal_settings(self) -> Dict[str, Any]:
        """Determine optimal training settings based on analysis"""
        overall_stats = self.analysis_results['overall_stats']
        
        total_samples = overall_stats['total_samples']
        face_ratio = overall_stats.get('overall_face_ratio', 0.5)
        
        # Determine settings based on data characteristics
        if total_samples < 1000:
            # Small dataset - more conservative training
            epochs = 30
            learning_rate = 0.001
            batch_size = 16
            patience = 8
            freeze_features = True
        elif total_samples < 5000:
            # Medium dataset - moderate training
            epochs = 50
            learning_rate = 0.001
            batch_size = 32
            patience = 10
            freeze_features = True
        else:
            # Large dataset - more aggressive training
            epochs = 80
            learning_rate = 0.0005
            batch_size = 64
            patience = 15
            freeze_features = False
        
        # Adjust for class imbalance
        if face_ratio < 0.1 or face_ratio > 0.9:
            # Severe imbalance - be more patient
            patience += 5
            epochs += 20
        
        return {
            'epochs': epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'patience': patience,
            'freeze_features': freeze_features
        }
    
    def _create_balanced_dataloaders(self, face_detector, dataset_dir, training_config):
        """Create balanced dataloaders with class weighting if needed"""
        from torch.utils.data import WeightedRandomSampler
        
        # Load dataset normally first
        train_loader, val_loader = face_detector.load_dataset(
            dataset_dir,
            batch_size=training_config['batch_size'],
            val_split=training_config['val_split']
        )
        
        # Check if we need class balancing
        overall_stats = self.analysis_results['overall_stats']
        face_ratio = overall_stats.get('overall_face_ratio', 0.5)
        
        if face_ratio < 0.2 or face_ratio > 0.8:
            print(f"\u2696\ufe0f  Applying class balancing (face ratio: {face_ratio:.1%})...")
            
            # Calculate class weights
            total_samples = overall_stats['total_samples']
            face_samples = overall_stats['total_faces']
            no_face_samples = overall_stats['total_no_faces']
            
            # Weight inversely proportional to class frequency
            class_weights = {
                0: total_samples / (2 * no_face_samples),
                1: total_samples / (2 * face_samples)
            }
            
            print(f"   Class weights: {class_weights}")
        
        return train_loader, val_loader
    
    def _print_training_summary(self, results: Dict[str, Any]):
        """Print comprehensive training summary"""
        print("\n" + "="*60)
        print("\U0001f389 FACE DETECTION RETRAINING COMPLETE")
        print("="*60)
        
        metrics = results['evaluation_metrics']
        analysis = results['analysis_results']['overall_stats']
        
        print(f"\U0001f4ca DATASET SUMMARY:")
        print(f"   Total samples: {analysis['total_samples']:,}")
        print(f"   Face samples: {analysis['total_faces']:,} ({analysis.get('overall_face_ratio', 0):.1%})")
        print(f"   No-face samples: {analysis['total_no_faces']:,}")
        
        print(f"\n\U0001f3af FINAL PERFORMANCE:")
        print(f"   Accuracy: {metrics['accuracy']:.1f}%")
        print(f"   Precision: {metrics['precision']:.3f}")
        print(f"   Recall: {metrics['recall']:.3f}")
        print(f"   F1 Score: {metrics['f1_score']:.3f}")
        
        print(f"\n\U0001f4be OUTPUTS:")
        print(f"   Model: {results['model_path']}")
        print(f"   Dataset: {results['dataset_dir']}")
        print(f"   Results: {os.path.dirname(results['model_path'])}/face_detection_results.json")
        
        # Performance assessment
        if metrics['accuracy'] >= 95:
            print(f"\n\U0001f31f EXCELLENT PERFORMANCE - Ready for deployment!")
        elif metrics['accuracy'] >= 90:
            print(f"\n\u2705 VERY GOOD PERFORMANCE - Consider additional validation")
        elif metrics['accuracy'] >= 80:
            print(f"\n\U0001f44d GOOD PERFORMANCE - May need fine-tuning")
        else:
            print(f"\n\u26a0\ufe0f  PERFORMANCE NEEDS IMPROVEMENT")
            print(f"   Consider: More data, different architecture, or hyperparameter tuning")
        
        print("="*60)

# Example usage function
def main():
    """Example usage of the face detection analyzer"""
    
    # Your clip mapping
    clip_mapping = {
        'clip1_MLP': 'data/clip01/in/clip1_codes_MLP.csv',
        'clip2_AHKJ': 'data/clip02/in/clip2_codes_AHKJ.csv',
        'clip3_MLP': 'data/clip03/in/clip3_codes_MLP.csv'
    }
    
    # Your video list
    video_list = [
        "data/clip01/in/clip1_MLP.mp4",
        "data/clip02/in/clip2_AHKJ.mp4",
        "data/clip03/in/clip3_MLP.mp4"
    ]
    
    # Create analyzer
    analyzer = FaceDetectionAnalyzer(
        clip_csv_mapping=clip_mapping,
        timestamp_column='onset_milliseconds',
        face_column='has_faces'
    )
    
    # 1. Investigate distribution
    print("Step 1: Investigating face column distribution...")
    analysis_results = analyzer.investigate_face_column_distribution(verbose=True)
    
    # 2. Create visualizations
    print("\nStep 2: Creating distribution plots...")
    analyzer.plot_distribution_analysis(save_path='face_distribution_analysis.png')
    
    # 3. Retrain with optimal settings
    print("\nStep 3: Retraining face detector...")
    training_results = analyzer.retrain_face_detector(
        video_list=video_list,
        base_output_dir='data/face_detection_retrain',
        interval_ms=100,
        # Custom overrides (optional)
        model_config={
            'backbone': 'mobilenet',  # Fast for face detection
            'img_size': 224
        },
        training_config={
            'epochs': 60,  # Override if needed
            'lr': 0.001
        },
        skip_extraction=False,  # Set to True if frames already exist
        verbose=True
    )
    
    print(f"\n\U0001f389 Complete! Check results in: data/face_detection_retrain/")
    return training_results

if __name__ == "__main__":
    results = main()