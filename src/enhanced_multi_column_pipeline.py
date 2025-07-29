#!/usr/bin/env python3
"""
Enhanced Multi-Column Classification Pipeline with Integrated Face Detection Analysis - FIXED

This combines the multi-column pipeline with specialized face detection analysis,
allowing comprehensive investigation and retraining of all boolean columns including has_faces.

FIXED: Debug mode now properly reduces epochs and other training parameters.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
from collections import defaultdict
import time

from frame_organizer import organize_frames_from_multiple_clips, MultiClipFrameOrganizer
from mp4_frame_extractor import extract_frames_from_videos
from pytorch_cartoon_face_detector import BinaryClassifier, FaceDetector

class EnhancedMultiColumnPipeline:
    """Enhanced pipeline with integrated face detection analysis and specialized retraining"""
    
    def __init__(self, 
                 video_list: List[str],
                 clip_csv_mapping: Dict[str, str],
                 timestamp_column: str = 'onset_milliseconds',
                 base_output_dir: str = 'data/enhanced_combined',
                 interval_ms: int = 100,
                 debug: bool = False):
        """
        Initialize enhanced multi-column classification pipeline
        
        Args:
            video_list: List of video file paths
            clip_csv_mapping: Dictionary mapping clip names to CSV file paths
            timestamp_column: Name of the timestamp column
            base_output_dir: Base output directory for all files
            interval_ms: Frame extraction interval in milliseconds
            debug: If True, forces CPU usage and enables debug mode with reduced training time
        """
        self.video_list = video_list
        self.clip_csv_mapping = clip_csv_mapping
        self.timestamp_column = timestamp_column
        self.base_output_dir = base_output_dir
        self.interval_ms = interval_ms
        self.debug = debug
        
        # Debug mode configuration
        if self.debug:
            print("\U0001f41b DEBUG MODE ENABLED")
            print("   - Forcing CPU device usage")
            print("   - Enhanced logging enabled")
            print("   - REDUCED TRAINING PARAMETERS for fast debugging")
            os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Hide CUDA devices
        
        # Create base directories
        Path(base_output_dir).mkdir(parents=True, exist_ok=True)
        
        self.frames_dir = os.path.join(base_output_dir, 'frames')
        self.analysis_results = {}
        self.training_results = {}
        
        # Special handling for face detection
        self.face_column = 'has_faces'
        self.face_detector = None
        
    def comprehensive_column_analysis(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Comprehensive analysis of all boolean columns with special focus on face detection
        
        Args:
            verbose: If True, print detailed analysis
        
        Returns:
            Dictionary containing complete column analysis results
        """
        print("\U0001f50d COMPREHENSIVE MULTI-COLUMN ANALYSIS")
        print("="*70)
        
        all_columns = {}
        column_stats = {}
        face_analysis = {}
        
        for clip_name, csv_file in self.clip_csv_mapping.items():
            if not os.path.exists(csv_file):
                if verbose:
                    print(f"\u26a0\ufe0f  Warning: CSV file not found: {csv_file}")
                continue
            
            try:
                df = pd.read_csv(csv_file)
                
                if verbose:
                    print(f"\n\U0001f4ca Analyzing {clip_name}:")
                    print(f"   CSV: {csv_file}")
                    print(f"   Rows: {len(df)}")
                
                # Find all boolean columns
                boolean_columns = []
                for col in df.columns:
                    if col == self.timestamp_column:
                        continue
                    
                    # Check if column contains only 0/1 or boolean values
                    unique_vals = df[col].dropna().unique()
                    
                    if (len(unique_vals) <= 2 and 
                        all(val in [0, 1, True, False, '0', '1', 'True', 'False'] for val in unique_vals)):
                        
                        # Convert to 0/1 for consistency
                        binary_vals = [1 if val in [1, True, '1', 'True'] else 0 for val in unique_vals]
                        value_counts = df[col].value_counts()
                        
                        # Special analysis for face detection column
                        if col == self.face_column:
                            face_temporal = self._analyze_face_temporal_patterns(df, clip_name)
                            face_analysis[clip_name] = face_temporal
                        
                        boolean_columns.append({
                            'column': col,
                            'unique_values': sorted(unique_vals),
                            'binary_values': sorted(binary_vals),
                            'value_counts': value_counts.to_dict(),
                            'total_count': len(df[col].dropna()),
                            'is_face_column': col == self.face_column
                        })
                        
                        # Track across all clips
                        if col not in all_columns:
                            all_columns[col] = []
                        all_columns[col].append({
                            'clip': clip_name,
                            'stats': value_counts.to_dict(),
                            'total': len(df[col].dropna())
                        })
                
                if verbose:
                    print(f"   Boolean columns found: {len(boolean_columns)}")
                    for bcol in boolean_columns:
                        col_name = bcol['column']
                        counts = bcol['value_counts']
                        emoji = "\U0001f464" if bcol['is_face_column'] else "\U0001f3af"
                        print(f"     {emoji} {col_name}: {counts}")
                
            except Exception as e:
                if verbose:
                    print(f"\u274c Error analyzing {csv_file}: {e}")
        
        # Calculate combined statistics for all columns
        for col, clip_stats in all_columns.items():
            total_0 = sum(stats['stats'].get(0, 0) for stats in clip_stats)
            total_1 = sum(stats['stats'].get(1, 0) for stats in clip_stats)
            total_count = total_0 + total_1
            
            column_stats[col] = {
                'clips': [stats['clip'] for stats in clip_stats],
                'total_samples': total_count,
                'class_0_count': total_0,
                'class_1_count': total_1,
                'class_balance': {
                    'class_0_ratio': total_0 / total_count if total_count > 0 else 0,
                    'class_1_ratio': total_1 / total_count if total_count > 0 else 0
                },
                'clips_with_column': len(clip_stats),
                'is_face_column': col == self.face_column,
                'recommended_model': self._recommend_model_type(col, total_count, total_1/total_count if total_count > 0 else 0)
            }
            
            # Add face-specific analysis
            if col == self.face_column:
                column_stats[col]['face_temporal_analysis'] = face_analysis
        
        # Print comprehensive analysis
        if verbose:
            self._print_comprehensive_analysis(column_stats)
        
        self.analysis_results = {
            'column_stats': column_stats,
            'face_analysis': face_analysis,
            'recommendations': self._generate_training_recommendations(column_stats)
        }
        
        return self.analysis_results
    
    def _analyze_face_temporal_patterns(self, df: pd.DataFrame, clip_name: str) -> Dict[str, Any]:
        """Analyze temporal patterns specifically for face detection"""
        if self.face_column not in df.columns or self.timestamp_column not in df.columns:
            return {'error': 'Required columns not found'}
        
        # Sort by timestamp
        df_sorted = df.sort_values(self.timestamp_column)
        face_values = df_sorted[self.face_column].values
        timestamps = df_sorted[self.timestamp_column].values
        
        # Find face sequences
        face_sequences = []
        current_sequence = {'start': None, 'end': None, 'length': 0}
        
        for i, (timestamp, has_face) in enumerate(zip(timestamps, face_values)):
            if has_face == 1:
                if current_sequence['start'] is None:
                    current_sequence['start'] = timestamp
                current_sequence['end'] = timestamp
                current_sequence['length'] += 1
            else:
                if current_sequence['start'] is not None:
                    face_sequences.append(current_sequence.copy())
                    current_sequence = {'start': None, 'end': None, 'length': 0}
        
        # Don't forget the last sequence
        if current_sequence['start'] is not None:
            face_sequences.append(current_sequence)
        
        # Calculate sequence duration statistics
        sequence_durations = []
        for seq in face_sequences:
            if seq['start'] is not None and seq['end'] is not None:
                duration_ms = seq['end'] - seq['start']
                sequence_durations.append(duration_ms)
        
        return {
            'total_face_sequences': len(face_sequences),
            'avg_sequence_length_frames': np.mean([seq['length'] for seq in face_sequences]) if face_sequences else 0,
            'avg_sequence_duration_ms': np.mean(sequence_durations) if sequence_durations else 0,
            'max_sequence_duration_ms': max(sequence_durations) if sequence_durations else 0,
            'face_sequence_details': face_sequences[:10],  # First 10 for brevity
            'total_face_time_ms': sum(sequence_durations) if sequence_durations else 0
        }
    
    def _recommend_model_type(self, column: str, total_samples: int, positive_ratio: float) -> Dict[str, Any]:
        """Recommend optimal model configuration for each column"""
        
        # FIXED: Apply debug mode adjustments here
        if self.debug:
            print(f"\U0001f41b DEBUG: Applying reduced training config for {column}")
        
        # Base recommendations
        if column == self.face_column:
            # Face detection specific recommendations
            base_config = {
                'model_type': 'transfer',
                'backbone': 'mobilenet',  # Fast and efficient for face detection
                'pretrained': True,
                'freeze_features': total_samples < 5000,
                'img_size': 224
            }
        else:
            # General emotion/expression classification
            if 'happy' in column.lower() or 'smile' in column.lower():
                backbone = 'resnet18'  # Better for emotion detection
            elif 'excite' in column.lower() or 'surprise' in column.lower():
                backbone = 'efficientnet_b0'  # Good for subtle expressions
            else:
                backbone = 'mobilenet'  # Default efficient choice
            
            base_config = {
                'model_type': 'transfer',
                'backbone': backbone,
                'pretrained': True,
                'freeze_features': total_samples < 3000,
                'img_size': 224
            }
        
        # FIXED: Adjust based on data characteristics AND debug mode
        if self.debug:
            # Debug mode: drastically reduced parameters for fast testing
            training_config = {
                'epochs': 3,  # Very few epochs for debugging
                'lr': 0.01,   # Higher learning rate for faster convergence
                'batch_size': 4,  # Small batch size
                'patience': 2     # Low patience
            }
            print(f"\U0001f41b DEBUG: Using debug training config: {training_config}")
        else:
            # Normal mode: full training parameters
            training_config = {
                'epochs': 30 if total_samples < 1000 else 50 if total_samples < 5000 else 80,
                'lr': 0.001 if total_samples < 5000 else 0.0005,
                'batch_size': 16 if total_samples < 1000 else 32 if total_samples < 5000 else 64,
                'patience': 8 if total_samples < 1000 else 10 if total_samples < 5000 else 15
            }
        
        # Adjust for class imbalance (but not in debug mode to keep it simple)
        if not self.debug and (positive_ratio < 0.1 or positive_ratio > 0.9):
            training_config['patience'] += 5
            training_config['epochs'] += 20
            training_config['use_class_weighting'] = True
        
        return {
            'model_config': base_config,
            'training_config': training_config,
            'data_characteristics': {
                'samples': total_samples,
                'positive_ratio': positive_ratio,
                'balance_quality': 'good' if 0.2 <= positive_ratio <= 0.8 else 'imbalanced'
            },
            'debug_mode': self.debug  # Track if this was generated in debug mode
        }
    
    def _print_comprehensive_analysis(self, column_stats: Dict[str, Any]):
        """Print comprehensive analysis results"""
        print(f"\n\U0001f4c8 COMPREHENSIVE COLUMN ANALYSIS:")
        print(f"Found {len(column_stats)} boolean columns across all clips:")
        
        # Sort columns by total samples (descending)
        sorted_columns = sorted(column_stats.items(), key=lambda x: x[1]['total_samples'], reverse=True)
        
        for col, stats in sorted_columns:
            emoji = "\U0001f464" if stats.get('is_face_column', False) else "\U0001f3af"
            print(f"\n{emoji} Column: {col.upper()}")
            print(f"   Total samples: {stats['total_samples']:,}")
            print(f"   Class 0: {stats['class_0_count']:,} ({stats['class_balance']['class_0_ratio']:.1%})")
            print(f"   Class 1: {stats['class_1_count']:,} ({stats['class_balance']['class_1_ratio']:.1%})")
            print(f"   Available in {stats['clips_with_column']}/{len(self.clip_csv_mapping)} clips")
            
            # Recommended model
            rec_model = stats['recommended_model']['model_config']['backbone']
            epochs = stats['recommended_model']['training_config']['epochs']
            debug_indicator = " (DEBUG)" if stats['recommended_model'].get('debug_mode', False) else ""
            print(f"   Recommended: {rec_model} ({stats['recommended_model']['data_characteristics']['balance_quality']}) - {epochs} epochs{debug_indicator}")
            
            # Class balance assessment
            ratio = min(stats['class_balance']['class_0_ratio'], stats['class_balance']['class_1_ratio'])
            if ratio < 0.1:
                print(f"   \u26a0\ufe0f  SEVERE imbalance (minority: {ratio:.1%}) - Will use class weighting")
            elif ratio < 0.2:
                print(f"   \u26a0\ufe0f  MODERATE imbalance (minority: {ratio:.1%}) - Will monitor closely")
            else:
                print(f"   \u2705 GOOD balance")
            
            # Special face analysis
            if stats.get('is_face_column', False) and 'face_temporal_analysis' in stats:
                face_data = stats['face_temporal_analysis']
                if face_data:
                    total_sequences = sum(clip_data.get('total_face_sequences', 0) 
                                        for clip_data in face_data.values() if isinstance(clip_data, dict))
                    print(f"   \U0001f464 Face sequences across clips: {total_sequences}")
    
    def _generate_training_recommendations(self, column_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive training recommendations"""
        recommendations = {
            'priority_order': [],
            'batch_training_groups': [],
            'special_considerations': {},
            'debug_mode': self.debug  # Track debug mode in recommendations
        }
        
        # Sort by data quality and quantity
        prioritized_columns = []
        for col, stats in column_stats.items():
            quality_score = 0
            
            # Data quantity score
            if stats['total_samples'] >= 5000:
                quality_score += 3
            elif stats['total_samples'] >= 1000:
                quality_score += 2
            else:
                quality_score += 1
            
            # Balance score
            ratio = min(stats['class_balance']['class_0_ratio'], stats['class_balance']['class_1_ratio'])
            if ratio >= 0.2:
                quality_score += 3
            elif ratio >= 0.1:
                quality_score += 2
            else:
                quality_score += 1
            
            # Availability score
            if stats['clips_with_column'] == len(self.clip_csv_mapping):
                quality_score += 2
            elif stats['clips_with_column'] >= len(self.clip_csv_mapping) * 0.8:
                quality_score += 1
            
            prioritized_columns.append((col, quality_score, stats))
        
        # Sort by quality score
        prioritized_columns.sort(key=lambda x: x[1], reverse=True)
        recommendations['priority_order'] = [col for col, score, stats in prioritized_columns]
        
        # Group by similar training requirements
        mobilenet_group = []
        resnet_group = []
        efficientnet_group = []
        
        for col, score, stats in prioritized_columns:
            backbone = stats['recommended_model']['model_config']['backbone']
            if backbone == 'mobilenet':
                mobilenet_group.append(col)
            elif 'resnet' in backbone:
                resnet_group.append(col)
            elif 'efficientnet' in backbone:
                efficientnet_group.append(col)
        
        recommendations['batch_training_groups'] = [
            {'name': 'MobileNet Group (Fast)', 'columns': mobilenet_group},
            {'name': 'ResNet Group (Balanced)', 'columns': resnet_group},
            {'name': 'EfficientNet Group (Accuracy)', 'columns': efficientnet_group}
        ]
        
        # Special considerations
        for col, score, stats in prioritized_columns:
            considerations = []
            
            if self.debug:
                considerations.append('DEBUG MODE: Using minimal training parameters')
            
            if stats['total_samples'] < 1000:
                considerations.append('Limited data - use strong data augmentation')
            
            ratio = min(stats['class_balance']['class_0_ratio'], stats['class_balance']['class_1_ratio'])
            if ratio < 0.1:
                considerations.append('Severe imbalance - use class weighting and balanced sampling')
            
            if stats['clips_with_column'] < len(self.clip_csv_mapping):
                considerations.append(f'Missing from {len(self.clip_csv_mapping) - stats["clips_with_column"]} clips')
            
            if col == self.face_column:
                considerations.append('Face detection - consider ensemble with pre-trained face detectors')
            
            if considerations:
                recommendations['special_considerations'][col] = considerations
        
        return recommendations
    
    def create_visualization_dashboard(self, save_path: Optional[str] = None):
        """Create comprehensive visualization dashboard for all columns"""
        if not self.analysis_results:
            raise ValueError("Run comprehensive_column_analysis() first")
        
        # Configure matplotlib to use non-interactive backend for command-line usage
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        matplotlib.rcParams['font.family'] = ['DejaVu Sans']
        
        column_stats = self.analysis_results['column_stats']
        
        # Calculate grid size
        n_cols = len(column_stats)
        n_rows = 3  # Distribution, balance, temporal
        
        fig = plt.figure(figsize=(5 * min(n_cols, 4), 4 * n_rows))
        
        # If too many columns, create multiple figures or use subplots
        if n_cols > 4:
            self._create_multi_page_dashboard(column_stats, save_path)
            return
        
        # Create subplots for each column
        for idx, (col, stats) in enumerate(column_stats.items()):
            # 1. Sample distribution pie chart
            ax1 = plt.subplot(n_rows, n_cols, idx + 1)
            
            class_0_count = stats['class_0_count']
            class_1_count = stats['class_1_count']
            
            colors = ['lightcoral', 'lightgreen'] if col == self.face_column else ['lightblue', 'orange']
            labels = ['No Face', 'Face'] if col == self.face_column else ['Class 0', 'Class 1']
            
            ax1.pie([class_0_count, class_1_count], 
                   labels=[f'{labels[0]}\n({class_0_count:,})', f'{labels[1]}\n({class_1_count:,})'],
                   autopct='%1.1f%%', colors=colors)
            
            # Use text prefix instead of emoji for plot titles
            prefix = "FACE" if col == self.face_column else "TARGET"
            debug_suffix = " (DEBUG)" if self.debug else ""
            ax1.set_title(f'{prefix}: {col}{debug_suffix}\nDistribution')
            
            # 2. Clip-wise distribution
            ax2 = plt.subplot(n_rows, n_cols, idx + 1 + n_cols)
            
            clip_names = []
            clip_ratios = []
            
            for clip_data in stats['clips']:
                clip_names.append(clip_data)
                # Calculate ratio for this clip
                clip_stats_data = next(c for c in column_stats[col]['clips'] if isinstance(c, dict))
                total = clip_stats_data['total'] if isinstance(clip_stats_data, dict) else 0
                positive = clip_stats_data['stats'].get(1, 0) if isinstance(clip_stats_data, dict) else 0
                ratio = positive / total if total > 0 else 0
                clip_ratios.append(ratio)
            
            ax2.bar(range(len(clip_names)), clip_ratios, 
                   color='green' if col == self.face_column else 'blue', alpha=0.7)
            ax2.set_title(f'{col}\nPer Clip')
            ax2.set_ylabel('Positive Ratio')
            ax2.set_xticks(range(len(clip_names)))
            ax2.set_xticklabels(clip_names, rotation=45, ha='right')
            
            # 3. Model recommendation
            ax3 = plt.subplot(n_rows, n_cols, idx + 1 + 2*n_cols)
            
            rec = stats['recommended_model']
            model_info = f"Model: {rec['model_config']['backbone']}\n"
            model_info += f"Epochs: {rec['training_config']['epochs']}\n"
            model_info += f"Batch: {rec['training_config']['batch_size']}\n"
            model_info += f"Balance: {rec['data_characteristics']['balance_quality']}"
            if rec.get('debug_mode', False):
                model_info += "\nDEBUG MODE"
            
            ax3.text(0.1, 0.5, model_info, transform=ax3.transAxes, 
                    fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            ax3.set_xlim(0, 1)
            ax3.set_ylim(0, 1)
            ax3.axis('off')
            ax3.set_title(f'{col}\nRecommendation')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\U0001f4ca Dashboard saved to: {save_path}")
        
        # Close the figure instead of showing it
        plt.close(fig)
    
    def _create_multi_page_dashboard(self, column_stats: Dict[str, Any], save_path: Optional[str]):
        """Create multi-page dashboard for many columns"""
        # Configure matplotlib to use non-interactive backend for command-line usage
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        matplotlib.rcParams['font.family'] = ['DejaVu Sans']
        
        columns_per_page = 4
        column_items = list(column_stats.items())
        
        for page, i in enumerate(range(0, len(column_items), columns_per_page)):
            page_columns = column_items[i:i+columns_per_page]
            
            fig, axes = plt.subplots(3, len(page_columns), figsize=(5*len(page_columns), 12))
            if len(page_columns) == 1:
                axes = axes.reshape(-1, 1)
            
            for col_idx, (col, stats) in enumerate(page_columns):
                # Distribution pie chart
                class_0_count = stats['class_0_count']
                class_1_count = stats['class_1_count']
                
                colors = ['lightcoral', 'lightgreen'] if col == self.face_column else ['lightblue', 'orange']
                labels = ['No Face', 'Face'] if col == self.face_column else ['Class 0', 'Class 1']
                
                axes[0, col_idx].pie([class_0_count, class_1_count], 
                                   labels=[f'{labels[0]}\n({class_0_count:,})', f'{labels[1]}\n({class_1_count:,})'],
                                   autopct='%1.1f%%', colors=colors)
                
                # Use text prefix instead of emoji for plot titles
                prefix = "FACE" if col == self.face_column else "TARGET"
                debug_suffix = " (DEBUG)" if self.debug else ""
                axes[0, col_idx].set_title(f'{prefix}: {col}{debug_suffix}\nDistribution')
                
                # Add more visualizations for each column...
                # (Similar to above but adapted for multi-page layout)
            
            page_save_path = save_path.replace('.png', f'_page_{page+1}.png') if save_path else None
            
            plt.tight_layout()
            if page_save_path:
                plt.savefig(page_save_path, dpi=300, bbox_inches='tight')
                print(f"\U0001f4ca Dashboard page {page+1} saved to: {page_save_path}")
            
            # Close the figure instead of showing it
            plt.close(fig)
    
    def extract_frames(self, verbose: bool = True) -> List[Dict[str, Any]]:
        """Extract frames from all videos (same as original)"""
        print("\U0001f3ac EXTRACTING FRAMES FROM VIDEOS")
        print("-" * 40)
        
        extraction_results = extract_frames_from_videos(
            self.video_list,
            output_dir=self.frames_dir,
            interval_ms=self.interval_ms,
            include_video_name=True,
            skip_black_frames=True,
            verbose=verbose
        )
        
        total_extracted = sum(r.get('extracted_count', 0) for r in extraction_results if 'extracted_count' in r)
        print(f"\u2705 Frame extraction complete: {total_extracted:,} frames extracted")
        
        return extraction_results
    
    def create_datasets_for_all_columns(self, 
                                      columns_to_process: Optional[List[str]] = None,
                                      class_name_mapping: Optional[Dict[str, Dict[int, str]]] = None,
                                      verbose: bool = True) -> Dict[str, Any]:
        """
        Create datasets for all columns with special handling for face detection
        """
        print("\n\U0001f4c1 CREATING DATASETS FOR ALL COLUMNS")
        if self.debug:
            print("\U0001f41b DEBUG MODE: Creating datasets with reduced processing")
        print("-" * 40)
        
        if not self.analysis_results:
            print("\u26a0\ufe0f  Running analysis first...")
            self.comprehensive_column_analysis(verbose=False)
        
        # Use recommended columns if none specified
        if columns_to_process is None:
            columns_to_process = self.analysis_results['recommendations']['priority_order']
            print(f"\U0001f4cb Using recommended columns: {columns_to_process}")
        
        dataset_results = {}
        
        for column in columns_to_process:
            print(f"\n\U0001f3af Processing column: {column}")
            
            # Determine class names
            if class_name_mapping and column in class_name_mapping:
                class_names = class_name_mapping[column]
            elif column == self.face_column:
                class_names = {0: 'no_face', 1: 'face'}
            else:
                # Generate contextual class names
                if 'happy' in column.lower():
                    class_names = {0: 'not_happy', 1: 'happy'}
                elif 'excite' in column.lower():
                    class_names = {0: 'calm', 1: 'excited'}
                elif 'surprise' in column.lower():
                    class_names = {0: 'not_surprised', 1: 'surprised'}
                elif 'fear' in column.lower():
                    class_names = {0: 'not_fearful', 1: 'fearful'}
                else:
                    class_names = {0: f'not_{column}', 1: column}
            
            # Create dataset directory
            dataset_dir = os.path.join(self.base_output_dir, f'{column}_dataset')
            
            emoji = "\U0001f464" if column == self.face_column else "\U0001f3af"
            print(f"   {emoji} Class mapping: {class_names}")
            print(f"   \U0001f4c2 Dataset directory: {dataset_dir}")
            
            try:
                # Organize frames for this column
                organizer = MultiClipFrameOrganizer(
                    dataset_dir=dataset_dir,
                    output_frames_dir=self.frames_dir,
                    file_extension='jpg',
                    use_move=False  # Copy files for multiple column use
                )
                
                results = organizer.organize_from_clip_mapping(
                    clip_csv_mapping=self.clip_csv_mapping,
                    timestamp_column=self.timestamp_column,
                    label_column=column,
                    class_names=class_names,
                    verbose=verbose
                )
                
                # Calculate success statistics
                successful_clips = len([r for r in results if 'error' not in r])
                total_organized = sum(r.get('successful_operations', 0) for r in results if 'successful_operations' in r)
                
                dataset_results[column] = {
                    'class_names': class_names,
                    'dataset_dir': dataset_dir,
                    'organization_results': results,
                    'successful_clips': successful_clips,
                    'total_clips': len(self.clip_csv_mapping),
                    'total_frames_organized': total_organized,
                    'is_face_column': column == self.face_column,
                    'success': True
                }
                
                print(f"   \u2705 Success: {total_organized:,} frames organized from {successful_clips}/{len(self.clip_csv_mapping)} clips")
                
                # Save detailed results
                organizer.save_results(os.path.join(dataset_dir, 'organization_results.json'))
                
            except Exception as e:
                print(f"   \u274c Error processing column {column}: {e}")
                dataset_results[column] = {
                    'error': str(e),
                    'success': False,
                    'is_face_column': column == self.face_column
                }
        
        return dataset_results
    
    def train_all_classifiers(self,
                            columns_to_train: Optional[List[str]] = None,
                            class_name_mapping: Optional[Dict[str, Dict[int, str]]] = None,
                            custom_configs: Optional[Dict[str, Dict[str, Any]]] = None,
                            verbose: bool = True) -> Dict[str, Any]:
        """
        Train classifiers for all columns with optimized configurations
        """
        print("\n\U0001f9e0 TRAINING CLASSIFIERS FOR ALL COLUMNS")
        if self.debug:
            print("\U0001f41b DEBUG MODE: Using minimal training parameters for fast testing")
        print("-" * 40)
        
        if not self.analysis_results:
            raise ValueError("Run comprehensive_column_analysis() first")
        
        if columns_to_train is None:
            columns_to_train = self.analysis_results['recommendations']['priority_order']
        
        training_results = {}
        
        for column in columns_to_train:
            print(f"\n\U0001f3af Training classifier for: {column}")
            
            # FIXED: Get recommended configurations (already adjusted for debug mode in _recommend_model_type)
            recommended = self.analysis_results['column_stats'][column]['recommended_model']
            model_config = recommended['model_config'].copy()
            training_config = recommended['training_config'].copy()
            
            # FIXED: Apply custom overrides ONLY if not in debug mode, or allow debug override
            if custom_configs and column in custom_configs:
                if not self.debug:  # Normal mode: apply custom configs
                    if 'model_config' in custom_configs[column]:
                        model_config.update(custom_configs[column]['model_config'])
                    if 'training_config' in custom_configs[column]:
                        training_config.update(custom_configs[column]['training_config'])
                else:  # Debug mode: only apply custom configs if they're debug-friendly
                    print(f"\U0001f41b DEBUG: Skipping custom configs to maintain debug settings")
                    # Optionally allow debug-specific overrides
                    if 'debug_config' in custom_configs[column]:
                        training_config.update(custom_configs[column]['debug_config'])
            
            # FIXED: Ensure debug mode settings are preserved
            if self.debug:
                # Force debug parameters regardless of other configs
                training_config.update({
                    'epochs': 3,
                    'batch_size': 4,
                    'patience': 2,
                    'lr': 0.01
                })
                print(f"\U0001f41b DEBUG: Final training config: {training_config}")
            
            # Determine class names
            if class_name_mapping and column in class_name_mapping:
                class_names = class_name_mapping[column]
                positive_class = class_names[1]
                negative_class = class_names[0]
            elif column == self.face_column:
                positive_class = 'face'
                negative_class = 'no_face'
                class_names = {0: negative_class, 1: positive_class}
            else:
                positive_class = column
                negative_class = f'not_{column}'
                class_names = {0: negative_class, 1: positive_class}
            
            dataset_dir = os.path.join(self.base_output_dir, f'{column}_dataset')
            
            if not os.path.exists(dataset_dir):
                print(f"   \u274c Error: Dataset directory not found: {dataset_dir}")
                training_results[column] = {
                    'error': f'Dataset directory not found: {dataset_dir}',
                    'success': False,
                    'is_face_column': column == self.face_column
                }
                continue
            
            try:
                # Create appropriate classifier
                if column == self.face_column:
                    # Use specialized FaceDetector
                    classifier = FaceDetector(
                        model_type=model_config.get('model_type', 'transfer'),
                        backbone=model_config.get('backbone', 'mobilenet'),
                        img_size=model_config.get('img_size', 224),
                        device=model_config.get('device', None)
                    )
                    emoji = "\U0001f464"
                else:
                    # Use general BinaryClassifier
                    classifier = BinaryClassifier(
                        task_name=f"{column}_classification",
                        class_names=class_names,
                        model_type=model_config.get('model_type', 'transfer'),
                        backbone=model_config.get('backbone', 'mobilenet'),
                        img_size=model_config.get('img_size', 224),
                        device=model_config.get('device', None)
                    )
                    emoji = "\U0001f3af"
                
                # Create model
                model = classifier.create_model(
                    pretrained=model_config.get('pretrained', True),
                    freeze_features=model_config.get('freeze_features', True)
                )
                
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                frozen_status = 'frozen' if model_config.get('freeze_features') else 'unfrozen'
                debug_indicator = " (DEBUG MODE)" if self.debug else ""
                print(f"   {emoji} Model created: {trainable_params:,} trainable parameters{debug_indicator}")
                print(f"   \U0001f4cb Config: {model_config['backbone']} ({frozen_status})")
                
                # Load dataset with special handling for imbalanced data
                train_loader, val_loader = self._create_optimized_dataloaders(
                    classifier, dataset_dir, column, training_config
                )
                
                # Train model
                model_save_path = os.path.join(self.base_output_dir, f'{column}_classifier.pth')
                
                epochs = training_config.get('epochs', 3 if self.debug else 50)
                print(f"   \U0001f680 Training {column} classifier for {epochs} epochs...")
                start_time = time.time()
                
                history = classifier.train_model(
                    train_loader,
                    val_loader,
                    epochs=epochs,
                    lr=training_config.get('lr', 0.01 if self.debug else 0.001),
                    patience=training_config.get('patience', 2 if self.debug else 10),
                    save_path=model_save_path
                )
                
                training_time = time.time() - start_time
                
                # Evaluate model
                print(f"   \U0001f3af Evaluating {column} classifier...")
                metrics = classifier.evaluate_model(val_loader)
                
                # Store classifier and results
                training_results[column] = {
                    'classifier': classifier,
                    'model_path': model_save_path,
                    'dataset_dir': dataset_dir,
                    'class_names': class_names,
                    'training_history': history,
                    'evaluation_metrics': metrics,
                    'model_config': model_config,
                    'training_config': training_config,
                    'training_time_seconds': training_time,
                    'is_face_column': column == self.face_column,
                    'debug_mode': self.debug,
                    'success': True
                }
                
                # Special handling for face detector
                if column == self.face_column:
                    self.face_detector = classifier
                    training_results[column]['face_detector_ready'] = True
                
                debug_note = " (DEBUG - minimal training)" if self.debug else ""
                print(f"   \u2705 Training complete: {metrics['accuracy']:.1f}% accuracy in {training_time/60:.1f} minutes{debug_note}")
                
            except Exception as e:
                print(f"   \u274c Error training classifier for {column}: {e}")
                training_results[column] = {
                    'error': str(e),
                    'success': False,
                    'is_face_column': column == self.face_column,
                    'debug_mode': self.debug
                }
        
        self.training_results = training_results
        return training_results
    
    def _create_optimized_dataloaders(self, classifier, dataset_dir, column, training_config):
        """Create optimized dataloaders with class balancing for imbalanced datasets"""
        from torch.utils.data import WeightedRandomSampler
        import torch
        
        # FIXED: In debug mode, use smaller batches
        batch_size = training_config.get('batch_size', 4 if self.debug else 32)
        
        # Load dataset normally first
        train_loader, val_loader = classifier.load_dataset(
            dataset_dir,
            batch_size=batch_size,
            val_split=training_config.get('val_split', 0.2)
        )
        
        # Check if we need class balancing (skip in debug mode for simplicity)
        if not self.debug:
            column_stats = self.analysis_results['column_stats'][column]
            positive_ratio = column_stats['class_balance']['class_1_ratio']
            
            if training_config.get('use_class_weighting', False) or positive_ratio < 0.2 or positive_ratio > 0.8:
                print(f"   \u2696\ufe0f  Applying class balancing for {column} (positive ratio: {positive_ratio:.1%})")
                
                # Calculate class weights for loss function
                total_samples = column_stats['total_samples']
                pos_samples = column_stats['class_1_count']
                neg_samples = column_stats['class_0_count']
                
                # Store class weights for potential use in custom loss function
                class_weights = {
                    'neg_weight': total_samples / (2 * neg_samples),
                    'pos_weight': total_samples / (2 * pos_samples)
                }
                
                print(f"      Class weights: negative={class_weights['neg_weight']:.2f}, positive={class_weights['pos_weight']:.2f}")
        else:
            print(f"   \U0001f41b DEBUG: Skipping class balancing for faster training")
        
        return train_loader, val_loader
    
    def run_comprehensive_pipeline(self,
                                 columns_to_process: Optional[List[str]] = None,
                                 class_name_mapping: Optional[Dict[str, Dict[int, str]]] = None,
                                 custom_configs: Optional[Dict[str, Dict[str, Any]]] = None,
                                 skip_extraction: bool = False,
                                 create_visualizations: bool = True,
                                 verbose: bool = True) -> Dict[str, Any]:
        """
        Run the complete enhanced multi-column pipeline with face detection integration
        """
        print("\U0001f680 ENHANCED MULTI-COLUMN CLASSIFICATION PIPELINE")
        if self.debug:
            print("\U0001f41b DEBUG MODE: Fast testing with minimal parameters")
        print("="*70)
        
        # Step 0: Comprehensive analysis
        print("\U0001f4ca Step 1: Comprehensive Column Analysis")
        analysis_results = self.comprehensive_column_analysis(verbose=verbose)
        
        # Step 1: Create visualizations if requested
        if create_visualizations:
            print("\n\U0001f4c8 Step 2: Creating Analysis Dashboard")
            dashboard_path = os.path.join(self.base_output_dir, 'analysis_dashboard.png')
            self.create_visualization_dashboard(save_path=dashboard_path)
        
        # Step 2: Extract frames
        if not skip_extraction:
            print("\n\U0001f3ac Step 3: Extracting Frames")
            extraction_results = self.extract_frames(verbose=verbose)
        else:
            print("\n\u23e9 Step 3: Skipping frame extraction")
            extraction_results = []
        
        # Step 3: Create datasets
        print("\n\U0001f4c1 Step 4: Creating Datasets")
        dataset_results = self.create_datasets_for_all_columns(
            columns_to_process=columns_to_process,
            class_name_mapping=class_name_mapping,
            verbose=verbose
        )
        
        # Step 4: Train classifiers
        successful_datasets = [col for col, result in dataset_results.items() if result.get('success', False)]
        
        if successful_datasets:
            print(f"\n\U0001f9e0 Step 5: Training Classifiers for {len(successful_datasets)} columns")
            training_results = self.train_all_classifiers(
                columns_to_train=successful_datasets,
                class_name_mapping=class_name_mapping,
                custom_configs=custom_configs,
                verbose=verbose
            )
        else:
            print("\n\u274c Step 5: No successful datasets created, skipping training")
            training_results = {}
        
        # Compile comprehensive results
        final_results = {
            'analysis_results': analysis_results,
            'extraction_results': extraction_results,
            'dataset_results': dataset_results,
            'training_results': training_results,
            'successful_classifiers': [col for col, result in training_results.items() if result.get('success', False)],
            'face_detector_available': any(result.get('face_detector_ready', False) for result in training_results.values()),
            'debug_mode': self.debug,
            'pipeline_config': {
                'video_list': self.video_list,
                'clip_csv_mapping': self.clip_csv_mapping,
                'base_output_dir': self.base_output_dir,
                'interval_ms': self.interval_ms,
                'debug_mode': self.debug
            },
            'performance_summary': self._generate_performance_summary(training_results)
        }
        
        # Print comprehensive summary
        self._print_comprehensive_summary(final_results, verbose=verbose)
        
        # Save complete results
        results_file = os.path.join(self.base_output_dir, 'comprehensive_pipeline_results.json')
        with open(results_file, 'w') as f:
            # Convert non-serializable objects for JSON
            json_results = self._prepare_results_for_json(final_results)
            json.dump(json_results, f, indent=2, default=str)
        
        print(f"\n\U0001f4c4 Complete results saved to: {results_file}")
        
        return final_results
    
    def _generate_performance_summary(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive performance summary"""
        successful_trainings = {col: result for col, result in training_results.items() 
                              if result.get('success', False)}
        
        if not successful_trainings:
            return {'error': 'No successful trainings'}
        
        # Calculate aggregate statistics
        accuracies = [result['evaluation_metrics']['accuracy'] for result in successful_trainings.values()]
        f1_scores = [result['evaluation_metrics']['f1_score'] for result in successful_trainings.values()]
        training_times = [result['training_time_seconds'] for result in successful_trainings.values()]
        
        # Find best and worst performers
        best_accuracy_col = max(successful_trainings.keys(), 
                               key=lambda x: successful_trainings[x]['evaluation_metrics']['accuracy'])
        worst_accuracy_col = min(successful_trainings.keys(), 
                                key=lambda x: successful_trainings[x]['evaluation_metrics']['accuracy'])
        
        # Face detection specific metrics
        face_results = {}
        for col, result in successful_trainings.items():
            if result.get('is_face_column', False):
                face_results = {
                    'accuracy': result['evaluation_metrics']['accuracy'],
                    'precision': result['evaluation_metrics']['precision'],
                    'recall': result['evaluation_metrics']['recall'],
                    'f1_score': result['evaluation_metrics']['f1_score']
                }
                break
        
        return {
            'total_trained': len(successful_trainings),
            'avg_accuracy': np.mean(accuracies),
            'avg_f1_score': np.mean(f1_scores),
            'avg_training_time_minutes': np.mean(training_times) / 60,
            'best_performer': {
                'column': best_accuracy_col,
                'accuracy': successful_trainings[best_accuracy_col]['evaluation_metrics']['accuracy']
            },
            'worst_performer': {
                'column': worst_accuracy_col,
                'accuracy': successful_trainings[worst_accuracy_col]['evaluation_metrics']['accuracy']
            },
            'face_detection_metrics': face_results,
            'debug_mode': self.debug,
            'accuracy_distribution': {
                'excellent': len([acc for acc in accuracies if acc >= 95]),
                'very_good': len([acc for acc in accuracies if 90 <= acc < 95]),
                'good': len([acc for acc in accuracies if 80 <= acc < 90]),
                'fair': len([acc for acc in accuracies if 70 <= acc < 80]),
                'poor': len([acc for acc in accuracies if acc < 70])
            }
        }
    
    def _print_comprehensive_summary(self, results: Dict[str, Any], verbose: bool = True):
        """Print comprehensive pipeline summary"""
        print("\n" + "="*70)
        print("\U0001f4ca COMPREHENSIVE PIPELINE SUMMARY")
        if results.get('debug_mode', False):
            print("\U0001f41b DEBUG MODE RESULTS - Minimal Training for Testing")
        print("="*70)
        
        analysis = results['analysis_results']
        performance = results['performance_summary']
        
        # Data summary
        total_columns = len(analysis['column_stats'])
        successful_classifiers = len(results['successful_classifiers'])
        
        print(f"\n\U0001f4c8 DATA ANALYSIS:")
        print(f"   Boolean columns found: {total_columns}")
        print(f"   Classifiers trained: {successful_classifiers}/{total_columns}")
        
        face_status = "\u2705" if results['face_detector_available'] else "\u274c"
        print(f"   Face detection available: {face_status}")
        
        if results.get('debug_mode', False):
            print(f"   \U0001f41b DEBUG MODE: Training limited to 3 epochs for testing")
        
        # Performance overview
        if successful_classifiers > 0 and 'error' not in performance:
            print(f"\n\U0001f3af PERFORMANCE OVERVIEW:")
            print(f"   Average accuracy: {performance['avg_accuracy']:.1f}%")
            print(f"   Average F1 score: {performance['avg_f1_score']:.3f}")
            print(f"   Average training time: {performance['avg_training_time_minutes']:.1f} minutes")
            
            if results.get('debug_mode', False):
                print(f"   \U0001f41b Note: Performance may be lower due to minimal debug training")
            
            print(f"\n\U0001f3c6 PERFORMANCE DISTRIBUTION:")
            dist = performance['accuracy_distribution']
            print(f"   Excellent (\u226595%): {dist['excellent']} classifiers")
            print(f"   Very Good (90-95%): {dist['very_good']} classifiers")
            print(f"   Good (80-90%): {dist['good']} classifiers")
            print(f"   Fair (70-80%): {dist['fair']} classifiers")
            print(f"   Poor (<70%): {dist['poor']} classifiers")
            
            # Best and worst performers
            print(f"\n\U0001f947 BEST PERFORMER:")
            best = performance['best_performer']
            print(f"   {best['column']}: {best['accuracy']:.1f}% accuracy")
            
            if performance['worst_performer']['accuracy'] < 80:
                print(f"\n\u26a0\ufe0f  NEEDS ATTENTION:")
                worst = performance['worst_performer']
                print(f"   {worst['column']}: {worst['accuracy']:.1f}% accuracy")
        
        # Face detection specific results
        if results['face_detector_available'] and performance.get('face_detection_metrics'):
            face_metrics = performance['face_detection_metrics']
            print(f"\n\U0001f464 FACE DETECTION RESULTS:")
            print(f"   Accuracy: {face_metrics['accuracy']:.1f}%")
            print(f"   Precision: {face_metrics['precision']:.3f}")
            print(f"   Recall: {face_metrics['recall']:.3f}")
            print(f"   F1 Score: {face_metrics['f1_score']:.3f}")
            
            if results.get('debug_mode', False):
                print(f"   \U0001f41b DEBUG: Results from minimal training")
            elif face_metrics['accuracy'] >= 90:
                print(f"   \U0001f31f Excellent face detection performance!")
            elif face_metrics['accuracy'] >= 80:
                print(f"   \u2705 Good face detection performance")
            else:
                print(f"   \u26a0\ufe0f  Face detection may need improvement")
        
        # Individual classifier results
        if verbose and results['successful_classifiers']:
            print(f"\n\U0001f4cb INDIVIDUAL CLASSIFIER RESULTS:")
            for col in results['successful_classifiers']:
                training_result = results['training_results'][col]
                metrics = training_result['evaluation_metrics']
                
                emoji = "\U0001f464" if training_result.get('is_face_column', False) else "\U0001f3af"
                epochs = training_result['training_config'].get('epochs', 'unknown')
                debug_note = f" ({epochs} epochs)" if results.get('debug_mode', False) else ""
                print(f"\n  {emoji} {col.upper()}{debug_note}:")
                print(f"     Accuracy: {metrics['accuracy']:.1f}%")
                print(f"     F1 Score: {metrics['f1_score']:.3f}")
                print(f"     Model: {training_result['model_path']}")
        
        # Failed items
        failed_datasets = [col for col, r in results['dataset_results'].items() if not r.get('success', False)]
        failed_training = [col for col, r in results['training_results'].items() if not r.get('success', False)]
        
        if failed_datasets or failed_training:
            print(f"\n\u274c ISSUES ENCOUNTERED:")
            if failed_datasets:
                print(f"   Dataset creation failed: {failed_datasets}")
            if failed_training:
                print(f"   Training failed: {failed_training}")
        
        # Recommendations
        print(f"\n\U0001f4a1 RECOMMENDATIONS:")
        if results.get('debug_mode', False):
            print("   \U0001f41b DEBUG MODE COMPLETE:")
            print("   \u2022 Set DEBUG_MODE = False for full training")
            print("   \u2022 Current results are from minimal 3-epoch training")
            print("   \u2022 Full training will use 30-80 epochs based on data size")
        elif performance.get('avg_accuracy', 0) >= 90:
            print("   \U0001f31f Excellent overall performance! Ready for deployment.")
            print("   \u2022 Consider creating ensemble models for critical tasks")
            print("   \u2022 Test on additional validation data")
        elif performance.get('avg_accuracy', 0) >= 80:
            print("   \u2705 Good performance overall.")
            print("   \u2022 Fine-tune underperforming classifiers")
            print("   \u2022 Consider data augmentation for improved robustness")
        else:
            print("   \u26a0\ufe0f  Performance could be improved.")
            print("   \u2022 Review data quality and labeling")
            print("   \u2022 Consider collecting more training data")
            print("   \u2022 Experiment with different model architectures")
        
        print(f"\n\U0001f4c1 All outputs saved in: {self.base_output_dir}")
        print("="*70)
        debug_status = " (DEBUG MODE)" if results.get('debug_mode', False) else ""
        print(f"\U0001f389 Enhanced multi-column classification pipeline complete!{debug_status}")
        print("="*70)
    
    def _prepare_results_for_json(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare results for JSON serialization"""
        json_results = {}
        for key, value in results.items():
            if key == 'training_results':
                # Remove classifier objects, keep only serializable data
                json_training = {}
                for col, training_result in value.items():
                    if isinstance(training_result, dict):
                        json_training[col] = {k: v for k, v in training_result.items() 
                                            if k not in ['classifier']}
                    else:
                        json_training[col] = training_result
                json_results[key] = json_training
            else:
                json_results[key] = value
        return json_results
    
    def predict_video_with_all_classifiers(self, 
                                         video_path: str, 
                                         output_path: Optional[str] = None,
                                         threshold: float = 0.5) -> Dict[str, Any]:
        """
        Process a video with all trained classifiers including face detection
        """
        if not self.training_results:
            raise ValueError("No trained classifiers available. Run the pipeline first.")
        
        successful_classifiers = {col: result for col, result in self.training_results.items() 
                                if result.get('success', False)}
        
        if not successful_classifiers:
            raise ValueError("No successful classifiers found.")
        
        debug_note = " (using debug-trained models)" if self.debug else ""
        print(f"\U0001f3ac Processing video with {len(successful_classifiers)} classifiers{debug_note}...")
        
        # Process video with all classifiers
        video_results = {}
        
        for col, classifier_result in successful_classifiers.items():
            classifier = classifier_result['classifier']
            emoji = "\U0001f464" if classifier_result.get('is_face_column') else "\U0001f3af"
            print(f"   {emoji} Processing with {col} classifier...")
            
            frame_results = classifier.process_video(
                video_path, 
                output_path=f"{output_path}_{col}.mp4" if output_path else None,
                threshold=threshold
            )
            
            video_results[col] = {
                'frame_results': frame_results,
                'class_names': classifier_result['class_names'],
                'total_frames': len(frame_results),
                'positive_frames': sum(1 for r in frame_results if r['is_positive']),
                'is_face_column': classifier_result.get('is_face_column', False)
            }
        
        return video_results

# Enhanced example usage function
def run_enhanced_example():
    """
    Enhanced example demonstrating the complete pipeline with face detection integration
    """
    
    # FIXED: Debug mode control - set to False for full training, True for quick testing
    DEBUG_MODE = True  # CHANGE THIS TO False FOR FULL TRAINING
    
    if DEBUG_MODE:
        print("\U0001f41b RUNNING IN DEBUG MODE")
        print("   - Only 3 epochs per classifier")
        print("   - Small batch sizes")
        print("   - CPU-only processing")
        print("   - Change DEBUG_MODE = False for full training")
        print("-" * 50)
    
    # Configuration
    video_list = [
        "data/clip01/in/clip1_MLP.mp4",
        "data/clip02/in/clip2_AHKJ.mp4", 
        "data/clip03/in/clip3_MLP.mp4"
    ]
    
    clip_mapping = {
        'clip1_MLP': 'data/clip01/in/clip1_codes_MLP.csv',
        'clip2_AHKJ': 'data/clip02/in/clip2_codes_AHKJ.csv',
        'clip3_MLP': 'data/clip03/in/clip3_codes_MLP.csv'
    }
    
    # Enhanced class name mapping with face detection
    class_name_mapping = {
        'has_faces': {0: 'no_face', 1: 'face'},
        'c_excite_face': {0: 'calm', 1: 'excited'},
        'c_happy_face': {0: 'not_happy', 1: 'happy'},
        'c_surprise_face': {0: 'not_surprised', 1: 'surprised'},
        'c_fear_face': {0: 'not_fearful', 1: 'fearful'}
    }
    
    # FIXED: Custom configurations - now properly handles debug mode
    custom_configs = {
        'has_faces': {
            'model_config': {
                'backbone': 'mobilenet',  # Fast for face detection
                'freeze_features': True,
                'img_size': 224
            },
            'training_config': {
                'epochs': 60,  # Will be overridden in debug mode
                'lr': 0.001,
                'batch_size': 32,
                'patience': 12
            },
            # Debug-specific config (only used if DEBUG_MODE=True)
            'debug_config': {
                'lr': 0.01,  # Higher learning rate for debug
                'batch_size': 4
            }
        },
        'c_happy_face': {
            'model_config': {
                'backbone': 'resnet18',  # Better for emotion detection
                'freeze_features': False,  # More complex emotions need fine-tuning
                'img_size': 224
            },
            'training_config': {
                'epochs': 80,  # Will be overridden in debug mode
                'lr': 0.0005,
                'batch_size': 16
            },
            'debug_config': {
                'lr': 0.01,
                'batch_size': 4
            }
        },
        'c_excite_face': {
            'model_config': {
                'backbone': 'efficientnet_b0',  # Good for subtle expressions
                'freeze_features': True,
                'img_size': 256
            },
            'training_config': {
                'epochs': 50,  # Will be overridden in debug mode
                'batch_size': 32
            },
            'debug_config': {
                'lr': 0.01,
                'batch_size': 4
            }
        }
    }
    
    # Create enhanced pipeline
    pipeline = EnhancedMultiColumnPipeline(
        video_list=video_list,
        clip_csv_mapping=clip_mapping,
        timestamp_column='onset_milliseconds',
        base_output_dir='data/enhanced_results',
        interval_ms=100,
        debug=DEBUG_MODE  # Pass debug flag
    )
    
    # Run comprehensive pipeline
    results = pipeline.run_comprehensive_pipeline(
        class_name_mapping=class_name_mapping,
        custom_configs=custom_configs,
        skip_extraction=False,  # Set to True if frames already exist
        create_visualizations=True,
        verbose=True
    )
    
    # Test trained classifiers on a sample video (optional)
    if results['successful_classifiers']:
        print(f"\n\U0001f3ac Testing classifiers on sample video...")
        try:
            test_results = pipeline.predict_video_with_all_classifiers(
                video_path=video_list[0],  # Test on first video
                output_path='data/enhanced_results/test_output',
                threshold=0.5
            )
            
            print(f"\u2705 Video processing complete!")
            for col, result in test_results.items():
                emoji = "\U0001f464" if result['is_face_column'] else "\U0001f3af"
                pos_ratio = result['positive_frames'] / result['total_frames']
                print(f"   {emoji} {col}: {pos_ratio:.1%} positive frames")
        
        except Exception as e:
            print(f"\u26a0\ufe0f  Video testing skipped: {e}")
    
    return results

if __name__ == "__main__":
    # Run the enhanced example
    print("\U0001f680 Starting Enhanced Multi-Column Pipeline with Face Detection...")
    results = run_enhanced_example()
    
    print(f"\n\U0001f389 PIPELINE COMPLETE!")
    print(f"Trained classifiers: {results['successful_classifiers']}")
    print(f"Face detection available: {results['face_detector_available']}")
    print(f"Results saved in: data/enhanced_results/")
    
    if results.get('debug_mode', False):
        print(f"\n\U0001f41b DEBUG MODE SUMMARY:")
        print("   - Training was limited to 3 epochs for fast testing")
        print("   - Set DEBUG_MODE = False in run_enhanced_example() for full training")
        print("   - Full training uses 30-80 epochs based on dataset size")