#!/usr/bin/env python3
"""
Column Analysis Module

This module handles comprehensive analysis of all boolean columns
with special focus on face detection patterns and temporal analysis.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Any, Optional
from collections import defaultdict


class FaceTemporalAnalyzer:
    """Specialized analyzer for face detection temporal patterns"""
    
    def __init__(self, timestamp_column: str = 'onset_milliseconds'):
        self.timestamp_column = timestamp_column
        self.face_column = 'has_faces'
    
    def analyze_face_temporal_patterns(self, df: pd.DataFrame, clip_name: str) -> Dict[str, Any]:
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


class ModelRecommendationEngine:
    """Engine for recommending optimal model configurations"""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.face_column = 'has_faces'
    
    def recommend_model_type(self, column: str, total_samples: int, positive_ratio: float) -> Dict[str, Any]:
        """Recommend optimal model configuration for each column"""
        
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
        
        # Adjust based on data characteristics AND debug mode
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


class ColumnAnalyzer:
    """Main analyzer for comprehensive column analysis"""
    
    def __init__(self, 
                 clip_csv_mapping: Dict[str, str],
                 timestamp_column: str = 'onset_milliseconds',
                 debug: bool = False):
        self.clip_csv_mapping = clip_csv_mapping
        self.timestamp_column = timestamp_column
        self.debug = debug
        self.face_column = 'has_faces'
        
        # Initialize sub-analyzers
        self.face_analyzer = FaceTemporalAnalyzer(timestamp_column)
        self.recommendation_engine = ModelRecommendationEngine(debug)
    
    def analyze_all_columns(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Comprehensive analysis of all boolean columns with special focus on face detection
        """
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
                boolean_columns = self._find_boolean_columns(df, verbose)
                
                # Special analysis for face detection column
                if self.face_column in df.columns:
                    face_temporal = self.face_analyzer.analyze_face_temporal_patterns(df, clip_name)
                    face_analysis[clip_name] = face_temporal
                
                # Track columns across all clips
                self._track_columns_across_clips(boolean_columns, clip_name, all_columns)
                
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
                'recommended_model': self.recommendation_engine.recommend_model_type(
                    col, total_count, total_1/total_count if total_count > 0 else 0
                )
            }
            
            # Add face-specific analysis
            if col == self.face_column:
                column_stats[col]['face_temporal_analysis'] = face_analysis
        
        # Print comprehensive analysis
        if verbose:
            self._print_comprehensive_analysis(column_stats)
        
        return {
            'column_stats': column_stats,
            'face_analysis': face_analysis,
            'recommendations': self._generate_training_recommendations(column_stats)
        }
    
    def _find_boolean_columns(self, df: pd.DataFrame, verbose: bool) -> List[Dict[str, Any]]:
        """Find all boolean columns in a DataFrame"""
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
                
                boolean_columns.append({
                    'column': col,
                    'unique_values': sorted(unique_vals),
                    'binary_values': sorted(binary_vals),
                    'value_counts': value_counts.to_dict(),
                    'total_count': len(df[col].dropna()),
                    'is_face_column': col == self.face_column
                })
        
        if verbose:
            print(f"   Boolean columns found: {len(boolean_columns)}")
            for bcol in boolean_columns:
                col_name = bcol['column']
                counts = bcol['value_counts']
                emoji = "\U0001f464" if bcol['is_face_column'] else "\U0001f3af"
                print(f"     {emoji} {col_name}: {counts}")
        
        return boolean_columns
    
    def _track_columns_across_clips(self, boolean_columns: List[Dict[str, Any]], 
                                   clip_name: str, all_columns: Dict[str, List]):
        """Track column statistics across all clips"""
        for bcol in boolean_columns:
            col = bcol['column']
            if col not in all_columns:
                all_columns[col] = []
            all_columns[col].append({
                'clip': clip_name,
                'stats': bcol['value_counts'],
                'total': bcol['total_count']
            })
    
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
            'debug_mode': self.debug
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