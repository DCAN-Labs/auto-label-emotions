#!/usr/bin/env python3
"""
Core Enhanced Multi-Column Classification Pipeline

This module contains the main pipeline orchestrator that coordinates
all components of the enhanced multi-column classification system.
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from frame_organizer import organize_frames_from_multiple_clips, MultiClipFrameOrganizer
from mp4_frame_extractor import extract_frames_from_videos

from .analysis import ColumnAnalyzer
from .visualization import DashboardGenerator
from .training import ModelTrainer
from .utils import ConfigurationManager, ResultsManager


class EnhancedMultiColumnPipeline:
    """
    Enhanced pipeline orchestrator for multi-column classification with 
    integrated face detection analysis and specialized retraining
    """
    
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
        
        # Initialize components
        self.analyzer = ColumnAnalyzer(
            clip_csv_mapping=clip_csv_mapping,
            timestamp_column=timestamp_column,
            debug=debug
        )
        
        self.dashboard_generator = DashboardGenerator(debug=debug)
        
        self.model_trainer = ModelTrainer(
            base_output_dir=base_output_dir,
            debug=debug
        )
        
        self.config_manager = ConfigurationManager(debug=debug)
        self.results_manager = ResultsManager(base_output_dir=base_output_dir)
        
        # Results storage
        self.analysis_results = {}
        self.training_results = {}
        
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
        
        self.analysis_results = self.analyzer.analyze_all_columns(verbose=verbose)
        return self.analysis_results
    
    def create_visualization_dashboard(self, save_path: Optional[str] = None):
        """Create comprehensive visualization dashboard for all columns"""
        if not self.analysis_results:
            raise ValueError("Run comprehensive_column_analysis() first")
        
        if save_path is None:
            save_path = os.path.join(self.base_output_dir, 'analysis_dashboard.png')
        
        self.dashboard_generator.create_dashboard(
            column_stats=self.analysis_results['column_stats'],
            save_path=save_path
        )
    
    def extract_frames(self, verbose: bool = True) -> List[Dict[str, Any]]:
        """Extract frames from all videos"""
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
            class_names = self.config_manager.get_class_names(
                column=column,
                class_name_mapping=class_name_mapping
            )
            
            # Create dataset directory
            dataset_dir = os.path.join(self.base_output_dir, f'{column}_dataset')
            
            emoji = "\U0001f464" if column == 'has_faces' else "\U0001f3af"
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
                    'is_face_column': column == 'has_faces',
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
                    'is_face_column': column == 'has_faces'
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
        
        self.training_results = self.model_trainer.train_all_classifiers(
            columns_to_train=columns_to_train,
            analysis_results=self.analysis_results,
            clip_csv_mapping=self.clip_csv_mapping,
            class_name_mapping=class_name_mapping,
            custom_configs=custom_configs,
            verbose=verbose
        )
        
        return self.training_results
    
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
        
        # Step 1: Comprehensive analysis
        print("\U0001f4ca Step 1: Comprehensive Column Analysis")
        analysis_results = self.comprehensive_column_analysis(verbose=verbose)
        
        # Step 2: Create visualizations if requested
        if create_visualizations:
            print("\n\U0001f4c8 Step 2: Creating Analysis Dashboard")
            self.create_visualization_dashboard()
        
        # Step 3: Extract frames
        if not skip_extraction:
            print("\n\U0001f3ac Step 3: Extracting Frames")
            extraction_results = self.extract_frames(verbose=verbose)
        else:
            print("\n\u23e9 Step 3: Skipping frame extraction")
            extraction_results = []
        
        # Step 4: Create datasets
        print("\n\U0001f4c1 Step 4: Creating Datasets")
        dataset_results = self.create_datasets_for_all_columns(
            columns_to_process=columns_to_process,
            class_name_mapping=class_name_mapping,
            verbose=verbose
        )
        
        # Step 5: Train classifiers
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
            'performance_summary': self.results_manager.generate_performance_summary(training_results)
        }
        
        # Print comprehensive summary
        self.results_manager.print_comprehensive_summary(final_results, verbose=verbose)
        
        # Save complete results
        self.results_manager.save_results(final_results)
        
        return final_results
    
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