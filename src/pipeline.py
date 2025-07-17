# Multi-Column Boolean Classification Pipeline
import pandas as pd
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json

from frame_organizer import organize_frames_from_multiple_clips, MultiClipFrameOrganizer
from mp4_frame_extractor import extract_frames_from_videos
from pytorch_cartoon_face_detector import BinaryClassifier

class MultiColumnClassificationPipeline:
    def __init__(self, 
                 video_list: List[str],
                 clip_csv_mapping: Dict[str, str],
                 timestamp_column: str = 'onset_milliseconds',
                 base_output_dir: str = 'data/combined',
                 interval_ms: int = 100):
        """
        Initialize multi-column classification pipeline
        
        Args:
            video_list: List of video file paths
            clip_csv_mapping: Dictionary mapping clip names to CSV file paths
            timestamp_column: Name of the timestamp column
            base_output_dir: Base output directory for all files
            interval_ms: Frame extraction interval in milliseconds
        """
        self.video_list = video_list
        self.clip_csv_mapping = clip_csv_mapping
        self.timestamp_column = timestamp_column
        self.base_output_dir = base_output_dir
        self.interval_ms = interval_ms
        
        # Create base directories
        Path(base_output_dir).mkdir(parents=True, exist_ok=True)
        
        self.frames_dir = os.path.join(base_output_dir, 'frames')
        self.results = {
            'extraction': None,
            'columns_analyzed': {},
            'classifiers': {},
            'datasets_created': []
        }
    
    def analyze_csv_columns(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Analyze all CSV files to find boolean columns
        
        Args:
            verbose: If True, print detailed analysis
        
        Returns:
            Dictionary containing column analysis results
        """
        print("\U0001f50d ANALYZING CSV COLUMNS FOR BOOLEAN CLASSIFICATION")
        print("="*60)
        
        all_columns = {}
        column_stats = {}
        
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
                
                # Find potential boolean columns
                boolean_columns = []
                for col in df.columns:
                    if col == self.timestamp_column:
                        continue
                    
                    # Check if column contains only 0/1 or boolean values
                    unique_vals = df[col].dropna().unique()
                    
                    # Check for binary columns (0/1, True/False, or similar)
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
                            'total_count': len(df[col].dropna())
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
                        print(f"     \u2022 {col_name}: {counts}")
                
                self.results['columns_analyzed'][clip_name] = {
                    'csv_file': csv_file,
                    'total_rows': len(df),
                    'boolean_columns': boolean_columns
                }
                
            except Exception as e:
                if verbose:
                    print(f"\u274c Error analyzing {csv_file}: {e}")
                self.results['columns_analyzed'][clip_name] = {
                    'csv_file': csv_file,
                    'error': str(e)
                }
        
        # Calculate combined statistics
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
                'clips_with_column': len(clip_stats)
            }
        
        if verbose:
            print(f"\n\U0001f4c8 OVERALL COLUMN ANALYSIS:")
            print(f"Found {len(column_stats)} boolean columns across all clips:")
            
            for col, stats in column_stats.items():
                print(f"\n\U0001f3af Column: {col}")
                print(f"   Total samples: {stats['total_samples']:,}")
                print(f"   Class 0: {stats['class_0_count']:,} ({stats['class_balance']['class_0_ratio']:.1%})")
                print(f"   Class 1: {stats['class_1_count']:,} ({stats['class_balance']['class_1_ratio']:.1%})")
                print(f"   Available in {stats['clips_with_column']}/{len(self.clip_csv_mapping)} clips")
                
                # Check for class imbalance
                ratio = min(stats['class_balance']['class_0_ratio'], stats['class_balance']['class_1_ratio'])
                if ratio < 0.1:
                    print(f"   \u26a0\ufe0f  Warning: Severe class imbalance (minority class: {ratio:.1%})")
                elif ratio < 0.2:
                    print(f"   \u26a0\ufe0f  Warning: Class imbalance (minority class: {ratio:.1%})")
                else:
                    print(f"   \u2705 Good class balance")
        
        print("="*60)
        return column_stats
    
    def extract_frames(self, verbose: bool = True) -> List[Dict[str, Any]]:
        """Extract frames from all videos"""
        print("\U0001f4f9 STEP 1: EXTRACTING FRAMES FROM VIDEOS")
        print("-" * 40)
        
        extraction_results = extract_frames_from_videos(
            self.video_list,
            output_dir=self.frames_dir,
            interval_ms=self.interval_ms,
            include_video_name=True,
            skip_black_frames=True,
            verbose=verbose
        )
        
        self.results['extraction'] = extraction_results
        
        total_extracted = sum(r.get('extracted_count', 0) for r in extraction_results if 'extracted_count' in r)
        print(f"\u2705 Frame extraction complete: {total_extracted:,} frames extracted")
        
        return extraction_results
    
    def create_datasets_for_columns(self, 
                                  columns_to_process: List[str],
                                  class_name_mapping: Optional[Dict[str, Dict[int, str]]] = None,
                                  verbose: bool = True) -> Dict[str, Any]:
        """
        Create datasets for multiple boolean columns
        
        Args:
            columns_to_process: List of column names to create datasets for
            class_name_mapping: Optional mapping of column names to class names
                                Format: {'column_name': {0: 'negative_class', 1: 'positive_class'}}
            verbose: If True, print detailed information
        
        Returns:
            Dictionary containing dataset creation results
        """
        print("\n\U0001f4ca STEP 2: CREATING DATASETS FOR MULTIPLE COLUMNS")
        print("-" * 40)
        
        dataset_results = {}
        
        for column in columns_to_process:
            print(f"\n\U0001f3af Processing column: {column}")
            
            # Determine class names
            if class_name_mapping and column in class_name_mapping:
                class_names = class_name_mapping[column]
            else:
                # Generate default class names
                class_names = {0: f'not_{column}', 1: column}
            
            # Create dataset directory for this column
            dataset_dir = os.path.join(self.base_output_dir, f'{column}_dataset')
            
            print(f"   Class mapping: {class_names}")
            print(f"   Dataset directory: {dataset_dir}")
            
            try:
                # Organize frames for this column
                organizer = MultiClipFrameOrganizer(
                    dataset_dir=dataset_dir,
                    output_frames_dir=self.frames_dir,
                    file_extension='jpg',
                    use_move=False  # Copy files so we can use them for multiple columns
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
                    'success': True
                }
                
                print(f"   \u2705 Success: {total_organized:,} frames organized from {successful_clips}/{len(self.clip_csv_mapping)} clips")
                
                # Save detailed results
                organizer.save_results(os.path.join(dataset_dir, 'organization_results.json'))
                
                self.results['datasets_created'].append(column)
                
            except Exception as e:
                print(f"   \u274c Error processing column {column}: {e}")
                dataset_results[column] = {
                    'error': str(e),
                    'success': False
                }
        
        return dataset_results
    
    def train_classifiers(self,
                         columns_to_train: List[str],
                         model_configs: Optional[Dict[str, Dict[str, Any]]] = None,
                         training_configs: Optional[Dict[str, Dict[str, Any]]] = None,
                         class_name_mapping: Optional[Dict[str, Dict[int, str]]] = None,
                         verbose: bool = True) -> Dict[str, Any]:
        """
        Train classifiers for multiple columns
        
        Args:
            columns_to_train: List of column names to train classifiers for
            model_configs: Optional model configurations per column
            training_configs: Optional training configurations per column
            class_name_mapping: Optional class name mapping per column
            verbose: If True, print detailed information
        
        Returns:
            Dictionary containing training results for each classifier
        """
        print("\n\U0001f9e0 STEP 3: TRAINING CLASSIFIERS FOR MULTIPLE COLUMNS")
        print("-" * 40)
        
        training_results = {}
        
        for column in columns_to_train:
            print(f"\n\U0001f3af Training classifier for: {column}")
            
            # Get configurations
            model_config = model_configs.get(column, {}) if model_configs else {}
            training_config = training_configs.get(column, {}) if training_configs else {}
            
            # Determine class names
            if class_name_mapping and column in class_name_mapping:
                class_names = class_name_mapping[column]
                positive_class = class_names[1]
                negative_class = class_names[0]
            else:
                positive_class = column
                negative_class = f'not_{column}'
            
            dataset_dir = os.path.join(self.base_output_dir, f'{column}_dataset')
            
            if not os.path.exists(dataset_dir):
                print(f"   \u274c Error: Dataset directory not found: {dataset_dir}")
                training_results[column] = {
                    'error': f'Dataset directory not found: {dataset_dir}',
                    'success': False
                }
                continue
            
            try:
                # Create classifier
                from pytorch_cartoon_face_detector import BinaryClassifier
                
                classifier = BinaryClassifier(
                    task_name=f"{column}_classification",
                    class_names={0: negative_class, 1: positive_class},
                    model_type=model_config.get('model_type', 'transfer'),
                    backbone=model_config.get('backbone', 'mobilenet'),
                    img_size=model_config.get('img_size', 224),
                    device=model_config.get('device', None)
                )
                
                # Create model
                model = classifier.create_model(
                    pretrained=model_config.get('pretrained', True),
                    freeze_features=model_config.get('freeze_features', True)
                )
                
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"   Model created: {trainable_params:,} trainable parameters")
                
                # Load dataset
                train_loader, val_loader = classifier.load_dataset(
                    dataset_dir,
                    batch_size=training_config.get('batch_size', 32),
                    val_split=training_config.get('val_split', 0.2)
                )
                
                # Train model
                model_save_path = os.path.join(self.base_output_dir, f'{column}_classifier.pth')
                
                history = classifier.train_model(
                    train_loader,
                    val_loader,
                    epochs=training_config.get('epochs', 50),
                    lr=training_config.get('lr', 0.001),
                    patience=training_config.get('patience', 10),
                    save_path=model_save_path
                )
                
                # Evaluate model
                metrics = classifier.evaluate_model(val_loader)
                
                # Store classifier and results
                self.results['classifiers'][column] = classifier
                
                training_results[column] = {
                    'classifier': classifier,
                    'model_path': model_save_path,
                    'dataset_dir': dataset_dir,
                    'class_names': {0: negative_class, 1: positive_class},
                    'training_history': history,
                    'evaluation_metrics': metrics,
                    'model_config': model_config,
                    'training_config': training_config,
                    'success': True
                }
                
                print(f"   \u2705 Training complete: {metrics['accuracy']:.1f}% accuracy")
                
            except Exception as e:
                print(f"   \u274c Error training classifier for {column}: {e}")
                training_results[column] = {
                    'error': str(e),
                    'success': False
                }
        
        return training_results
    
    def run_complete_pipeline(self,
                            columns_to_process: Optional[List[str]] = None,
                            class_name_mapping: Optional[Dict[str, Dict[int, str]]] = None,
                            model_configs: Optional[Dict[str, Dict[str, Any]]] = None,
                            training_configs: Optional[Dict[str, Dict[str, Any]]] = None,
                            skip_extraction: bool = False,
                            verbose: bool = True) -> Dict[str, Any]:
        """
        Run the complete multi-column classification pipeline
        
        Args:
            columns_to_process: List of columns to process (if None, will analyze and ask)
            class_name_mapping: Custom class names for each column
            model_configs: Model configurations for each column
            training_configs: Training configurations for each column
            skip_extraction: Skip frame extraction if frames already exist
            verbose: Print detailed information
        
        Returns:
            Complete pipeline results
        """
        print("\U0001f680 MULTI-COLUMN CLASSIFICATION PIPELINE")
        print("="*60)
        
        # Step 0: Analyze columns if not specified
        if columns_to_process is None:
            column_stats = self.analyze_csv_columns(verbose=verbose)
            
            # Show available columns and let user choose
            available_columns = list(column_stats.keys())
            if not available_columns:
                print("\u274c No boolean columns found in CSV files!")
                return {'error': 'No boolean columns found'}
            
            print(f"\n\U0001f4cb Available boolean columns: {available_columns}")
            
            # Auto-select columns with good balance and sufficient data
            recommended_columns = []
            for col, stats in column_stats.items():
                if (stats['total_samples'] >= 100 and  # Minimum samples
                    min(stats['class_balance']['class_0_ratio'], stats['class_balance']['class_1_ratio']) >= 0.1):  # Not too imbalanced
                    recommended_columns.append(col)
            
            if recommended_columns:
                print(f"\U0001f3af Recommended columns (good balance & sufficient data): {recommended_columns}")
                columns_to_process = recommended_columns
            else:
                columns_to_process = available_columns[:3]  # Take first 3 as fallback
                print(f"\u26a0\ufe0f  Using first {len(columns_to_process)} columns as no ideal candidates found")
        
        # Step 1: Extract frames
        if not skip_extraction:
            extraction_results = self.extract_frames(verbose=verbose)
        else:
            print("\u23ed\ufe0f  Skipping frame extraction (skip_extraction=True)")
            extraction_results = []
        
        # Step 2: Create datasets
        dataset_results = self.create_datasets_for_columns(
            columns_to_process=columns_to_process,
            class_name_mapping=class_name_mapping,
            verbose=verbose
        )
        
        # Step 3: Train classifiers
        successful_datasets = [col for col, result in dataset_results.items() if result.get('success', False)]
        
        if successful_datasets:
            training_results = self.train_classifiers(
                columns_to_train=successful_datasets,
                model_configs=model_configs,
                training_configs=training_configs,
                class_name_mapping=class_name_mapping,
                verbose=verbose
            )
        else:
            print("\u274c No successful datasets created, skipping training")
            training_results = {}
        
        # Compile final results
        final_results = {
            'extraction_results': extraction_results,
            'columns_processed': columns_to_process,
            'dataset_results': dataset_results,
            'training_results': training_results,
            'successful_classifiers': [col for col, result in training_results.items() if result.get('success', False)],
            'pipeline_config': {
                'video_list': self.video_list,
                'clip_csv_mapping': self.clip_csv_mapping,
                'base_output_dir': self.base_output_dir,
                'interval_ms': self.interval_ms
            }
        }
        
        # Print final summary
        self._print_pipeline_summary(final_results, verbose=verbose)
        
        # Save complete results
        results_file = os.path.join(self.base_output_dir, 'pipeline_results.json')
        with open(results_file, 'w') as f:
            # Convert non-serializable objects for JSON
            json_results = self._prepare_results_for_json(final_results)
            json.dump(json_results, f, indent=2, default=str)
        
        print(f"\U0001f4c1 Complete results saved to: {results_file}")
        
        return final_results
    
    def _prepare_results_for_json(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare results for JSON serialization by removing non-serializable objects"""
        json_results = {}
        for key, value in results.items():
            if key == 'training_results':
                # Remove classifier objects, keep only serializable data
                json_training = {}
                for col, training_result in value.items():
                    if isinstance(training_result, dict):
                        json_training[col] = {k: v for k, v in training_result.items() if k != 'classifier'}
                    else:
                        json_training[col] = training_result
                json_results[key] = json_training
            else:
                json_results[key] = value
        return json_results
    
    def _print_pipeline_summary(self, results: Dict[str, Any], verbose: bool = True):
        """Print comprehensive pipeline summary"""
        print("\n" + "="*60)
        print("\U0001f4ca MULTI-COLUMN PIPELINE SUMMARY")
        print("="*60)
        
        # Extraction summary
        extraction_count = sum(r.get('extracted_count', 0) for r in results['extraction_results'] if isinstance(r, dict))
        print(f"\U0001f4f9 Frame extraction: {extraction_count:,} frames extracted")
        
        # Dataset creation summary
        successful_datasets = sum(1 for r in results['dataset_results'].values() if r.get('success', False))
        total_datasets = len(results['dataset_results'])
        print(f"\U0001f4ca Dataset creation: {successful_datasets}/{total_datasets} successful")
        
        # Training summary
        successful_classifiers = len(results['successful_classifiers'])
        total_training_attempts = len(results['training_results'])
        print(f"\U0001f9e0 Classifier training: {successful_classifiers}/{total_training_attempts} successful")
        
        # Individual classifier results
        if results['successful_classifiers']:
            print(f"\n\U0001f3c6 SUCCESSFUL CLASSIFIERS:")
            for col in results['successful_classifiers']:
                training_result = results['training_results'][col]
                metrics = training_result['evaluation_metrics']
                class_names = training_result['class_names']
                
                print(f"\n  \U0001f3af {col.upper()}:")
                print(f"     Classes: {class_names[0]} / {class_names[1]}")
                print(f"     Accuracy: {metrics['accuracy']:.1f}%")
                print(f"     Precision: {metrics['precision']:.3f}")
                print(f"     Recall: {metrics['recall']:.3f}")
                print(f"     F1 Score: {metrics['f1_score']:.3f}")
                print(f"     Model: {training_result['model_path']}")
        
        # Failed items
        failed_datasets = [col for col, r in results['dataset_results'].items() if not r.get('success', False)]
        failed_training = [col for col, r in results['training_results'].items() if not r.get('success', False)]
        
        if failed_datasets or failed_training:
            print(f"\n\u274c FAILURES:")
            if failed_datasets:
                print(f"   Dataset creation failed: {failed_datasets}")
            if failed_training:
                print(f"   Training failed: {failed_training}")
        
        print(f"\n\U0001f4c1 All results saved in: {self.base_output_dir}")
        print("="*60)
        print("\U0001f389 Multi-column classification pipeline complete!")
        print("="*60)

# Example usage function
def run_multi_column_example():
    """Example of running the multi-column pipeline"""
    
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
    
    # Custom class names for different emotions
    class_name_mapping = {
        'c_excite_face': {0: 'calm', 1: 'excited'},
        'c_happy_face': {0: 'not_happy', 1: 'happy'},
        'c_surprise_face': {0: 'not_surprised', 1: 'surprised'},
        'has_faces': {0: 'no_face', 1: 'face'}
    }
    
    # Different model configurations for different tasks
    model_configs = {
        'c_excite_face': {
            'model_type': 'transfer',
            'backbone': 'mobilenet',
            'pretrained': True,
            'freeze_features': True
        },
        'c_happy_face': {
            'model_type': 'transfer', 
            'backbone': 'resnet18',
            'pretrained': True,
            'freeze_features': False  # Unfreeze for more complex emotion
        },
        'has_faces': {
            'model_type': 'cnn',  # Custom CNN for face detection
        }
    }
    
    # Different training configurations
    training_configs = {
        'c_excite_face': {
            'epochs': 30,
            'lr': 0.001,
            'batch_size': 32
        },
        'c_happy_face': {
            'epochs': 50,
            'lr': 0.0001,  # Lower learning rate for unfrozen model
            'batch_size': 16
        }
    }
    
    # Create and run pipeline
    pipeline = MultiColumnClassificationPipeline(
        video_list=video_list,
        clip_csv_mapping=clip_mapping,
        timestamp_column='onset_milliseconds',
        base_output_dir='data/multi_column_results',
        interval_ms=100
    )
    
    # Option 1: Let pipeline analyze and recommend columns
    # results = pipeline.run_complete_pipeline(
    #     class_name_mapping=class_name_mapping,
    #     model_configs=model_configs,
    #     training_configs=training_configs,
    #     verbose=True
    # )
    
    # Option 2: Specify exact columns to process
    results = pipeline.run_complete_pipeline(
        columns_to_process=['c_excite_face', 'c_happy_face', 'c_fear_face'],
        class_name_mapping=class_name_mapping,
        model_configs=model_configs,
        training_configs=training_configs,
        verbose=True
    )
    
    return results

if __name__ == "__main__":
    # Run the example
    results = run_multi_column_example()
    
    print("\n\U0001f3af PIPELINE COMPLETE!")
    print("Trained classifiers for:", results['successful_classifiers'])
