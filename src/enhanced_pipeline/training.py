#!/usr/bin/env python3
"""
Training Module

This module handles model training, classifier management, and 
optimized configurations for all column types including face detection.
"""

import os
import time
from typing import Dict, List, Any, Optional, Tuple
from pytorch_cartoon_face_detector import BinaryClassifier, FaceDetector


class DataLoaderOptimizer:
    """Optimizer for creating balanced and efficient data loaders"""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
    
    def create_optimized_dataloaders(self, 
                                   classifier, 
                                   dataset_dir: str, 
                                   column: str, 
                                   training_config: Dict[str, Any],
                                   analysis_results: Dict[str, Any]):
        """Create optimized dataloaders with class balancing for imbalanced datasets"""
        from torch.utils.data import WeightedRandomSampler
        import torch
        
        # Use appropriate batch size for debug mode
        batch_size = training_config.get('batch_size', 4 if self.debug else 32)
        
        # Load dataset normally first
        train_loader, val_loader = classifier.load_dataset(
            dataset_dir,
            batch_size=batch_size,
            val_split=training_config.get('val_split', 0.2)
        )
        
        # Check if we need class balancing (skip in debug mode for simplicity)
        if not self.debug:
            column_stats = analysis_results['column_stats'][column]
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


class ClassifierFactory:
    """Factory for creating appropriate classifiers based on column type"""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.face_column = 'has_faces'
    
    def create_classifier(self, 
                         column: str, 
                         model_config: Dict[str, Any],
                         class_names: Dict[int, str]) -> Tuple[Any, str]:
        """
        Create appropriate classifier based on column type
        
        Returns:
            Tuple of (classifier, emoji_indicator)
        """
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
        
        return classifier, emoji


class ModelTrainer:
    """Main trainer for handling all classifier training operations"""
    
    def __init__(self, base_output_dir: str, debug: bool = False):
        self.base_output_dir = base_output_dir
        self.debug = debug
        self.face_column = 'has_faces'
        
        # Initialize components
        self.dataloader_optimizer = DataLoaderOptimizer(debug)
        self.classifier_factory = ClassifierFactory(debug)
        
        # Store trained models
        self.trained_classifiers = {}
        self.face_detector = None
    
    def train_single_classifier(self,
                              column: str,
                              model_config: Dict[str, Any],
                              training_config: Dict[str, Any],
                              class_names: Dict[int, str],
                              analysis_results: Dict[str, Any],
                              verbose: bool = True) -> Dict[str, Any]:
        """
        Train a single classifier with optimized configuration
        """
        dataset_dir = os.path.join(self.base_output_dir, f'{column}_dataset')
        
        if not os.path.exists(dataset_dir):
            return {
                'error': f'Dataset directory not found: {dataset_dir}',
                'success': False,
                'is_face_column': column == self.face_column
            }
        
        try:
            # Create appropriate classifier
            classifier, emoji = self.classifier_factory.create_classifier(
                column, model_config, class_names
            )
            
            # Create model
            model = classifier.create_model(
                pretrained=model_config.get('pretrained', True),
                freeze_features=model_config.get('freeze_features', True)
            )
            
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            frozen_status = 'frozen' if model_config.get('freeze_features') else 'unfrozen'
            debug_indicator = " (DEBUG MODE)" if self.debug else ""
            
            if verbose:
                print(f"   {emoji} Model created: {trainable_params:,} trainable parameters{debug_indicator}")
                print(f"   \U0001f4cb Config: {model_config['backbone']} ({frozen_status})")
            
            # Load dataset with special handling for imbalanced data
            train_loader, val_loader = self.dataloader_optimizer.create_optimized_dataloaders(
                classifier, dataset_dir, column, training_config, analysis_results
            )
            
            # Train model
            model_save_path = os.path.join(self.base_output_dir, f'{column}_classifier.pth')
            
            epochs = training_config.get('epochs', 3 if self.debug else 50)
            if verbose:
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
            if verbose:
                print(f"   \U0001f3af Evaluating {column} classifier...")
            metrics = classifier.evaluate_model(val_loader)
            
            # Store results
            result = {
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
                result['face_detector_ready'] = True
            
            # Store in trained classifiers
            self.trained_classifiers[column] = classifier
            
            debug_note = " (DEBUG - minimal training)" if self.debug else ""
            if verbose:
                print(f"   \u2705 Training complete: {metrics['accuracy']:.1f}% accuracy in {training_time/60:.1f} minutes{debug_note}")
            
            return result
            
        except Exception as e:
            return {
                'error': str(e),
                'success': False,
                'is_face_column': column == self.face_column,
                'debug_mode': self.debug
            }
    
    def train_all_classifiers(self,
                            columns_to_train: List[str],
                            analysis_results: Dict[str, Any],
                            clip_csv_mapping: Dict[str, str],
                            class_name_mapping: Optional[Dict[str, Dict[int, str]]] = None,
                            custom_configs: Optional[Dict[str, Dict[str, Any]]] = None,
                            verbose: bool = True) -> Dict[str, Any]:
        """
        Train classifiers for all specified columns with optimized configurations
        """
        training_results = {}
        
        for column in columns_to_train:
            print(f"\n\U0001f3af Training classifier for: {column}")
            
            # Get recommended configurations (already adjusted for debug mode)
            recommended = analysis_results['column_stats'][column]['recommended_model']
            model_config = recommended['model_config'].copy()
            training_config = recommended['training_config'].copy()
            
            # Apply custom overrides with debug mode consideration
            if custom_configs and column in custom_configs:
                if not self.debug:  # Normal mode: apply custom configs
                    if 'model_config' in custom_configs[column]:
                        model_config.update(custom_configs[column]['model_config'])
                    if 'training_config' in custom_configs[column]:
                        training_config.update(custom_configs[column]['training_config'])
                else:  # Debug mode: only apply debug-specific configs
                    print(f"\U0001f41b DEBUG: Skipping custom configs to maintain debug settings")
                    if 'debug_config' in custom_configs[column]:
                        training_config.update(custom_configs[column]['debug_config'])
            
            # Ensure debug mode settings are preserved
            if self.debug:
                training_config.update({
                    'epochs': 3,
                    'batch_size': 4,
                    'patience': 2,
                    'lr': 0.01
                })
                print(f"\U0001f41b DEBUG: Final training config: {training_config}")
            
            # Determine class names
            class_names = self._get_class_names(column, class_name_mapping)
            
            # Train the classifier
            result = self.train_single_classifier(
                column=column,
                model_config=model_config,
                training_config=training_config,
                class_names=class_names,
                analysis_results=analysis_results,
                verbose=verbose
            )
            
            training_results[column] = result
        
        return training_results
    
    def _get_class_names(self, 
                        column: str, 
                        class_name_mapping: Optional[Dict[str, Dict[int, str]]]) -> Dict[int, str]:
        """Get appropriate class names for a column"""
        if class_name_mapping and column in class_name_mapping:
            return class_name_mapping[column]
        elif column == self.face_column:
            return {0: 'no_face', 1: 'face'}
        else:
            # Generate contextual class names
            if 'happy' in column.lower():
                return {0: 'not_happy', 1: 'happy'}
            elif 'excite' in column.lower():
                return {0: 'calm', 1: 'excited'}
            elif 'surprise' in column.lower():
                return {0: 'not_surprised', 1: 'surprised'}
            elif 'fear' in column.lower():
                return {0: 'not_fearful', 1: 'fearful'}
            else:
                return {0: f'not_{column}', 1: column}
    
    def get_trained_classifier(self, column: str):
        """Get a specific trained classifier"""
        return self.trained_classifiers.get(column)
    
    def get_face_detector(self):
        """Get the trained face detector if available"""
        return self.face_detector
    
    def save_all_models(self, save_dir: Optional[str] = None):
        """Save all trained models to disk"""
        if save_dir is None:
            save_dir = self.base_output_dir
        
        saved_models = {}
        
        for column, classifier in self.trained_classifiers.items():
            model_path = os.path.join(save_dir, f'{column}_classifier.pth')
            try:
                classifier.save_model(model_path)
                saved_models[column] = model_path
                print(f"\u2705 Saved {column} classifier to: {model_path}")
            except Exception as e:
                print(f"\u274c Failed to save {column} classifier: {e}")
                saved_models[column] = f"Error: {e}"
        
        return saved_models
    
    def load_trained_classifiers(self, 
                                model_paths: Dict[str, str],
                                analysis_results: Dict[str, Any],
                                class_name_mapping: Optional[Dict[str, Dict[int, str]]] = None) -> Dict[str, Any]:
        """Load previously trained classifiers from disk"""
        loaded_results = {}
        
        for column, model_path in model_paths.items():
            if not os.path.exists(model_path):
                loaded_results[column] = {
                    'error': f'Model file not found: {model_path}',
                    'success': False
                }
                continue
            
            try:
                # Get model configuration from analysis results
                if column not in analysis_results['column_stats']:
                    loaded_results[column] = {
                        'error': f'No analysis data found for column: {column}',
                        'success': False
                    }
                    continue
                
                recommended = analysis_results['column_stats'][column]['recommended_model']
                model_config = recommended['model_config']
                class_names = self._get_class_names(column, class_name_mapping)
                
                # Create classifier
                classifier, emoji = self.classifier_factory.create_classifier(
                    column, model_config, class_names
                )
                
                # Load the trained model
                classifier.load_model(model_path)
                
                # Store the loaded classifier
                self.trained_classifiers[column] = classifier
                
                if column == self.face_column:
                    self.face_detector = classifier
                
                loaded_results[column] = {
                    'classifier': classifier,
                    'model_path': model_path,
                    'class_names': class_names,
                    'is_face_column': column == self.face_column,
                    'success': True
                }
                
                print(f"   {emoji} Loaded {column} classifier from: {model_path}")
                
            except Exception as e:
                loaded_results[column] = {
                    'error': f'Failed to load classifier: {e}',
                    'success': False
                }
                print(f"\u274c Failed to load {column} classifier: {e}")
        
        return loaded_results


class BatchTrainingManager:
    """Manager for coordinating batch training operations"""
    
    def __init__(self, trainer: ModelTrainer, debug: bool = False):
        self.trainer = trainer
        self.debug = debug
    
    def train_by_groups(self,
                       analysis_results: Dict[str, Any],
                       clip_csv_mapping: Dict[str, str],
                       class_name_mapping: Optional[Dict[str, Dict[int, str]]] = None,
                       custom_configs: Optional[Dict[str, Dict[str, Any]]] = None,
                       verbose: bool = True) -> Dict[str, Any]:
        """
        Train classifiers in optimized groups based on model architecture
        """
        recommendations = analysis_results['recommendations']
        training_groups = recommendations['batch_training_groups']
        
        all_results = {}
        
        for group in training_groups:
            group_name = group['name']
            columns = group['columns']
            
            if not columns:
                continue
            
            print(f"\n\U0001f504 Training {group_name}: {len(columns)} classifiers")
            if self.debug:
                print("\U0001f41b DEBUG: Using minimal parameters for group training")
            
            # Train all classifiers in this group
            group_results = self.trainer.train_all_classifiers(
                columns_to_train=columns,
                analysis_results=analysis_results,
                clip_csv_mapping=clip_csv_mapping,
                class_name_mapping=class_name_mapping,
                custom_configs=custom_configs,
                verbose=verbose
            )
            
            # Merge results
            all_results.update(group_results)
            
            # Print group summary
            successful = len([r for r in group_results.values() if r.get('success', False)])
            avg_accuracy = 0
            if successful > 0:
                accuracies = [r['evaluation_metrics']['accuracy'] 
                            for r in group_results.values() if r.get('success', False)]
                avg_accuracy = sum(accuracies) / len(accuracies)
            
            print(f"   \u2705 {group_name} complete: {successful}/{len(columns)} successful (avg: {avg_accuracy:.1f}%)")
        
        return all_results
    
    def train_priority_order(self,
                           analysis_results: Dict[str, Any],
                           clip_csv_mapping: Dict[str, str],
                           max_models: Optional[int] = None,
                           class_name_mapping: Optional[Dict[str, Dict[int, str]]] = None,
                           custom_configs: Optional[Dict[str, Dict[str, Any]]] = None,
                           verbose: bool = True) -> Dict[str, Any]:
        """
        Train classifiers in priority order, optionally stopping after max_models
        """
        priority_order = analysis_results['recommendations']['priority_order']
        
        if max_models:
            priority_order = priority_order[:max_models]
            print(f"\U0001f3af Training top {max_models} priority classifiers")
        else:
            print(f"\U0001f3af Training all {len(priority_order)} classifiers in priority order")
        
        return self.trainer.train_all_classifiers(
            columns_to_train=priority_order,
            analysis_results=analysis_results,
            clip_csv_mapping=clip_csv_mapping,
            class_name_mapping=class_name_mapping,
            custom_configs=custom_configs,
            verbose=verbose
        )


class ClassifierManager:
    """
    Unified manager for all classifier operations including creation, training, 
    loading, saving, and inference across multiple columns.
    """
    
    def __init__(self, base_output_dir: str, debug: bool = False):
        self.base_output_dir = base_output_dir
        self.debug = debug
        
        # Initialize components
        self.trainer = ModelTrainer(base_output_dir, debug)
        self.batch_trainer = BatchTrainingManager(self.trainer, debug)
        
        # Track all classifiers
        self.classifiers = {}
        self.face_detector = None
        
    def train_classifiers(self,
                         columns_to_train: List[str],
                         analysis_results: Dict[str, Any],
                         clip_csv_mapping: Dict[str, str],
                         class_name_mapping: Optional[Dict[str, Dict[int, str]]] = None,
                         custom_configs: Optional[Dict[str, Dict[str, Any]]] = None,
                         use_batch_training: bool = False,
                         verbose: bool = True) -> Dict[str, Any]:
        """
        Train classifiers for specified columns
        
        Args:
            columns_to_train: List of column names to train classifiers for
            analysis_results: Results from column analysis
            clip_csv_mapping: Mapping of clip names to CSV files
            class_name_mapping: Custom class name mappings
            custom_configs: Custom model and training configurations
            use_batch_training: If True, use batch training by groups
            verbose: Enable verbose output
            
        Returns:
            Dictionary containing training results for all classifiers
        """
        if use_batch_training:
            results = self.batch_trainer.train_by_groups(
                analysis_results=analysis_results,
                clip_csv_mapping=clip_csv_mapping,
                class_name_mapping=class_name_mapping,
                custom_configs=custom_configs,
                verbose=verbose
            )
        else:
            results = self.trainer.train_all_classifiers(
                columns_to_train=columns_to_train,
                analysis_results=analysis_results,
                clip_csv_mapping=clip_csv_mapping,
                class_name_mapping=class_name_mapping,
                custom_configs=custom_configs,
                verbose=verbose
            )
        
        # Store successful classifiers
        for col, result in results.items():
            if result.get('success', False) and 'classifier' in result:
                self.classifiers[col] = result['classifier']
                
                # Track face detector separately
                if result.get('is_face_column', False):
                    self.face_detector = result['classifier']
        
        return results
    
    def load_classifiers(self,
                        model_paths: Dict[str, str],
                        analysis_results: Dict[str, Any],
                        class_name_mapping: Optional[Dict[str, Dict[int, str]]] = None) -> Dict[str, Any]:
        """
        Load previously trained classifiers from disk
        
        Args:
            model_paths: Dictionary mapping column names to model file paths
            analysis_results: Results from column analysis (for model configs)
            class_name_mapping: Custom class name mappings
            
        Returns:
            Dictionary containing loading results for all classifiers
        """
        results = self.trainer.load_trained_classifiers(
            model_paths=model_paths,
            analysis_results=analysis_results,
            class_name_mapping=class_name_mapping
        )
        
        # Store successfully loaded classifiers
        for col, result in results.items():
            if result.get('success', False) and 'classifier' in result:
                self.classifiers[col] = result['classifier']
                
                if result.get('is_face_column', False):
                    self.face_detector = result['classifier']
        
        return results
    
    def save_all_classifiers(self, save_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Save all trained classifiers to disk
        
        Args:
            save_dir: Directory to save models (uses base_output_dir if None)
            
        Returns:
            Dictionary mapping column names to saved model paths
        """
        return self.trainer.save_all_models(save_dir)
    
    def get_classifier(self, column: str):
        """Get a specific trained classifier"""
        return self.classifiers.get(column)
    
    def get_face_detector(self):
        """Get the trained face detector if available"""
        return self.face_detector
    
    def get_all_classifiers(self) -> Dict[str, Any]:
        """Get all trained classifiers"""
        return self.classifiers.copy()
    
    def predict_single_frame(self, 
                           image_path: str, 
                           columns: Optional[List[str]] = None,
                           threshold: float = 0.5) -> Dict[str, Any]:
        """
        Run inference on a single frame with specified classifiers
        
        Args:
            image_path: Path to the image file
            columns: List of columns to run inference for (all if None)
            threshold: Classification threshold
            
        Returns:
            Dictionary with predictions for each classifier
        """
        if columns is None:
            columns = list(self.classifiers.keys())
        
        results = {}
        
        for col in columns:
            if col not in self.classifiers:
                results[col] = {'error': f'Classifier not available for {col}'}
                continue
                
            try:
                classifier = self.classifiers[col]
                prediction = classifier.predict_single_image(image_path, threshold=threshold)
                results[col] = {
                    'prediction': prediction,
                    'is_face_column': col == 'has_faces',
                    'success': True
                }
            except Exception as e:
                results[col] = {
                    'error': str(e),
                    'success': False
                }
        
        return results
    
    def predict_video(self,
                     video_path: str,
                     columns: Optional[List[str]] = None,
                     output_path: Optional[str] = None,
                     threshold: float = 0.5) -> Dict[str, Any]:
        """
        Run inference on a video with specified classifiers
        
        Args:
            video_path: Path to the video file
            columns: List of columns to run inference for (all if None)
            output_path: Base path for output videos (optional)
            threshold: Classification threshold
            
        Returns:
            Dictionary with video processing results for each classifier
        """
        if columns is None:
            columns = list(self.classifiers.keys())
        
        results = {}
        
        for col in columns:
            if col not in self.classifiers:
                results[col] = {'error': f'Classifier not available for {col}'}
                continue
            
            try:
                classifier = self.classifiers[col]
                video_output_path = f"{output_path}_{col}.mp4" if output_path else None
                
                frame_results = classifier.process_video(
                    video_path=video_path,
                    output_path=video_output_path,
                    threshold=threshold
                )
                
                results[col] = {
                    'frame_results': frame_results,
                    'total_frames': len(frame_results),
                    'positive_frames': sum(1 for r in frame_results if r.get('is_positive', False)),
                    'is_face_column': col == 'has_faces',
                    'success': True
                }
                
            except Exception as e:
                results[col] = {
                    'error': str(e),
                    'success': False
                }
        
        return results
    
    def get_classifier_summary(self) -> Dict[str, Any]:
        """
        Get summary information about all managed classifiers
        
        Returns:
            Dictionary with classifier summary information
        """
        summary = {
            'total_classifiers': len(self.classifiers),
            'face_detector_available': self.face_detector is not None,
            'available_columns': list(self.classifiers.keys()),
            'debug_mode': self.debug,
            'classifiers': {}
        }
        
        for col, classifier in self.classifiers.items():
            try:
                # Try to get basic info about each classifier
                summary['classifiers'][col] = {
                    'type': type(classifier).__name__,
                    'is_face_detector': col == 'has_faces',
                    'available': True
                }
            except Exception as e:
                summary['classifiers'][col] = {
                    'error': str(e),
                    'available': False
                }
        
        return summary
    
    def validate_classifiers(self) -> Dict[str, Any]:
        """
        Validate all classifiers to ensure they're working properly
        
        Returns:
            Dictionary with validation results for each classifier
        """
        validation_results = {}
        
        for col, classifier in self.classifiers.items():
            try:
                # Basic validation - check if classifier has required methods
                required_methods = ['predict_single_image', 'process_video']
                missing_methods = [method for method in required_methods 
                                 if not hasattr(classifier, method)]
                
                if missing_methods:
                    validation_results[col] = {
                        'valid': False,
                        'error': f'Missing methods: {missing_methods}'
                    }
                else:
                    validation_results[col] = {
                        'valid': True,
                        'type': type(classifier).__name__,
                        'is_face_detector': col == 'has_faces'
                    }
                    
            except Exception as e:
                validation_results[col] = {
                    'valid': False,
                    'error': str(e)
                }
        
        return validation_results