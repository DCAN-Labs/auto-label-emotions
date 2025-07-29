#!/usr/bin/env python3
"""
Utilities Module

This module contains utility classes for configuration management,
results handling, and other common operations.
"""

import json
import os
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path


class ConfigurationManager:
    """Manager for handling model and training configurations"""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.face_column = 'has_faces'
    
    def get_class_names(self, 
                       column: str, 
                       class_name_mapping: Optional[Dict[str, Dict[int, str]]] = None) -> Dict[int, str]:
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
    
    def load_config_from_file(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Apply debug mode adjustments if needed
            if self.debug and 'training_configs' in config:
                for col_config in config['training_configs'].values():
                    if 'training_config' in col_config:
                        col_config['training_config'].update({
                            'epochs': 3,
                            'batch_size': 4,
                            'patience': 2,
                            'lr': 0.01
                        })
                print("\U0001f41b DEBUG: Applied debug adjustments to loaded config")
            
            return config
            
        except Exception as e:
            print(f"\u274c Error loading config from {config_path}: {e}")
            return {}
    
    def save_config_to_file(self, config: Dict[str, Any], config_path: str):
        """Save configuration to JSON file"""
        try:
            # Ensure directory exists
            Path(config_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            
            print(f"\U0001f4be Configuration saved to: {config_path}")
            
        except Exception as e:
            print(f"\u274c Error saving config to {config_path}: {e}")
    
    def create_default_config(self, columns: List[str]) -> Dict[str, Any]:
        """Create default configuration for given columns"""
        config = {
            'general_settings': {
                'debug_mode': self.debug,
                'interval_ms': 100,
                'val_split': 0.2
            },
            'model_configs': {},
            'training_configs': {},
            'class_mappings': {}
        }
        
        for column in columns:
            # Default model config
            if column == self.face_column:
                backbone = 'mobilenet'
            elif 'happy' in column.lower():
                backbone = 'resnet18'
            elif 'excite' in column.lower():
                backbone = 'efficientnet_b0'
            else:
                backbone = 'mobilenet'
            
            config['model_configs'][column] = {
                'model_type': 'transfer',
                'backbone': backbone,
                'pretrained': True,
                'freeze_features': True,
                'img_size': 224
            }
            
            # Default training config
            if self.debug:
                training_config = {
                    'epochs': 3,
                    'lr': 0.01,
                    'batch_size': 4,
                    'patience': 2
                }
            else:
                training_config = {
                    'epochs': 50,
                    'lr': 0.001,
                    'batch_size': 32,
                    'patience': 10
                }
            
            config['training_configs'][column] = training_config
            
            # Default class mapping
            config['class_mappings'][column] = self.get_class_names(column)
        
        return config


class ResultsManager:
    """Manager for handling pipeline results and summaries"""
    
    def __init__(self, base_output_dir: str):
        self.base_output_dir = base_output_dir
        Path(base_output_dir).mkdir(parents=True, exist_ok=True)
    
    def generate_performance_summary(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
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
        
        # Check if any results are from debug mode
        debug_mode = any(result.get('debug_mode', False) for result in successful_trainings.values())
        
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
            'debug_mode': debug_mode,
            'accuracy_distribution': {
                'excellent': len([acc for acc in accuracies if acc >= 95]),
                'very_good': len([acc for acc in accuracies if 90 <= acc < 95]),
                'good': len([acc for acc in accuracies if 80 <= acc < 90]),
                'fair': len([acc for acc in accuracies if 70 <= acc < 80]),
                'poor': len([acc for acc in accuracies if acc < 70])
            }
        }
    
    def print_comprehensive_summary(self, results: Dict[str, Any], verbose: bool = True):
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
    
    def save_results(self, results: Dict[str, Any]):
        """Save complete results to JSON file"""
        results_file = os.path.join(self.base_output_dir, 'comprehensive_pipeline_results.json')
        
        try:
            # Convert non-serializable objects for JSON
            json_results = self._prepare_results_for_json(results)
            
            with open(results_file, 'w') as f:
                json.dump(json_results, f, indent=2, default=str)
            
            print(f"\n\U0001f4c4 Complete results saved to: {results_file}")
            
        except Exception as e:
            print(f"\u274c Error saving results: {e}")
    
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
    
    def generate_report(self, results: Dict[str, Any], format: str = 'markdown') -> str:
        """Generate a formatted report of pipeline results"""
        if format == 'markdown':
            return self._generate_markdown_report(results)
        elif format == 'html':
            return self._generate_html_report(results)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _generate_markdown_report(self, results: Dict[str, Any]) -> str:
        """Generate markdown report"""
        report = []
        report.append("# Enhanced Multi-Column Classification Pipeline Report\n")
        
        if results.get('debug_mode', False):
            report.append("**\u26a0\ufe0f DEBUG MODE RESULTS** - Training was limited for testing purposes\n")
        
        # Executive Summary
        performance = results['performance_summary']
        if 'error' not in performance:
            report.append("## Executive Summary\n")
            report.append(f"- **Total Classifiers Trained:** {performance['total_trained']}")
            report.append(f"- **Average Accuracy:** {performance['avg_accuracy']:.1f}%")
            report.append(f"- **Average F1 Score:** {performance['avg_f1_score']:.3f}")
            report.append(f"- **Average Training Time:** {performance['avg_training_time_minutes']:.1f} minutes")
            report.append("")
        
        # Individual Results
        if results['successful_classifiers']:
            report.append("## Individual Classifier Results\n")
            report.append("| Column | Accuracy | F1 Score | Type |")
            report.append("|--------|----------|----------|------|")
            
            for col in results['successful_classifiers']:
                training_result = results['training_results'][col]
                metrics = training_result['evaluation_metrics']
                col_type = "Face Detection" if training_result.get('is_face_column', False) else "Classification"
                
                report.append(f"| {col} | {metrics['accuracy']:.1f}% | {metrics['f1_score']:.3f} | {col_type} |")
            
            report.append("")
        
        # Recommendations
        report.append("## Recommendations\n")
        if results.get('debug_mode', False):
            report.append("- **Switch to Full Training:** Set DEBUG_MODE = False for production training")
            report.append("- **Current Limitations:** Results are from minimal 3-epoch training")
        elif performance.get('avg_accuracy', 0) >= 90:
            report.append("- **Excellent Performance:** Ready for deployment")
            report.append("- **Next Steps:** Consider ensemble models and additional validation")
        elif performance.get('avg_accuracy', 0) >= 80:
            report.append("- **Good Performance:** Fine-tune underperforming classifiers")
            report.append("- **Improvements:** Consider data augmentation for robustness")
        else:
            report.append("- **Performance Issues:** Review data quality and labeling")
            report.append("- **Next Steps:** Collect more training data or try different architectures")
        
        return "\n".join(report)
    
    def _generate_html_report(self, results: Dict[str, Any]) -> str:
        """Generate HTML report"""
        html = []
        html.append("<!DOCTYPE html>")
        html.append("<html><head><title>Pipeline Report</title>")
        html.append("<style>")
        html.append("body { font-family: Arial, sans-serif; margin: 40px; }")
        html.append("table { border-collapse: collapse; width: 100%; }")
        html.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
        html.append("th { background-color: #f2f2f2; }")
        html.append(".debug { background-color: #fff3cd; padding: 10px; border: 1px solid #ffeaa7; }")
        html.append("</style></head><body>")
        
        html.append("<h1>Enhanced Multi-Column Classification Pipeline Report</h1>")
        
        if results.get('debug_mode', False):
            html.append('<div class="debug"><strong>\u26a0\ufe0f DEBUG MODE RESULTS</strong> - Training was limited for testing purposes</div>')
        
        # Add executive summary and results tables here
        performance = results['performance_summary']
        if 'error' not in performance:
            html.append("<h2>Executive Summary</h2>")
            html.append("<ul>")
            html.append(f"<li><strong>Total Classifiers Trained:</strong> {performance['total_trained']}</li>")
            html.append(f"<li><strong>Average Accuracy:</strong> {performance['avg_accuracy']:.1f}%</li>")
            html.append(f"<li><strong>Average F1 Score:</strong> {performance['avg_f1_score']:.3f}</li>")
            html.append(f"<li><strong>Average Training Time:</strong> {performance['avg_training_time_minutes']:.1f} minutes</li>")
            html.append("</ul>")
        
        html.append("</body></html>")
        return "\n".join(html)
    
    def save_report(self, results: Dict[str, Any], format: str = 'markdown'):
        """Save formatted report to file"""
        report_content = self.generate_report(results, format)
        
        if format == 'markdown':
            filename = 'pipeline_report.md'
        elif format == 'html':
            filename = 'pipeline_report.html'
        else:
            filename = 'pipeline_report.txt'
        
        report_path = os.path.join(self.base_output_dir, filename)
        
        try:
            with open(report_path, 'w') as f:
                f.write(report_content)
            print(f"\U0001f4c4 Report saved to: {report_path}")
        except Exception as e:
            print(f"\u274c Error saving report: {e}")


class FileManager:
    """Manager for file operations and directory handling"""
    
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        Path(base_dir).mkdir(parents=True, exist_ok=True)
    
    def create_directory_structure(self, columns: List[str]):
        """Create organized directory structure for pipeline outputs"""
        directories = [
            'frames',
            'analysis',
            'models',
            'reports',
            'visualizations'
        ]
        
        # Create base directories
        for dir_name in directories:
            Path(self.base_dir, dir_name).mkdir(parents=True, exist_ok=True)
        
        # Create column-specific directories
        for column in columns:
            dataset_dir = Path(self.base_dir, f'{column}_dataset')
            dataset_dir.mkdir(parents=True, exist_ok=True)
            
            # Create train/val subdirectories
            for class_dir in ['train', 'val']:
                Path(dataset_dir, class_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"\U0001f4c1 Directory structure created in: {self.base_dir}")
    
    def cleanup_temporary_files(self):
        """Clean up temporary files and empty directories"""
        import shutil
        
        temp_patterns = ['*.tmp', '*.temp', '*.log']
        cleaned_count = 0
        
        for pattern in temp_patterns:
            for temp_file in Path(self.base_dir).rglob(pattern):
                try:
                    temp_file.unlink()
                    cleaned_count += 1
                except Exception as e:
                    print(f"\u26a0\ufe0f  Could not remove {temp_file}: {e}")
        
        # Remove empty directories
        for path in Path(self.base_dir).rglob('*'):
            if path.is_dir() and not any(path.iterdir()):
                try:
                    path.rmdir()
                    cleaned_count += 1
                except Exception:
                    pass
        
        if cleaned_count > 0:
            print(f"\U0001f9f9 Cleaned up {cleaned_count} temporary files/directories")
    
    def get_disk_usage(self) -> Dict[str, Any]:
        """Get disk usage statistics for the pipeline directory"""
        total_size = 0
        file_count = 0
        dir_breakdown = {}
        
        for path in Path(self.base_dir).rglob('*'):
            if path.is_file():
                size = path.stat().st_size
                total_size += size
                file_count += 1
                
                # Track size by subdirectory
                relative_path = path.relative_to(self.base_dir)
                top_dir = str(relative_path).split('/')[0] if '/' in str(relative_path) else str(relative_path)
                
                if top_dir not in dir_breakdown:
                    dir_breakdown[top_dir] = {'size': 0, 'files': 0}
                dir_breakdown[top_dir]['size'] += size
                dir_breakdown[top_dir]['files'] += 1
        
        return {
            'total_size_mb': total_size / (1024 * 1024),
            'total_files': file_count,
            'directory_breakdown': dir_breakdown
        }
    
    def archive_results(self, archive_name: Optional[str] = None) -> str:
        """Create compressed archive of all pipeline results"""
        import zipfile
        from datetime import datetime
        
        if archive_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_name = f"pipeline_results_{timestamp}.zip"
        
        archive_path = os.path.join(self.base_dir, archive_name)
        
        try:
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in Path(self.base_dir).rglob('*'):
                    if file_path.is_file() and file_path != Path(archive_path):
                        arcname = file_path.relative_to(self.base_dir)
                        zipf.write(file_path, arcname)
            
            archive_size = Path(archive_path).stat().st_size / (1024 * 1024)
            print(f"\U0001f4e6 Results archived to: {archive_path} ({archive_size:.1f} MB)")
            return archive_path
            
        except Exception as e:
            print(f"\u274c Error creating archive: {e}")
            return ""


class LoggingManager:
    """Manager for pipeline logging and progress tracking"""
    
    def __init__(self, base_dir: str, debug: bool = False):
        self.base_dir = base_dir
        self.debug = debug
        self.log_file = os.path.join(base_dir, 'pipeline.log')
        
        # Ensure log directory exists
        Path(base_dir).mkdir(parents=True, exist_ok=True)
    
    def log_message(self, message: str, level: str = 'INFO'):
        """Log a message with timestamp"""
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}\n"
        
        try:
            with open(self.log_file, 'a') as f:
                f.write(log_entry)
        except Exception as e:
            print(f"\u26a0\ufe0f  Logging error: {e}")
        
        # Also print to console if debug mode
        if self.debug:
            print(f"\U0001f4dd {log_entry.strip()}")
    
    def log_error(self, error: str, context: str = ""):
        """Log an error with context"""
        full_message = f"ERROR in {context}: {error}" if context else f"ERROR: {error}"
        self.log_message(full_message, 'ERROR')
    
    def log_performance(self, operation: str, duration: float, details: str = ""):
        """Log performance metrics"""
        message = f"PERFORMANCE - {operation}: {duration:.2f}s"
        if details:
            message += f" ({details})"
        self.log_message(message, 'PERF')
    
    def get_log_summary(self) -> Dict[str, Any]:
        """Get summary of logged events"""
        if not os.path.exists(self.log_file):
            return {'error': 'No log file found'}
        
        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
            
            summary = {
                'total_entries': len(lines),
                'errors': len([l for l in lines if 'ERROR' in l]),
                'warnings': len([l for l in lines if 'WARNING' in l]),
                'performance_entries': len([l for l in lines if 'PERF' in l])
            }
            
            # Get recent errors
            recent_errors = [l.strip() for l in lines if 'ERROR' in l][-5:]
            summary['recent_errors'] = recent_errors
            
            return summary
            
        except Exception as e:
            return {'error': f'Failed to read log: {e}'}
    
    def clear_logs(self):
        """Clear the log file"""
        try:
            if os.path.exists(self.log_file):
                os.remove(self.log_file)
            print("\U0001f9f9 Log file cleared")
        except Exception as e:
            print(f"\u274c Error clearing logs: {e}")