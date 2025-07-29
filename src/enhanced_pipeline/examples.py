#!/usr/bin/env python3
"""
Examples Module

This module contains example usage patterns and demo functions
for the enhanced multi-column classification pipeline.
"""

from typing import Dict, List, Any, Optional
from .core_pipeline import EnhancedMultiColumnPipeline


def run_enhanced_example(debug_mode: bool = True) -> Dict[str, Any]:
    """
    Enhanced example demonstrating the complete pipeline with face detection integration
    
    Args:
        debug_mode: If True, uses minimal training parameters for quick testing
    
    Returns:
        Dictionary containing all pipeline results
    """
    
    if debug_mode:
        print("\U0001f41b RUNNING IN DEBUG MODE")
        print("   - Only 3 epochs per classifier")
        print("   - Small batch sizes")
        print("   - CPU-only processing")
        print("   - Change debug_mode=False for full training")
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
    
    # Custom configurations with debug mode handling
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
            # Debug-specific config (only used if debug_mode=True)
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
        debug=debug_mode  # Pass debug flag
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


def run_minimal_example() -> Dict[str, Any]:
    """
    Minimal example with basic configuration for quick testing
    """
    print("\U0001f680 RUNNING MINIMAL EXAMPLE")
    print("   - Single video, single column")
    print("   - Fast execution for testing")
    print("-" * 50)
    
    # Minimal configuration
    video_list = ["data/clip01/in/clip1_MLP.mp4"]
    
    clip_mapping = {
        'clip1_MLP': 'data/clip01/in/clip1_codes_MLP.csv'
    }
    
    # Only process face detection for minimal example
    class_name_mapping = {
        'has_faces': {0: 'no_face', 1: 'face'}
    }
    
    # Create pipeline with debug mode enabled
    pipeline = EnhancedMultiColumnPipeline(
        video_list=video_list,
        clip_csv_mapping=clip_mapping,
        timestamp_column='onset_milliseconds',
        base_output_dir='data/minimal_results',
        interval_ms=100,
        debug=True  # Always debug for minimal example
    )
    
    # Run with minimal processing
    results = pipeline.run_comprehensive_pipeline(
        columns_to_process=['has_faces'],  # Only process face detection
        class_name_mapping=class_name_mapping,
        skip_extraction=False,
        create_visualizations=True,
        verbose=True
    )
    
    return results


def run_production_example() -> Dict[str, Any]:
    """
    Production-ready example with full training and comprehensive configuration
    """
    print("\U0001f3ed RUNNING PRODUCTION EXAMPLE")
    print("   - Full training with all columns")
    print("   - Optimized configurations")
    print("   - Comprehensive validation")
    print("-" * 50)
    
    # Full configuration for production
    video_list = [
        "data/clip01/in/clip1_MLP.mp4",
        "data/clip02/in/clip2_AHKJ.mp4", 
        "data/clip03/in/clip3_MLP.mp4",
        # Add more videos as needed
    ]
    
    clip_mapping = {
        'clip1_MLP': 'data/clip01/in/clip1_codes_MLP.csv',
        'clip2_AHKJ': 'data/clip02/in/clip2_codes_AHKJ.csv',
        'clip3_MLP': 'data/clip03/in/clip3_codes_MLP.csv',
        # Add more mappings as needed
    }
    
    # Comprehensive class mapping
    class_name_mapping = {
        'has_faces': {0: 'no_face', 1: 'face'},
        'c_excite_face': {0: 'calm', 1: 'excited'},
        'c_happy_face': {0: 'not_happy', 1: 'happy'},
        'c_surprise_face': {0: 'not_surprised', 1: 'surprised'},
        'c_fear_face': {0: 'not_fearful', 1: 'fearful'},
        'c_anger_face': {0: 'not_angry', 1: 'angry'},
        'c_disgust_face': {0: 'not_disgusted', 1: 'disgusted'},
        'c_sad_face': {0: 'not_sad', 1: 'sad'}
    }
    
    # Production-optimized configurations
    custom_configs = {
        'has_faces': {
            'model_config': {
                'backbone': 'mobilenet',
                'freeze_features': False,  # Allow fine-tuning for production
                'img_size': 224
            },
            'training_config': {
                'epochs': 100,  # More epochs for production
                'lr': 0.0005,
                'batch_size': 64,
                'patience': 15,
                'use_class_weighting': True
            }
        },
        'c_happy_face': {
            'model_config': {
                'backbone': 'resnet18',
                'freeze_features': False,
                'img_size': 224
            },
            'training_config': {
                'epochs': 120,
                'lr': 0.0003,
                'batch_size': 32,
                'patience': 20
            }
        },
        'c_excite_face': {
            'model_config': {
                'backbone': 'efficientnet_b0',
                'freeze_features': False,
                'img_size': 256
            },
            'training_config': {
                'epochs': 150,
                'lr': 0.0001,
                'batch_size': 16,
                'patience': 25
            }
        }
    }
    
    # Create production pipeline (no debug mode)
    pipeline = EnhancedMultiColumnPipeline(
        video_list=video_list,
        clip_csv_mapping=clip_mapping,
        timestamp_column='onset_milliseconds',
        base_output_dir='data/production_results',
        interval_ms=50,  # Higher temporal resolution for production
        debug=False  # Full training mode
    )
    
    # Run comprehensive production pipeline
    results = pipeline.run_comprehensive_pipeline(
        class_name_mapping=class_name_mapping,
        custom_configs=custom_configs,
        skip_extraction=False,
        create_visualizations=True,
        verbose=True
    )
    
    # Comprehensive testing on all videos
    if results['successful_classifiers']:
        print(f"\n\U0001f3ac Testing all classifiers on all videos...")
        
        for i, video_path in enumerate(video_list):
            print(f"\n   Testing video {i+1}: {video_path}")
            try:
                test_results = pipeline.predict_video_with_all_classifiers(
                    video_path=video_path,
                    output_path=f'data/production_results/test_output_video_{i+1}',
                    threshold=0.5
                )
                
                print(f"   \u2705 Video {i+1} processing complete!")
                for col, result in test_results.items():
                    emoji = "\U0001f464" if result['is_face_column'] else "\U0001f3af"
                    pos_ratio = result['positive_frames'] / result['total_frames']
                    print(f"      {emoji} {col}: {pos_ratio:.1%} positive frames")
            
            except Exception as e:
                print(f"   \u26a0\ufe0f  Video {i+1} testing failed: {e}")
    
    return results


def run_batch_training_example() -> Dict[str, Any]:
    """
    Example demonstrating batch training by model groups
    """
    print("\U0001f4e6 RUNNING BATCH TRAINING EXAMPLE")
    print("   - Training in optimized groups")
    print("   - Efficient resource utilization")
    print("-" * 50)
    
    from .training import BatchTrainingManager
    
    # Standard configuration
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
    
    # Create pipeline
    pipeline = EnhancedMultiColumnPipeline(
        video_list=video_list,
        clip_csv_mapping=clip_mapping,
        timestamp_column='onset_milliseconds',
        base_output_dir='data/batch_results',
        interval_ms=100,
        debug=False  # Use full training for batch example
    )
    
    # Run analysis first
    analysis_results = pipeline.comprehensive_column_analysis(verbose=True)
    
    # Extract frames and create datasets
    extraction_results = pipeline.extract_frames(verbose=True)
    dataset_results = pipeline.create_datasets_for_all_columns(verbose=True)
    
    # Use batch training manager
    batch_trainer = BatchTrainingManager(pipeline.model_trainer, debug=False)
    
    # Train by groups for efficiency
    training_results = batch_trainer.train_by_groups(
        analysis_results=analysis_results,
        clip_csv_mapping=clip_mapping,
        verbose=True
    )
    
    # Compile results
    final_results = {
        'analysis_results': analysis_results,
        'extraction_results': extraction_results,
        'dataset_results': dataset_results,
        'training_results': training_results,
        'successful_classifiers': [col for col, result in training_results.items() if result.get('success', False)],
        'batch_training': True
    }
    
    return final_results


def run_custom_configuration_example() -> Dict[str, Any]:
    """
    Example showing how to use custom configurations and advanced features
    """
    print("\u2699\ufe0f RUNNING CUSTOM CONFIGURATION EXAMPLE")
    print("   - Advanced configuration management")
    print("   - Custom model architectures")
    print("   - Specialized training strategies")
    print("-" * 50)
    
    from .utils import ConfigurationManager
    
    # Create configuration manager
    config_manager = ConfigurationManager(debug=False)
    
    # Define columns to process
    columns = ['has_faces', 'c_happy_face', 'c_excite_face']
    
    # Create custom configuration
    custom_config = config_manager.create_default_config(columns)
    
    # Modify for specific requirements
    custom_config['model_configs']['has_faces'].update({
        'backbone': 'efficientnet_b1',  # More powerful backbone
        'img_size': 256,
        'freeze_features': False
    })
    
    custom_config['training_configs']['c_happy_face'].update({
        'epochs': 200,  # Extended training for emotion detection
        'lr': 0.0001,
        'use_class_weighting': True
    })
    
    # Save custom configuration
    config_path = 'data/custom_results/custom_config.json'
    config_manager.save_config_to_file(custom_config, config_path)
    
    # Standard pipeline setup
    video_list = ["data/clip01/in/clip1_MLP.mp4"]
    clip_mapping = {'clip1_MLP': 'data/clip01/in/clip1_codes_MLP.csv'}
    
    # Create pipeline
    pipeline = EnhancedMultiColumnPipeline(
        video_list=video_list,
        clip_csv_mapping=clip_mapping,
        timestamp_column='onset_milliseconds',
        base_output_dir='data/custom_results',
        interval_ms=100,
        debug=False
    )
    
    # Run with custom configuration
    results = pipeline.run_comprehensive_pipeline(
        columns_to_process=columns,
        class_name_mapping=custom_config['class_mappings'],
        custom_configs={
            col: {
                'model_config': custom_config['model_configs'][col],
                'training_config': custom_config['training_configs'][col]
            } for col in columns
        },
        verbose=True
    )
    
    return results


# Main execution function
def main():
    """
    Main function to run examples based on command line arguments or user choice
    """
    import sys
    
    examples = {
        'minimal': run_minimal_example,
        'debug': lambda: run_enhanced_example(debug_mode=True),
        'production': run_production_example,
        'batch': run_batch_training_example,
        'custom': run_custom_configuration_example,
        'full': lambda: run_enhanced_example(debug_mode=False)
    }
    
    if len(sys.argv) > 1:
        example_name = sys.argv[1].lower()
        if example_name in examples:
            print(f"\U0001f680 Running {example_name} example...")
            results = examples[example_name]()
            print(f"\u2705 {example_name.title()} example completed!")
            return results
        else:
            print(f"\u274c Unknown example: {example_name}")
            print(f"Available examples: {list(examples.keys())}")
            return None
    else:
        # Interactive mode
        print("\U0001f4cb Available Examples:")
        for i, (name, _) in enumerate(examples.items(), 1):
            print(f"   {i}. {name}")
        
        try:
            choice = int(input("\nSelect example (1-{}): ".format(len(examples))))
            if 1 <= choice <= len(examples):
                example_name = list(examples.keys())[choice - 1]
                print(f"\n\U0001f680 Running {example_name} example...")
                results = examples[example_name]()
                print(f"\u2705 {example_name.title()} example completed!")
                return results
            else:
                print("\u274c Invalid choice")
                return None
        except (ValueError, KeyboardInterrupt):
            print("\n\U0001f44b Goodbye!")
            return None


if __name__ == "__main__":
    main()