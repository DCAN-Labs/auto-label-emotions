#!/usr/bin/env python3
"""
Main Usage Example - Enhanced with Configuration File Support

This shows how to use configuration files for flexible pipeline execution.
"""

import argparse
import json
import yaml
import sys
import os
from pathlib import Path
from typing import Dict, List, Any

from enhanced_pipeline import (
    ColumnAnalyzer, 
    DashboardGenerator, 
    ModelTrainer,
    ConfigurationManager,
    ResultsManager
)
from enhanced_pipeline.core_pipeline import EnhancedMultiColumnPipeline


def load_config_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON or YAML file"""
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    file_ext = Path(config_path).suffix.lower()
    
    try:
        with open(config_path, 'r') as f:
            if file_ext == '.json':
                config = json.load(f)
            elif file_ext in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config file format: {file_ext}")
        
        print(f"\u2705 Loaded configuration from: {config_path}")
        return config
        
    except Exception as e:
        raise ValueError(f"Error loading config file {config_path}: {e}")


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize configuration"""
    
    required_keys = ['video_list', 'clip_mapping']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    # Validate video files exist
    missing_videos = []
    for video_path in config['video_list']:
        if not os.path.exists(video_path):
            missing_videos.append(video_path)
    
    if missing_videos:
        print(f"\u26a0\ufe0f  Warning: Missing video files:")
        for video in missing_videos:
            print(f"   \u274c {video}")
        
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Validate CSV files exist
    missing_csvs = []
    for clip_name, csv_path in config['clip_mapping'].items():
        if not os.path.exists(csv_path):
            missing_csvs.append(f"{clip_name}: {csv_path}")
    
    if missing_csvs:
        print(f"\u26a0\ufe0f  Warning: Missing CSV files:")
        for csv_info in missing_csvs:
            print(f"   \u274c {csv_info}")
        
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Set defaults for pipeline settings
    default_settings = {
        'base_output_dir': 'data/my_results',
        'debug': False,
        'frame_interval_ms': 100,
        'skip_extraction': False,
        'create_visualizations': True,
        'verbose': True
    }
    
    if 'pipeline_settings' not in config:
        config['pipeline_settings'] = {}
    
    for key, default_value in default_settings.items():
        if key not in config['pipeline_settings']:
            config['pipeline_settings'][key] = default_value
    
    print(f"\U0001f4ca Configuration validated:")
    print(f"   Videos: {len(config['video_list'])}")
    print(f"   CSV mappings: {len(config['clip_mapping'])}")
    print(f"   Output directory: {config['pipeline_settings']['base_output_dir']}")
    
    return config


def create_config_template(template_path: str):
    """Create a template configuration file"""
    
    template_config = {
        "video_list": [
            "data/clip01/in/clip1_MLP.mp4",
            "data/clip02/in/clip2_AHKJ.mp4",
            "data/clip03/in/clip3_MLP.mp4",
            "data/clip04/in/clip4_MLP.mp4"
        ],
        "clip_mapping": {
            "clip1_MLP": "data/clip01/in/clip1_codes_MLP.csv",
            "clip2_AHKJ": "data/clip02/in/clip2_codes_AHKJ.csv",
            "clip3_MLP": "data/clip03/in/clip3_codes_MLP.csv",
            "clip4_MLP": "data/clip04/in/clip4_codes_MLP.csv"
        },
        "pipeline_settings": {
            "base_output_dir": "data/my_results",
            "debug": False,
            "frame_interval_ms": 100,
            "skip_extraction": False,
            "create_visualizations": True,
            "verbose": True
        }
    }
    
    file_ext = Path(template_path).suffix.lower()
    
    with open(template_path, 'w') as f:
        if file_ext == '.json':
            json.dump(template_config, f, indent=2)
        elif file_ext in ['.yaml', '.yml']:
            yaml.dump(template_config, f, default_flow_style=False, indent=2)
        else:
            # Default to JSON
            json.dump(template_config, f, indent=2)
    
    print(f"\U0001f4dd Template configuration created: {template_path}")
    print("   Edit this file with your actual video paths and settings")


def run_pipeline_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run the pipeline using configuration"""
    
    print("\n\u2699\ufe0f Running Enhanced Multi-Column Pipeline...")
    print("="*60)
    
    # Extract configuration
    video_list = config['video_list']
    clip_mapping = config['clip_mapping']
    settings = config['pipeline_settings']
    
    # Create pipeline
    pipeline = EnhancedMultiColumnPipeline(
        video_list=video_list,
        clip_csv_mapping=clip_mapping,
        base_output_dir=settings['base_output_dir'],
        debug=settings['debug']
    )
    
    # Run pipeline
    results = pipeline.run_comprehensive_pipeline(
        skip_extraction=settings['skip_extraction'],
        create_visualizations=settings['create_visualizations'],
        verbose=settings['verbose']
    )
    
    print(f"\n\u2705 Pipeline complete! Trained {len(results['successful_classifiers'])} classifiers")
    return results


def run_analysis_only(config: Dict[str, Any]):
    """Run only the analysis phase"""
    
    print("\n\U0001f4ca Running Analysis Only...")
    print("="*40)
    
    analyzer = ColumnAnalyzer(
        clip_csv_mapping=config['clip_mapping'],
        debug=config['pipeline_settings']['debug']
    )
    
    analysis_results = analyzer.analyze_all_columns(
        verbose=config['pipeline_settings']['verbose']
    )
    
    # Create dashboard
    dashboard_gen = DashboardGenerator(debug=config['pipeline_settings']['debug'])
    output_path = os.path.join(
        config['pipeline_settings']['base_output_dir'], 
        'analysis_dashboard.png'
    )
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    dashboard_gen.create_dashboard(
        column_stats=analysis_results['column_stats'],
        save_path=output_path
    )
    
    print(f"\u2705 Analysis complete! Dashboard saved to: {output_path}")


def main():
    """Main function with command-line argument support"""
    
    parser = argparse.ArgumentParser(
        description="Enhanced Multi-Column Classification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with configuration file
  python main.py --config my_config.json
  
  # Create template configuration
  python main.py --create-template config_template.json
  
  # Run analysis only
  python main.py --config my_config.json --analysis-only
  
  # Run with default settings (no config file)
  python main.py --default
        """
    )
    
    # Main action arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--config', '-c', type=str, 
                      help='Path to configuration file (JSON or YAML)')
    group.add_argument('--create-template', '-t', type=str,
                      help='Create template configuration file')
    group.add_argument('--default', action='store_true',
                      help='Run with default hardcoded settings')
    
    # Optional arguments
    parser.add_argument('--analysis-only', action='store_true',
                       help='Run only analysis phase (no training)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    try:
        # Handle template creation
        if args.create_template:
            create_config_template(args.create_template)
            return
        
        # Handle configuration loading
        if args.config:
            config = load_config_file(args.config)
            config = validate_config(config)
            
            # Override settings from command line
            if args.verbose:
                config['pipeline_settings']['verbose'] = True
            if args.debug:
                config['pipeline_settings']['debug'] = True
            
        elif args.default:
            # Use hardcoded defaults
            print("\U0001f527 Using default hardcoded configuration...")
            config = {
                'video_list': [
                    "data/clip01/in/clip1_MLP.mp4",
                    "data/clip02/in/clip2_AHKJ.mp4",
                    "data/clip03/in/clip3_MLP.mp4",
                    "data/clip04/in/clip4_MLP.mp4"
                ],
                'clip_mapping': {
                    'clip1_MLP': 'data/clip01/in/clip1_codes_MLP.csv',
                    'clip2_AHKJ': 'data/clip02/in/clip2_codes_AHKJ.csv',
                    'clip3_MLP': 'data/clip03/in/clip3_codes_MLP.csv',
                    'clip4_MLP': 'data/clip04/in/clip4_codes_MLP.csv'
                },
                'pipeline_settings': {
                    'base_output_dir': 'data/my_results',
                    'debug': args.debug,
                    'frame_interval_ms': 100,
                    'skip_extraction': False,
                    'create_visualizations': True,
                    'verbose': args.verbose
                }
            }
        
        # Run the appropriate pipeline phase
        if args.analysis_only:
            run_analysis_only(config)
        else:
            results = run_pipeline_from_config(config)
            
            # Print summary
            print(f"\n\U0001f389 SUCCESS! Pipeline Results:")
            print(f"   Successful classifiers: {len(results.get('successful_classifiers', []))}")
            print(f"   Failed classifiers: {len(results.get('failed_classifiers', []))}")
            print(f"   Results saved to: {config['pipeline_settings']['base_output_dir']}")
        
    except KeyboardInterrupt:
        print(f"\n\U0001f6d1 Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\U0001f4a5 Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()