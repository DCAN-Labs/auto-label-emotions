#!/usr/bin/env python3
"""
Main Usage Example - Using the Modular Enhanced Pipeline

This shows how to use the new modular structure instead of the monolithic file.
"""


from enhanced_pipeline import (
    ColumnAnalyzer, 
    DashboardGenerator, 
    ModelTrainer,
    ConfigurationManager,
    ResultsManager
)
from enhanced_pipeline.core_pipeline import EnhancedMultiColumnPipeline

def main():
    """Main execution with multiple usage patterns"""
    
    # CUSTOM WAY: Build pipeline
    print("\n\u2699\ufe0f Running with custom configuration...")
    
    # 1. Setup configuration
    video_list = [
        "data/clip01/in/clip1_MLP.mp4",
        "data/clip02/in/clip2_AHKJ.mp4",
        "data/clip03/in/clip3_MLP.mp4",
        "data/clip04/in/clip4_MLP.mp4"
    ]
    
    clip_mapping = {
        'clip1_MLP': 'data/clip01/in/clip1_codes_MLP.csv',
        'clip2_AHKJ': 'data/clip02/in/clip2_codes_AHKJ.csv',
        'clip3_MLP': 'data/clip03/in/clip3_codes_MLP.csv',
        'clip4_MLP': 'data/clip04/in/clip4_codes_MLP.csv'
    }
    
    # 2. Create and run pipeline
    pipeline = EnhancedMultiColumnPipeline(
        video_list=video_list,
        clip_csv_mapping=clip_mapping,
        base_output_dir='data/my_results',
        debug=False  # Change to False for production
    )
    
    # 3. Run complete pipeline
    results = pipeline.run_comprehensive_pipeline(
        skip_extraction=False,
        create_visualizations=True,
        verbose=True
    )
    
    print(f"\u2705 Pipeline complete! Trained {len(results['successful_classifiers'])} classifiers")
    return results

def advanced_usage_example():
    """Example showing advanced usage of individual components"""
    
    # Use individual components for custom workflows
    clip_mapping = {
        'clip1_MLP': 'data/clip01/in/clip1_codes_MLP.csv'
    }
    
    # 1. Analysis only
    analyzer = ColumnAnalyzer(
        clip_csv_mapping=clip_mapping,
        debug=True
    )
    analysis_results = analyzer.analyze_all_columns(verbose=True)
    
    # 2. Visualization only
    dashboard_gen = DashboardGenerator(debug=True)
    dashboard_gen.create_dashboard(
        column_stats=analysis_results['column_stats'],
        save_path='custom_dashboard.png'
    )
    
    # 3. Configuration management
    config_manager = ConfigurationManager(debug=True)
    columns = list(analysis_results['column_stats'].keys())
    config = config_manager.create_default_config(columns)
    config_manager.save_config_to_file(config, 'my_config.json')
    
    print("\u2705 Advanced usage complete!")

if __name__ == "__main__":
    # Run the main example
    results = main()
    
    # Optionally run advanced example
    # advanced_usage_example()