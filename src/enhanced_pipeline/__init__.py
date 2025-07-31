"""
Enhanced Multi-Column Classification Pipeline Package

A comprehensive machine learning pipeline for multi-column classification with
integrated face detection analysis and specialized retraining capabilities.
"""

from .core_pipeline import EnhancedMultiColumnPipeline
from .analysis import ColumnAnalyzer, FaceTemporalAnalyzer
from .visualization import DashboardGenerator, MetricsVisualizer
from .training import ModelTrainer, ClassifierFactory, BatchTrainingManager, ClassifierManager
from .utils import ConfigurationManager, ResultsManager, FileManager, LoggingManager
from .prediction import ModelLoader, VideoPredictor, BatchPredictor
from .examples import run_enhanced_example

__version__ = "1.0.0"
__author__ = "Paul Reiners"

__all__ = [
    'EnhancedMultiColumnPipeline',
    'ColumnAnalyzer',
    'FaceTemporalAnalyzer', 
    'DashboardGenerator',
    'MetricsVisualizer',
    'ModelTrainer',
    'ClassifierFactory',
    'BatchTrainingManager',
    'ClassifierManager',
    'ConfigurationManager',
    'ResultsManager',
    'FileManager',
    'LoggingManager',
    'ModelLoader',
    'VideoPredictor', 
    'BatchPredictor',
    'run_enhanced_example'
]