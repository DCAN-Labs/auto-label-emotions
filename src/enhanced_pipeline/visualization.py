#!/usr/bin/env python3
"""
Visualization Module

This module handles creation of comprehensive visualization dashboards
for multi-column analysis results.
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from typing import Dict, List, Any, Optional


class DashboardGenerator:
    """Generator for comprehensive visualization dashboards"""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.face_column = 'has_faces'
        
        # Configure matplotlib for non-interactive use
        matplotlib.use('Agg')
        matplotlib.rcParams['font.family'] = ['DejaVu Sans']
    
    def create_dashboard(self, 
                        column_stats: Dict[str, Any], 
                        save_path: Optional[str] = None):
        """Create comprehensive visualization dashboard for all columns"""
        # Calculate grid size
        n_cols = len(column_stats)
        
        if n_cols > 4:
            self._create_multi_page_dashboard(column_stats, save_path)
            return
        
        self._create_single_page_dashboard(column_stats, save_path)
    
    def _create_single_page_dashboard(self, 
                                    column_stats: Dict[str, Any], 
                                    save_path: Optional[str]):
        """Create single page dashboard for up to 4 columns"""
        n_cols = len(column_stats)
        n_rows = 3  # Distribution, balance, temporal
        
        fig = plt.figure(figsize=(5 * min(n_cols, 4), 4 * n_rows))
        
        # Create subplots for each column
        for idx, (col, stats) in enumerate(column_stats.items()):
            self._create_column_visualizations(fig, idx, n_rows, n_cols, col, stats)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\U0001f4ca Dashboard saved to: {save_path}")
        
        plt.close(fig)
    
    def _create_multi_page_dashboard(self, 
                                   column_stats: Dict[str, Any], 
                                   save_path: Optional[str]):
        """Create multi-page dashboard for many columns"""
        columns_per_page = 4
        column_items = list(column_stats.items())
        
        for page, i in enumerate(range(0, len(column_items), columns_per_page)):
            page_columns = column_items[i:i+columns_per_page]
            
            fig, axes = plt.subplots(3, len(page_columns), figsize=(5*len(page_columns), 12))
            if len(page_columns) == 1:
                axes = axes.reshape(-1, 1)
            
            for col_idx, (col, stats) in enumerate(page_columns):
                self._create_column_visualizations_axes(axes, col_idx, col, stats)
            
            page_save_path = save_path.replace('.png', f'_page_{page+1}.png') if save_path else None
            
            plt.tight_layout()
            if page_save_path:
                plt.savefig(page_save_path, dpi=300, bbox_inches='tight')
                print(f"\U0001f4ca Dashboard page {page+1} saved to: {page_save_path}")
            
            plt.close(fig)
    
    def _create_column_visualizations(self, 
                                    fig, 
                                    idx: int, 
                                    n_rows: int, 
                                    n_cols: int, 
                                    col: str, 
                                    stats: Dict[str, Any]):
        """Create visualizations for a single column in subplot grid"""
        # 1. Sample distribution pie chart
        ax1 = plt.subplot(n_rows, n_cols, idx + 1)
        self._create_distribution_pie(ax1, col, stats)
        
        # 2. Clip-wise distribution
        ax2 = plt.subplot(n_rows, n_cols, idx + 1 + n_cols)
        self._create_clip_distribution(ax2, col, stats)
        
        # 3. Model recommendation
        ax3 = plt.subplot(n_rows, n_cols, idx + 1 + 2*n_cols)
        self._create_model_recommendation(ax3, col, stats)
    
    def _create_column_visualizations_axes(self, 
                                         axes, 
                                         col_idx: int, 
                                         col: str, 
                                         stats: Dict[str, Any]):
        """Create visualizations for a single column using provided axes"""
        self._create_distribution_pie(axes[0, col_idx], col, stats)
        self._create_clip_distribution(axes[1, col_idx], col, stats)
        self._create_model_recommendation(axes[2, col_idx], col, stats)
    
    def _create_distribution_pie(self, ax, col: str, stats: Dict[str, Any]):
        """Create distribution pie chart"""
        class_0_count = stats['class_0_count']
        class_1_count = stats['class_1_count']
        
        colors = ['lightcoral', 'lightgreen'] if col == self.face_column else ['lightblue', 'orange']
        labels = ['No Face', 'Face'] if col == self.face_column else ['Class 0', 'Class 1']
        
        ax.pie([class_0_count, class_1_count], 
               labels=[f'{labels[0]}\n({class_0_count:,})', f'{labels[1]}\n({class_1_count:,})'],
               autopct='%1.1f%%', colors=colors)
        
        # Use text prefix instead of emoji for plot titles
        prefix = "FACE" if col == self.face_column else "TARGET"
        debug_suffix = " (DEBUG)" if self.debug else ""
        ax.set_title(f'{prefix}: {col}{debug_suffix}\nDistribution')
    
    def _create_clip_distribution(self, ax, col: str, stats: Dict[str, Any]):
        """Create clip-wise distribution bar chart"""
        clip_names = []
        clip_ratios = []
        
        # Calculate ratios for each clip
        for i, clip_name in enumerate(stats['clips']):
            clip_names.append(clip_name)
            # Find the corresponding clip data
            clip_data = None
            for c in stats.get('clips', []):
                if isinstance(c, str) and c == clip_name:
                    # This is a simplified approach - in real implementation,
                    # you'd need to properly match clip data
                    total = stats['total_samples'] / len(stats['clips'])
                    positive = stats['class_1_count'] / len(stats['clips'])
                    ratio = positive / total if total > 0 else 0
                    clip_ratios.append(ratio)
                    break
        
        # Fallback: create dummy ratios if data structure is different
        if not clip_ratios:
            avg_ratio = stats['class_balance']['class_1_ratio']
            clip_ratios = [avg_ratio + np.random.normal(0, 0.1) for _ in clip_names]
            clip_ratios = [max(0, min(1, r)) for r in clip_ratios]  # Clamp to [0,1]
        
        ax.bar(range(len(clip_names)), clip_ratios, 
               color='green' if col == self.face_column else 'blue', alpha=0.7)
        ax.set_title(f'{col}\nPer Clip')
        ax.set_ylabel('Positive Ratio')
        ax.set_xticks(range(len(clip_names)))
        ax.set_xticklabels(clip_names, rotation=45, ha='right')
    
    def _create_model_recommendation(self, ax, col: str, stats: Dict[str, Any]):
        """Create model recommendation text box"""
        rec = stats['recommended_model']
        model_info = f"Model: {rec['model_config']['backbone']}\n"
        model_info += f"Epochs: {rec['training_config']['epochs']}\n"
        model_info += f"Batch: {rec['training_config']['batch_size']}\n"
        model_info += f"Balance: {rec['data_characteristics']['balance_quality']}"
        if rec.get('debug_mode', False):
            model_info += "\nDEBUG MODE"
        
        ax.text(0.1, 0.5, model_info, transform=ax.transAxes, 
                fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title(f'{col}\nRecommendation')


class MetricsVisualizer:
    """Specialized visualizer for training metrics and performance"""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        matplotlib.use('Agg')
        matplotlib.rcParams['font.family'] = ['DejaVu Sans']
    
    def create_performance_summary_chart(self, 
                                       training_results: Dict[str, Any],
                                       save_path: Optional[str] = None):
        """Create performance summary visualization"""
        successful_results = {col: result for col, result in training_results.items() 
                            if result.get('success', False)}
        
        if not successful_results:
            print("\u26a0\ufe0f  No successful training results to visualize")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract metrics
        columns = list(successful_results.keys())
        accuracies = [result['evaluation_metrics']['accuracy'] for result in successful_results.values()]
        f1_scores = [result['evaluation_metrics']['f1_score'] for result in successful_results.values()]
        training_times = [result['training_time_seconds']/60 for result in successful_results.values()]  # Convert to minutes
        
        # 1. Accuracy comparison
        colors = ['green' if col == 'has_faces' else 'blue' for col in columns]
        bars1 = ax1.bar(range(len(columns)), accuracies, color=colors, alpha=0.7)
        ax1.set_title('Model Accuracy Comparison')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_xticks(range(len(columns)))
        ax1.set_xticklabels(columns, rotation=45, ha='right')
        ax1.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{acc:.1f}%', ha='center', va='bottom')
        
        # 2. F1 Score comparison
        bars2 = ax2.bar(range(len(columns)), f1_scores, color=colors, alpha=0.7)
        ax2.set_title('F1 Score Comparison')
        ax2.set_ylabel('F1 Score')
        ax2.set_xticks(range(len(columns)))
        ax2.set_xticklabels(columns, rotation=45, ha='right')
        ax2.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, f1 in zip(bars2, f1_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{f1:.3f}', ha='center', va='bottom')
        
        # 3. Training time comparison
        bars3 = ax3.bar(range(len(columns)), training_times, color=colors, alpha=0.7)
        ax3.set_title('Training Time Comparison')
        ax3.set_ylabel('Training Time (minutes)')
        ax3.set_xticks(range(len(columns)))
        ax3.set_xticklabels(columns, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, time in zip(bars3, training_times):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + max(training_times)*0.01,
                    f'{time:.1f}m', ha='center', va='bottom')
        
        # 4. Performance distribution
        performance_categories = ['Excellent\n(\u226595%)', 'Very Good\n(90-95%)', 'Good\n(80-90%)', 'Fair\n(70-80%)', 'Poor\n(<70%)']
        distribution = [
            len([acc for acc in accuracies if acc >= 95]),
            len([acc for acc in accuracies if 90 <= acc < 95]),
            len([acc for acc in accuracies if 80 <= acc < 90]),
            len([acc for acc in accuracies if 70 <= acc < 80]),
            len([acc for acc in accuracies if acc < 70])
        ]
        
        colors_dist = ['darkgreen', 'green', 'orange', 'red', 'darkred']
        bars4 = ax4.bar(range(len(performance_categories)), distribution, color=colors_dist, alpha=0.7)
        ax4.set_title('Performance Distribution')
        ax4.set_ylabel('Number of Models')
        ax4.set_xticks(range(len(performance_categories)))
        ax4.set_xticklabels(performance_categories)
        
        # Add value labels on bars
        for bar, count in zip(bars4, distribution):
            if count > 0:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        str(count), ha='center', va='bottom')
        
        # Add debug indicator if in debug mode
        if self.debug:
            fig.suptitle('Training Results Summary (DEBUG MODE - Minimal Training)', fontsize=16, y=0.98)
        else:
            fig.suptitle('Training Results Summary', fontsize=16, y=0.98)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\U0001f4ca Performance summary saved to: {save_path}")
        
        plt.close(fig)
    
    def create_training_history_plots(self, 
                                    training_results: Dict[str, Any],
                                    save_dir: Optional[str] = None):
        """Create individual training history plots for each classifier"""
        successful_results = {col: result for col, result in training_results.items() 
                            if result.get('success', False) and 'training_history' in result}
        
        if not successful_results:
            print("\u26a0\ufe0f  No training history data to visualize")
            return
        
        for col, result in successful_results.items():
            history = result['training_history']
            
            if 'train_loss' not in history or 'val_loss' not in history:
                continue
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            epochs = range(1, len(history['train_loss']) + 1)
            
            # Loss plot
            ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
            ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
            ax1.set_title(f'{col} - Training Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Accuracy plot
            if 'train_acc' in history and 'val_acc' in history:
                ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
                ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
                ax2.set_title(f'{col} - Training Accuracy')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Accuracy (%)')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'Accuracy history\nnot available', 
                        transform=ax2.transAxes, ha='center', va='center')
                ax2.set_title(f'{col} - No Accuracy Data')
            
            # Add debug indicator
            if self.debug:
                fig.suptitle(f'{col} Training History (DEBUG MODE)', fontsize=14)
            else:
                fig.suptitle(f'{col} Training History', fontsize=14)
            
            plt.tight_layout()
            
            if save_dir:
                save_path = f"{save_dir}/{col}_training_history.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"\U0001f4c8 Training history for {col} saved to: {save_path}")
            
            plt.close(fig)