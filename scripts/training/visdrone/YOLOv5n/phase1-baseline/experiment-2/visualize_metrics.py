#!/usr/bin/env python3
"""
Metrics Visualization Dashboard for YOLOv5n VisDrone Experiment 2
================================================================

This script provides comprehensive visualization of training metrics from:
1. Phase-level metrics (from metrics_tracking logs)
2. Epoch-level metrics (from results.csv files)
3. Comparative analysis between phases

Usage:
    python visualize_metrics.py                    # Visualize latest run
    python visualize_metrics.py --compare exp1     # Compare with Experiment 1
    python visualize_metrics.py --export           # Export plots to files
"""

import argparse
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10

# Project paths
SCRIPT_DIR = Path(__file__).parent
LOGS_DIR = SCRIPT_DIR / "logs&results"
TRAINING_DIR = LOGS_DIR / "training"


class MetricsVisualizer:
    """Comprehensive metrics visualization for progressive training"""
    
    def __init__(self, export_plots: bool = False):
        """Initialize visualizer"""
        self.export_plots = export_plots
        self.plots_dir = LOGS_DIR / "visualizations"
        if export_plots:
            self.plots_dir.mkdir(exist_ok=True)
            
    def find_latest_metrics_log(self) -> Optional[Path]:
        """Find the most recent metrics tracking log"""
        metrics_files = list(LOGS_DIR.glob("metrics_tracking_*.log"))
        if not metrics_files:
            print("No metrics tracking logs found")
            return None
        return max(metrics_files, key=lambda f: f.stat().st_mtime)
        
    def parse_metrics_log(self, log_file: Path) -> Dict:
        """Parse phase-level metrics from tracking log"""
        metrics = {
            'phases': [],
            'mAP_0.5': [],
            'mAP_0.5:0.95': [],
            'precision': [],
            'recall': [],
            'duration': [],
            'status': []
        }
        
        with open(log_file, 'r') as f:
            for line in f:
                if 'Phase=' in line:
                    try:
                        # Extract metrics using regex
                        phase = re.search(r'Phase=(\w+)', line).group(1)
                        status = re.search(r'Status=(\w+)', line).group(1)
                        duration = float(re.search(r'Duration=([\d.]+)h', line).group(1))
                        
                        # Extract performance metrics (handle N/A values)
                        map50_match = re.search(r'mAP@0.5=([\d.]+|N/A)', line)
                        map50 = float(map50_match.group(1)) if map50_match and map50_match.group(1) != 'N/A' else None
                        
                        map50_95_match = re.search(r'mAP@0.5:0.95=([\d.]+|N/A)', line)
                        map50_95 = float(map50_95_match.group(1)) if map50_95_match and map50_95_match.group(1) != 'N/A' else None
                        
                        precision_match = re.search(r'Precision=([\d.]+|N/A)', line)
                        precision = float(precision_match.group(1)) if precision_match and precision_match.group(1) != 'N/A' else None
                        
                        recall_match = re.search(r'Recall=([\d.]+|N/A)', line)
                        recall = float(recall_match.group(1)) if recall_match and recall_match.group(1) != 'N/A' else None
                        
                        metrics['phases'].append(phase)
                        metrics['status'].append(status)
                        metrics['duration'].append(duration)
                        metrics['mAP_0.5'].append(map50)
                        metrics['mAP_0.5:0.95'].append(map50_95)
                        metrics['precision'].append(precision)
                        metrics['recall'].append(recall)
                        
                    except Exception as e:
                        print(f"Error parsing line: {line}")
                        print(f"Error: {e}")
                        
        return metrics
        
    def load_epoch_metrics(self, phase_name: str) -> Optional[pd.DataFrame]:
        """Load epoch-level metrics from results.csv"""
        results_file = TRAINING_DIR / phase_name / "results.csv"
        if not results_file.exists():
            # Try quicktest naming
            results_file = TRAINING_DIR / f"{phase_name}_quicktest" / "results.csv"
            
        if results_file.exists():
            try:
                df = pd.read_csv(results_file)
                # Clean column names
                df.columns = [col.strip() for col in df.columns]
                return df
            except Exception as e:
                print(f"Error loading {results_file}: {e}")
                return None
        return None
        
    def plot_phase_progression(self, metrics: Dict):
        """Plot phase-level progression"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle('Progressive Training - Phase-Level Analysis', fontsize=16, fontweight='bold')
        
        # Filter out None values for plotting
        valid_indices = [i for i, v in enumerate(metrics['mAP_0.5']) if v is not None]
        
        if not valid_indices:
            print("No valid metrics to plot")
            return
            
        phases = [metrics['phases'][i] for i in valid_indices]
        
        # mAP@0.5 progression
        ax = axes[0, 0]
        map_values = [metrics['mAP_0.5'][i] for i in valid_indices]
        bars = ax.bar(phases, map_values, color='steelblue')
        ax.set_title('mAP@0.5 Progression', fontweight='bold')
        ax.set_ylabel('mAP@0.5')
        ax.set_ylim([0, max(map_values) * 1.2] if map_values else [0, 1])
        
        # Add value labels on bars
        for bar, val in zip(bars, map_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        
        # Add percentage improvement
        if len(map_values) > 1:
            for i in range(1, len(map_values)):
                improvement = ((map_values[i] - map_values[i-1]) / map_values[i-1]) * 100
                ax.text(i, map_values[i]/2, f'+{improvement:.1f}%', 
                       ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        
        # Precision progression
        ax = axes[0, 1]
        prec_values = [metrics['precision'][i] for i in valid_indices]
        bars = ax.bar(phases, prec_values, color='green')
        ax.set_title('Precision Progression', fontweight='bold')
        ax.set_ylabel('Precision')
        ax.set_ylim([0, max(prec_values) * 1.2] if prec_values else [0, 1])
        
        for bar, val in zip(bars, prec_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        
        # Recall progression
        ax = axes[0, 2]
        recall_values = [metrics['recall'][i] for i in valid_indices]
        bars = ax.bar(phases, recall_values, color='orange')
        ax.set_title('Recall Progression', fontweight='bold')
        ax.set_ylabel('Recall')
        ax.set_ylim([0, max(recall_values) * 1.2] if recall_values else [0, 1])
        
        for bar, val in zip(bars, recall_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        
        # mAP@0.5:0.95 progression
        ax = axes[1, 0]
        map95_values = [metrics['mAP_0.5:0.95'][i] for i in valid_indices if metrics['mAP_0.5:0.95'][i] is not None]
        if map95_values:
            bars = ax.bar(phases[:len(map95_values)], map95_values, color='purple')
            ax.set_title('mAP@0.5:0.95 Progression', fontweight='bold')
            ax.set_ylabel('mAP@0.5:0.95')
            ax.set_ylim([0, max(map95_values) * 1.2])
            
            for bar, val in zip(bars, map95_values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                       f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        
        # Training duration
        ax = axes[1, 1]
        duration_values = [metrics['duration'][i] for i in valid_indices]
        bars = ax.bar(phases, duration_values, color='coral')
        ax.set_title('Training Duration per Phase', fontweight='bold')
        ax.set_ylabel('Hours')
        
        for bar, val in zip(bars, duration_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.2f}h', ha='center', va='bottom', fontsize=9)
        
        # F1 Score (calculated from precision and recall)
        ax = axes[1, 2]
        f1_scores = []
        for i in valid_indices:
            p = metrics['precision'][i]
            r = metrics['recall'][i]
            if p and r and (p + r) > 0:
                f1 = 2 * (p * r) / (p + r)
                f1_scores.append(f1)
            else:
                f1_scores.append(0)
                
        bars = ax.bar(phases, f1_scores, color='crimson')
        ax.set_title('F1 Score Progression', fontweight='bold')
        ax.set_ylabel('F1 Score')
        ax.set_ylim([0, max(f1_scores) * 1.2] if f1_scores else [0, 1])
        
        for bar, val in zip(bars, f1_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if self.export_plots:
            plt.savefig(self.plots_dir / 'phase_progression.png', dpi=300, bbox_inches='tight')
            print(f"Saved phase progression plot to {self.plots_dir / 'phase_progression.png'}")
            
        plt.show()
        
    def plot_epoch_curves(self):
        """Plot epoch-level training curves for all phases"""
        phases = ['exp2_phase2a', 'exp2_phase2b', 'exp2_phase2c']
        phase_labels = ['Phase 2A', 'Phase 2B', 'Phase 2C']
        colors = ['blue', 'green', 'red']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle('Epoch-Level Training Curves', fontsize=16, fontweight='bold')
        
        all_data = {}
        
        for phase, label, color in zip(phases, phase_labels, colors):
            df = self.load_epoch_metrics(phase)
            if df is not None:
                all_data[label] = df
                
                # Adjust epoch numbers for continuous plotting
                if label == 'Phase 2B' and 'Phase 2A' in all_data:
                    df['epoch'] = df.index + len(all_data['Phase 2A'])
                elif label == 'Phase 2C' and 'Phase 2B' in all_data:
                    if 'Phase 2A' in all_data:
                        df['epoch'] = df.index + len(all_data['Phase 2A']) + len(all_data['Phase 2B'])
                    else:
                        df['epoch'] = df.index + len(all_data['Phase 2B'])
                else:
                    df['epoch'] = df.index
        
        if not all_data:
            print("No epoch-level data found")
            return
            
        # Plot mAP@0.5
        ax = axes[0, 0]
        for label, df in all_data.items():
            map_col = [c for c in df.columns if 'map_0.5' in c.lower() and 'map_0.5:0.95' not in c.lower()]
            if map_col:
                ax.plot(df['epoch'], df[map_col[0]], marker='o', label=label, linewidth=2)
        ax.set_title('mAP@0.5 Over Epochs', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('mAP@0.5')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot Precision
        ax = axes[0, 1]
        for label, df in all_data.items():
            prec_col = [c for c in df.columns if 'precision' in c.lower()]
            if prec_col:
                ax.plot(df['epoch'], df[prec_col[0]], marker='o', label=label, linewidth=2)
        ax.set_title('Precision Over Epochs', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Precision')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot Recall
        ax = axes[0, 2]
        for label, df in all_data.items():
            recall_col = [c for c in df.columns if 'recall' in c.lower()]
            if recall_col:
                ax.plot(df['epoch'], df[recall_col[0]], marker='o', label=label, linewidth=2)
        ax.set_title('Recall Over Epochs', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Recall')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot Box Loss
        ax = axes[1, 0]
        for label, df in all_data.items():
            box_col = [c for c in df.columns if 'box_loss' in c.lower()]
            if box_col:
                ax.plot(df['epoch'], df[box_col[0]], marker='o', label=label, linewidth=2)
        ax.set_title('Box Loss Over Epochs', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Box Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot Object Loss
        ax = axes[1, 1]
        for label, df in all_data.items():
            obj_col = [c for c in df.columns if 'obj_loss' in c.lower()]
            if obj_col:
                ax.plot(df['epoch'], df[obj_col[0]], marker='o', label=label, linewidth=2)
        ax.set_title('Object Loss Over Epochs', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Object Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot Learning Rate
        ax = axes[1, 2]
        for label, df in all_data.items():
            lr_col = [c for c in df.columns if 'lr0' in c.lower()]
            if lr_col:
                ax.plot(df['epoch'], df[lr_col[0]], marker='o', label=label, linewidth=2)
        ax.set_title('Learning Rate Schedule', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        
        if self.export_plots:
            plt.savefig(self.plots_dir / 'epoch_curves.png', dpi=300, bbox_inches='tight')
            print(f"Saved epoch curves plot to {self.plots_dir / 'epoch_curves.png'}")
            
        plt.show()
        
    def generate_summary_report(self, metrics: Dict):
        """Generate a text summary report"""
        report = []
        report.append("="*80)
        report.append("EXPERIMENT 2 METRICS SUMMARY REPORT")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        if metrics['phases']:
            report.append("PHASE-LEVEL PERFORMANCE:")
            report.append("-"*40)
            
            for i, phase in enumerate(metrics['phases']):
                report.append(f"\n{phase}:")
                report.append(f"  Status: {metrics['status'][i]}")
                report.append(f"  Duration: {metrics['duration'][i]:.2f} hours")
                
                if metrics['mAP_0.5'][i] is not None:
                    report.append(f"  mAP@0.5: {metrics['mAP_0.5'][i]:.4f}")
                if metrics['precision'][i] is not None:
                    report.append(f"  Precision: {metrics['precision'][i]:.4f}")
                if metrics['recall'][i] is not None:
                    report.append(f"  Recall: {metrics['recall'][i]:.4f}")
                if metrics['mAP_0.5:0.95'][i] is not None:
                    report.append(f"  mAP@0.5:0.95: {metrics['mAP_0.5:0.95'][i]:.4f}")
                    
                # Calculate improvement
                if i > 0 and metrics['mAP_0.5'][i] is not None and metrics['mAP_0.5'][i-1] is not None:
                    improvement = ((metrics['mAP_0.5'][i] - metrics['mAP_0.5'][i-1]) / metrics['mAP_0.5'][i-1]) * 100
                    report.append(f"  Improvement from previous: {improvement:+.2f}%")
            
            # Overall statistics
            valid_map = [m for m in metrics['mAP_0.5'] if m is not None]
            if valid_map:
                report.append(f"\nOVERALL STATISTICS:")
                report.append("-"*40)
                report.append(f"Total training time: {sum(metrics['duration']):.2f} hours")
                report.append(f"Best mAP@0.5: {max(valid_map):.4f}")
                report.append(f"Total improvement: {((valid_map[-1] - valid_map[0]) / valid_map[0]) * 100:+.2f}%")
                
        report_text = "\n".join(report)
        
        # Save report
        report_file = LOGS_DIR / f"metrics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report_text)
            
        print(report_text)
        print(f"\nReport saved to: {report_file}")
        
        return report_text
        

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Visualize YOLOv5n VisDrone training metrics')
    parser.add_argument('--export', action='store_true', help='Export plots to files')
    parser.add_argument('--metrics-log', type=str, help='Path to specific metrics log file')
    parser.add_argument('--no-epoch', action='store_true', help='Skip epoch-level plots')
    
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = MetricsVisualizer(export_plots=args.export)
    
    # Find or use specified metrics log
    if args.metrics_log:
        metrics_log = Path(args.metrics_log)
    else:
        metrics_log = visualizer.find_latest_metrics_log()
        
    if not metrics_log or not metrics_log.exists():
        print("No metrics log found. Please run training first.")
        return
        
    print(f"Loading metrics from: {metrics_log}")
    
    # Parse and visualize phase-level metrics
    metrics = visualizer.parse_metrics_log(metrics_log)
    
    if metrics['phases']:
        print(f"Found {len(metrics['phases'])} phases in metrics log")
        visualizer.plot_phase_progression(metrics)
        visualizer.generate_summary_report(metrics)
    else:
        print("No phase metrics found in log")
        
    # Plot epoch-level curves if available
    if not args.no_epoch:
        print("\nGenerating epoch-level curves...")
        visualizer.plot_epoch_curves()
    
    print("\nVisualization complete!")
    

if __name__ == "__main__":
    main()