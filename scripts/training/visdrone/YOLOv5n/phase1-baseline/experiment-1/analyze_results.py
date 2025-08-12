#!/usr/bin/env python3
"""
YOLOv5n VisDrone Phase 1 Results Analyzer - Experiment 1
=========================================================

This script analyzes and processes results from Phase 1 baseline training,
validation, and weather testing. It generates comprehensive reports, visualizations,
and summary statistics for thesis documentation.

Usage:
    python analyze_results.py [--experiment-dir path] [--output-format json,md,csv]
    
    --experiment-dir: Path to experiment results directory (default: logs&results)
    --output-format: Output formats (default: json,md)
"""

import argparse
import os
import sys
import json
import csv
import re
from datetime import datetime
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union
import numpy as np


class ResultsAnalyzer:
    """Comprehensive results analyzer for Phase 1 baseline experiments"""
    
    def __init__(self, experiment_dir: Path, output_formats: List[str] = None):
        self.experiment_dir = Path(experiment_dir)
        self.output_formats = output_formats or ['json', 'md']
        self.setup_paths()
        self.analysis_results = {}
        self.found_results = {}
        
    def setup_paths(self):
        """Setup analysis paths and create output directory"""
        self.script_dir = Path(__file__).parent
        self.analysis_output = self.script_dir / "analysis_output"
        self.analysis_output.mkdir(parents=True, exist_ok=True)
        
        print(f"[SETUP] Analyzing results from: {self.experiment_dir}")
        print(f"[SETUP] Analysis output directory: {self.analysis_output}")
        
    def discover_results(self):
        """Discover and catalog all available results"""
        print("[DISCOVERY] Scanning for experiment results...")
        
        self.found_results = {
            'training': [],
            'validation': [],
            'weather_testing': [],
            'experiment_summaries': []
        }
        
        if not self.experiment_dir.exists():
            print(f"[ERROR] Experiment directory not found: {self.experiment_dir}")
            return False
        
        # Scan for training results
        training_dir = self.experiment_dir / "training"
        if training_dir.exists():
            # Look for training summaries
            for file in training_dir.rglob("training_summary_*.json"):
                self.found_results['training'].append(file)
            
            # Look for YOLOv5 results
            for file in training_dir.rglob("results.csv"):
                self.found_results['training'].append(file)
                
        # Scan for validation results  
        validation_dir = self.experiment_dir / "validation"
        if validation_dir.exists():
            for file in validation_dir.rglob("validation_summary.json"):
                self.found_results['validation'].append(file)
            for file in validation_dir.rglob("metrics.json"):
                self.found_results['validation'].append(file)
                
        # Scan for weather testing results
        weather_dir = self.experiment_dir / "weather_testing"
        if weather_dir.exists():
            for file in weather_dir.rglob("weather_testing_complete_*.json"):
                self.found_results['weather_testing'].append(file)
            for file in weather_dir.rglob("weather_comparison_*.json"):
                self.found_results['weather_testing'].append(file)
                
        # Scan for experiment summaries
        for file in self.experiment_dir.rglob("experiment_summary_*.json"):
            self.found_results['experiment_summaries'].append(file)
            
        # Report discovery results
        total_files = sum(len(files) for files in self.found_results.values())
        print(f"[DISCOVERY] Found {total_files} result files:")
        for category, files in self.found_results.items():
            print(f"  {category}: {len(files)} files")
            
        return total_files > 0
        
    def analyze_training_results(self):
        """Analyze training results and extract key metrics"""
        print("[ANALYSIS] Analyzing training results...")
        
        training_analysis = {
            'summaries': [],
            'learning_curves': [],
            'convergence_analysis': {},
            'performance_metrics': {}
        }
        
        # Process training summaries
        for summary_file in self.found_results['training']:
            if 'training_summary' in summary_file.name:
                try:
                    with open(summary_file, 'r') as f:
                        summary = json.load(f)
                    training_analysis['summaries'].append(summary)
                    print(f"  Processed training summary: {summary_file.name}")
                except Exception as e:
                    print(f"  Error processing {summary_file}: {e}")
            
            elif 'results.csv' in summary_file.name:
                try:
                    # Read YOLOv5 training results CSV
                    df = pd.read_csv(summary_file)
                    learning_curve = {
                        'file': str(summary_file),
                        'epochs': df['epoch'].tolist() if 'epoch' in df.columns else [],
                        'train_loss': df['train/box_loss'].tolist() if 'train/box_loss' in df.columns else [],
                        'val_loss': df['val/box_loss'].tolist() if 'val/box_loss' in df.columns else [],
                        'map50': df['metrics/mAP_0.5'].tolist() if 'metrics/mAP_0.5' in df.columns else [],
                        'map50_95': df['metrics/mAP_0.5:0.95'].tolist() if 'metrics/mAP_0.5:0.95' in df.columns else []
                    }
                    training_analysis['learning_curves'].append(learning_curve)
                    print(f"  Processed learning curve: {summary_file.name}")
                except Exception as e:
                    print(f"  Error processing {summary_file}: {e}")
        
        # Analyze convergence patterns
        if training_analysis['learning_curves']:
            self.analyze_convergence_patterns(training_analysis)
        
        self.analysis_results['training'] = training_analysis
        
    def analyze_convergence_patterns(self, training_analysis):
        """Analyze training convergence patterns"""
        print("  Analyzing convergence patterns...")
        
        convergence_stats = {
            'best_epochs': [],
            'convergence_speed': [],
            'final_performance': []
        }
        
        for curve in training_analysis['learning_curves']:
            if curve['map50']:
                # Find best epoch
                best_map50 = max(curve['map50'])
                best_epoch = curve['map50'].index(best_map50)
                convergence_stats['best_epochs'].append(best_epoch)
                
                # Calculate convergence speed (epochs to reach 90% of best performance)
                target_performance = best_map50 * 0.9
                convergence_epoch = None
                for i, map50 in enumerate(curve['map50']):
                    if map50 >= target_performance:
                        convergence_epoch = i
                        break
                
                if convergence_epoch is not None:
                    convergence_stats['convergence_speed'].append(convergence_epoch)
                
                # Final performance
                if len(curve['map50']) > 0:
                    convergence_stats['final_performance'].append(curve['map50'][-1])
        
        # Calculate statistics
        if convergence_stats['best_epochs']:
            training_analysis['convergence_analysis'] = {
                'average_best_epoch': np.mean(convergence_stats['best_epochs']),
                'std_best_epoch': np.std(convergence_stats['best_epochs']),
                'average_convergence_speed': np.mean(convergence_stats['convergence_speed']) if convergence_stats['convergence_speed'] else None,
                'average_final_performance': np.mean(convergence_stats['final_performance']),
                'best_performance_achieved': max(max(curve['map50']) for curve in training_analysis['learning_curves'] if curve['map50'])
            }
        
    def analyze_validation_results(self):
        """Analyze validation results"""
        print("[ANALYSIS] Analyzing validation results...")
        
        validation_analysis = {
            'summaries': [],
            'metrics_comparison': {},
            'performance_distribution': {}
        }
        
        # Process validation summaries
        for validation_file in self.found_results['validation']:
            try:
                with open(validation_file, 'r') as f:
                    validation_data = json.load(f)
                validation_analysis['summaries'].append(validation_data)
                print(f"  Processed validation file: {validation_file.name}")
            except Exception as e:
                print(f"  Error processing {validation_file}: {e}")
        
        # Extract and compare metrics
        if validation_analysis['summaries']:
            self.extract_validation_metrics(validation_analysis)
        
        self.analysis_results['validation'] = validation_analysis
        
    def extract_validation_metrics(self, validation_analysis):
        """Extract validation metrics for comparison"""
        print("  Extracting validation metrics...")
        
        metrics_data = {
            'map50': [],
            'map50_95': [],
            'precision': [],
            'recall': []
        }
        
        for summary in validation_analysis['summaries']:
            if 'metrics' in summary:
                metrics = summary['metrics']
                for metric in metrics_data.keys():
                    if metric.upper() in metrics or f"mAP_{metric.split('map')[1]}" in metrics:
                        # Handle different metric naming conventions
                        value = None
                        if 'mAP_50' in metrics:
                            if metric == 'map50':
                                value = metrics['mAP_50']
                        elif 'mAP_50_95' in metrics:
                            if metric == 'map50_95':
                                value = metrics['mAP_50_95']
                        elif metric in metrics:
                            value = metrics[metric]
                        
                        if value is not None:
                            metrics_data[metric].append(value)
        
        # Calculate statistics
        validation_analysis['metrics_comparison'] = {}
        for metric, values in metrics_data.items():
            if values:
                validation_analysis['metrics_comparison'][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
        
    def analyze_weather_testing_results(self):
        """Analyze weather testing results"""
        print("[ANALYSIS] Analyzing weather testing results...")
        
        weather_analysis = {
            'complete_tests': [],
            'degradation_analysis': {},
            'condition_comparison': {},
            'robustness_metrics': {}
        }
        
        # Process weather testing results
        for weather_file in self.found_results['weather_testing']:
            try:
                with open(weather_file, 'r') as f:
                    weather_data = json.load(f)
                weather_analysis['complete_tests'].append(weather_data)
                print(f"  Processed weather file: {weather_file.name}")
            except Exception as e:
                print(f"  Error processing {weather_file}: {e}")
        
        # Analyze degradation patterns
        if weather_analysis['complete_tests']:
            self.analyze_weather_degradation(weather_analysis)
        
        self.analysis_results['weather_testing'] = weather_analysis
        
    def analyze_weather_degradation(self, weather_analysis):
        """Analyze weather-induced performance degradation"""
        print("  Analyzing weather degradation patterns...")
        
        degradation_stats = {
            'by_condition': {},
            'overall_stats': {},
            'degradation_ranking': []
        }
        
        for test_data in weather_analysis['complete_tests']:
            if 'weather_results' in test_data:
                baseline_map50 = None
                
                # Find baseline (clean) performance
                for condition, result in test_data['weather_results'].items():
                    if condition == 'clean' and 'metrics' in result:
                        baseline_map50 = result['metrics'].get('map50', 0)
                        break
                
                if baseline_map50:
                    # Calculate degradation for each condition
                    for condition, result in test_data['weather_results'].items():
                        if condition != 'clean' and 'metrics' in result:
                            condition_map50 = result['metrics'].get('map50', 0)
                            degradation_pct = ((baseline_map50 - condition_map50) / baseline_map50) * 100
                            
                            if condition not in degradation_stats['by_condition']:
                                degradation_stats['by_condition'][condition] = []
                            
                            degradation_stats['by_condition'][condition].append(degradation_pct)
        
        # Calculate condition statistics
        for condition, degradations in degradation_stats['by_condition'].items():
            if degradations:
                degradation_stats['by_condition'][condition] = {
                    'mean_degradation': np.mean(degradations),
                    'std_degradation': np.std(degradations),
                    'max_degradation': np.max(degradations),
                    'min_degradation': np.min(degradations),
                    'count': len(degradations)
                }
        
        # Create degradation ranking
        condition_means = []
        for condition, stats in degradation_stats['by_condition'].items():
            if isinstance(stats, dict) and 'mean_degradation' in stats:
                condition_means.append((condition, stats['mean_degradation']))
        
        degradation_stats['degradation_ranking'] = sorted(condition_means, 
                                                         key=lambda x: x[1], 
                                                         reverse=True)
        
        weather_analysis['degradation_analysis'] = degradation_stats
        
    def generate_performance_summary(self):
        """Generate overall performance summary"""
        print("[ANALYSIS] Generating performance summary...")
        
        summary = {
            'experiment_overview': {},
            'key_findings': [],
            'performance_benchmarks': {},
            'protocol_compliance': {},
            'recommendations': []
        }
        
        # Extract key performance metrics
        if 'validation' in self.analysis_results:
            val_results = self.analysis_results['validation']
            if 'metrics_comparison' in val_results:
                metrics = val_results['metrics_comparison']
                
                if 'map50' in metrics:
                    map50_mean = metrics['map50']['mean']
                    summary['performance_benchmarks']['baseline_map50'] = map50_mean
                    
                    # Compare with recovery analysis expectations
                    if 0.17 <= map50_mean <= 0.25:
                        summary['key_findings'].append(f"Baseline mAP@0.5 ({map50_mean:.3f}) within expected range (17-25%)")
                    elif map50_mean > 0.25:
                        summary['key_findings'].append(f"Baseline mAP@0.5 ({map50_mean:.3f}) exceeds expectations (>25%)")
                    else:
                        summary['key_findings'].append(f"Baseline mAP@0.5 ({map50_mean:.3f}) below expected range (<17%)")
        
        # Extract weather robustness findings
        if 'weather_testing' in self.analysis_results:
            weather_results = self.analysis_results['weather_testing']
            if 'degradation_analysis' in weather_results:
                degradation = weather_results['degradation_analysis']
                
                if 'degradation_ranking' in degradation:
                    ranking = degradation['degradation_ranking']
                    if ranking:
                        worst_condition = ranking[0]
                        summary['key_findings'].append(
                            f"Most challenging condition: {worst_condition[0]} ({worst_condition[1]:.1f}% degradation)"
                        )
                        
                        # Calculate average degradation
                        avg_degradation = np.mean([deg for _, deg in ranking])
                        summary['performance_benchmarks']['average_weather_degradation'] = avg_degradation
                        
                        if 80 <= avg_degradation <= 95:
                            summary['key_findings'].append(f"Weather degradation ({avg_degradation:.1f}%) within expected range (80-95%)")
        
        # Protocol compliance check
        if 'training' in self.analysis_results:
            training_results = self.analysis_results['training']
            if 'summaries' in training_results and training_results['summaries']:
                summary_data = training_results['summaries'][0]
                if 'configuration' in summary_data:
                    config = summary_data['configuration']
                    
                    # Check augmentation compliance
                    if config.get('augmentation') == 'disabled':
                        summary['protocol_compliance']['zero_augmentation'] = True
                        summary['key_findings'].append("Phase 1 protocol compliance: Zero augmentation confirmed")
                    
                    # Check optimizer
                    if config.get('optimizer') == 'SGD':
                        summary['protocol_compliance']['sgd_optimizer'] = True
                        summary['key_findings'].append("Optimization applied: SGD optimizer used (recovery analysis finding)")
        
        self.analysis_results['summary'] = summary
        
    def create_visualizations(self):
        """Create performance visualizations"""
        print("[ANALYSIS] Creating visualizations...")
        
        try:
            # Set style for better plots
            plt.style.use('seaborn-v0_8')
            
            # Create learning curves plot
            if 'training' in self.analysis_results:
                self.plot_learning_curves()
            
            # Create weather degradation plot
            if 'weather_testing' in self.analysis_results:
                self.plot_weather_degradation()
            
            # Create metrics comparison plot
            if 'validation' in self.analysis_results:
                self.plot_metrics_comparison()
                
        except Exception as e:
            print(f"  Warning: Could not create visualizations: {e}")
            
    def plot_learning_curves(self):
        """Plot training learning curves"""
        training_data = self.analysis_results['training']
        
        if 'learning_curves' in training_data and training_data['learning_curves']:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Training Learning Curves', fontsize=16)
            
            for curve in training_data['learning_curves']:
                if curve['epochs'] and curve['map50']:
                    epochs = curve['epochs']
                    
                    # Plot mAP@0.5
                    if curve['map50']:
                        axes[0, 0].plot(epochs, curve['map50'], label='mAP@0.5')
                    
                    # Plot mAP@0.5:0.95
                    if curve['map50_95']:
                        axes[0, 1].plot(epochs, curve['map50_95'], label='mAP@0.5:0.95')
                    
                    # Plot training loss
                    if curve['train_loss']:
                        axes[1, 0].plot(epochs, curve['train_loss'], label='Train Loss')
                    
                    # Plot validation loss
                    if curve['val_loss']:
                        axes[1, 1].plot(epochs, curve['val_loss'], label='Val Loss')
            
            # Configure subplots
            axes[0, 0].set_title('mAP@0.5')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('mAP@0.5')
            axes[0, 0].grid(True)
            
            axes[0, 1].set_title('mAP@0.5:0.95')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('mAP@0.5:0.95')
            axes[0, 1].grid(True)
            
            axes[1, 0].set_title('Training Loss')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True)
            
            axes[1, 1].set_title('Validation Loss')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            plt.savefig(self.analysis_output / 'learning_curves.png', dpi=300, bbox_inches='tight')
            plt.close()
            
    def plot_weather_degradation(self):
        """Plot weather degradation analysis"""
        weather_data = self.analysis_results['weather_testing']
        
        if 'degradation_analysis' in weather_data:
            degradation = weather_data['degradation_analysis']
            
            if 'degradation_ranking' in degradation and degradation['degradation_ranking']:
                conditions = [item[0] for item in degradation['degradation_ranking']]
                degradations = [item[1] for item in degradation['degradation_ranking']]
                
                plt.figure(figsize=(10, 6))
                bars = plt.bar(conditions, degradations, color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'])
                plt.title('Performance Degradation by Weather Condition', fontsize=14)
                plt.xlabel('Weather Condition')
                plt.ylabel('Performance Degradation (%)')
                plt.xticks(rotation=45)
                
                # Add value labels on bars
                for bar, value in zip(bars, degradations):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                            f'{value:.1f}%', ha='center', va='bottom')
                
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(self.analysis_output / 'weather_degradation.png', dpi=300, bbox_inches='tight')
                plt.close()
                
    def plot_metrics_comparison(self):
        """Plot validation metrics comparison"""
        validation_data = self.analysis_results['validation']
        
        if 'metrics_comparison' in validation_data:
            metrics = validation_data['metrics_comparison']
            
            metric_names = list(metrics.keys())
            mean_values = [metrics[metric]['mean'] for metric in metric_names]
            std_values = [metrics[metric]['std'] for metric in metric_names]
            
            plt.figure(figsize=(8, 6))
            bars = plt.bar(metric_names, mean_values, yerr=std_values, capsize=5,
                          color=['#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
            plt.title('Validation Metrics Summary', fontsize=14)
            plt.xlabel('Metric')
            plt.ylabel('Value')
            
            # Add value labels
            for bar, value in zip(bars, mean_values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.analysis_output / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
    def save_analysis_results(self):
        """Save analysis results in requested formats"""
        print("[OUTPUT] Saving analysis results...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON format
        if 'json' in self.output_formats:
            json_file = self.analysis_output / f"analysis_results_{timestamp}.json"
            with open(json_file, 'w') as f:
                json.dump(self.analysis_results, f, indent=2, default=str)
            print(f"  JSON results saved: {json_file}")
        
        # Save Markdown format
        if 'md' in self.output_formats:
            md_file = self.analysis_output / f"analysis_report_{timestamp}.md"
            self.generate_markdown_report(md_file)
            print(f"  Markdown report saved: {md_file}")
        
        # Save CSV format for metrics
        if 'csv' in self.output_formats:
            self.save_csv_summaries(timestamp)
            
    def generate_markdown_report(self, output_file):
        """Generate comprehensive markdown report"""
        
        with open(output_file, 'w') as f:
            f.write("# YOLOv5n VisDrone Phase 1 Baseline - Analysis Report\\n\\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
            
            # Executive Summary
            if 'summary' in self.analysis_results:
                summary = self.analysis_results['summary']
                f.write("## Executive Summary\\n\\n")
                
                if 'key_findings' in summary:
                    f.write("### Key Findings\\n\\n")
                    for finding in summary['key_findings']:
                        f.write(f"- {finding}\\n")
                    f.write("\\n")
                
                if 'performance_benchmarks' in summary:
                    f.write("### Performance Benchmarks\\n\\n")
                    benchmarks = summary['performance_benchmarks']
                    f.write("| Metric | Value |\\n")
                    f.write("|--------|-------|\\n")
                    for metric, value in benchmarks.items():
                        f.write(f"| {metric.replace('_', ' ').title()} | {value:.3f} |\\n")
                    f.write("\\n")
            
            # Training Analysis
            if 'training' in self.analysis_results:
                f.write("## Training Analysis\\n\\n")
                training = self.analysis_results['training']
                
                if 'convergence_analysis' in training and training['convergence_analysis']:
                    conv = training['convergence_analysis']
                    f.write("### Convergence Analysis\\n\\n")
                    f.write(f"- **Average Best Epoch**: {conv.get('average_best_epoch', 'N/A'):.1f}\\n")
                    f.write(f"- **Best Performance Achieved**: {conv.get('best_performance_achieved', 'N/A'):.3f}\\n")
                    f.write(f"- **Average Final Performance**: {conv.get('average_final_performance', 'N/A'):.3f}\\n\\n")
            
            # Validation Analysis
            if 'validation' in self.analysis_results:
                f.write("## Validation Analysis\\n\\n")
                validation = self.analysis_results['validation']
                
                if 'metrics_comparison' in validation:
                    f.write("### Metrics Summary\\n\\n")
                    f.write("| Metric | Mean | Std | Min | Max |\\n")
                    f.write("|--------|------|-----|-----|-----|\\n")
                    
                    for metric, stats in validation['metrics_comparison'].items():
                        f.write(f"| {metric} | {stats['mean']:.3f} | {stats['std']:.3f} | "
                               f"{stats['min']:.3f} | {stats['max']:.3f} |\\n")
                    f.write("\\n")
            
            # Weather Testing Analysis
            if 'weather_testing' in self.analysis_results:
                f.write("## Weather Testing Analysis\\n\\n")
                weather = self.analysis_results['weather_testing']
                
                if 'degradation_analysis' in weather:
                    degradation = weather['degradation_analysis']
                    
                    if 'degradation_ranking' in degradation:
                        f.write("### Performance Degradation by Condition\\n\\n")
                        f.write("| Condition | Degradation (%) |\\n")
                        f.write("|-----------|-----------------|\\n")
                        
                        for condition, deg in degradation['degradation_ranking']:
                            f.write(f"| {condition.title()} | {deg:.1f}% |\\n")
                        f.write("\\n")
            
            f.write("---\\n\\n")
            f.write("*Report generated by YOLOv5n VisDrone Phase 1 Results Analyzer*\\n")
            
    def save_csv_summaries(self, timestamp):
        """Save key metrics in CSV format"""
        
        # Save validation metrics
        if 'validation' in self.analysis_results:
            validation = self.analysis_results['validation']
            if 'metrics_comparison' in validation:
                csv_file = self.analysis_output / f"validation_metrics_{timestamp}.csv"
                
                with open(csv_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Metric', 'Mean', 'Std', 'Min', 'Max', 'Count'])
                    
                    for metric, stats in validation['metrics_comparison'].items():
                        writer.writerow([
                            metric,
                            f"{stats['mean']:.6f}",
                            f"{stats['std']:.6f}",
                            f"{stats['min']:.6f}",
                            f"{stats['max']:.6f}",
                            stats['count']
                        ])
                
                print(f"  CSV metrics saved: {csv_file}")
        
        # Save weather degradation data
        if 'weather_testing' in self.analysis_results:
            weather = self.analysis_results['weather_testing']
            if 'degradation_analysis' in weather and 'degradation_ranking' in weather['degradation_analysis']:
                csv_file = self.analysis_output / f"weather_degradation_{timestamp}.csv"
                
                with open(csv_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Condition', 'Degradation_Percent'])
                    
                    for condition, degradation in weather['degradation_analysis']['degradation_ranking']:
                        writer.writerow([condition, f"{degradation:.2f}"])
                
                print(f"  CSV degradation saved: {csv_file}")
                
    def run_comprehensive_analysis(self):
        """Run complete analysis workflow"""
        print("[ANALYZER] Starting comprehensive results analysis...")
        
        try:
            # Discover available results
            if not self.discover_results():
                print("[ERROR] No results found to analyze")
                return False
            
            # Run analysis modules
            self.analyze_training_results()
            self.analyze_validation_results()
            self.analyze_weather_testing_results()
            
            # Generate summary
            self.generate_performance_summary()
            
            # Create visualizations
            self.create_visualizations()
            
            # Save results
            self.save_analysis_results()
            
            print("[ANALYZER] Analysis completed successfully")
            return True
            
        except Exception as e:
            print(f"[ERROR] Analysis failed: {str(e)}")
            return False


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='YOLOv5n VisDrone Phase 1 Results Analyzer')
    parser.add_argument('--experiment-dir', type=str, default='logs&results',
                       help='Path to experiment results directory (default: logs&results)')
    parser.add_argument('--output-format', type=str, default='json,md',
                       help='Output formats: json,md,csv (default: json,md)')
    
    args = parser.parse_args()
    
    # Parse output formats
    output_formats = [fmt.strip() for fmt in args.output_format.split(',')]
    
    # Resolve experiment directory path
    script_dir = Path(__file__).parent
    if Path(args.experiment_dir).is_absolute():
        experiment_dir = Path(args.experiment_dir)
    else:
        experiment_dir = script_dir / args.experiment_dir
    
    try:
        analyzer = ResultsAnalyzer(experiment_dir, output_formats)
        success = analyzer.run_comprehensive_analysis()
        
        if success:
            print("\\n[SUCCESS] Analysis completed successfully!")
            print(f"Results saved in: {analyzer.analysis_output}")
        else:
            print("\\n[ERROR] Analysis failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\\nAnalysis failed with error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()