#!/usr/bin/env python3
"""
YOLOv5n VisDrone Phase 1 Baseline - Experiment 2 Multi-Weather Validation Script
================================================================================

This script performs comprehensive validation across multiple weather conditions
to track model robustness throughout training. It validates on clean data every
epoch and weather conditions every 5 epochs as per Experiment 2 design.

Usage:
    python validate_multi_weather.py --weights path/to/best.pt
    python validate_multi_weather.py --weights path/to/best.pt --weather-only
    python validate_multi_weather.py --phase 2a  # Validate specific phase results
"""

import argparse
import os
import sys
import subprocess
import json
import time
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import yaml
from typing import Dict, List, Tuple, Optional

# Project paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.parent.parent.parent
YOLO_PATH = PROJECT_ROOT / "models" / "yolov5n" / "baseline" / "yolov5"
CONFIG_DIR = PROJECT_ROOT / "configs" / "yolov5n-visdrone"
LOGS_DIR = SCRIPT_DIR / "logs&results"
DATA_ROOT = PROJECT_ROOT / "data"

# Add YOLOv5 to path
sys.path.append(str(YOLO_PATH))


class MultiWeatherValidator:
    """Multi-weather validation manager for Experiment 2"""
    
    def __init__(self, weights_path: str, conf_thres: float = 0.001, 
                 iou_thres: float = 0.6, batch_size: int = 16):
        """
        Initialize the multi-weather validator
        
        Args:
            weights_path: Path to model weights
            conf_thres: Confidence threshold for detections
            iou_thres: IoU threshold for NMS
            batch_size: Batch size for validation
        """
        self.weights_path = Path(weights_path).resolve()
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.batch_size = batch_size
        self.setup_paths()
        self.setup_logging()
        self.define_validation_conditions()
        
    def setup_paths(self):
        """Setup all required paths"""
        self.yolo_val = YOLO_PATH / "val.py"
        self.data_config_base = CONFIG_DIR / "visdrone_experiment2.yaml"
        self.validation_dir = LOGS_DIR / "validation"
        self.validation_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamp for this validation run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.validation_dir / f"multi_weather_{self.timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_file = self.run_dir / "validation.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("="*80)
        self.logger.info("YOLOv5n VisDrone Experiment 2 - Multi-Weather Validation")
        self.logger.info("="*80)
        self.logger.info(f"Weights: {self.weights_path}")
        self.logger.info(f"Confidence threshold: {self.conf_thres}")
        self.logger.info(f"IoU threshold: {self.iou_thres}")
        self.logger.info(f"Batch size: {self.batch_size}")
        
    def define_validation_conditions(self):
        """Define all validation conditions with their configurations"""
        self.conditions = {
            'clean': {
                'name': 'Clean (Baseline)',
                'data_path': DATA_ROOT / "raw" / "visdrone" / "VisDrone2019-DET-val",
                'weight': 0.4,
                'frequency': 'every_epoch',
                'description': 'Original clean validation dataset'
            },
            'fog': {
                'name': 'Fog Synthetic',
                'data_path': DATA_ROOT / "synthetic_test" / "VisDrone2019-DET-test-fog",
                'weight': 0.2,
                'frequency': 'every_5_epochs',
                'description': 'Synthetic fog augmented validation'
            },
            'rain': {
                'name': 'Rain Synthetic',
                'data_path': DATA_ROOT / "synthetic_test" / "VisDrone2019-DET-test-rain",
                'weight': 0.2,
                'frequency': 'every_5_epochs',
                'description': 'Synthetic rain augmented validation'
            },
            'night': {
                'name': 'Night Synthetic',
                'data_path': DATA_ROOT / "synthetic_test" / "VisDrone2019-DET-test-night",
                'weight': 0.2,
                'frequency': 'every_5_epochs',
                'description': 'Synthetic night/low-light validation'
            },
            'mixed': {
                'name': 'Mixed Conditions',
                'data_path': DATA_ROOT / "synthetic_test" / "VisDrone2019-DET-test-mixed",
                'weight': 0.0,  # Not included in composite score
                'frequency': 'every_10_epochs',
                'description': 'Mixed weather conditions validation'
            }
        }
        
    def create_temp_data_config(self, condition: str) -> Path:
        """
        Create temporary data configuration for specific condition
        
        Args:
            condition: Weather condition name
            
        Returns:
            Path to temporary configuration file
        """
        condition_info = self.conditions[condition]
        
        # Load base config
        with open(self.data_config_base, 'r') as f:
            base_config = yaml.safe_load(f)
            
        # Modify for specific condition
        if condition == 'clean':
            # Use standard validation set
            base_config['val'] = 'images/val'
        else:
            # Point to synthetic test data
            base_config['path'] = str(condition_info['data_path'].parent.parent)
            base_config['val'] = str(condition_info['data_path'].name / 'images')
            
        # Save temporary config
        temp_config = self.run_dir / f"data_{condition}.yaml"
        with open(temp_config, 'w') as f:
            yaml.dump(base_config, f)
            
        return temp_config
        
    def validate_condition(self, condition: str) -> Dict:
        """
        Run validation on specific weather condition
        
        Args:
            condition: Weather condition to validate
            
        Returns:
            Dictionary containing validation metrics
        """
        self.logger.info(f"\n[VALIDATION] Testing on {condition.upper()} condition...")
        condition_info = self.conditions[condition]
        
        # Create temporary data config
        temp_config = self.create_temp_data_config(condition)
        
        # Prepare output directory
        output_dir = self.run_dir / f"{condition}_results"
        output_dir.mkdir(exist_ok=True)
        
        # Build validation command
        cmd = [
            sys.executable,
            str(self.yolo_val),
            "--weights", str(self.weights_path),
            "--data", str(temp_config),
            "--batch-size", str(self.batch_size),
            "--imgsz", "640",
            "--conf-thres", str(self.conf_thres),
            "--iou-thres", str(self.iou_thres),
            "--device", "0",
            "--project", str(self.run_dir),
            "--name", f"{condition}_results",
            "--exist-ok",
            "--save-json",
            "--save-txt",
            "--save-conf",
            "--verbose",
            "--plots"
        ]
        
        # Run validation
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                cwd=str(YOLO_PATH),
                capture_output=True,
                text=True,
                check=True
            )
            
            duration = time.time() - start_time
            
            # Parse results from output
            metrics = self.parse_validation_output(result.stdout, condition)
            
            # Add metadata
            metrics.update({
                'condition': condition,
                'condition_name': condition_info['name'],
                'duration_seconds': duration,
                'timestamp': datetime.now().isoformat(),
                'success': True
            })
            
            self.logger.info(f"[VALIDATION] {condition.upper()} Results:")
            self.logger.info(f"  mAP@0.5      : {metrics.get('map50', 0):.4f}")
            self.logger.info(f"  mAP@0.5:0.95 : {metrics.get('map50_95', 0):.4f}")
            self.logger.info(f"  Precision    : {metrics.get('precision', 0):.4f}")
            self.logger.info(f"  Recall       : {metrics.get('recall', 0):.4f}")
            
            return metrics
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Validation failed for {condition}: {str(e)}")
            if e.stderr:
                self.logger.error(f"Error output: {e.stderr}")
                
            return {
                'condition': condition,
                'success': False,
                'error': str(e)
            }
            
    def parse_validation_output(self, output: str, condition: str) -> Dict:
        """
        Parse validation metrics from YOLOv5 output
        
        Args:
            output: Standard output from validation
            condition: Weather condition name
            
        Returns:
            Dictionary of parsed metrics
        """
        metrics = {
            'map50': 0.0,
            'map50_95': 0.0,
            'precision': 0.0,
            'recall': 0.0
        }
        
        try:
            lines = output.split('\n')
            for line in lines:
                if 'all' in line and len(line.split()) > 10:
                    # Parse the "all" summary line
                    parts = line.split()
                    if len(parts) >= 10:
                        # Find indices of metrics (handling various formats)
                        for i, part in enumerate(parts):
                            try:
                                if i >= 2:  # Skip first two columns (Class, Images)
                                    val = float(part)
                                    if i == 4:  # Precision
                                        metrics['precision'] = val
                                    elif i == 5:  # Recall
                                        metrics['recall'] = val
                                    elif i == 6:  # mAP@0.5
                                        metrics['map50'] = val
                                    elif i == 7:  # mAP@0.5:0.95
                                        metrics['map50_95'] = val
                            except ValueError:
                                continue
                                
            # Also try to parse from COCO eval if available
            if 'COCO mAP' in output:
                for line in lines:
                    if 'Average Precision' in line and '@[ IoU=0.50:0.95' in line:
                        try:
                            metrics['map50_95'] = float(line.split('=')[-1].strip())
                        except:
                            pass
                    elif 'Average Precision' in line and '@[ IoU=0.50' in line:
                        try:
                            metrics['map50'] = float(line.split('=')[-1].strip())
                        except:
                            pass
                            
        except Exception as e:
            self.logger.warning(f"Could not parse metrics for {condition}: {e}")
            
        return metrics
        
    def calculate_composite_score(self, results: Dict[str, Dict]) -> float:
        """
        Calculate weighted composite score across conditions
        
        Args:
            results: Dictionary of results per condition
            
        Returns:
            Composite mAP score
        """
        composite_score = 0.0
        total_weight = 0.0
        
        for condition, metrics in results.items():
            if metrics.get('success', False) and condition in self.conditions:
                weight = self.conditions[condition]['weight']
                map50 = metrics.get('map50', 0)
                composite_score += weight * map50
                total_weight += weight
                
        if total_weight > 0:
            composite_score /= total_weight
            
        return composite_score
        
    def calculate_degradation(self, baseline_metrics: Dict, condition_metrics: Dict) -> Dict:
        """
        Calculate performance degradation from baseline
        
        Args:
            baseline_metrics: Metrics from clean condition
            condition_metrics: Metrics from weather condition
            
        Returns:
            Dictionary of degradation metrics
        """
        degradation = {}
        
        for metric in ['map50', 'map50_95', 'precision', 'recall']:
            baseline_val = baseline_metrics.get(metric, 0)
            condition_val = condition_metrics.get(metric, 0)
            
            if baseline_val > 0:
                abs_drop = baseline_val - condition_val
                rel_drop = (abs_drop / baseline_val) * 100
                
                degradation[metric] = {
                    'baseline': baseline_val,
                    'condition': condition_val,
                    'absolute_drop': abs_drop,
                    'relative_drop_percent': rel_drop
                }
            else:
                degradation[metric] = {
                    'baseline': baseline_val,
                    'condition': condition_val,
                    'absolute_drop': 0,
                    'relative_drop_percent': 0
                }
                
        return degradation
        
    def run_full_validation(self, conditions_to_test: Optional[List[str]] = None) -> Dict:
        """
        Run validation on all specified conditions
        
        Args:
            conditions_to_test: List of conditions to test (None = all)
            
        Returns:
            Dictionary containing all results
        """
        if conditions_to_test is None:
            conditions_to_test = list(self.conditions.keys())
            
        self.logger.info(f"[VALIDATION] Testing {len(conditions_to_test)} conditions...")
        
        all_results = {}
        baseline_metrics = None
        
        # Always test clean first to get baseline
        if 'clean' in conditions_to_test:
            conditions_to_test.remove('clean')
            conditions_to_test.insert(0, 'clean')
            
        # Run validation for each condition
        for condition in conditions_to_test:
            results = self.validate_condition(condition)
            all_results[condition] = results
            
            if condition == 'clean' and results.get('success', False):
                baseline_metrics = results
                
        # Calculate degradation for weather conditions
        if baseline_metrics:
            for condition, results in all_results.items():
                if condition != 'clean' and results.get('success', False):
                    degradation = self.calculate_degradation(baseline_metrics, results)
                    all_results[condition]['degradation'] = degradation
                    
        # Calculate composite score
        composite_score = self.calculate_composite_score(all_results)
        
        # Save comprehensive results
        self.save_results(all_results, composite_score)
        
        return all_results
        
    def save_results(self, results: Dict, composite_score: float):
        """
        Save validation results to files
        
        Args:
            results: All validation results
            composite_score: Weighted composite score
        """
        # Create summary
        summary = {
            'experiment': 'phase1_baseline_experiment2_validation',
            'timestamp': datetime.now().isoformat(),
            'weights': str(self.weights_path),
            'composite_score': composite_score,
            'conditions_tested': list(results.keys()),
            'results': results
        }
        
        # Save JSON summary
        summary_file = self.run_dir / "validation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        self.logger.info(f"[SAVE] Results saved to: {summary_file}")
        
        # Create comparison table
        self.create_comparison_table(results, composite_score)
        
    def create_comparison_table(self, results: Dict, composite_score: float):
        """
        Create a comparison table of results across conditions
        
        Args:
            results: All validation results
            composite_score: Weighted composite score
        """
        # Prepare data for table
        table_data = []
        
        for condition, metrics in results.items():
            if metrics.get('success', False):
                row = {
                    'Condition': condition.upper(),
                    'mAP@0.5': f"{metrics.get('map50', 0):.4f}",
                    'mAP@0.5:0.95': f"{metrics.get('map50_95', 0):.4f}",
                    'Precision': f"{metrics.get('precision', 0):.4f}",
                    'Recall': f"{metrics.get('recall', 0):.4f}"
                }
                
                if 'degradation' in metrics:
                    deg = metrics['degradation']['map50']['relative_drop_percent']
                    row['Degradation'] = f"{deg:.1f}%"
                else:
                    row['Degradation'] = '-'
                    
                table_data.append(row)
                
        # Create DataFrame and save
        if table_data:
            df = pd.DataFrame(table_data)
            csv_file = self.run_dir / "comparison_table.csv"
            df.to_csv(csv_file, index=False)
            
            # Log table
            self.logger.info("\n[RESULTS] Performance Comparison Table:")
            self.logger.info("-" * 80)
            for _, row in df.iterrows():
                self.logger.info(f"{row['Condition']:10} | mAP: {row['mAP@0.5']:7} | "
                               f"Precision: {row['Precision']:7} | "
                               f"Recall: {row['Recall']:7} | "
                               f"Degradation: {row['Degradation']:7}")
            self.logger.info("-" * 80)
            self.logger.info(f"Composite Score: {composite_score:.4f}")
            
    def validate_phase_results(self, phase: str):
        """
        Validate results from a specific training phase
        
        Args:
            phase: Phase identifier (2a, 2b, 2c)
        """
        # Find weights from phase
        phase_name = f"exp2_phase{phase}"
        weights_path = LOGS_DIR / "training" / phase_name / "weights" / "best.pt"
        
        if not weights_path.exists():
            weights_path = LOGS_DIR / "training" / phase_name / "weights" / "last.pt"
            
        if not weights_path.exists():
            self.logger.error(f"No weights found for phase {phase}")
            return
            
        self.weights_path = weights_path
        self.logger.info(f"Validating Phase {phase.upper()} with weights: {weights_path}")
        
        # Run full validation
        self.run_full_validation()


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='YOLOv5n VisDrone Experiment 2 Multi-Weather Validation')
    parser.add_argument('--weights', type=str, required=False,
                       help='Path to model weights')
    parser.add_argument('--phase', type=str, choices=['2a', '2b', '2c'],
                       help='Validate specific training phase results')
    parser.add_argument('--conf-thres', type=float, default=0.001,
                       help='Confidence threshold (default: 0.001)')
    parser.add_argument('--iou-thres', type=float, default=0.6,
                       help='IoU threshold for NMS (default: 0.6)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for validation (default: 16)')
    parser.add_argument('--weather-only', action='store_true',
                       help='Only test weather conditions (skip clean)')
    parser.add_argument('--conditions', nargs='+',
                       choices=['clean', 'fog', 'rain', 'night', 'mixed'],
                       help='Specific conditions to test')
    
    args = parser.parse_args()
    
    if args.phase:
        # Validate phase results
        validator = MultiWeatherValidator(
            weights_path="",  # Will be set by validate_phase_results
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            batch_size=args.batch_size
        )
        validator.validate_phase_results(args.phase)
    elif args.weights:
        # Validate with specific weights
        validator = MultiWeatherValidator(
            weights_path=args.weights,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            batch_size=args.batch_size
        )
        
        # Determine conditions to test
        if args.conditions:
            conditions = args.conditions
        elif args.weather_only:
            conditions = ['fog', 'rain', 'night', 'mixed']
        else:
            conditions = None  # Test all
            
        validator.run_full_validation(conditions)
    else:
        parser.error("Either --weights or --phase must be specified")


if __name__ == "__main__":
    main()