#!/usr/bin/env python3
"""
YOLOv5n VisDrone Phase 1 Baseline Weather Testing Script - Experiment 2
========================================================================

This script tests trained Phase 1 Experiment 2 models against all synthetic weather
conditions to evaluate robustness and performance degradation. Tests are performed
sequentially on: Clean, Fog, Rain, Night, and Mixed synthetic datasets.

Optimized for Experiment 2's progressive training results.

Usage:
    python test_weather_conditions.py --weights path/to/best.pt
    python test_weather_conditions.py --phase 2c  # Test final phase results
    python test_weather_conditions.py --weights path/to/best.pt --save-images
"""

import argparse
import os
import sys
import json
import yaml
import time
import logging
import subprocess
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

# Add YOLOv5 to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.parent.parent.parent
YOLO_PATH = PROJECT_ROOT / "models" / "yolov5n" / "baseline" / "yolov5"
sys.path.append(str(YOLO_PATH))

try:
    import torch
    from utils.general import check_requirements, colorstr
    from utils.torch_utils import select_device
    from utils.metrics import ap_per_class
except ImportError as e:
    print(f"Error importing YOLOv5 modules: {e}")
    print(f"Make sure YOLOv5 is properly installed at: {YOLO_PATH}")
    sys.exit(1)


class WeatherTester:
    """Weather Conditions Tester for Experiment 2 Models"""
    
    def __init__(self, weights_path: str = None, phase: str = None, 
                 conf_thres: float = 0.001, iou_thres: float = 0.6,
                 save_images: bool = False):
        """
        Initialize weather tester
        
        Args:
            weights_path: Path to model weights
            phase: Training phase to test (2a, 2b, 2c)
            conf_thres: Confidence threshold
            iou_thres: IoU threshold for NMS
            save_images: Whether to save detection images
        """
        # Handle phase-based weights
        if phase:
            self.weights_path = self.get_phase_weights(phase)
        else:
            self.weights_path = Path(weights_path).resolve()
            
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.save_images = save_images
        
        self.setup_paths()
        self.setup_logging()
        self.load_config()
        self.define_weather_conditions()
        
    def get_phase_weights(self, phase: str) -> Path:
        """Get weights path for specific training phase"""
        # Try different naming patterns (quicktest vs full)
        possible_names = [
            f"exp2_phase{phase}_quicktest",  # Quick test mode
            f"exp2_phase{phase}",           # Full mode
        ]
        
        for phase_name in possible_names:
            weights_path = SCRIPT_DIR / "logs&results" / "training" / phase_name / "weights" / "best.pt"
            if weights_path.exists():
                return weights_path
                
            # Try last.pt as fallback
            weights_path = SCRIPT_DIR / "logs&results" / "training" / phase_name / "weights" / "last.pt"
            if weights_path.exists():
                return weights_path
        
        # If nothing found, provide helpful error message
        training_dir = SCRIPT_DIR / "logs&results" / "training"
        available_dirs = [d.name for d in training_dir.iterdir() if d.is_dir()]
        raise FileNotFoundError(
            f"No weights found for phase {phase}. "
            f"Tried: {possible_names}. "
            f"Available directories: {available_dirs}"
        )
        
    def setup_paths(self):
        """Setup all required paths"""
        self.script_dir = SCRIPT_DIR
        self.project_root = PROJECT_ROOT
        self.yolo_path = YOLO_PATH
        self.config_dir = PROJECT_ROOT / "configs" / "yolov5n-visdrone"
        self.logs_dir = self.script_dir / "logs&results" / "weather_testing"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Synthetic test data paths
        self.synthetic_data_root = self.project_root / "data" / "synthetic_test"
        self.clean_data_root = self.project_root / "data" / "raw" / "visdrone"
        
        print(f"[SETUP] Weather testing for Experiment 2")
        print(f"[SETUP] YOLOv5 path: {self.yolo_path}")
        print(f"[SETUP] Synthetic data root: {self.synthetic_data_root}")
        print(f"[SETUP] Logs directory: {self.logs_dir}")
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.logs_dir / f"weather_testing_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("="*70)
        self.logger.info("YOLOv5n VisDrone Phase 1 Weather Testing - Experiment 2")
        self.logger.info("="*70)
        self.logger.info(f"Weights: {self.weights_path}")
        self.logger.info(f"Confidence threshold: {self.conf_thres}")
        self.logger.info(f"IoU threshold: {self.iou_thres}")
        self.logger.info(f"Save images: {self.save_images}")
        self.logger.info(f"Log file: {log_file}")
        
    def load_config(self):
        """Load and validate data configuration"""
        self.logger.info("[CONFIG] Loading data configuration...")
        
        # Load base config to get class information
        base_config = self.config_dir / "visdrone_experiment2.yaml"
        
        if not base_config.exists():
            self.logger.error(f"Data config not found: {base_config}")
            raise FileNotFoundError(f"Data configuration not found: {base_config}")
            
        with open(base_config, 'r') as f:
            self.data_config = yaml.safe_load(f)
            
        self.class_names = self.data_config['names']
        self.num_classes = self.data_config['nc']
        
        self.logger.info(f"[CONFIG] Loaded {self.num_classes} classes")
        
    def define_weather_conditions(self):
        """Define weather test conditions"""
        self.weather_conditions = {
            'clean': {
                'name': 'Clean (Baseline)',
                'description': 'Original clean test dataset',
                'data_path': self.clean_data_root / "VisDrone2019-DET-test-dev",
                'yaml_path': 'test-dev'
            },
            'fog': {
                'name': 'Fog Synthetic',
                'description': 'Synthetic fog augmented test dataset',
                'data_path': self.synthetic_data_root / "VisDrone2019-DET-test-fog",
                'yaml_path': 'synthetic_fog'
            },
            'rain': {
                'name': 'Rain Synthetic', 
                'description': 'Synthetic rain augmented test dataset',
                'data_path': self.synthetic_data_root / "VisDrone2019-DET-test-rain",
                'yaml_path': 'synthetic_rain'
            },
            'night': {
                'name': 'Night Synthetic',
                'description': 'Synthetic night/low-light test dataset',
                'data_path': self.synthetic_data_root / "VisDrone2019-DET-test-night",
                'yaml_path': 'synthetic_night'
            },
            'mixed': {
                'name': 'Mixed Conditions',
                'description': 'Mixed weather conditions test dataset',
                'data_path': self.synthetic_data_root / "VisDrone2019-DET-test-mixed",
                'yaml_path': 'synthetic_mixed'
            }
        }
        
        # Validate paths exist
        self.valid_conditions = {}
        for condition, info in self.weather_conditions.items():
            if info['data_path'].exists():
                self.valid_conditions[condition] = info
                self.logger.info(f"[CONFIG] Found {condition} dataset: {info['data_path']}")
            else:
                self.logger.warning(f"[CONFIG] Missing {condition} dataset: {info['data_path']}")
                
    def create_temp_data_yaml(self, condition: str) -> Path:
        """Create temporary data YAML for specific condition"""
        condition_info = self.valid_conditions[condition]
        
        # For clean (baseline), use the raw VisDrone structure
        if condition == 'clean':
            # Clean data is in raw/visdrome/VisDrone2019-DET-test-dev
            test_images_path = condition_info['data_path'] / 'images'
            
            # Create temp config with absolute paths to avoid path resolution issues
            temp_config = {
                'path': str(condition_info['data_path'].parent),  # Points to visdrone folder
                'train': str(condition_info['data_path'].parent / 'VisDrone2019-DET-train' / 'images'),
                'val': str(condition_info['data_path'].parent / 'VisDrone2019-DET-val' / 'images'),
                'test': str(test_images_path),  # Absolute path to test images
                'nc': self.num_classes,
                'names': self.class_names
            }
        else:
            # For synthetic weather conditions
            test_images_path = condition_info['data_path'] / 'images'
            
            # Use absolute paths for synthetic datasets
            temp_config = {
                'path': str(condition_info['data_path'].parent),  # Points to synthetic_test folder
                'train': str(self.clean_data_root / 'VisDrone2019-DET-train' / 'images'),
                'val': str(self.clean_data_root / 'VisDrone2019-DET-val' / 'images'),
                'test': str(test_images_path),  # Absolute path to synthetic test images
                'nc': self.num_classes,
                'names': self.class_names
            }
        
        # Save temp YAML (always overwrite to ensure fresh config)
        temp_yaml_path = self.logs_dir / f"temp_data_{condition}.yaml"
        
        # Force overwrite even if file exists
        if temp_yaml_path.exists():
            self.logger.info(f"Overwriting existing temp YAML: {temp_yaml_path}")
            
        with open(temp_yaml_path, 'w') as f:
            yaml.dump(temp_config, f, default_flow_style=False)
            
        # Log the created config for debugging
        self.logger.info(f"Created temp YAML for {condition}:")
        self.logger.info(f"  Path: {temp_config['path']}")
        self.logger.info(f"  Train: {temp_config['train']}")
        self.logger.info(f"  Val: {temp_config['val']}")
        self.logger.info(f"  Test: {temp_config['test']}")
        
        # Verify paths exist
        if not Path(temp_config['test']).exists():
            self.logger.warning(f"Test path does not exist: {temp_config['test']}")
        if not Path(temp_config['val']).exists():
            self.logger.warning(f"Val path does not exist: {temp_config['val']}")
        
        return temp_yaml_path
        
    def test_weather_condition(self, condition: str, condition_info: Dict) -> Dict:
        """Test model on specific weather condition"""
        self.logger.info(f"[TESTING] Starting {condition} test...")
        
        # Create temporary data config
        temp_data_yaml = self.create_temp_data_yaml(condition)
        
        # Setup results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = self.logs_dir / f"{condition}_results_{timestamp}"
        
        # Build test command
        cmd = [
            sys.executable,
            str(self.yolo_path / "val.py"),
            "--weights", str(self.weights_path),
            "--data", str(temp_data_yaml),
            "--batch-size", "8",  # Increased for faster testing
            "--imgsz", "640",
            "--conf-thres", str(self.conf_thres),
            "--iou-thres", str(self.iou_thres),
            "--task", "test",
            "--device", "0",
            "--project", str(self.logs_dir),
            "--name", f"{condition}_results_{timestamp}",
            "--exist-ok",
            "--save-json",
            "--save-txt",
            "--save-conf",
            "--verbose"
        ]
        
        if self.save_images:
            cmd.extend(["--save-hybrid", "--plots"])
            
        # Run test
        start_time = time.time()
        
        try:
            self.logger.info(f"[WEATHER-{condition.upper()}] Running YOLOv5 validation...")
            self.logger.info(f"[WEATHER-{condition.upper()}] This may take a few minutes...")
            
            # Run with real-time output
            process = subprocess.Popen(
                cmd,
                cwd=str(self.yolo_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Stream output in real-time
            output_lines = []
            for line in iter(process.stdout.readline, ''):
                if line:
                    print(line.rstrip())  # Show in terminal
                    output_lines.append(line.rstrip())
                    
            process.wait()
            
            # Check return code
            if process.returncode != 0:
                raise subprocess.CalledProcessError(
                    process.returncode, cmd, 
                    output='\n'.join(output_lines)
                )
            
            duration = time.time() - start_time
            
            # Parse results
            output_text = '\n'.join(output_lines)
            metrics = self.parse_test_output(output_text, condition)
            
            # Log results
            self.logger.info(f"[TESTING] {condition} test completed in {duration:.2f} seconds")
            
            # Create result dictionary
            test_result = {
                'condition': condition,
                'condition_info': {
                    'name': condition_info['name'],
                    'description': condition_info['description'],
                    'data_path': str(condition_info['data_path']),
                    'image_count': self.count_images(condition_info['data_path'])
                },
                'metrics': metrics,
                'duration_seconds': duration,
                'timestamp': datetime.now().isoformat(),
                'results_directory': str(results_dir)
            }
            
            return test_result
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"[ERROR] Test failed for {condition}: {str(e)}")
            if hasattr(e, 'output') and e.output:
                self.logger.error(f"[ERROR] Output: {e.output[:500]}...")  # Show first 500 chars
            return {
                'condition': condition,
                'error': str(e),
                'success': False
            }
            
    def parse_test_output(self, output: str, condition: str) -> Dict:
        """Parse test metrics from YOLOv5 output"""
        metrics = {
            'map50': 0.0,
            'map50_95': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'condition': condition
        }
        
        try:
            lines = output.split('\n')
            for line in lines:
                # Log the actual output line for debugging
                if condition.upper() in line.upper() or 'all' in line:
                    self.logger.info(f"[WEATHER-{condition.upper()}] {line}")
                    
                # Parse metrics from "all" summary line
                if 'all' in line and len(line.split()) >= 6:
                    parts = line.split()
                    self.logger.debug(f"Parsing line parts: {parts}")
                    try:
                        # YOLOv5 output format: 'all' Images Instances P R mAP@.5 mAP@.5:.95
                        # From log: all 1610 75102 0.351 0.191 0.152 0.0715
                        if parts[0].strip() == 'all':
                            metrics['precision'] = float(parts[3]) if len(parts) > 3 else 0.0
                            metrics['recall'] = float(parts[4]) if len(parts) > 4 else 0.0
                            metrics['map50'] = float(parts[5]) if len(parts) > 5 else 0.0
                            metrics['map50_95'] = float(parts[6]) if len(parts) > 6 else 0.0
                            self.logger.info(f"Successfully parsed metrics from: {line.strip()}")
                    except (ValueError, IndexError) as e:
                        self.logger.warning(f"Could not parse metrics line: {line}")
                        self.logger.warning(f"Parts: {parts}")
                        
        except Exception as e:
            self.logger.error(f"[PARSING] Error parsing output for {condition}: {e}")
            
        # Log parsed metrics
        self.logger.info(f"[METRICS] {condition.upper()} Results:")
        self.logger.info(f"  mAP@0.5      : {metrics['map50']:.4f} ({metrics['map50']*100:.2f}%)")
        self.logger.info(f"  mAP@0.5:0.95 : {metrics['map50_95']:.4f} ({metrics['map50_95']*100:.2f}%)")
        self.logger.info(f"  Precision    : {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        self.logger.info(f"  Recall       : {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        
        return metrics
        
    def count_images(self, data_path: Path) -> int:
        """Count images in dataset"""
        image_dir = data_path / "images"
        if image_dir.exists():
            return len(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
        return 0
        
    def calculate_degradation(self, baseline_metrics: Dict, condition_metrics: Dict) -> Dict:
        """Calculate performance degradation from baseline"""
        degradation = {}
        
        for metric in ['map50', 'map50_95', 'precision', 'recall']:
            baseline_val = baseline_metrics.get(metric, 0)
            condition_val = condition_metrics.get(metric, 0)
            
            if baseline_val > 0:
                absolute_drop = baseline_val - condition_val
                relative_drop = (absolute_drop / baseline_val) * 100
            else:
                absolute_drop = 0
                relative_drop = 0
                
            degradation[metric] = {
                'absolute_drop': absolute_drop,
                'relative_drop_percent': relative_drop,
                'baseline_value': baseline_val,
                'condition_value': condition_val
            }
            
        return degradation
        
    def run_weather_testing(self) -> Dict:
        """Run comprehensive weather testing"""
        self.logger.info("[TESTING] Starting comprehensive weather testing...")
        
        try:
            # Validate environment
            self.validate_environment()
            
            # Store all results
            all_results = {}
            baseline_metrics = None
            
            # Test each weather condition
            for condition in ['clean', 'fog', 'rain', 'night', 'mixed']:
                if condition in self.valid_conditions:
                    condition_info = self.valid_conditions[condition]
                    
                    self.logger.info(f"\n[TESTING] {'='*50}")
                    self.logger.info(f"[TESTING] Testing: {condition_info['name']}")
                    self.logger.info(f"[TESTING] {'='*50}")
                    
                    try:
                        results = self.test_weather_condition(condition, condition_info)
                        all_results[condition] = results
                        
                        # Store baseline metrics for degradation calculation
                        if condition == 'clean':
                            baseline_metrics = results['metrics']
                            
                    except Exception as e:
                        self.logger.error(f"[TESTING] Failed to test {condition}: {str(e)}")
                        all_results[condition] = {
                            'condition': condition,
                            'error': str(e),
                            'success': False
                        }
                else:
                    self.logger.warning(f"[TESTING] Skipping {condition} - dataset not available")
                    
            # Calculate degradation analysis
            if baseline_metrics:
                self.logger.info(f"\n[ANALYSIS] Calculating performance degradation...")
                self.calculate_comprehensive_analysis(all_results, baseline_metrics)
                
            # Save comprehensive results
            self.save_comprehensive_results(all_results, baseline_metrics)
            
            return all_results
            
        except Exception as e:
            self.logger.error(f"[ERROR] Weather testing failed: {str(e)}")
            raise
            
    def validate_environment(self):
        """Validate testing environment"""
        # Check weights exist
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {self.weights_path}")
            
        # Check YOLOv5 val.py exists
        val_script = self.yolo_path / "val.py"
        if not val_script.exists():
            raise FileNotFoundError(f"YOLOv5 val.py not found: {val_script}")
            
        # Check GPU availability
        if torch.cuda.is_available():
            self.logger.info(f"[SETUP] GPU available: {torch.cuda.get_device_name(0)}")
        else:
            self.logger.warning("[SETUP] No GPU available, using CPU (slower)")
            
    def calculate_comprehensive_analysis(self, all_results: Dict, baseline_metrics: Dict):
        """Calculate and log comprehensive degradation analysis"""
        self.logger.info("[ANALYSIS] Performance Degradation Analysis:")
        self.logger.info("-" * 80)
        
        # Create comparison table
        conditions_order = ['clean', 'fog', 'rain', 'night', 'mixed']
        
        for condition in conditions_order:
            if condition in all_results and 'metrics' in all_results[condition]:
                metrics = all_results[condition]['metrics']
                
                if condition == 'clean':
                    self.logger.info(f"{condition.upper():10} (BASELINE):")
                    self.logger.info(f"  mAP@0.5: {metrics['map50']:.4f} ({metrics['map50']*100:.2f}%)")
                else:
                    # Calculate degradation
                    degradation = self.calculate_degradation(baseline_metrics, metrics)
                    map50_deg = degradation['map50']
                    
                    self.logger.info(f"{condition.upper():10}:")
                    self.logger.info(f"  mAP@0.5: {metrics['map50']:.4f} ({metrics['map50']*100:.2f}%) "
                                   f"[{map50_deg['relative_drop_percent']:+.1f}%]")
                    
                    # Store degradation in results
                    all_results[condition]['degradation'] = degradation
                    
        self.logger.info("-" * 80)
        
    def save_comprehensive_results(self, all_results: Dict, baseline_metrics: Dict):
        """Save comprehensive weather testing results"""
        self.logger.info("[SAVE] Saving comprehensive weather testing results...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create comprehensive summary
        summary = {
            "experiment": "phase1_baseline_experiment2_weather_testing",
            "timestamp": datetime.now().isoformat(),
            "model_info": {
                "weights": str(self.weights_path),
                "model_type": "yolov5n",
                "dataset": "visdrone_synthetic_weather"
            },
            "testing_config": {
                "conf_threshold": self.conf_thres,
                "iou_threshold": self.iou_thres,
                "conditions_tested": list(all_results.keys())
            },
            "baseline_metrics": baseline_metrics,
            "weather_results": all_results
        }
        
        # Calculate summary statistics
        if baseline_metrics:
            degradations = []
            for condition, results in all_results.items():
                if condition != 'clean' and 'degradation' in results:
                    degradations.append(results['degradation']['map50']['relative_drop_percent'])
                    
            if degradations:
                summary["summary_statistics"] = {
                    "map50": {
                        "mean_degradation_percent": np.mean(degradations),
                        "max_degradation_percent": np.max(degradations),
                        "min_degradation_percent": np.min(degradations)
                    }
                }
                
        # Save complete results
        complete_file = self.logs_dir / f"weather_testing_complete_{timestamp}.json"
        with open(complete_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        self.logger.info(f"[SAVE] Complete results saved to: {complete_file}")
        
        # Save comparison table
        self.save_comparison_table(all_results, baseline_metrics, timestamp)
        
    def save_comparison_table(self, all_results: Dict, baseline_metrics: Dict, timestamp: str):
        """Save weather comparison table"""
        comparison = {
            "baseline": baseline_metrics,
            "conditions": {}
        }
        
        for condition, results in all_results.items():
            if 'metrics' in results:
                comparison["conditions"][condition] = {
                    "metrics": results['metrics']
                }
                if 'degradation' in results:
                    comparison["conditions"][condition]["degradation"] = results['degradation']
                    
        comparison_file = self.logs_dir / f"weather_comparison_{timestamp}.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
            
        self.logger.info(f"[SAVE] Comparison table saved to: {comparison_file}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='YOLOv5n VisDrone Experiment 2 Weather Testing')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--weights', type=str,
                      help='Path to trained model weights')
    group.add_argument('--phase', type=str, choices=['2a', '2b', '2c'],
                      help='Test specific training phase results')
    
    parser.add_argument('--conf-thres', type=float, default=0.001,
                       help='Confidence threshold for detection (default: 0.001)')
    parser.add_argument('--iou-thres', type=float, default=0.6,
                       help='IoU threshold for NMS (default: 0.6)')
    parser.add_argument('--save-images', action='store_true',
                       help='Save detection result images')
    
    args = parser.parse_args()
    
    try:
        tester = WeatherTester(
            weights_path=args.weights,
            phase=args.phase,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            save_images=args.save_images
        )
        
        results = tester.run_weather_testing()
        
        print("\n" + "="*70)
        print("Weather testing completed successfully!")
        print(f"Tested {len(results)} weather conditions")
        
        # Print summary
        if 'clean' in results and 'metrics' in results['clean']:
            baseline = results['clean']['metrics']
            print(f"\nBaseline (Clean): mAP@0.5 = {baseline['map50']:.4f}")
            
            for condition in ['fog', 'rain', 'night', 'mixed']:
                if condition in results and 'metrics' in results[condition]:
                    metrics = results[condition]['metrics']
                    if 'degradation' in results[condition]:
                        deg = results[condition]['degradation']['map50']['relative_drop_percent']
                        print(f"{condition.capitalize():8}: mAP@0.5 = {metrics['map50']:.4f} ({deg:+.1f}%)")
                    else:
                        print(f"{condition.capitalize():8}: mAP@0.5 = {metrics['map50']:.4f}")
                        
        print("="*70)
        
    except Exception as e:
        print(f"\nWeather testing failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()