#!/usr/bin/env python3
"""
YOLOv5n VisDrone Phase 1 Baseline Weather Testing Script - Experiment 1
========================================================================

This script tests trained Phase 1 baseline models against all synthetic weather
conditions to evaluate robustness and performance degradation. Tests are performed
sequentially on: Clean, Fog, Rain, Night, and Mixed synthetic datasets.

Usage:
    python test_weather_conditions.py --weights path/to/best.pt
    python test_weather_conditions.py --weights path/to/best.pt --conf-thres 0.25
"""

import argparse
import os
import sys
import json
import yaml
import time
import logging
from datetime import datetime
from pathlib import Path
import numpy as np

# Add YOLOv5 to path
YOLO_PATH = Path(__file__).parent.parent.parent.parent.parent.parent / "models" / "yolov5n" / "baseline" / "yolov5"
sys.path.append(str(YOLO_PATH))

try:
    import torch
    import val as yolo_val
    from utils.general import check_requirements, colorstr
    from utils.torch_utils import select_device
    from utils.metrics import ap_per_class
except ImportError as e:
    print(f"Error importing YOLOv5 modules: {e}")
    print(f"Make sure YOLOv5 is properly installed at: {YOLO_PATH}")
    sys.exit(1)


class WeatherTester:
    """Weather Conditions Tester for Phase 1 Baseline Models"""
    
    def __init__(self, weights_path, conf_thres=0.001, iou_thres=0.6):
        self.weights_path = Path(weights_path)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.setup_paths()
        self.setup_logging()
        self.load_config()
        self.define_weather_conditions()
        
    def setup_paths(self):
        """Setup all required paths"""
        self.script_dir = Path(__file__).parent
        self.project_root = self.script_dir.parent.parent.parent.parent.parent.parent
        self.yolo_path = self.project_root / "models" / "yolov5n" / "baseline" / "yolov5"
        self.config_dir = self.project_root / "configs" / "yolov5n-visdrone"
        self.logs_dir = self.script_dir / "logs&results" / "weather_testing"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Synthetic test data paths
        self.synthetic_data_root = self.project_root / "data" / "synthetic_test"
        
        print(f"[SETUP] Weather testing script directory: {self.script_dir}")
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
        self.logger.info("YOLOv5n VisDrone Phase 1 Weather Conditions Testing - Experiment 1")
        self.logger.info("="*70)
        self.logger.info(f"Weights: {self.weights_path}")
        self.logger.info(f"Confidence threshold: {self.conf_thres}")
        self.logger.info(f"IoU threshold: {self.iou_thres}")
        self.logger.info(f"Log file: {log_file}")
        
    def load_config(self):
        """Load and validate data configuration"""
        self.logger.info("[CONFIG] Loading data configuration...")
        
        # Load base config to get class information
        base_config = self.config_dir / "visdrone_experiment1.yaml"
        if not base_config.exists():
            self.logger.error(f"Base config not found: {base_config}")
            raise FileNotFoundError(f"Base config not found: {base_config}")
        
        with open(base_config, 'r') as f:
            self.base_data_cfg = yaml.safe_load(f)
            
        self.logger.info(f"[CONFIG] Base config loaded: {base_config}")
        self.logger.info(f"[CONFIG] Number of classes: {self.base_data_cfg['nc']}")
        self.logger.info(f"[CONFIG] Class names: {self.base_data_cfg['names']}")
        
    def define_weather_conditions(self):
        """Define weather conditions and their data paths"""
        # Using absolute paths for all weather conditions
        base_path = Path("C:/Users/burak/OneDrive/Desktop/Git-Repos/drone-obj-detect-lightweight-ai")
        
        self.weather_conditions = {
            'clean': {
                'name': 'Clean (Baseline)',
                'description': 'Original clean test dataset',
                'data_path': base_path / "data" / "raw" / "visdrone" / "VisDrone2019-DET-test-dev"
            },
            'fog': {
                'name': 'Fog Synthetic',
                'description': 'Synthetic fog augmented test dataset', 
                'data_path': base_path / "data" / "synthetic_test" / "VisDrone2019-DET-test-fog"
            },
            'rain': {
                'name': 'Rain Synthetic',
                'description': 'Synthetic rain augmented test dataset',
                'data_path': base_path / "data" / "synthetic_test" / "VisDrone2019-DET-test-rain"
            },
            'night': {
                'name': 'Night Synthetic',
                'description': 'Synthetic night/low-light test dataset',
                'data_path': base_path / "data" / "synthetic_test" / "VisDrone2019-DET-test-night"
            },
            'mixed': {
                'name': 'Mixed Conditions',
                'description': 'Mixed weather conditions test dataset',
                'data_path': base_path / "data" / "synthetic_test" / "VisDrone2019-DET-test-mixed"
            }
        }
        
        self.logger.info("[WEATHER] Defined weather conditions:")
        for condition, info in self.weather_conditions.items():
            self.logger.info(f"  {condition}: {info['name']} - {info['data_path']}")
            
    def validate_environment(self):
        """Validate testing environment"""
        self.logger.info("[VALIDATION] Validating environment...")
        
        # Check weights file
        if not self.weights_path.exists():
            self.logger.error(f"Weights file not found: {self.weights_path}")
            raise FileNotFoundError(f"Weights file not found: {self.weights_path}")
        
        self.logger.info(f"[VALIDATION] Weights file: {self.weights_path}")
        
        # Check GPU availability
        device = select_device('')
        self.logger.info(f"[VALIDATION] Device: {device}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            self.logger.info(f"[VALIDATION] GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            self.logger.info("[VALIDATION] Using CPU for testing")
        
        # Validate weather condition data paths
        self.logger.info("[VALIDATION] Checking weather condition datasets...")
        valid_conditions = {}
        
        for condition, info in self.weather_conditions.items():
            data_path = info['data_path']
            if data_path.exists():
                # Check for images subdirectory
                images_path = data_path / "images"
                if images_path.exists():
                    image_count = len(list(images_path.glob("*.jpg")))
                    valid_conditions[condition] = {**info, 'image_count': image_count}
                    self.logger.info(f"  [OK] {condition}: {image_count} images found at {data_path}")
                else:
                    self.logger.warning(f"  [WARNING] {condition}: images directory not found at {images_path}")
            else:
                self.logger.warning(f"  [ERROR] {condition}: dataset not found at {data_path}")
        
        if not valid_conditions:
            self.logger.error("[VALIDATION] No valid weather condition datasets found!")
            raise FileNotFoundError("No valid weather condition datasets found!")
        
        self.valid_conditions = valid_conditions
        self.logger.info(f"[VALIDATION] Found {len(valid_conditions)} valid weather conditions")
        self.logger.info("[VALIDATION] Environment validation completed")
        
    def create_temp_config(self, condition, data_path):
        """Create temporary data configuration for specific weather condition"""
        temp_config = {
            'path': str(data_path),
            'train': str(data_path / "images"),  # Using same for all splits in testing
            'val': str(data_path / "images"),
            'test': str(data_path / "images"),
            'nc': self.base_data_cfg['nc'],
            'names': self.base_data_cfg['names']
        }
        
        # Create temporary config file
        temp_config_path = self.logs_dir / f"temp_config_{condition}.yaml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(temp_config, f, default_flow_style=False)
            
        return temp_config_path
        
    def test_weather_condition(self, condition, condition_info):
        """Test model on specific weather condition"""
        self.logger.info(f"[TESTING] Testing condition: {condition_info['name']}")
        self.logger.info(f"[TESTING] Description: {condition_info['description']}")
        self.logger.info(f"[TESTING] Dataset: {condition_info['data_path']}")
        
        try:
            # Create temporary config for this condition
            temp_config = self.create_temp_config(condition, condition_info['data_path'])
            
            # Create results directory for this condition
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = self.logs_dir / f"{condition}_results_{timestamp}"
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare validation arguments
            args = argparse.Namespace(
                # Core parameters
                data=str(temp_config),
                weights=str(self.weights_path),
                batch_size=32,
                imgsz=640,
                
                # Threshold parameters
                conf_thres=self.conf_thres,
                iou_thres=self.iou_thres,
                
                # Output parameters
                task='test',
                device='',
                workers=8,
                single_cls=False,
                augment=False,
                verbose=True,
                save_txt=True,
                save_hybrid=False,
                save_conf=True,
                save_json=True,
                
                # Results directory
                project=str(results_dir.parent),
                name=results_dir.name,
                exist_ok=True,
                
                # Additional parameters
                half=False,
                dnn=False,
                plots=True,
                wandb_logger=None,
                compute_loss=None,
                callbacks=None,
            )
            
            # Record test start
            start_time = time.time()
            self.logger.info(f"[TESTING] Starting {condition} test at {datetime.now()}")
            
            # Change to YOLOv5 directory
            original_cwd = os.getcwd()
            os.chdir(self.yolo_path)
            
            try:
                # Run validation/testing
                results = yolo_val.run(**vars(args))
                
                # Test completed successfully
                end_time = time.time()
                duration = end_time - start_time
                self.logger.info(f"[TESTING] {condition} test completed in {duration:.2f} seconds")
                
                # Extract metrics
                metrics = self.extract_metrics(results, condition)
                self.log_condition_metrics(condition, metrics)
                
                # Save condition results
                condition_results = {
                    'condition': condition,
                    'condition_info': condition_info,
                    'metrics': metrics,
                    'duration_seconds': duration,
                    'timestamp': datetime.now().isoformat(),
                    'results_directory': str(results_dir)
                }
                
                # Clean up temporary config
                temp_config.unlink()
                
                return condition_results
                
            finally:
                # Restore original working directory
                os.chdir(original_cwd)
                
        except Exception as e:
            self.logger.error(f"[TESTING] Failed testing {condition}: {str(e)}")
            raise
            
    def extract_metrics(self, results, condition):
        """Extract metrics from validation results"""
        self.logger.info(f"[METRICS] Extracting metrics for {condition}...")
        
        # Extract main metrics (same format as validation script)
        metrics = {
            'map50': float(results[0][0]) if len(results) > 0 and len(results[0]) > 0 else 0.0,
            'map50_95': float(results[0][1]) if len(results) > 0 and len(results[0]) > 1 else 0.0,
            'precision': float(results[0][2]) if len(results) > 0 and len(results[0]) > 2 else 0.0,
            'recall': float(results[0][3]) if len(results) > 0 and len(results[0]) > 3 else 0.0,
        }
        
        # Handle alternative results format
        if isinstance(results, tuple) and len(results) >= 4:
            metrics.update({
                'map50': float(results[0]) if results[0] is not None else 0.0,
                'map50_95': float(results[1]) if results[1] is not None else 0.0,
                'precision': float(results[2]) if results[2] is not None else 0.0,
                'recall': float(results[3]) if results[3] is not None else 0.0,
            })
        
        return metrics
        
    def log_condition_metrics(self, condition, metrics):
        """Log metrics for specific condition"""
        self.logger.info(f"[METRICS] {condition.upper()} Results:")
        self.logger.info(f"  mAP@0.5      : {metrics['map50']:.4f} ({metrics['map50']*100:.2f}%)")
        self.logger.info(f"  mAP@0.5:0.95 : {metrics['map50_95']:.4f} ({metrics['map50_95']*100:.2f}%)")
        self.logger.info(f"  Precision    : {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        self.logger.info(f"  Recall       : {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        
    def calculate_degradation(self, baseline_metrics, condition_metrics):
        """Calculate performance degradation compared to baseline"""
        degradation = {}
        
        for metric in ['map50', 'map50_95', 'precision', 'recall']:
            baseline_val = baseline_metrics.get(metric, 0.0)
            condition_val = condition_metrics.get(metric, 0.0)
            
            if baseline_val > 0:
                absolute_drop = baseline_val - condition_val
                relative_drop = (absolute_drop / baseline_val) * 100
                degradation[metric] = {
                    'absolute_drop': absolute_drop,
                    'relative_drop_percent': relative_drop,
                    'baseline_value': baseline_val,
                    'condition_value': condition_val
                }
            else:
                degradation[metric] = {
                    'absolute_drop': 0.0,
                    'relative_drop_percent': 0.0,
                    'baseline_value': baseline_val,
                    'condition_value': condition_val
                }
                
        return degradation
        
    def run_weather_testing(self):
        """Execute comprehensive weather testing"""
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
            
    def calculate_comprehensive_analysis(self, all_results, baseline_metrics):
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
        
    def save_comprehensive_results(self, all_results, baseline_metrics):
        """Save comprehensive weather testing results"""
        self.logger.info("[SAVE] Saving comprehensive weather testing results...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create comprehensive summary
        summary = {
            "experiment": "phase1_baseline_experiment1_weather_testing",
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
            "weather_results": all_results,
            "summary_statistics": {}
        }
        
        # Calculate summary statistics
        if baseline_metrics:
            successful_tests = [r for r in all_results.values() if 'metrics' in r]
            if successful_tests:
                # Average degradation
                avg_degradations = {}
                for metric in ['map50', 'map50_95', 'precision', 'recall']:
                    degradations = []
                    for result in successful_tests:
                        if 'degradation' in result and metric in result['degradation']:
                            degradations.append(result['degradation'][metric]['relative_drop_percent'])
                    
                    if degradations:
                        avg_degradations[metric] = {
                            'mean_degradation_percent': np.mean(degradations),
                            'max_degradation_percent': np.max(degradations),
                            'min_degradation_percent': np.min(degradations)
                        }
                
                summary["summary_statistics"] = avg_degradations
        
        # Save complete results
        results_file = self.logs_dir / f"weather_testing_complete_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        self.logger.info(f"[SAVE] Complete results saved to: {results_file}")
        
        # Save simplified comparison table
        comparison_file = self.logs_dir / f"weather_comparison_{timestamp}.json"
        comparison = {"baseline": baseline_metrics, "conditions": {}}
        
        for condition, result in all_results.items():
            if 'metrics' in result:
                comparison["conditions"][condition] = {
                    "metrics": result['metrics'],
                    "degradation": result.get('degradation', {})
                }
        
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
            
        self.logger.info(f"[SAVE] Comparison table saved to: {comparison_file}")
        
        return results_file


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='YOLOv5n VisDrone Phase 1 Weather Conditions Testing')
    parser.add_argument('--weights', type=str, required=True,
                       help='Path to trained model weights (best.pt)')
    parser.add_argument('--conf-thres', type=float, default=0.001,
                       help='Confidence threshold for detection (default: 0.001)')
    parser.add_argument('--iou-thres', type=float, default=0.6,
                       help='IoU threshold for NMS (default: 0.6)')
    
    args = parser.parse_args()
    
    try:
        tester = WeatherTester(
            weights_path=args.weights,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres
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