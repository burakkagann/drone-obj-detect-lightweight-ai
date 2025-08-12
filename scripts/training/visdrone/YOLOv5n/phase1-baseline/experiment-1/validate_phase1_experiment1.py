#!/usr/bin/env python3
"""
YOLOv5n VisDrone Phase 1 Baseline Validation Script - Experiment 1
===================================================================

This script performs comprehensive validation of trained Phase 1 baseline models
using the clean VisDrone test dataset. It generates detailed performance metrics
including mAP@0.5, mAP@0.5:0.95, precision, and recall.

Usage:
    python validate_phase1_experiment1.py --weights path/to/best.pt
    python validate_phase1_experiment1.py --weights path/to/best.pt --conf-thres 0.25
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


class Phase1Validator:
    """Phase 1 Baseline Validator for comprehensive model evaluation"""
    
    def __init__(self, weights_path, conf_thres=0.001, iou_thres=0.6, task='val'):
        self.weights_path = Path(weights_path)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.task = task
        self.setup_paths()
        self.setup_logging()
        self.load_config()
        
    def setup_paths(self):
        """Setup all required paths"""
        self.script_dir = Path(__file__).parent
        self.project_root = self.script_dir.parent.parent.parent.parent.parent.parent
        self.yolo_path = self.project_root / "models" / "yolov5n" / "baseline" / "yolov5"
        self.config_dir = self.project_root / "configs" / "yolov5n-visdrone"
        self.logs_dir = self.script_dir / "logs&results" / "validation"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Data config
        self.data_config = self.config_dir / "visdrone_experiment1.yaml"
        
        print(f"[SETUP] Validation script directory: {self.script_dir}")
        print(f"[SETUP] YOLOv5 path: {self.yolo_path}")
        print(f"[SETUP] Config directory: {self.config_dir}")
        print(f"[SETUP] Logs directory: {self.logs_dir}")
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.logs_dir / f"validation_{timestamp}.log"
        
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
        self.logger.info("YOLOv5n VisDrone Phase 1 Baseline Validation - Experiment 1")
        self.logger.info("="*70)
        self.logger.info(f"Weights: {self.weights_path}")
        self.logger.info(f"Confidence threshold: {self.conf_thres}")
        self.logger.info(f"IoU threshold: {self.iou_thres}")
        self.logger.info(f"Log file: {log_file}")
        
    def load_config(self):
        """Load and validate data configuration"""
        self.logger.info("[CONFIG] Loading data configuration...")
        
        if not self.data_config.exists():
            self.logger.error(f"Data config not found: {self.data_config}")
            raise FileNotFoundError(f"Data config not found: {self.data_config}")
        
        with open(self.data_config, 'r') as f:
            self.data_cfg = yaml.safe_load(f)
            
        self.logger.info(f"[CONFIG] Data config loaded: {self.data_config}")
        self.logger.info(f"[CONFIG] Number of classes: {self.data_cfg['nc']}")
        self.logger.info(f"[CONFIG] Class names: {self.data_cfg['names']}")
        
    def validate_environment(self):
        """Validate validation environment"""
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
            self.logger.info("[VALIDATION] Using CPU for validation")
        
        # Validate test data path
        if self.task == 'val':
            test_path = Path(self.data_cfg['val'])
        else:
            test_path = Path(self.data_cfg['test']) if 'test' in self.data_cfg else Path(self.data_cfg['val'])
            
        if not test_path.exists():
            self.logger.error(f"Test data not found: {test_path}")
            raise FileNotFoundError(f"Test data not found: {test_path}")
            
        self.logger.info(f"[VALIDATION] Test data: {test_path}")
        self.logger.info("[VALIDATION] Environment validation completed")
        
    def prepare_validation_args(self):
        """Prepare validation arguments"""
        self.logger.info("[VALIDATION] Preparing validation arguments...")
        
        # Create results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = self.logs_dir / f"validation_results_{timestamp}"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        args = argparse.Namespace(
            # Core parameters
            data=str(self.data_config),
            weights=str(self.weights_path),
            batch_size=32,
            imgsz=640,
            
            # Threshold parameters
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            
            # Output parameters
            task=self.task,
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
            
            # Callback parameters
            callbacks=None,
        )
        
        self.logger.info(f"[VALIDATION] Results directory: {results_dir}")
        self.logger.info(f"[VALIDATION] Batch size: {args.batch_size}")
        self.logger.info(f"[VALIDATION] Image size: {args.imgsz}")
        self.logger.info(f"[VALIDATION] Save results: {args.save_txt}")
        
        return args, results_dir
        
    def run_validation(self):
        """Execute validation process"""
        self.logger.info("[VALIDATION] Starting validation process...")
        
        try:
            # Validate environment
            self.validate_environment()
            
            # Prepare arguments
            args, results_dir = self.prepare_validation_args()
            
            # Record validation start
            start_time = time.time()
            self.logger.info(f"[VALIDATION] Validation started at {datetime.now()}")
            
            # Change to YOLOv5 directory
            original_cwd = os.getcwd()
            os.chdir(self.yolo_path)
            
            try:
                # Run validation
                results = yolo_val.run(**vars(args))
                
                # Validation completed successfully
                end_time = time.time()
                duration = end_time - start_time
                self.logger.info(f"[VALIDATION] Validation completed successfully")
                self.logger.info(f"[VALIDATION] Duration: {duration:.2f} seconds")
                
                # Extract and log metrics
                metrics = self.extract_metrics(results)
                self.log_detailed_metrics(metrics)
                
                # Save comprehensive results
                self.save_validation_results(args, results, metrics, duration, results_dir)
                
                return metrics
                
            except Exception as e:
                self.logger.error(f"[VALIDATION] Validation failed: {str(e)}")
                raise
            finally:
                # Restore original working directory
                os.chdir(original_cwd)
                
        except Exception as e:
            self.logger.error(f"[ERROR] Validation process failed: {str(e)}")
            raise
            
    def extract_metrics(self, results):
        """Extract detailed metrics from validation results"""
        self.logger.info("[METRICS] Extracting validation metrics...")
        
        # Extract main metrics
        metrics = {
            'map50': float(results[0][0]) if len(results) > 0 and len(results[0]) > 0 else 0.0,
            'map50_95': float(results[0][1]) if len(results) > 0 and len(results[0]) > 1 else 0.0,
            'precision': float(results[0][2]) if len(results) > 0 and len(results[0]) > 2 else 0.0,
            'recall': float(results[0][3]) if len(results) > 0 and len(results[0]) > 3 else 0.0,
        }
        
        # Handle case where results format might be different
        if isinstance(results, tuple) and len(results) >= 4:
            metrics.update({
                'map50': float(results[0]) if results[0] is not None else 0.0,
                'map50_95': float(results[1]) if results[1] is not None else 0.0,
                'precision': float(results[2]) if results[2] is not None else 0.0,
                'recall': float(results[3]) if results[3] is not None else 0.0,
            })
        
        # Additional metrics if available
        if len(results) > 4:
            metrics['class_maps'] = results[4] if results[4] is not None else []
            
        return metrics
        
    def log_detailed_metrics(self, metrics):
        """Log detailed metrics"""
        self.logger.info("[METRICS] Validation Results Summary:")
        self.logger.info("-" * 50)
        self.logger.info(f"mAP@0.5      : {metrics['map50']:.4f} ({metrics['map50']*100:.2f}%)")
        self.logger.info(f"mAP@0.5:0.95 : {metrics['map50_95']:.4f} ({metrics['map50_95']*100:.2f}%)")
        self.logger.info(f"Precision    : {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        self.logger.info(f"Recall       : {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        self.logger.info("-" * 50)
        
        # Per-class metrics if available
        if 'class_maps' in metrics and metrics['class_maps']:
            self.logger.info("Per-Class mAP@0.5:")
            for i, class_map in enumerate(metrics['class_maps']):
                if i < len(self.data_cfg['names']):
                    class_name = self.data_cfg['names'][i]
                    self.logger.info(f"  {class_name:15}: {class_map:.4f}")
                    
    def save_validation_results(self, args, results, metrics, duration, results_dir):
        """Save comprehensive validation results"""
        self.logger.info("[SAVE] Saving validation results...")
        
        # Create comprehensive results summary
        summary = {
            "experiment": "phase1_baseline_experiment1_validation",
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": duration,
            "model_info": {
                "weights": str(self.weights_path),
                "model_type": "yolov5n",
                "dataset": "visdrone",
                "task": self.task
            },
            "validation_config": {
                "conf_threshold": self.conf_thres,
                "iou_threshold": self.iou_thres,
                "batch_size": args.batch_size,
                "image_size": args.imgsz,
                "augmentation": args.augment
            },
            "metrics": {
                "mAP_50": metrics['map50'],
                "mAP_50_95": metrics['map50_95'],
                "precision": metrics['precision'],
                "recall": metrics['recall'],
                "mAP_50_percentage": metrics['map50'] * 100,
                "mAP_50_95_percentage": metrics['map50_95'] * 100,
                "precision_percentage": metrics['precision'] * 100,
                "recall_percentage": metrics['recall'] * 100
            },
            "dataset_info": {
                "num_classes": self.data_cfg['nc'],
                "class_names": self.data_cfg['names']
            },
            "results_directory": str(results_dir)
        }
        
        # Add per-class metrics if available
        if 'class_maps' in metrics and metrics['class_maps']:
            summary["per_class_metrics"] = {}
            for i, class_map in enumerate(metrics['class_maps']):
                if i < len(self.data_cfg['names']):
                    class_name = self.data_cfg['names'][i]
                    summary["per_class_metrics"][class_name] = {
                        "mAP_50": float(class_map),
                        "mAP_50_percentage": float(class_map) * 100
                    }
        
        # Save to JSON
        results_file = results_dir / "validation_summary.json"
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        self.logger.info(f"[SAVE] Results saved to: {results_file}")
        
        # Also save a simple metrics file for easy parsing
        metrics_file = results_dir / "metrics.json"
        simple_metrics = {
            "mAP@0.5": metrics['map50'],
            "mAP@0.5:0.95": metrics['map50_95'],
            "precision": metrics['precision'],
            "recall": metrics['recall']
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(simple_metrics, f, indent=2)
            
        self.logger.info(f"[SAVE] Simple metrics saved to: {metrics_file}")
        
        return results_file


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='YOLOv5n VisDrone Phase 1 Baseline Validation')
    parser.add_argument('--weights', type=str, required=True,
                       help='Path to trained model weights (best.pt)')
    parser.add_argument('--conf-thres', type=float, default=0.001,
                       help='Confidence threshold for detection (default: 0.001)')
    parser.add_argument('--iou-thres', type=float, default=0.6,
                       help='IoU threshold for NMS (default: 0.6)')
    parser.add_argument('--task', type=str, default='val', choices=['val', 'test'],
                       help='Validation task (default: val)')
    
    args = parser.parse_args()
    
    try:
        validator = Phase1Validator(
            weights_path=args.weights,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            task=args.task
        )
        
        metrics = validator.run_validation()
        
        print("\n" + "="*70)
        print("Validation completed successfully!")
        print(f"mAP@0.5: {metrics['map50']:.4f} ({metrics['map50']*100:.2f}%)")
        print(f"mAP@0.5:0.95: {metrics['map50_95']:.4f} ({metrics['map50_95']*100:.2f}%)")
        print(f"Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"Recall: {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print("="*70)
        
    except Exception as e:
        print(f"\nValidation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()