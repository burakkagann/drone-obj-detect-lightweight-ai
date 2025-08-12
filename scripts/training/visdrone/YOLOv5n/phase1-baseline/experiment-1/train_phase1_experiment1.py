#!/usr/bin/env python3
"""
YOLOv5n VisDrone Phase 1 Baseline Training Script - Experiment 1
================================================================

This script implements the optimized Phase 1 baseline training configuration
based on recovery analysis findings from previous experiments.

Key Optimizations Applied:
- SGD optimizer (outperformed AdamW)
- Fixed batch size 8 for stability  
- 50 epochs training duration (optimal convergence)
- Loss weights: cls=0.3, obj=0.7 (best performing ratio)
- Zero augmentation (Phase 1 baseline requirement)
- Comprehensive logging and validation

Usage:
    python train_phase1_experiment1.py [--QuickTest]
    
    --QuickTest: Run 5 epochs for configuration validation
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

# Add YOLOv5 to path
YOLO_PATH = Path(__file__).parent.parent.parent.parent.parent.parent / "models" / "yolov5n" / "baseline" / "yolov5"
sys.path.append(str(YOLO_PATH))

try:
    import torch
    import train as yolo_train
    from utils.general import check_requirements, check_img_size, increment_path
    from utils.torch_utils import select_device
except ImportError as e:
    print(f"Error importing YOLOv5 modules: {e}")
    print(f"Make sure YOLOv5 is properly installed at: {YOLO_PATH}")
    sys.exit(1)


class Phase1Trainer:
    """Phase 1 Baseline Trainer with optimized configuration"""
    
    def __init__(self, quick_test=False):
        self.quick_test = quick_test
        self.setup_paths()
        self.setup_logging()
        self.load_config()
        
    def setup_paths(self):
        """Setup all required paths"""
        self.script_dir = Path(__file__).parent
        self.project_root = self.script_dir.parent.parent.parent.parent.parent.parent
        self.yolo_path = self.project_root / "models" / "yolov5n" / "baseline" / "yolov5"
        self.config_dir = self.project_root / "configs" / "yolov5n-visdrone"
        self.logs_dir = self.script_dir / "logs&results" / "training"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Data paths
        self.data_config = self.config_dir / "visdrone_experiment1.yaml"
        self.hyp_config = self.config_dir / "experiment1_training_config.yaml"
        
        print(f"[SETUP] Script directory: {self.script_dir}")
        print(f"[SETUP] Project root: {self.project_root}")
        print(f"[SETUP] YOLOv5 path: {self.yolo_path}")
        print(f"[SETUP] Config directory: {self.config_dir}")
        print(f"[SETUP] Logs directory: {self.logs_dir}")
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "quicktest" if self.quick_test else "full"
        log_file = self.logs_dir / f"training_{mode}_{timestamp}.log"
        
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
        self.logger.info("YOLOv5n VisDrone Phase 1 Baseline Training - Experiment 1")
        self.logger.info("="*70)
        self.logger.info(f"Mode: {'Quick Test (5 epochs)' if self.quick_test else 'Full Training (50 epochs)'}")
        self.logger.info(f"Log file: {log_file}")
        
    def load_config(self):
        """Load and validate training configuration"""
        self.logger.info("[CONFIG] Loading training configuration...")
        
        # Check if config files exist
        if not self.data_config.exists():
            self.logger.error(f"Data config not found: {self.data_config}")
            raise FileNotFoundError(f"Data config not found: {self.data_config}")
            
        if not self.hyp_config.exists():
            self.logger.error(f"Hyperparameter config not found: {self.hyp_config}")
            raise FileNotFoundError(f"Hyperparameter config not found: {self.hyp_config}")
        
        # Load configurations
        with open(self.data_config, 'r') as f:
            self.data_cfg = yaml.safe_load(f)
            
        with open(self.hyp_config, 'r') as f:
            self.hyp_cfg = yaml.safe_load(f)
            
        self.logger.info(f"[CONFIG] Data config loaded: {self.data_config}")
        self.logger.info(f"[CONFIG] Hyperparameter config loaded: {self.hyp_config}")
        
    def validate_environment(self):
        """Validate training environment"""
        self.logger.info("[VALIDATION] Validating training environment...")
        
        # Check GPU availability
        device = select_device('')
        self.logger.info(f"[VALIDATION] Device: {device}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            self.logger.info(f"[VALIDATION] GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            self.logger.warning("[VALIDATION] No GPU available, using CPU")
        
        # Validate data paths
        train_path = Path(self.data_cfg['train'])
        val_path = Path(self.data_cfg['val'])
        
        if not train_path.exists():
            self.logger.error(f"[VALIDATION] Training data not found: {train_path}")
            raise FileNotFoundError(f"Training data not found: {train_path}")
            
        if not val_path.exists():
            self.logger.error(f"[VALIDATION] Validation data not found: {val_path}")
            raise FileNotFoundError(f"Validation data not found: {val_path}")
            
        self.logger.info(f"[VALIDATION] Training data: {train_path}")
        self.logger.info(f"[VALIDATION] Validation data: {val_path}")
        self.logger.info(f"[VALIDATION] Number of classes: {self.data_cfg['nc']}")
        self.logger.info(f"[VALIDATION] Class names: {self.data_cfg['names']}")
        
        # Validate model weights
        weights_path = self.yolo_path / "yolov5n.pt"
        if not weights_path.exists():
            self.logger.error(f"[VALIDATION] Model weights not found: {weights_path}")
            raise FileNotFoundError(f"Model weights not found: {weights_path}")
            
        self.logger.info(f"[VALIDATION] Model weights: {weights_path}")
        self.logger.info("[VALIDATION] Environment validation completed successfully")
        
    def prepare_training_args(self):
        """Prepare training arguments based on optimized configuration"""
        self.logger.info("[TRAINING] Preparing training arguments...")
        
        # Create results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "quicktest" if self.quick_test else "baseline"
        project_name = f"phase1_{mode}_exp1_{timestamp}"
        results_dir = self.logs_dir / project_name
        
        # Training arguments based on recovery analysis optimizations
        args = argparse.Namespace(
            # Core training parameters
            weights=str(self.yolo_path / "yolov5n.pt"),
            cfg='',
            data=str(self.data_config),
            hyp=str(self.hyp_config),
            epochs=5 if self.quick_test else 50,
            batch_size=8,  # Fixed batch size for stability
            imgsz=640,
            
            # Optimization parameters (from recovery analysis)
            optimizer='SGD',  # SGD outperformed AdamW
            lr0=0.01,         # Optimal learning rate
            lrf=0.01,         # Learning rate factor
            momentum=0.937,
            weight_decay=0.0005,
            
            # Loss weights (optimal configuration)
            cls=0.3,  # Class loss weight
            obj=0.7,  # Object loss weight
            
            # Output and logging
            project=str(results_dir.parent),
            name=project_name,
            exist_ok=False,
            
            # Device and workers
            device='',
            workers=8,
            
            # No augmentation for Phase 1 baseline
            rect=False,
            resume=False,
            nosave=False,
            noval=False,
            noautoanchor=False,
            noplots=False,
            evolve=None,
            bucket='',
            cache=None,
            image_weights=False,
            
            # Advanced settings
            multi_scale=False,
            single_cls=False,
            adam=False,
            sync_bn=False,
            
            # Evaluation settings
            val=True,
            save_period=-1,
            
            # Additional parameters
            artifact_alias='latest',
            local_rank=-1,
            entity=None,
            upload_dataset=False,
            bbox_interval=-1,
            save_period=-1,
            quad=False,
            cos_lr=False,
            label_smoothing=0.0,
            patience=300,
            freeze=[],
            save_dir=str(results_dir)
        )
        
        # Log configuration
        self.logger.info("[TRAINING] Training Configuration:")
        self.logger.info(f"  Epochs: {args.epochs}")
        self.logger.info(f"  Batch Size: {args.batch_size}")
        self.logger.info(f"  Optimizer: {args.optimizer}")
        self.logger.info(f"  Learning Rate: {args.lr0}")
        self.logger.info(f"  Loss Weights - cls: {args.cls}, obj: {args.obj}")
        self.logger.info(f"  Results Directory: {results_dir}")
        
        return args
        
    def run_training(self):
        """Execute the training process"""
        self.logger.info("[TRAINING] Starting training process...")
        
        try:
            # Validate environment
            self.validate_environment()
            
            # Prepare arguments
            args = self.prepare_training_args()
            
            # Record training start
            start_time = time.time()
            self.logger.info(f"[TRAINING] Training started at {datetime.now()}")
            
            # Change to YOLOv5 directory
            original_cwd = os.getcwd()
            os.chdir(self.yolo_path)
            
            try:
                # Run training
                yolo_train.run(**vars(args))
                
                # Training completed successfully
                end_time = time.time()
                duration = end_time - start_time
                self.logger.info(f"[TRAINING] Training completed successfully")
                self.logger.info(f"[TRAINING] Total duration: {duration/3600:.2f} hours")
                
                # Save training summary
                self.save_training_summary(args, duration, success=True)
                
            except Exception as e:
                self.logger.error(f"[TRAINING] Training failed: {str(e)}")
                self.save_training_summary(args, time.time() - start_time, success=False, error=str(e))
                raise
            finally:
                # Restore original working directory
                os.chdir(original_cwd)
                
        except Exception as e:
            self.logger.error(f"[ERROR] Training process failed: {str(e)}")
            raise
            
    def save_training_summary(self, args, duration, success=True, error=None):
        """Save training summary"""
        summary = {
            "experiment": "phase1_baseline_experiment1",
            "timestamp": datetime.now().isoformat(),
            "mode": "quicktest" if self.quick_test else "full_training",
            "success": success,
            "duration_hours": duration / 3600,
            "configuration": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "optimizer": args.optimizer,
                "learning_rate": args.lr0,
                "loss_weights": {"cls": args.cls, "obj": args.obj},
                "model": "yolov5n",
                "dataset": "visdrone",
                "augmentation": "disabled"
            },
            "paths": {
                "weights": args.weights,
                "data_config": str(self.data_config),
                "hyp_config": str(self.hyp_config),
                "results_dir": args.save_dir
            }
        }
        
        if error:
            summary["error"] = error
            
        summary_file = self.logs_dir / f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        self.logger.info(f"[SUMMARY] Training summary saved: {summary_file}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='YOLOv5n VisDrone Phase 1 Baseline Training')
    parser.add_argument('--QuickTest', action='store_true', 
                       help='Run quick test with 5 epochs for configuration validation')
    args = parser.parse_args()
    
    try:
        trainer = Phase1Trainer(quick_test=args.QuickTest)
        trainer.run_training()
        print("\n" + "="*70)
        print("Training completed successfully!")
        print("="*70)
    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()