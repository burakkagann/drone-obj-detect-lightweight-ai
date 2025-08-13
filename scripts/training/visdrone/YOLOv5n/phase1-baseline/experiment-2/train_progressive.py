#!/usr/bin/env python3
"""
YOLOv5n VisDrone Phase 1 Baseline - Experiment 2 Progressive Training Script
============================================================================

This script implements the 3-phase progressive training strategy for Experiment 2:
- Phase 2A (Epochs 1-30): Foundation building with lr=0.001
- Phase 2B (Epochs 31-70): Optimization with lr=0.0005
- Phase 2C (Epochs 71-100): Fine-tuning with lr=0.0001

Usage:
    python train_progressive.py --phase 2a  # Start Phase 2A
    python train_progressive.py --phase 2b  # Continue with Phase 2B
    python train_progressive.py --phase 2c  # Finish with Phase 2C
    python train_progressive.py --phase all # Run all phases sequentially
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
import shutil
import torch
import yaml

# Project paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.parent.parent.parent
YOLO_PATH = PROJECT_ROOT / "models" / "yolov5n" / "baseline" / "yolov5"
CONFIG_DIR = PROJECT_ROOT / "configs" / "yolov5n-visdrone"
LOGS_DIR = SCRIPT_DIR / "logs&results"


class ProgressiveTrainer:
    """Progressive training manager for Experiment 2"""
    
    def __init__(self, phase='all', batch_size=16, device='0', quick_test=False):
        self.phase = phase
        self.batch_size = batch_size
        self.device = device
        self.quick_test = quick_test
        self.setup_paths()
        self.setup_logging()
        self.load_configs()
        
    def setup_paths(self):
        """Setup all required paths"""
        self.yolo_train = YOLO_PATH / "train.py"
        self.yolo_val = YOLO_PATH / "val.py"
        
        # Phase-specific paths
        if self.quick_test:
            # Quick test configuration - 5 epochs total
            self.phase_configs = {
                '2a': {
                    'hyp': CONFIG_DIR / "experiment2a_phase1_hyp.yaml",
                    'epochs': 2,  # Quick test: 2 epochs
                    'name': 'exp2_phase2a_quicktest',
                    'resume': None
                },
                '2b': {
                    'hyp': CONFIG_DIR / "experiment2b_phase1_hyp.yaml",
                    'epochs': 2,  # Quick test: 2 epochs
                    'name': 'exp2_phase2b_quicktest',
                    'resume': 'exp2_phase2a_quicktest'
                },
                '2c': {
                    'hyp': CONFIG_DIR / "experiment2c_phase_hyp.yaml",
                    'epochs': 1,  # Quick test: 1 epoch
                    'name': 'exp2_phase2c_quicktest',
                    'resume': 'exp2_phase2b_quicktest'
                }
            }
        else:
            # Full configuration
            self.phase_configs = {
                '2a': {
                    'hyp': CONFIG_DIR / "experiment2a_phase1_hyp.yaml",
                    'epochs': 30,
                    'name': 'exp2_phase2a',
                    'resume': None
                },
                '2b': {
                    'hyp': CONFIG_DIR / "experiment2b_phase1_hyp.yaml",
                    'epochs': 40,
                    'name': 'exp2_phase2b',
                    'resume': 'exp2_phase2a'  # Will be resolved to actual path
                },
                '2c': {
                    'hyp': CONFIG_DIR / "experiment2c_phase_hyp.yaml",
                    'epochs': 30,
                    'name': 'exp2_phase2c',
                    'resume': 'exp2_phase2b'  # Will be resolved to actual path
                }
            }
        
        self.data_config = CONFIG_DIR / "visdrone_experiment2.yaml"
        self.model_config = YOLO_PATH / "models" / "yolov5n.yaml"
        self.pretrained_weights = YOLO_PATH / "yolov5n.pt"
        
        # Results directories
        self.training_dir = LOGS_DIR / "training"
        self.training_dir.mkdir(parents=True, exist_ok=True)
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = LOGS_DIR / f"progressive_training_{timestamp}.log"
        
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
        self.logger.info("YOLOv5n VisDrone Experiment 2 - Progressive Training")
        if self.quick_test:
            self.logger.info("*** QUICK TEST MODE - 5 EPOCHS TOTAL ***")
        self.logger.info("="*80)
        self.logger.info(f"Phase: {self.phase}")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Quick test: {self.quick_test}")
        
    def load_configs(self):
        """Load and validate all configuration files"""
        self.logger.info("[CONFIG] Validating configuration files...")
        
        # Check all required files exist
        required_files = [
            self.yolo_train,
            self.data_config,
            self.model_config,
            self.pretrained_weights
        ]
        
        for phase_key, phase_info in self.phase_configs.items():
            required_files.append(phase_info['hyp'])
            
        for file_path in required_files:
            if not file_path.exists():
                self.logger.error(f"Missing required file: {file_path}")
                raise FileNotFoundError(f"Required file not found: {file_path}")
                
        self.logger.info("[CONFIG] All configuration files validated successfully")
        
    def train_phase(self, phase_key):
        """Train a specific phase"""
        phase_info = self.phase_configs[phase_key]
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Starting Phase {phase_key.upper()} Training")
        self.logger.info(f"{'='*80}")
        
        # Build training command
        cmd = [
            sys.executable,
            str(self.yolo_train),
            "--epochs", str(phase_info['epochs']),
            "--batch-size", str(self.batch_size),
            "--cfg", str(self.model_config),
            "--hyp", str(phase_info['hyp']),
            "--data", str(self.data_config),
            "--device", str(self.device),
            "--project", str(self.training_dir),
            "--name", phase_info['name'],
            "--exist-ok",
            "--patience", "30",
            "--save-period", "5",
            "--workers", "8",
            "--rect",  # Rectangular training
            "--image-weights", "False",  # Disabled for Phase 1
            "--multi-scale",  # Enable multi-scale training
            "--single-cls", "False",
            "--quad", "False",
            "--cos-lr", "True" if phase_key in ['2b', '2c'] else "False",
            "--label-smoothing", "0.0" if phase_key == '2a' else ("0.05" if phase_key == '2b' else "0.1"),
            "--cache", "ram" if self.check_ram_availability() else "disk"
        ]
        
        # Add weights (pretrained for 2a, resume for 2b/2c)
        if phase_key == '2a':
            cmd.extend(["--weights", str(self.pretrained_weights)])
        else:
            # Find the last checkpoint from previous phase
            prev_phase = self.phase_configs[phase_info['resume']]
            checkpoint_path = self.training_dir / prev_phase['name'] / "weights" / "last.pt"
            
            if not checkpoint_path.exists():
                # Try best.pt if last.pt doesn't exist
                checkpoint_path = self.training_dir / prev_phase['name'] / "weights" / "best.pt"
                
            if not checkpoint_path.exists():
                self.logger.error(f"Cannot find checkpoint from {prev_phase['name']}")
                raise FileNotFoundError(f"Previous phase checkpoint not found: {checkpoint_path}")
                
            cmd.extend(["--weights", str(checkpoint_path)])
            self.logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        
        # Log command
        self.logger.info(f"Training command: {' '.join(cmd)}")
        
        # Execute training
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                cwd=str(YOLO_PATH),
                capture_output=True,
                text=True,
                check=True
            )
            
            # Log output
            if result.stdout:
                with open(LOGS_DIR / f"phase_{phase_key}_stdout.log", 'w') as f:
                    f.write(result.stdout)
                    
            if result.stderr:
                with open(LOGS_DIR / f"phase_{phase_key}_stderr.log", 'w') as f:
                    f.write(result.stderr)
                    
            duration = (time.time() - start_time) / 3600
            self.logger.info(f"Phase {phase_key.upper()} completed successfully in {duration:.2f} hours")
            
            # Save phase summary
            self.save_phase_summary(phase_key, duration, "success")
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Training failed for phase {phase_key}: {str(e)}")
            if e.stderr:
                self.logger.error(f"Error output: {e.stderr}")
            self.save_phase_summary(phase_key, 0, "failed", str(e))
            return False
            
    def save_phase_summary(self, phase_key, duration, status, error=None):
        """Save summary for each phase"""
        phase_info = self.phase_configs[phase_key]
        summary = {
            "phase": phase_key.upper(),
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "duration_hours": duration,
            "configuration": {
                "epochs": phase_info['epochs'],
                "batch_size": self.batch_size,
                "hyp_config": str(phase_info['hyp']),
                "name": phase_info['name']
            }
        }
        
        if error:
            summary["error"] = error
            
        # Get training metrics if available
        results_file = self.training_dir / phase_info['name'] / "results.csv"
        if results_file.exists():
            try:
                import pandas as pd
                df = pd.read_csv(results_file)
                if len(df) > 0:
                    last_row = df.iloc[-1]
                    summary["final_metrics"] = {
                        "mAP_0.5": float(last_row['   metrics/mAP_0.5']) if '   metrics/mAP_0.5' in df.columns else None,
                        "precision": float(last_row['metrics/precision']) if 'metrics/precision' in df.columns else None,
                        "recall": float(last_row['metrics/recall']) if 'metrics/recall' in df.columns else None
                    }
            except Exception as e:
                self.logger.warning(f"Could not read results.csv: {e}")
                
        # Save summary
        summary_file = LOGS_DIR / f"phase_{phase_key}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        self.logger.info(f"Phase summary saved to: {summary_file}")
        
    def check_ram_availability(self):
        """Check if there's enough RAM for caching"""
        try:
            import psutil
            available_gb = psutil.virtual_memory().available / (1024**3)
            return available_gb > 16  # Need at least 16GB free RAM for caching
        except:
            return False
            
    def run_all_phases(self):
        """Run all three phases sequentially"""
        phases = ['2a', '2b', '2c']
        
        for phase_key in phases:
            success = self.train_phase(phase_key)
            if not success:
                self.logger.error(f"Phase {phase_key} failed. Stopping progressive training.")
                return False
                
            # Optional: Run validation after each phase
            self.validate_phase(phase_key)
            
        self.logger.info("\n" + "="*80)
        self.logger.info("All phases completed successfully!")
        self.logger.info("="*80)
        
        # Save final experiment summary
        self.save_experiment_summary()
        
        return True
        
    def validate_phase(self, phase_key):
        """Run validation on completed phase"""
        phase_info = self.phase_configs[phase_key]
        weights_path = self.training_dir / phase_info['name'] / "weights" / "best.pt"
        
        if not weights_path.exists():
            self.logger.warning(f"Best weights not found for phase {phase_key}, skipping validation")
            return
            
        self.logger.info(f"Running validation for phase {phase_key}...")
        
        cmd = [
            sys.executable,
            str(self.yolo_val),
            "--weights", str(weights_path),
            "--data", str(self.data_config),
            "--batch-size", str(self.batch_size),
            "--imgsz", "640",
            "--conf-thres", "0.001",
            "--iou-thres", "0.6",
            "--device", str(self.device),
            "--project", str(LOGS_DIR / "validation"),
            "--name", f"phase_{phase_key}_val",
            "--exist-ok",
            "--save-json",
            "--save-txt",
            "--save-conf"
        ]
        
        try:
            subprocess.run(cmd, cwd=str(YOLO_PATH), check=True, capture_output=True)
            self.logger.info(f"Validation completed for phase {phase_key}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Validation failed for phase {phase_key}: {str(e)}")
            
    def save_experiment_summary(self):
        """Save overall experiment summary"""
        summary = {
            "experiment": "phase1_baseline_experiment2",
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "progressive_training": True,
                "total_epochs": 100,
                "phases": ["2A (1-30)", "2B (31-70)", "2C (71-100)"],
                "batch_size": self.batch_size,
                "optimizer": "SGD",
                "multi_scale": True,
                "augmentation": "disabled (Phase 1)"
            },
            "phases_completed": []
        }
        
        # Collect metrics from each phase
        for phase_key in ['2a', '2b', '2c']:
            phase_summary_file = LOGS_DIR / f"phase_{phase_key}_summary.json"
            if phase_summary_file.exists():
                with open(phase_summary_file, 'r') as f:
                    phase_data = json.load(f)
                    summary["phases_completed"].append(phase_data)
                    
        # Save final summary
        summary_file = LOGS_DIR / "experiment2_complete_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        self.logger.info(f"Experiment summary saved to: {summary_file}")
        
    def run(self):
        """Main execution method"""
        if self.phase.lower() == 'all':
            return self.run_all_phases()
        elif self.phase.lower() in ['2a', '2b', '2c']:
            success = self.train_phase(self.phase.lower())
            if success:
                self.validate_phase(self.phase.lower())
            return success
        else:
            self.logger.error(f"Invalid phase: {self.phase}. Must be '2a', '2b', '2c', or 'all'")
            return False


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='YOLOv5n VisDrone Experiment 2 Progressive Training')
    parser.add_argument('--phase', type=str, default='all',
                       choices=['2a', '2b', '2c', 'all'],
                       help='Training phase to execute (default: all)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training (default: 16)')
    parser.add_argument('--device', type=str, default='0',
                       help='CUDA device (default: 0)')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with 5 total epochs (2+2+1)')
    
    args = parser.parse_args()
    
    trainer = ProgressiveTrainer(
        phase=args.phase,
        batch_size=args.batch_size,
        device=args.device,
        quick_test=args.quick_test
    )
    
    success = trainer.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()