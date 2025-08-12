#!/usr/bin/env python3
"""
YOLOv5n VisDrone Phase 1 Baseline Complete Experiment Runner - Experiment 1
============================================================================

Master script that orchestrates the complete Phase 1 baseline experiment workflow:
1. Training (50 epochs or 5 for QuickTest)
2. Validation on clean dataset
3. Weather conditions testing on all synthetic datasets
4. Results compilation and analysis

This script implements the complete experimental protocol with optimized configuration
based on recovery analysis findings.

Usage:
    python run_complete_experiment1.py [--QuickTest] [--SkipTraining] [--weights path/to/weights]
    
    --QuickTest: Run 5 epochs training + validation only (no weather testing)
    --SkipTraining: Skip training, use existing weights for validation and weather testing
    --weights: Path to existing weights (required if --SkipTraining)
"""

import argparse
import os
import sys
import json
import subprocess
import time
import logging
from datetime import datetime
from pathlib import Path
import shutil


class CompleteExperimentRunner:
    """Complete Phase 1 Baseline Experiment Orchestrator"""
    
    def __init__(self, quick_test=False, skip_training=False, weights_path=None):
        self.quick_test = quick_test
        self.skip_training = skip_training
        self.weights_path = Path(weights_path) if weights_path else None
        self.setup_paths()
        self.setup_logging()
        self.experiment_results = {}
        
    def setup_paths(self):
        """Setup all required paths"""
        self.script_dir = Path(__file__).parent
        self.project_root = self.script_dir.parent.parent.parent.parent.parent.parent
        self.logs_dir = self.script_dir / "logs&results"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Script paths
        self.train_script = self.script_dir / "train_phase1_experiment1.py"
        self.validate_script = self.script_dir / "validate_phase1_experiment1.py"
        self.weather_test_script = self.script_dir / "test_weather_conditions.py"
        
        print(f"[SETUP] Experiment runner directory: {self.script_dir}")
        print(f"[SETUP] Project root: {self.project_root}")
        print(f"[SETUP] Logs directory: {self.logs_dir}")
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "quicktest" if self.quick_test else "complete"
        if self.skip_training:
            mode += "_skip_training"
        
        log_file = self.logs_dir / f"experiment_runner_{mode}_{timestamp}.log"
        
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
        self.logger.info("YOLOv5n VisDrone Phase 1 Complete Experiment Runner - Experiment 1")
        self.logger.info("="*80)
        self.logger.info(f"Mode: {mode}")
        self.logger.info(f"Quick Test: {self.quick_test}")
        self.logger.info(f"Skip Training: {self.skip_training}")
        if self.weights_path:
            self.logger.info(f"Using weights: {self.weights_path}")
        self.logger.info(f"Log file: {log_file}")
        
    def validate_environment(self):
        """Validate experiment environment"""
        self.logger.info("[VALIDATION] Validating experiment environment...")
        
        # Check if all required scripts exist
        required_scripts = [self.train_script, self.validate_script, self.weather_test_script]
        for script in required_scripts:
            if not script.exists():
                self.logger.error(f"Required script not found: {script}")
                raise FileNotFoundError(f"Required script not found: {script}")
        
        self.logger.info("[VALIDATION] All required scripts found")
        
        # If skip training, validate weights path
        if self.skip_training:
            if not self.weights_path or not self.weights_path.exists():
                self.logger.error("--SkipTraining requires valid --weights path")
                raise FileNotFoundError("--SkipTraining requires valid --weights path")
            self.logger.info(f"[VALIDATION] Using existing weights: {self.weights_path}")
        
        # Check Python environment
        python_executable = sys.executable
        self.logger.info(f"[VALIDATION] Python executable: {python_executable}")
        
        # Check if we can import required modules
        try:
            import torch
            import yaml
            self.logger.info(f"[VALIDATION] PyTorch version: {torch.__version__}")
            self.logger.info(f"[VALIDATION] CUDA available: {torch.cuda.is_available()}")
        except ImportError as e:
            self.logger.error(f"[VALIDATION] Failed to import required modules: {e}")
            raise
        
        self.logger.info("[VALIDATION] Environment validation completed")
        
    def run_training_phase(self):
        """Execute training phase"""
        if self.skip_training:
            self.logger.info("[TRAINING] Skipping training phase (using existing weights)")
            self.experiment_results['training'] = {
                'skipped': True,
                'weights_used': str(self.weights_path),
                'timestamp': datetime.now().isoformat()
            }
            return self.weights_path
        
        self.logger.info("[TRAINING] Starting training phase...")
        
        try:
            # Prepare training command
            cmd = [sys.executable, str(self.train_script)]
            if self.quick_test:
                cmd.append('--QuickTest')
            
            self.logger.info(f"[TRAINING] Running command: {' '.join(cmd)}")
            
            # Record start time
            start_time = time.time()
            
            # Execute training
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.script_dir)
            
            # Record end time
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                self.logger.info(f"[TRAINING] Training completed successfully in {duration/3600:.2f} hours")
                self.logger.info(f"[TRAINING] Training output logged")
                
                # Find the trained weights
                weights_path = self.find_best_weights()
                
                self.experiment_results['training'] = {
                    'success': True,
                    'duration_hours': duration / 3600,
                    'mode': 'quicktest' if self.quick_test else 'full',
                    'weights_path': str(weights_path),
                    'timestamp': datetime.now().isoformat()
                }
                
                return weights_path
                
            else:
                self.logger.error(f"[TRAINING] Training failed with return code: {result.returncode}")
                self.logger.error(f"[TRAINING] Error output: {result.stderr}")
                
                self.experiment_results['training'] = {
                    'success': False,
                    'error': result.stderr,
                    'return_code': result.returncode,
                    'timestamp': datetime.now().isoformat()
                }
                
                raise RuntimeError(f"Training failed: {result.stderr}")
                
        except Exception as e:
            self.logger.error(f"[TRAINING] Training phase failed: {str(e)}")
            self.experiment_results['training'] = {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            raise
            
    def find_best_weights(self):
        """Find the best weights from training"""
        self.logger.info("[TRAINING] Searching for best weights...")
        
        # Look for weights in training logs directory
        training_logs = self.logs_dir / "training"
        if not training_logs.exists():
            raise FileNotFoundError("Training logs directory not found")
        
        # Search for best.pt files
        best_weights = list(training_logs.rglob("best.pt"))
        
        if not best_weights:
            # Also check for weights.pt or last.pt
            last_weights = list(training_logs.rglob("last.pt"))
            if last_weights:
                weights_path = max(last_weights, key=lambda p: p.stat().st_mtime)
                self.logger.warning(f"[TRAINING] Using last.pt weights: {weights_path}")
                return weights_path
            else:
                raise FileNotFoundError("No trained weights found")
        
        # Use the most recent best.pt
        weights_path = max(best_weights, key=lambda p: p.stat().st_mtime)
        self.logger.info(f"[TRAINING] Found best weights: {weights_path}")
        
        return weights_path
        
    def run_validation_phase(self, weights_path):
        """Execute validation phase"""
        self.logger.info("[VALIDATION] Starting validation phase...")
        
        try:
            # Prepare validation command
            cmd = [
                sys.executable, str(self.validate_script),
                '--weights', str(weights_path)
            ]
            
            self.logger.info(f"[VALIDATION] Running command: {' '.join(cmd)}")
            
            # Record start time
            start_time = time.time()
            
            # Execute validation
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.script_dir)
            
            # Record end time
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                self.logger.info(f"[VALIDATION] Validation completed successfully in {duration:.2f} seconds")
                
                # Parse validation results if possible
                validation_metrics = self.parse_validation_output(result.stdout)
                
                self.experiment_results['validation'] = {
                    'success': True,
                    'duration_seconds': duration,
                    'weights_used': str(weights_path),
                    'metrics': validation_metrics,
                    'timestamp': datetime.now().isoformat()
                }
                
                return validation_metrics
                
            else:
                self.logger.error(f"[VALIDATION] Validation failed with return code: {result.returncode}")
                self.logger.error(f"[VALIDATION] Error output: {result.stderr}")
                
                self.experiment_results['validation'] = {
                    'success': False,
                    'error': result.stderr,
                    'return_code': result.returncode,
                    'timestamp': datetime.now().isoformat()
                }
                
                raise RuntimeError(f"Validation failed: {result.stderr}")
                
        except Exception as e:
            self.logger.error(f"[VALIDATION] Validation phase failed: {str(e)}")
            self.experiment_results['validation'] = {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            raise
            
    def parse_validation_output(self, output):
        """Parse validation output to extract metrics"""
        try:
            # Look for metrics in the output
            lines = output.split('\\n')
            metrics = {}
            
            for line in lines:
                if 'mAP@0.5:' in line:
                    # Try to extract metrics from summary lines
                    # This is a simple parser - could be enhanced
                    pass
                    
            # For now, return empty dict - metrics will be in saved files
            return {}
            
        except Exception as e:
            self.logger.warning(f"[VALIDATION] Could not parse validation output: {e}")
            return {}
            
    def run_weather_testing_phase(self, weights_path):
        """Execute weather testing phase"""
        if self.quick_test:
            self.logger.info("[WEATHER] Skipping weather testing in QuickTest mode")
            self.experiment_results['weather_testing'] = {
                'skipped': True,
                'reason': 'quicktest_mode',
                'timestamp': datetime.now().isoformat()
            }
            return None
        
        self.logger.info("[WEATHER] Starting weather testing phase...")
        
        try:
            # Prepare weather testing command
            cmd = [
                sys.executable, str(self.weather_test_script),
                '--weights', str(weights_path)
            ]
            
            self.logger.info(f"[WEATHER] Running command: {' '.join(cmd)}")
            
            # Record start time
            start_time = time.time()
            
            # Execute weather testing
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.script_dir)
            
            # Record end time
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                self.logger.info(f"[WEATHER] Weather testing completed successfully in {duration/60:.2f} minutes")
                
                self.experiment_results['weather_testing'] = {
                    'success': True,
                    'duration_minutes': duration / 60,
                    'weights_used': str(weights_path),
                    'timestamp': datetime.now().isoformat()
                }
                
                return True
                
            else:
                self.logger.error(f"[WEATHER] Weather testing failed with return code: {result.returncode}")
                self.logger.error(f"[WEATHER] Error output: {result.stderr}")
                
                self.experiment_results['weather_testing'] = {
                    'success': False,
                    'error': result.stderr,
                    'return_code': result.returncode,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Don't raise error for weather testing failure - continue with results compilation
                self.logger.warning("[WEATHER] Continuing despite weather testing failure")
                return False
                
        except Exception as e:
            self.logger.error(f"[WEATHER] Weather testing phase failed: {str(e)}")
            self.experiment_results['weather_testing'] = {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return False
            
    def compile_results(self):
        """Compile and save final experiment results"""
        self.logger.info("[RESULTS] Compiling final experiment results...")
        
        try:
            # Create comprehensive experiment summary
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            experiment_summary = {
                "experiment_id": f"phase1_baseline_experiment1_{timestamp}",
                "experiment_type": "phase1_baseline",
                "model": "yolov5n",
                "dataset": "visdrone",
                "timestamp": datetime.now().isoformat(),
                "configuration": {
                    "quick_test": self.quick_test,
                    "skip_training": self.skip_training,
                    "optimizations_applied": [
                        "SGD optimizer (vs AdamW)",
                        "batch_size=8 (fixed)",
                        "50 epochs duration",
                        "loss_weights cls=0.3, obj=0.7",
                        "zero_augmentation (Phase 1 requirement)"
                    ]
                },
                "phases_executed": {},
                "overall_success": True,
                "results": self.experiment_results
            }
            
            # Determine which phases were executed
            phases = []
            if not self.skip_training:
                phases.append("training")
            phases.append("validation")
            if not self.quick_test:
                phases.append("weather_testing")
                
            experiment_summary["phases_executed"] = phases
            
            # Check overall success
            overall_success = True
            for phase in phases:
                if phase in self.experiment_results:
                    if not self.experiment_results[phase].get('success', True):
                        overall_success = False
                        break
                else:
                    overall_success = False
                    break
                    
            experiment_summary["overall_success"] = overall_success
            
            # Save experiment summary
            summary_file = self.logs_dir / f"experiment_summary_{timestamp}.json"
            with open(summary_file, 'w') as f:
                json.dump(experiment_summary, f, indent=2)
                
            self.logger.info(f"[RESULTS] Experiment summary saved: {summary_file}")
            
            # Create a simple status file
            status = "SUCCESS" if overall_success else "PARTIAL_SUCCESS"
            status_file = self.logs_dir / f"experiment_status_{timestamp}.txt"
            with open(status_file, 'w') as f:
                f.write(f"Experiment Status: {status}\\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\\n")
                f.write(f"Phases: {', '.join(phases)}\\n")
                f.write(f"Mode: {'QuickTest' if self.quick_test else 'Complete'}\\n")
                
            self.logger.info(f"[RESULTS] Status file saved: {status_file}")
            
            return experiment_summary
            
        except Exception as e:
            self.logger.error(f"[RESULTS] Failed to compile results: {str(e)}")
            raise
            
    def run_complete_experiment(self):
        """Execute the complete experiment workflow"""
        self.logger.info("[EXPERIMENT] Starting complete experiment workflow...")
        
        try:
            # Validate environment
            self.validate_environment()
            
            # Phase 1: Training (or use existing weights)
            if self.skip_training:
                weights_path = self.weights_path
            else:
                weights_path = self.run_training_phase()
            
            # Phase 2: Validation
            self.run_validation_phase(weights_path)
            
            # Phase 3: Weather Testing (if not QuickTest)
            self.run_weather_testing_phase(weights_path)
            
            # Phase 4: Results Compilation
            final_results = self.compile_results()
            
            self.logger.info("[EXPERIMENT] Complete experiment workflow finished successfully")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"[EXPERIMENT] Experiment workflow failed: {str(e)}")
            
            # Still try to compile partial results
            try:
                self.compile_results()
            except:
                pass
                
            raise


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='YOLOv5n VisDrone Phase 1 Complete Experiment Runner')
    parser.add_argument('--QuickTest', action='store_true',
                       help='Run quick test: 5 epochs training + validation only (no weather testing)')
    parser.add_argument('--SkipTraining', action='store_true',
                       help='Skip training phase, use existing weights for validation and testing')
    parser.add_argument('--weights', type=str,
                       help='Path to existing weights (required if --SkipTraining)')
    
    args = parser.parse_args()
    
    # Validation
    if args.SkipTraining and not args.weights:
        print("Error: --SkipTraining requires --weights argument")
        sys.exit(1)
    
    try:
        runner = CompleteExperimentRunner(
            quick_test=args.QuickTest,
            skip_training=args.SkipTraining,
            weights_path=args.weights
        )
        
        results = runner.run_complete_experiment()
        
        print("\\n" + "="*80)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        mode = "Quick Test" if args.QuickTest else "Complete Experiment"
        print(f"Mode: {mode}")
        
        if args.SkipTraining:
            print(f"Used existing weights: {args.weights}")
        else:
            training_result = results['results'].get('training', {})
            if training_result.get('success'):
                duration = training_result.get('duration_hours', 0)
                print(f"Training: Completed in {duration:.2f} hours")
            
        validation_result = results['results'].get('validation', {})
        if validation_result.get('success'):
            print("Validation: Completed successfully")
            
        if not args.QuickTest:
            weather_result = results['results'].get('weather_testing', {})
            if weather_result.get('success'):
                print("Weather Testing: Completed successfully")
            elif not weather_result.get('skipped'):
                print("Weather Testing: Completed with issues")
        
        print(f"\\nResults saved in: {runner.logs_dir}")
        print("="*80)
        
    except Exception as e:
        print(f"\\nExperiment failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()