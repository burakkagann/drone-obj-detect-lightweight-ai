#!/usr/bin/env python3
"""
YOLOv5n VisDrone Phase 1 Baseline - Experiment 2 Complete Runner
================================================================

This script orchestrates the complete Experiment 2 pipeline:
1. Progressive training (3 phases)
2. Multi-weather validation after each phase
3. Final comprehensive weather testing
4. Results analysis and reporting

Usage:
    python run_experiment2.py --mode full  # Run complete experiment
    python run_experiment2.py --mode train  # Only training
    python run_experiment2.py --mode test  # Only testing (requires trained weights)
    python run_experiment2.py --mode analyze  # Analyze existing results
"""

import argparse
import os
import sys
import subprocess
import json
import time
import shutil
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

# Project paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.parent.parent.parent
LOGS_DIR = SCRIPT_DIR / "logs&results"


class Experiment2Runner:
    """Complete experiment runner for Experiment 2"""
    
    def __init__(self, mode: str = 'full', batch_size: int = 8, 
                 device: str = '0', skip_validation: bool = False,
                 quick_test: bool = False):
        """
        Initialize experiment runner
        
        Args:
            mode: Execution mode (full, train, test, analyze)
            batch_size: Batch size for training/testing
            device: CUDA device
            skip_validation: Skip validation after each phase
            quick_test: Run quick test with 5 total epochs
        """
        self.mode = mode
        self.batch_size = batch_size
        self.device = device
        self.skip_validation = skip_validation
        self.quick_test = quick_test
        
        self.setup_paths()
        self.setup_logging()
        self.experiment_start_time = time.time()
        
    def setup_paths(self):
        """Setup all required paths"""
        self.train_script = SCRIPT_DIR / "train_progressive.py"
        self.validate_script = SCRIPT_DIR / "validate_multi_weather.py"
        self.test_script = SCRIPT_DIR / "test_weather_conditions.py"
        
        self.training_dir = LOGS_DIR / "training"
        self.validation_dir = LOGS_DIR / "validation"
        self.testing_dir = LOGS_DIR / "weather_testing"
        
        # Create directories
        for dir_path in [self.training_dir, self.validation_dir, self.testing_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
    def setup_logging(self):
        """Setup comprehensive logging"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = LOGS_DIR / f"experiment2_runner_{timestamp}.log"
        
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
        self.logger.info("YOLOv5n VisDrone Phase 1 - Experiment 2 Runner")
        if self.quick_test:
            self.logger.info("*** QUICK TEST MODE - 5 EPOCHS TOTAL ***")
        self.logger.info("="*80)
        self.logger.info(f"Mode: {self.mode}")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Skip validation: {self.skip_validation}")
        self.logger.info(f"Quick test: {self.quick_test}")
        
    def run_training(self) -> bool:
        """Run progressive training for all phases"""
        self.logger.info("\n" + "="*80)
        self.logger.info("PHASE 1: PROGRESSIVE TRAINING")
        self.logger.info("="*80)
        
        cmd = [
            sys.executable,
            str(self.train_script),
            "--phase", "all",
            "--batch-size", str(self.batch_size),
            "--device", str(self.device)
        ]
        
        if self.quick_test:
            cmd.append("--quick-test")
        
        self.logger.info("Starting progressive training (3 phases)...")
        self.logger.info(f"Command: {' '.join(cmd)}")
        
        try:
            start_time = time.time()
            
            # Run training with real-time output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Stream output in real-time
            for line in iter(process.stdout.readline, ''):
                if line:
                    print(line.rstrip())  # Show in terminal only
                    # Note: subprocess handles its own logging, avoid duplication
                    
            process.wait()
            
            if process.returncode == 0:
                duration = (time.time() - start_time) / 3600
                self.logger.info(f"Training completed successfully in {duration:.2f} hours")
                
                # Run validation after training if not skipped
                if not self.skip_validation:
                    self.run_phase_validations()
                    
                return True
            else:
                self.logger.error(f"Training failed with return code: {process.returncode}")
                return False
                
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            return False
            
    def run_phase_validations(self):
        """Run validation for each training phase"""
        self.logger.info("\n" + "="*80)
        self.logger.info("PHASE 2: MULTI-WEATHER VALIDATION")
        self.logger.info("="*80)
        
        phases = ['2a', '2b', '2c']
        
        for phase in phases:
            self.logger.info(f"\nValidating Phase {phase.upper()} results...")
            
            cmd = [
                sys.executable,
                str(self.validate_script),
                "--phase", phase,
                "--batch-size", str(self.batch_size),
                "--conf-thres", "0.001",
                "--iou-thres", "0.6"
            ]
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                self.logger.info(f"Phase {phase.upper()} validation completed")
                
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Validation failed for phase {phase}: {str(e)}")
                
    def run_final_testing(self) -> bool:
        """Run comprehensive weather testing on final model"""
        self.logger.info("\n" + "="*80)
        self.logger.info("PHASE 3: COMPREHENSIVE WEATHER TESTING")
        self.logger.info("="*80)
        
        # Test the final phase (2c) model
        cmd = [
            sys.executable,
            str(self.test_script),
            "--phase", "2c",
            "--conf-thres", "0.001",
            "--iou-thres", "0.6"
        ]
        
        self.logger.info("Running comprehensive weather testing on final model...")
        self.logger.info(f"Command: {' '.join(cmd)}")
        
        try:
            start_time = time.time()
            
            # Run testing with real-time output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Stream output
            for line in iter(process.stdout.readline, ''):
                if line:
                    print(line.rstrip())  # Show in terminal only
                    # Note: subprocess handles its own logging, avoid duplication
                    
            process.wait()
            
            if process.returncode == 0:
                duration = (time.time() - start_time) / 60
                self.logger.info(f"Weather testing completed successfully in {duration:.2f} minutes")
                return True
            else:
                self.logger.error(f"Weather testing failed with return code: {process.returncode}")
                return False
                
        except Exception as e:
            self.logger.error(f"Weather testing failed: {str(e)}")
            return False
            
    def analyze_results(self):
        """Analyze and summarize all experiment results"""
        self.logger.info("\n" + "="*80)
        self.logger.info("PHASE 4: RESULTS ANALYSIS")
        self.logger.info("="*80)
        
        analysis = {
            "experiment": "phase1_baseline_experiment2",
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "mode": self.mode,
                "batch_size": self.batch_size,
                "device": self.device,
                "progressive_training": True,
                "phases": ["2A", "2B", "2C"],
                "total_epochs": 100
            },
            "results": {}
        }
        
        # Analyze training results
        self.logger.info("Analyzing training results...")
        training_results = self.analyze_training_results()
        if training_results:
            analysis["results"]["training"] = training_results
            
        # Analyze validation results  
        self.logger.info("Analyzing validation results...")
        validation_results = self.analyze_validation_results()
        if validation_results:
            analysis["results"]["validation"] = validation_results
            
        # Analyze weather testing results
        self.logger.info("Analyzing weather testing results...")
        weather_results = self.analyze_weather_results()
        if weather_results:
            analysis["results"]["weather_testing"] = weather_results
            
        # Generate comparative analysis with Experiment 1
        self.logger.info("Generating comparative analysis...")
        comparison = self.compare_with_experiment1()
        if comparison:
            analysis["comparison_with_exp1"] = comparison
            
        # Save complete analysis
        analysis_file = LOGS_DIR / f"experiment2_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
            
        self.logger.info(f"Analysis saved to: {analysis_file}")
        
        # Print summary
        self.print_summary(analysis)
        
    def analyze_training_results(self) -> Optional[Dict]:
        """Analyze training results from all phases"""
        results = {}
        
        phases = ['2a', '2b', '2c']
        phase_names = ['exp2_phase2a', 'exp2_phase2b', 'exp2_phase2c']
        
        for phase, phase_name in zip(phases, phase_names):
            results_csv = self.training_dir / phase_name / "results.csv"
            
            if results_csv.exists():
                try:
                    df = pd.read_csv(results_csv)
                    if len(df) > 0:
                        last_row = df.iloc[-1]
                        best_row = df.loc[df['   metrics/mAP_0.5'].idxmax()] if '   metrics/mAP_0.5' in df.columns else last_row
                        
                        results[phase] = {
                            "final_epoch": len(df),
                            "best_metrics": {
                                "mAP_0.5": float(best_row.get('   metrics/mAP_0.5', 0)),
                                "precision": float(best_row.get('metrics/precision', 0)),
                                "recall": float(best_row.get('metrics/recall', 0))
                            },
                            "final_losses": {
                                "box_loss": float(last_row.get('train/box_loss', 0)),
                                "obj_loss": float(last_row.get('train/obj_loss', 0)),
                                "cls_loss": float(last_row.get('train/cls_loss', 0))
                            }
                        }
                except Exception as e:
                    self.logger.warning(f"Could not analyze {phase} training results: {e}")
                    
        return results if results else None
        
    def analyze_validation_results(self) -> Optional[Dict]:
        """Analyze validation results"""
        results = {}
        
        # Find latest validation results
        validation_dirs = list(self.validation_dir.glob("multi_weather_*"))
        
        for val_dir in validation_dirs:
            summary_file = val_dir / "validation_summary.json"
            if summary_file.exists():
                try:
                    with open(summary_file, 'r') as f:
                        data = json.load(f)
                        phase = val_dir.name.split('_')[-1]  # Extract phase from dir name
                        results[phase] = {
                            "composite_score": data.get("composite_score", 0),
                            "conditions": data.get("conditions_tested", [])
                        }
                except Exception as e:
                    self.logger.warning(f"Could not read validation summary: {e}")
                    
        return results if results else None
        
    def analyze_weather_results(self) -> Optional[Dict]:
        """Analyze weather testing results"""
        # Find latest weather testing results
        weather_files = list(self.testing_dir.glob("weather_testing_complete_*.json"))
        
        if weather_files:
            latest_file = max(weather_files, key=lambda f: f.stat().st_mtime)
            
            try:
                with open(latest_file, 'r') as f:
                    data = json.load(f)
                    
                # Extract key metrics
                results = {
                    "baseline_map50": data.get("baseline_metrics", {}).get("map50", 0),
                    "weather_degradation": {}
                }
                
                for condition in ['fog', 'rain', 'night', 'mixed']:
                    if condition in data.get("weather_results", {}):
                        condition_data = data["weather_results"][condition]
                        if "degradation" in condition_data:
                            results["weather_degradation"][condition] = {
                                "map50": condition_data["metrics"].get("map50", 0),
                                "relative_drop": condition_data["degradation"]["map50"]["relative_drop_percent"]
                            }
                            
                return results
                
            except Exception as e:
                self.logger.warning(f"Could not analyze weather results: {e}")
                
        return None
        
    def compare_with_experiment1(self) -> Optional[Dict]:
        """Compare results with Experiment 1"""
        comparison = {
            "experiment1": {
                "clean_map50": 0.0736,
                "fog_map50": 0.0019,
                "rain_map50": 0.0028,
                "night_map50": 0.0023,
                "training_hours": 2.88
            },
            "experiment2": {},
            "improvements": {}
        }
        
        # Get Experiment 2 results
        weather_results = self.analyze_weather_results()
        if weather_results:
            exp2_baseline = weather_results.get("baseline_map50", 0)
            comparison["experiment2"]["clean_map50"] = exp2_baseline
            
            # Calculate improvements
            if exp2_baseline > 0:
                improvement = ((exp2_baseline - comparison["experiment1"]["clean_map50"]) / 
                             comparison["experiment1"]["clean_map50"]) * 100
                comparison["improvements"]["clean_map50_percent"] = improvement
                
        return comparison
        
    def print_summary(self, analysis: Dict):
        """Print experiment summary"""
        print("\n" + "="*80)
        print("EXPERIMENT 2 SUMMARY")
        print("="*80)
        
        # Training summary
        if "training" in analysis.get("results", {}):
            print("\nTraining Results:")
            for phase, metrics in analysis["results"]["training"].items():
                print(f"  Phase {phase.upper()}:")
                print(f"    Best mAP@0.5: {metrics['best_metrics']['mAP_0.5']:.4f}")
                print(f"    Precision: {metrics['best_metrics']['precision']:.4f}")
                print(f"    Recall: {metrics['best_metrics']['recall']:.4f}")
                
        # Weather testing summary
        if "weather_testing" in analysis.get("results", {}):
            weather = analysis["results"]["weather_testing"]
            print(f"\nWeather Testing Results:")
            print(f"  Baseline mAP@0.5: {weather.get('baseline_map50', 0):.4f}")
            
            if "weather_degradation" in weather:
                for condition, metrics in weather["weather_degradation"].items():
                    print(f"  {condition.capitalize()}: {metrics['map50']:.4f} "
                         f"({metrics['relative_drop']:.1f}% degradation)")
                    
        # Comparison with Experiment 1
        if "comparison_with_exp1" in analysis:
            comp = analysis["comparison_with_exp1"]
            if "improvements" in comp and "clean_map50_percent" in comp["improvements"]:
                improvement = comp["improvements"]["clean_map50_percent"]
                print(f"\nImprovement over Experiment 1:")
                print(f"  Clean mAP@0.5: {improvement:+.1f}%")
                
        print("="*80)
        
    def save_experiment_summary(self):
        """Save final experiment summary"""
        total_time = (time.time() - self.experiment_start_time) / 3600
        
        summary = {
            "experiment": "phase1_baseline_experiment2",
            "timestamp": datetime.now().isoformat(),
            "total_duration_hours": total_time,
            "status": "completed",
            "configuration": {
                "batch_size": self.batch_size,
                "device": self.device,
                "mode": self.mode,
                "progressive_training": True,
                "total_epochs": 100
            }
        }
        
        summary_file = LOGS_DIR / f"experiment2_final_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        self.logger.info(f"Experiment summary saved to: {summary_file}")
        
    def run(self):
        """Main execution method"""
        self.logger.info(f"Starting Experiment 2 in {self.mode} mode...")
        
        success = True
        
        if self.mode in ['full', 'train']:
            # Run training
            if not self.run_training():
                self.logger.error("Training failed")
                success = False
                
        if self.mode in ['full', 'test'] and success:
            # Run weather testing
            if not self.run_final_testing():
                self.logger.error("Weather testing failed")
                success = False
                
        if self.mode in ['full', 'analyze'] and success:
            # Analyze results
            self.analyze_results()
            
        # Save final summary
        self.save_experiment_summary()
        
        total_time = (time.time() - self.experiment_start_time) / 3600
        
        if success:
            self.logger.info("\n" + "="*80)
            self.logger.info(f"EXPERIMENT 2 COMPLETED SUCCESSFULLY!")
            self.logger.info(f"Total duration: {total_time:.2f} hours")
            self.logger.info("="*80)
        else:
            self.logger.error("\n" + "="*80)
            self.logger.error(f"EXPERIMENT 2 FAILED")
            self.logger.error(f"Duration before failure: {total_time:.2f} hours")
            self.logger.error("="*80)
            
        return success


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='YOLOv5n VisDrone Experiment 2 Runner')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['full', 'train', 'test', 'analyze'],
                       help='Execution mode (default: full)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for training/testing (default: 8)')
    parser.add_argument('--device', type=str, default='0',
                       help='CUDA device (default: 0)')
    parser.add_argument('--skip-validation', action='store_true',
                       help='Skip validation after each training phase')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with 5 total epochs (2+2+1)')
    
    args = parser.parse_args()
    
    # Create runner
    runner = Experiment2Runner(
        mode=args.mode,
        batch_size=args.batch_size,
        device=args.device,
        skip_validation=args.skip_validation,
        quick_test=args.quick_test
    )
    
    # Run experiment
    success = runner.run()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()