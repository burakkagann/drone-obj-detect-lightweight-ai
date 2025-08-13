#!/usr/bin/env python3
"""
Real-time Epoch Monitoring for YOLOv5n Training
===============================================

This script monitors training progress in real-time by watching the results.csv file
and logs metrics after each epoch to the metrics tracking log.

Usage:
    python epoch_monitor.py --phase 2a --name exp2_phase2a_quicktest
"""

import argparse
import time
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
from typing import Optional, Dict

# Project paths
SCRIPT_DIR = Path(__file__).parent
LOGS_DIR = SCRIPT_DIR / "logs&results"
TRAINING_DIR = LOGS_DIR / "training"


class EpochMonitor:
    """Monitor training progress epoch by epoch"""
    
    def __init__(self, phase: str, experiment_name: str, poll_interval: int = 10):
        """
        Initialize epoch monitor
        
        Args:
            phase: Training phase (2a, 2b, 2c)
            experiment_name: Name of the experiment directory
            poll_interval: Seconds between checking for updates
        """
        self.phase = phase.upper()
        self.experiment_name = experiment_name
        self.poll_interval = poll_interval
        
        self.results_file = TRAINING_DIR / experiment_name / "results.csv"
        self.last_epoch = -1
        self.setup_logging()
        
    def setup_logging(self):
        """Setup metrics logging"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Find or create metrics log
        metrics_logs = list(LOGS_DIR.glob("metrics_tracking_*.log"))
        if metrics_logs:
            # Use the most recent metrics log
            metrics_file = max(metrics_logs, key=lambda f: f.stat().st_mtime)
        else:
            metrics_file = LOGS_DIR / f"metrics_tracking_{timestamp}.log"
            
        # Setup logger
        self.logger = logging.getLogger('epoch_metrics')
        handler = logging.FileHandler(metrics_file, mode='a')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        # Also log to console
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(console)
        
    def extract_epoch_metrics(self, df: pd.DataFrame, epoch: int) -> Dict:
        """Extract metrics for a specific epoch"""
        if epoch >= len(df):
            return None
            
        row = df.iloc[epoch]
        metrics = {}
        
        # Clean column names and extract values
        for col in df.columns:
            col_clean = col.strip().lower()
            
            if col_clean == 'epoch':
                metrics['epoch'] = int(row[col])
            elif 'map_0.5' in col_clean and 'map_0.5:0.95' not in col_clean:
                metrics['mAP_0.5'] = float(row[col])
            elif 'map_0.5:0.95' in col_clean:
                metrics['mAP_0.5:0.95'] = float(row[col])
            elif 'precision' in col_clean and 'metrics' in col_clean:
                metrics['precision'] = float(row[col])
            elif 'recall' in col_clean and 'metrics' in col_clean:
                metrics['recall'] = float(row[col])
            elif 'train/box_loss' in col_clean:
                metrics['box_loss'] = float(row[col])
            elif 'train/obj_loss' in col_clean:
                metrics['obj_loss'] = float(row[col])
            elif 'train/cls_loss' in col_clean:
                metrics['cls_loss'] = float(row[col])
            elif 'val/box_loss' in col_clean:
                metrics['val_box_loss'] = float(row[col])
            elif 'lr0' in col_clean:
                metrics['lr'] = float(row[col])
                
        return metrics
        
    def log_epoch_metrics(self, metrics: Dict):
        """Log metrics for an epoch"""
        # Format epoch metrics message
        msg = (
            f"EPOCH Phase={self.phase} "
            f"Epoch={metrics.get('epoch', 'N/A')} "
            f"mAP@0.5={metrics.get('mAP_0.5', 0):.4f} "
            f"Precision={metrics.get('precision', 0):.4f} "
            f"Recall={metrics.get('recall', 0):.4f} "
            f"BoxLoss={metrics.get('box_loss', 0):.4f} "
            f"ObjLoss={metrics.get('obj_loss', 0):.4f} "
            f"ClsLoss={metrics.get('cls_loss', 0):.4f} "
            f"LR={metrics.get('lr', 0):.6f}"
        )
        
        self.logger.info(msg)
        
        # Check for improvements
        if self.last_epoch >= 0 and 'mAP_0.5' in metrics:
            if hasattr(self, 'last_map'):
                improvement = metrics['mAP_0.5'] - self.last_map
                if abs(improvement) > 0.001:
                    sign = "↑" if improvement > 0 else "↓"
                    self.logger.info(
                        f"  {sign} mAP@0.5 change: {improvement:+.4f} "
                        f"({(improvement/self.last_map)*100:+.1f}%)"
                    )
            self.last_map = metrics['mAP_0.5']
            
    def monitor(self):
        """Main monitoring loop"""
        self.logger.info(f"Starting epoch monitoring for {self.experiment_name}")
        self.logger.info(f"Watching: {self.results_file}")
        
        # Wait for results file to be created
        wait_count = 0
        while not self.results_file.exists():
            if wait_count == 0:
                print(f"Waiting for {self.results_file} to be created...")
            time.sleep(self.poll_interval)
            wait_count += 1
            if wait_count > 60:  # Timeout after 10 minutes
                self.logger.error("Timeout waiting for results file")
                return
                
        self.logger.info("Results file found, starting monitoring...")
        
        # Monitor for changes
        while True:
            try:
                # Read current results
                df = pd.read_csv(self.results_file)
                
                if len(df) > self.last_epoch + 1:
                    # New epochs detected
                    for epoch_idx in range(self.last_epoch + 1, len(df)):
                        metrics = self.extract_epoch_metrics(df, epoch_idx)
                        if metrics:
                            self.log_epoch_metrics(metrics)
                            self.last_epoch = epoch_idx
                            
                    # Check if training is complete
                    # This is a heuristic - adjust based on your training setup
                    if self.phase == '2A' and len(df) >= 30:
                        self.logger.info(f"Phase {self.phase} training appears complete")
                        break
                    elif self.phase == '2B' and len(df) >= 40:
                        self.logger.info(f"Phase {self.phase} training appears complete")
                        break
                    elif self.phase == '2C' and len(df) >= 30:
                        self.logger.info(f"Phase {self.phase} training appears complete")
                        break
                        
            except Exception as e:
                # File might be in use, wait and retry
                pass
                
            time.sleep(self.poll_interval)
            
        self.logger.info(f"Monitoring complete for {self.experiment_name}")
        

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Monitor YOLOv5n training epochs in real-time')
    parser.add_argument('--phase', type=str, required=True, 
                       choices=['2a', '2b', '2c'],
                       help='Training phase')
    parser.add_argument('--name', type=str, required=True,
                       help='Experiment directory name (e.g., exp2_phase2a_quicktest)')
    parser.add_argument('--interval', type=int, default=10,
                       help='Polling interval in seconds (default: 10)')
    
    args = parser.parse_args()
    
    # Create and run monitor
    monitor = EpochMonitor(
        phase=args.phase,
        experiment_name=args.name,
        poll_interval=args.interval
    )
    
    try:
        monitor.monitor()
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
        

if __name__ == "__main__":
    main()