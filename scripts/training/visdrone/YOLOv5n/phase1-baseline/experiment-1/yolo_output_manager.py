#!/usr/bin/env python3
"""
YOLOv5 Output Manager - Standardized Live Output and Recording
============================================================

This utility provides a standardized way to run YOLOv5 commands with:
- Live terminal display
- Complete output recording  
- Result parsing
- YOLOv5-aligned subprocess execution

Used by validation, weather testing, and orchestrator scripts.
"""

import subprocess
import sys
import json
import logging
from pathlib import Path
from datetime import datetime


class YOLOv5OutputManager:
    """Standardized manager for YOLOv5 subprocess execution with live output"""
    
    def __init__(self, logger=None):
        self.logger = logger or self._create_default_logger()
    
    def _create_default_logger(self):
        """Create a default logger if none provided"""
        logger = logging.getLogger("YOLOv5OutputManager")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def run_yolo_command(self, cmd, cwd, prefix="[YOLO]", timeout=1800):
        """
        Run YOLOv5 command with live output display and recording
        
        Args:
            cmd: List of command arguments
            cwd: Working directory (YOLOv5 directory)
            prefix: Display prefix for output lines
            timeout: Command timeout in seconds (default: 30 minutes)
            
        Returns:
            YOLOResult object with returncode, stdout, stderr
        """
        self.logger.info(f"{prefix} Running YOLOv5 command: {' '.join(cmd)}")
        
        # Start process
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr with stdout
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        output_lines = []
        
        # Stream output in real-time while capturing it
        try:
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    line = output.strip()
                    if line:  # Only process non-empty lines
                        # Display to terminal
                        print(f"{prefix} {line}")
                        # Log to file
                        self.logger.info(f"{prefix} {line}")
                        # Store for parsing
                        output_lines.append(line)
        
        except Exception as e:
            self.logger.error(f"Error reading YOLOv5 output: {e}")
        
        # Wait for process to complete
        try:
            return_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.logger.error(f"{prefix} Command timed out after {timeout} seconds")
            process.kill()
            return_code = -1
        
        # Create result object
        result = YOLOResult(return_code, output_lines)
        
        # Log completion status
        if result.returncode == 0:
            self.logger.info(f"{prefix} Command completed successfully")
        else:
            self.logger.error(f"{prefix} Command failed with return code: {result.returncode}")
        
        return result
    
    def parse_yolo_metrics(self, results_dir, stdout_output, context=""):
        """
        Parse YOLOv5 validation metrics from output and files
        
        Args:
            results_dir: Directory where YOLOv5 saves results
            stdout_output: Standard output from YOLOv5 command
            context: Context string for logging (e.g., "validation", "fog_test")
            
        Returns:
            Dictionary with parsed metrics
        """
        self.logger.info(f"[PARSING] Parsing YOLOv5 {context} results...")
        
        metrics = {
            'map50': 0.0,
            'map50_95': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'context': context
        }
        
        try:
            # Parse metrics from stdout output
            if stdout_output:
                lines = stdout_output.split('\n')
                for line in lines:
                    line = line.strip()
                    # Look for the summary line with overall metrics
                    if 'all' in line and len(line.split()) >= 6:
                        parts = line.split()
                        try:
                            # YOLOv5 output format: Class Images Instances P R mAP50 mAP50-95
                            if len(parts) >= 7:
                                metrics['precision'] = float(parts[4])
                                metrics['recall'] = float(parts[5]) 
                                metrics['map50'] = float(parts[6])
                                if len(parts) >= 8:
                                    metrics['map50_95'] = float(parts[7])
                        except (ValueError, IndexError):
                            continue
            
            # Try to read results from JSON file if available
            if results_dir:
                results_json = Path(results_dir) / "results.json"
                if results_json.exists():
                    with open(results_json, 'r') as f:
                        json_data = json.load(f)
                        if isinstance(json_data, dict):
                            metrics.update({
                                'map50': json_data.get('metrics/mAP_0.5', metrics['map50']),
                                'map50_95': json_data.get('metrics/mAP_0.5:0.95', metrics['map50_95']),
                                'precision': json_data.get('metrics/precision', metrics['precision']),
                                'recall': json_data.get('metrics/recall', metrics['recall'])
                            })
        
        except Exception as e:
            self.logger.warning(f"[PARSING] Could not parse some {context} metrics: {e}")
        
        return metrics
    
    def build_yolo_val_command(self, weights, data, batch_size=4, imgsz=640, 
                              conf_thres=0.001, iou_thres=0.6, task='val', 
                              device='', workers=2, project=None, name=None,
                              save_txt=True, save_conf=True, save_json=True,
                              verbose=True, augment=False, half=False):
        """
        Build YOLOv5 validation command with standardized arguments
        
        Returns:
            List of command arguments ready for subprocess
        """
        cmd = [
            sys.executable, "val.py",
            "--weights", str(weights),
            "--data", str(data),
            "--batch-size", str(batch_size),
            "--imgsz", str(imgsz),
            "--conf-thres", str(conf_thres),
            "--iou-thres", str(iou_thres),
            "--task", task,
            "--device", device,
            "--workers", str(workers)
        ]
        
        # Add optional project and name
        if project:
            cmd.extend(["--project", str(project)])
        if name:
            cmd.extend(["--name", name])
        
        # Add output options
        if save_txt:
            cmd.append("--save-txt")
        if save_conf:
            cmd.append("--save-conf")
        if save_json:
            cmd.append("--save-json")
        
        # Add boolean flags
        if verbose:
            cmd.append("--verbose")
        if augment:
            cmd.append("--augment")
        if half:
            cmd.append("--half")
        
        # Always add exist-ok for convenience
        cmd.append("--exist-ok")
        
        return cmd


class YOLOResult:
    """Result object for YOLOv5 command execution"""
    
    def __init__(self, returncode, stdout_lines):
        self.returncode = returncode
        self.stdout = '\n'.join(stdout_lines)
        self.stderr = ""  # We merge stderr with stdout
        self.output_lines = stdout_lines
    
    @property
    def success(self):
        """Check if command was successful"""
        return self.returncode == 0
    
    def get_metric_line(self):
        """Extract the line containing 'all' metrics summary"""
        for line in self.output_lines:
            if 'all' in line and len(line.split()) >= 6:
                return line
        return None
    
    def __repr__(self):
        status = "SUCCESS" if self.success else f"FAILED ({self.returncode})"
        return f"YOLOResult(status={status}, output_lines={len(self.output_lines)})"