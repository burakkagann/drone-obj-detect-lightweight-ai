#!/usr/bin/env python3
"""
YOLOv5n VisDrone Phase 1 Configuration Validator - Experiment 1
===============================================================

This script performs comprehensive validation of the experimental setup before
training execution. It verifies all paths, configurations, dependencies, and
environment settings to prevent runtime failures.

Usage:
    python validate_config.py [--verbose]
    
    --verbose: Enable detailed validation output
"""

import argparse
import os
import sys
import json
import yaml
import logging
from datetime import datetime
from pathlib import Path
import subprocess


class ConfigurationValidator:
    """Comprehensive configuration and environment validator"""
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.setup_paths()
        self.setup_logging()
        self.validation_results = {}
        self.errors = []
        self.warnings = []
        
    def setup_paths(self):
        """Setup all required paths"""
        self.script_dir = Path(__file__).parent
        self.project_root = self.script_dir.parent.parent.parent.parent.parent.parent
        self.yolo_path = self.project_root / "models" / "yolov5n" / "baseline" / "yolov5"
        self.config_dir = self.project_root / "configs" / "yolov5n-visdrone"
        self.data_root = self.project_root / "data"
        
        print(f"[SETUP] Configuration validator directory: {self.script_dir}")
        print(f"[SETUP] Project root: {self.project_root}")
        print(f"[SETUP] YOLOv5 path: {self.yolo_path}")
        print(f"[SETUP] Config directory: {self.config_dir}")
        
    def setup_logging(self):
        """Setup logging system"""
        log_level = logging.DEBUG if self.verbose else logging.INFO
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("="*70)
        self.logger.info("YOLOv5n VisDrone Phase 1 Configuration Validator")
        self.logger.info("="*70)
        
    def log_result(self, category, item, status, message="", details=None):
        """Log validation result"""
        if category not in self.validation_results:
            self.validation_results[category] = []
            
        result = {
            'item': item,
            'status': status,
            'message': message,
            'details': details or {}
        }
        
        self.validation_results[category].append(result)
        
        # Log to console
        status_symbol = "[OK]" if status == "PASS" else "[WARNING]" if status == "WARNING" else "[ERROR]"
        log_message = f"{status_symbol} {category}: {item}"
        if message:
            log_message += f" - {message}"
            
        if status == "PASS":
            self.logger.info(log_message)
        elif status == "WARNING":
            self.logger.warning(log_message)
            self.warnings.append(f"{category}: {item} - {message}")
        else:  # FAIL
            self.logger.error(log_message)
            self.errors.append(f"{category}: {item} - {message}")
            
    def validate_python_environment(self):
        """Validate Python environment and dependencies"""
        self.logger.info("[VALIDATION] Checking Python environment...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version.major == 3 and python_version.minor >= 8:
            self.log_result("Environment", "Python Version", "PASS", 
                          f"Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        else:
            self.log_result("Environment", "Python Version", "FAIL",
                          f"Python {python_version.major}.{python_version.minor} - requires Python 3.8+")
        
        # Check required packages
        required_packages = {
            'torch': '1.0.0',
            'torchvision': '0.2.0', 
            'numpy': '1.18.0',
            'opencv-python': '4.0.0',
            'PyYAML': '5.0.0',
            'pillow': '8.0.0'
        }
        
        for package, min_version in required_packages.items():
            try:
                if package == 'opencv-python':
                    import cv2
                    version = cv2.__version__
                    package_name = 'opencv-python'
                elif package == 'PyYAML':
                    import yaml
                    version = getattr(yaml, '__version__', 'unknown')
                    package_name = 'PyYAML'
                elif package == 'pillow':
                    from PIL import Image
                    version = getattr(Image, '__version__', 'unknown')
                    package_name = 'Pillow'
                else:
                    module = __import__(package)
                    version = getattr(module, '__version__', 'unknown')
                    package_name = package
                    
                self.log_result("Dependencies", package_name, "PASS", f"Version: {version}")
                
            except ImportError:
                self.log_result("Dependencies", package_name, "FAIL", "Package not installed")
                
        # Check CUDA availability
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                self.log_result("Hardware", "CUDA", "PASS", 
                              f"{gpu_count} GPU(s) - {gpu_name} ({gpu_memory:.1f}GB)")
            else:
                self.log_result("Hardware", "CUDA", "WARNING", 
                              "CUDA not available - will use CPU (slower training)")
        except Exception as e:
            self.log_result("Hardware", "CUDA", "FAIL", f"Error checking CUDA: {str(e)}")
            
    def validate_file_structure(self):
        """Validate required file and directory structure"""
        self.logger.info("[VALIDATION] Checking file structure...")
        
        # Check core directories
        core_dirs = {
            'Project Root': self.project_root,
            'YOLOv5 Directory': self.yolo_path,
            'Config Directory': self.config_dir,
            'Data Root': self.data_root
        }
        
        for name, path in core_dirs.items():
            if path.exists() and path.is_dir():
                self.log_result("File Structure", name, "PASS", str(path))
            else:
                self.log_result("File Structure", name, "FAIL", f"Directory not found: {path}")
        
        # Check required scripts
        required_scripts = {
            'Training Script': self.script_dir / "train_phase1_experiment1.py",
            'Validation Script': self.script_dir / "validate_phase1_experiment1.py",
            'Weather Testing Script': self.script_dir / "test_weather_conditions.py",
            'Master Script': self.script_dir / "run_complete_experiment1.py"
        }
        
        for name, path in required_scripts.items():
            if path.exists() and path.is_file():
                self.log_result("Scripts", name, "PASS", str(path))
            else:
                self.log_result("Scripts", name, "FAIL", f"Script not found: {path}")
        
        # Check YOLOv5 core files
        yolo_files = {
            'YOLOv5 Train Script': self.yolo_path / "train.py",
            'YOLOv5 Validation Script': self.yolo_path / "val.py",
            'YOLOv5n Model Config': self.yolo_path / "models" / "yolov5n.yaml",
            'YOLOv5n Weights': self.yolo_path / "yolov5n.pt"
        }
        
        for name, path in yolo_files.items():
            if path.exists():
                self.log_result("YOLOv5 Files", name, "PASS", str(path))
            else:
                self.log_result("YOLOv5 Files", name, "FAIL", f"File not found: {path}")
                
    def validate_configuration_files(self):
        """Validate configuration files"""
        self.logger.info("[VALIDATION] Checking configuration files...")
        
        # Check config files exist
        config_files = {
            'Training Config': self.config_dir / "experiment1_training_config.yaml",
            'Dataset Config': self.config_dir / "visdrone_experiment1.yaml"
        }
        
        for name, path in config_files.items():
            if path.exists():
                try:
                    with open(path, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    self.log_result("Configuration", name, "PASS", 
                                  f"Valid YAML: {len(config)} parameters")
                    
                    # Validate specific configurations
                    if name == "Training Config":
                        self.validate_training_config(config, path)
                    elif name == "Dataset Config":
                        self.validate_dataset_config(config, path)
                        
                except yaml.YAMLError as e:
                    self.log_result("Configuration", name, "FAIL", f"Invalid YAML: {str(e)}")
                except Exception as e:
                    self.log_result("Configuration", name, "FAIL", f"Error reading file: {str(e)}")
            else:
                self.log_result("Configuration", name, "FAIL", f"File not found: {path}")
                
    def validate_training_config(self, config, config_path):
        """Validate training configuration specifics"""
        
        # Check Phase 1 augmentation compliance (all must be 0.0)
        augmentation_params = [
            'hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate', 'scale', 
            'shear', 'perspective', 'flipud', 'fliplr', 'mosaic', 'mixup'
        ]
        
        augmentation_disabled = True
        for param in augmentation_params:
            if param in config:
                if config[param] != 0.0:
                    augmentation_disabled = False
                    self.log_result("Phase 1 Compliance", f"Augmentation {param}", "FAIL",
                                  f"Value {config[param]} must be 0.0 for Phase 1 baseline")
                    
        if augmentation_disabled:
            self.log_result("Phase 1 Compliance", "Zero Augmentation", "PASS",
                          "All augmentation parameters correctly set to 0.0")
        
        # Check optimized settings from recovery analysis
        optimal_settings = {
            'optimizer': 'SGD',
            'lr0': 0.01,
            'cls': 0.3,
            'obj': 0.7,
            'epochs': 50,
            'batch_size': 8
        }
        
        for setting, expected_value in optimal_settings.items():
            if setting in config:
                if config[setting] == expected_value:
                    self.log_result("Optimization", f"{setting}", "PASS",
                                  f"Optimal value: {config[setting]}")
                else:
                    self.log_result("Optimization", f"{setting}", "WARNING",
                                  f"Value {config[setting]} differs from optimal {expected_value}")
            else:
                self.log_result("Optimization", f"{setting}", "WARNING",
                              f"Parameter not found in config")
                
    def validate_dataset_config(self, config, config_path):
        """Validate dataset configuration"""
        
        # Check required dataset fields
        required_fields = ['path', 'train', 'val', 'test', 'nc', 'names']
        for field in required_fields:
            if field in config:
                self.log_result("Dataset Config", f"{field}", "PASS", f"Present")
            else:
                self.log_result("Dataset Config", f"{field}", "FAIL", f"Missing required field")
        
        # Validate class information
        if 'nc' in config and 'names' in config:
            expected_classes = 10
            if config['nc'] == expected_classes:
                self.log_result("Dataset Config", "Class Count", "PASS", f"{config['nc']} classes")
            else:
                self.log_result("Dataset Config", "Class Count", "FAIL",
                              f"Expected {expected_classes}, got {config['nc']}")
                
            if len(config['names']) == config['nc']:
                self.log_result("Dataset Config", "Class Names", "PASS", 
                              f"{len(config['names'])} names match nc={config['nc']}")
            else:
                self.log_result("Dataset Config", "Class Names", "FAIL",
                              f"Names count {len(config['names'])} != nc={config['nc']}")
                
    def validate_dataset_paths(self):
        """Validate dataset paths and data availability"""
        self.logger.info("[VALIDATION] Checking dataset paths...")
        
        # Load dataset config
        try:
            config_path = self.config_dir / "visdrone_experiment1.yaml"
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            self.log_result("Dataset Paths", "Config Loading", "FAIL",
                          f"Cannot load dataset config: {str(e)}")
            return
        
        # Check data splits
        data_splits = ['train', 'val', 'test']
        for split in data_splits:
            if split in config:
                # Handle both absolute and relative paths
                config_path = config[split]
                if Path(config_path).is_absolute():
                    # Path is absolute, use as-is
                    split_path = Path(config_path)
                elif config_path.startswith('../'):
                    # Path is relative to config file, resolve from config directory
                    split_path = (self.config_dir / config_path).resolve()
                else:
                    # Path is relative to project root
                    split_path = self.project_root / config_path
                if split_path.exists():
                    # Count images
                    if split_path.is_dir():
                        image_files = list(split_path.glob("*.jpg")) + list(split_path.glob("*.png"))
                        self.log_result("Dataset Paths", f"{split.upper()} Images", "PASS",
                                      f"{len(image_files)} images found at {split_path}")
                    else:
                        self.log_result("Dataset Paths", f"{split.upper()} Path", "FAIL",
                                      f"Not a directory: {split_path}")
                else:
                    self.log_result("Dataset Paths", f"{split.upper()} Path", "FAIL",
                                  f"Path not found: {split_path}")
        
        # Check synthetic test datasets
        synthetic_root = self.data_root / "synthetic_test"
        if synthetic_root.exists():
            synthetic_datasets = [
                "VisDrone2019-DET-test-fog",
                "VisDrone2019-DET-test-rain", 
                "VisDrone2019-DET-test-night",
                "VisDrone2019-DET-test-mixed"
            ]
            
            for dataset in synthetic_datasets:
                dataset_path = synthetic_root / dataset
                if dataset_path.exists():
                    images_path = dataset_path / "images"
                    if images_path.exists():
                        image_count = len(list(images_path.glob("*.jpg")))
                        self.log_result("Synthetic Data", dataset, "PASS",
                                      f"{image_count} images found")
                    else:
                        self.log_result("Synthetic Data", dataset, "WARNING",
                                      f"Images directory not found")
                else:
                    self.log_result("Synthetic Data", dataset, "WARNING",
                                  f"Dataset not found: {dataset_path}")
        else:
            self.log_result("Synthetic Data", "Root Directory", "WARNING",
                          f"Synthetic test data not found: {synthetic_root}")
            
    def validate_permissions(self):
        """Validate file and directory permissions"""
        self.logger.info("[VALIDATION] Checking permissions...")
        
        # Check write permissions for logs directory
        logs_dir = self.script_dir / "logs&results"
        try:
            logs_dir.mkdir(parents=True, exist_ok=True)
            test_file = logs_dir / "permission_test.tmp"
            test_file.write_text("test")
            test_file.unlink()
            self.log_result("Permissions", "Logs Directory", "PASS",
                          f"Write access confirmed: {logs_dir}")
        except Exception as e:
            self.log_result("Permissions", "Logs Directory", "FAIL",
                          f"No write access: {str(e)}")
        
        # Check read access to YOLOv5 directory
        try:
            files = list(self.yolo_path.glob("*.py"))
            self.log_result("Permissions", "YOLOv5 Directory", "PASS",
                          f"Read access confirmed: {len(files)} Python files found")
        except Exception as e:
            self.log_result("Permissions", "YOLOv5 Directory", "FAIL",
                          f"No read access: {str(e)}")
            
    def validate_disk_space(self):
        """Validate available disk space"""
        self.logger.info("[VALIDATION] Checking disk space...")
        
        try:
            # Check available space (Windows and Unix compatible)
            if os.name == 'nt':  # Windows
                import shutil
                total, used, free = shutil.disk_usage(self.project_root)
            else:  # Unix/Linux
                statvfs = os.statvfs(self.project_root)
                free = statvfs.f_bavail * statvfs.f_frsize
                
            free_gb = free / (1024**3)
            
            # Recommend at least 10GB free for training
            if free_gb >= 10:
                self.log_result("Resources", "Disk Space", "PASS",
                              f"{free_gb:.1f}GB available")
            elif free_gb >= 5:
                self.log_result("Resources", "Disk Space", "WARNING",
                              f"{free_gb:.1f}GB available - recommend >10GB")
            else:
                self.log_result("Resources", "Disk Space", "FAIL",
                              f"{free_gb:.1f}GB available - insufficient for training")
                
        except Exception as e:
            self.log_result("Resources", "Disk Space", "WARNING",
                          f"Could not check disk space: {str(e)}")
            
    def run_comprehensive_validation(self):
        """Run all validation checks"""
        self.logger.info("[VALIDATION] Starting comprehensive validation...")
        
        try:
            # Run all validation modules
            self.validate_python_environment()
            self.validate_file_structure()
            self.validate_configuration_files()
            self.validate_dataset_paths()
            self.validate_permissions()
            self.validate_disk_space()
            
            # Generate validation summary
            self.generate_validation_summary()
            
            return len(self.errors) == 0
            
        except Exception as e:
            self.logger.error(f"[VALIDATION] Validation failed: {str(e)}")
            return False
            
    def generate_validation_summary(self):
        """Generate comprehensive validation summary"""
        self.logger.info("\n" + "="*70)
        self.logger.info("VALIDATION SUMMARY")
        self.logger.info("="*70)
        
        # Count results by status
        total_checks = 0
        passed_checks = 0
        warning_checks = 0
        failed_checks = 0
        
        for category, results in self.validation_results.items():
            for result in results:
                total_checks += 1
                if result['status'] == 'PASS':
                    passed_checks += 1
                elif result['status'] == 'WARNING':
                    warning_checks += 1
                else:
                    failed_checks += 1
        
        self.logger.info(f"Total Checks: {total_checks}")
        self.logger.info(f"[OK] Passed: {passed_checks}")
        self.logger.info(f"[WARNING] Warnings: {warning_checks}")
        self.logger.info(f"[ERROR] Failed: {failed_checks}")
        
        # Overall status
        if failed_checks == 0:
            if warning_checks == 0:
                status = "[SUCCESS] READY FOR TRAINING"
                self.logger.info(f"\n{status}")
                self.logger.info("All validation checks passed!")
            else:
                status = "[WARNING] READY WITH WARNINGS"
                self.logger.info(f"\n{status}")
                self.logger.info("Training can proceed but review warnings.")
        else:
            status = "[ERROR] NOT READY"
            self.logger.error(f"\n{status}")
            self.logger.error("Critical issues must be resolved before training.")
        
        # Show critical errors
        if self.errors:
            self.logger.error("\nCRITICAL ERRORS:")
            for error in self.errors:
                self.logger.error(f"  - {error}")
        
        # Show warnings
        if self.warnings:
            self.logger.warning("\nWARNINGS:")
            for warning in self.warnings:
                self.logger.warning(f"  - {warning}")
        
        # Save validation report
        self.save_validation_report(status)
        
    def save_validation_report(self, status):
        """Save detailed validation report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report = {
            "validation_timestamp": datetime.now().isoformat(),
            "overall_status": status,
            "summary": {
                "total_checks": sum(len(results) for results in self.validation_results.values()),
                "passed": sum(1 for results in self.validation_results.values() 
                            for result in results if result['status'] == 'PASS'),
                "warnings": len(self.warnings),
                "errors": len(self.errors)
            },
            "detailed_results": self.validation_results,
            "errors": self.errors,
            "warnings": self.warnings,
            "environment_info": {
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "platform": sys.platform,
                "working_directory": str(Path.cwd())
            }
        }
        
        # Save to logs directory
        logs_dir = self.script_dir / "logs&results"
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = logs_dir / f"validation_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        self.logger.info(f"\nValidation report saved: {report_file}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='YOLOv5n VisDrone Phase 1 Configuration Validator')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose validation output')
    
    args = parser.parse_args()
    
    try:
        validator = ConfigurationValidator(verbose=args.verbose)
        success = validator.run_comprehensive_validation()
        
        if success:
            print("\n[SUCCESS] Configuration validation completed successfully!")
            print("You can proceed with training.")
            sys.exit(0)
        else:
            print("\n[ERROR] Configuration validation failed!")
            print("Please resolve the issues above before training.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\nValidation failed with error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()