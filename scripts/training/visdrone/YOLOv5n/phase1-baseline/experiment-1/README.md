# YOLOv5n VisDrone Phase 1 Baseline Experiment 1

## Overview

This directory contains the complete implementation for **Phase 1 Baseline Experiment 1** of the YOLOv5n VisDrone object detection research. This experiment establishes a true baseline performance using optimized configurations derived from extensive recovery analysis.

## Quick Start

### Prerequisites
- Activate the virtual environment: `yolov5n-visdrone-venv`
- Ensure all dependencies are installed (see validation script)
- Verify dataset paths are correct

### Running the Complete Experiment

```bash
# Full experiment (50 epochs + validation + weather testing)
python run_complete_experiment1.py

# Quick test (5 epochs + validation only)
python run_complete_experiment1.py --QuickTest

# Skip training, use existing weights
python run_complete_experiment1.py --SkipTraining --weights path/to/best.pt
```

### Running Individual Components

```bash
# 1. Validate configuration first
python validate_config.py --verbose

# 2. Run training only
python train_phase1_experiment1.py

# 3. Run validation only
python validate_phase1_experiment1.py --weights path/to/best.pt

# 4. Run weather testing only
python test_weather_conditions.py --weights path/to/best.pt

# 5. Analyze results
python analyze_results.py --output-format json,md,csv
```

## Experiment Configuration

### Optimized Settings (Recovery Analysis Based)
- **Optimizer**: SGD (outperformed AdamW)
- **Learning Rate**: 0.01 (optimal from recovery analysis)
- **Loss Weights**: cls=0.3, obj=0.7 (optimal ratio)
- **Training Duration**: 50 epochs (updated from 100)
- **Batch Size**: 8 (fixed for stability)
- **Augmentation**: All disabled (Phase 1 requirement)

### Expected Performance
- **Baseline mAP@0.5**: 17-25% (target based on recovery analysis)
- **Best Performance**: 24.66% mAP@0.5 (achieved in previous Run 1)
- **Weather Degradation**: 80-95% performance loss expected

## Directory Structure

```
experiment-1/
├── README.md                           # This file
├── run_complete_experiment1.py         # Master orchestration script
├── train_phase1_experiment1.py         # Optimized training script
├── validate_phase1_experiment1.py      # Comprehensive validation
├── test_weather_conditions.py          # Weather robustness testing
├── validate_config.py                  # Configuration validator
├── analyze_results.py                  # Results analysis and reporting
├── documentation/
│   ├── experiment1_protocol.md         # Detailed experimental protocol
│   └── experiment1_results_template.md # Results reporting template
└── logs&results/
    ├── training/                       # Training logs and checkpoints
    ├── validation/                     # Validation results
    ├── weather_testing/                # Weather condition test results
    └── experiment_summary_*.json       # Overall experiment summaries
```

## Configuration Files

Located in `configs/yolov5n-visdrone/`:
- **experiment1_training_config.yaml**: Optimized hyperparameters
- **visdrone_experiment1.yaml**: Dataset configuration with proper paths

## Script Details

### 1. Master Script (`run_complete_experiment1.py`)
Orchestrates the complete experimental workflow:
- **Phase 1**: Training (50 epochs or 5 for QuickTest)
- **Phase 2**: Validation on clean dataset
- **Phase 3**: Weather conditions testing (if not QuickTest)
- **Phase 4**: Results compilation

**Options:**
- `--QuickTest`: 5 epochs + validation only
- `--SkipTraining`: Use existing weights
- `--weights`: Path to existing weights (required with SkipTraining)

### 2. Training Script (`train_phase1_experiment1.py`)
Optimized training implementation:
- Recovery analysis optimizations applied
- Comprehensive logging and validation
- Environment validation before training
- Automatic best weights detection

**Options:**
- `--QuickTest`: 5 epochs for configuration validation

### 3. Validation Script (`validate_phase1_experiment1.py`)
Comprehensive model validation:
- Standard YOLO metrics (mAP@0.5, mAP@0.5:0.95, precision, recall)
- Per-class performance analysis
- Detailed results documentation

**Options:**
- `--weights`: Path to model weights (required)
- `--conf-thres`: Confidence threshold (default: 0.001)
- `--iou-thres`: IoU threshold for NMS (default: 0.6)

### 4. Weather Testing Script (`test_weather_conditions.py`)
Sequential weather condition testing:
- Tests: Clean, Fog, Rain, Night, Mixed conditions
- Same metrics as validation for consistency
- Comprehensive degradation analysis

**Options:**
- `--weights`: Path to model weights (required)
- `--conf-thres`: Confidence threshold (default: 0.001)
- `--iou-thres`: IoU threshold for NMS (default: 0.6)

### 5. Configuration Validator (`validate_config.py`)
Pre-training environment validation:
- Python environment and dependencies
- File structure and paths
- Configuration file validation
- Dataset availability
- Permissions and disk space

**Options:**
- `--verbose`: Detailed validation output

### 6. Results Analyzer (`analyze_results.py`)
Comprehensive results analysis:
- Training convergence analysis
- Validation metrics comparison
- Weather degradation patterns
- Performance visualizations
- Multiple output formats

**Options:**
- `--experiment-dir`: Results directory (default: logs&results)
- `--output-format`: json,md,csv (default: json,md)

## Key Features

### Optimization Implementation
✅ **SGD Optimizer**: Applied based on recovery analysis findings  
✅ **Optimal Loss Weights**: cls=0.3, obj=0.7 ratio implemented  
✅ **Learning Rate**: 0.01 initial rate (proven optimal)  
✅ **Training Duration**: 50 epochs (updated from 100)  
✅ **Zero Augmentation**: Strict Phase 1 compliance  

### Comprehensive Logging
- Timestamped logs for all operations
- Debug information and validation details
- GPU, memory, and performance monitoring
- Error handling with detailed diagnostics

### Robust Error Handling
- Environment validation before execution
- Comprehensive path checking
- Dependency verification
- Graceful failure handling with detailed logs

### Results Documentation
- JSON format for machine processing
- Markdown reports for human readability
- CSV exports for statistical analysis
- Visualization plots for presentations

## Troubleshooting

### Common Issues

1. **Path Not Found Errors**
   ```bash
   python validate_config.py --verbose
   ```

2. **CUDA/GPU Issues**
   - Check GPU availability in logs
   - Training will fall back to CPU if needed

3. **Memory Issues**
   - Batch size is fixed at 8 for stability
   - Reduce workers if needed

4. **Permission Errors**
   - Ensure write access to logs&results directory
   - Check dataset read permissions

### Validation Checklist

Before running experiments:
- [ ] Virtual environment activated
- [ ] All dependencies installed
- [ ] Dataset paths accessible
- [ ] GPU available (recommended)
- [ ] Sufficient disk space (>10GB)
- [ ] Configuration files valid

## Expected Outputs

### Training Results
- Best model weights (`best.pt`)
- Training logs and metrics
- Loss curves and convergence analysis
- Hyperparameter tracking

### Validation Results
- Performance metrics (mAP@0.5, mAP@0.5:0.95, precision, recall)
- Per-class analysis
- Confidence threshold optimization
- Detailed result summaries

### Weather Testing Results
- Performance on 5 weather conditions
- Degradation analysis vs baseline
- Robustness metrics
- Comparative performance tables

### Analysis Outputs
- Comprehensive performance summary
- Learning curve visualizations
- Weather degradation plots
- Statistical analysis reports

## Recovery Analysis Integration

This experiment implements learnings from comprehensive recovery analysis:

- **Configuration**: Based on best-performing Run 1 (24.66% mAP@0.5)
- **Optimizer**: SGD confirmed superior to AdamW
- **Loss Weights**: cls=0.3, obj=0.7 ratio validated
- **Duration**: 50 epochs based on convergence analysis
- **Performance Range**: 17-25% mAP@0.5 target established

## Protocol Compliance

### Phase 1 Requirements
- ✅ Zero augmentation (all parameters = 0.0)
- ✅ Baseline establishment on clean dataset
- ✅ Weather robustness evaluation
- ✅ Comprehensive documentation

### Success Criteria
- Training completion without errors
- mAP@0.5 between 17-25%
- Weather degradation 80-95%
- Complete results documentation

---

**Experiment ID**: phase1_baseline_experiment1  
**Model**: YOLOv5n  
**Dataset**: VisDrone  
**Phase**: Phase 1 - True Baseline  
**Status**: Implementation Complete