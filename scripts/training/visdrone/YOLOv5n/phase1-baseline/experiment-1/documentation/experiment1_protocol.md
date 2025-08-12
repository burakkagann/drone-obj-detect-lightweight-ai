# YOLOv5n VisDrone Phase 1 Baseline Experiment 1 Protocol

## Overview

This document defines the complete experimental protocol for Phase 1 baseline training of YOLOv5n on the VisDrone dataset. This experiment implements optimized configurations based on extensive recovery analysis from previous experimental runs.

## Experimental Objectives

### Primary Objectives
1. **Establish True Baseline Performance**: Train YOLOv5n without any data augmentation to establish clean baseline metrics
2. **Apply Optimized Configuration**: Implement learnings from recovery analysis to achieve best possible baseline performance
3. **Evaluate Weather Robustness**: Test baseline model against synthetic weather conditions (fog, rain, night, mixed)
4. **Document Performance Degradation**: Quantify performance loss under adverse conditions to justify Phase 2 augmentation

### Research Hypothesis
- **Baseline Performance**: YOLOv5n will achieve 17-25% mAP@0.5 on clean VisDrone test data
- **Weather Degradation**: 80-95% performance degradation expected under synthetic weather conditions
- **Optimization Impact**: SGD optimizer and optimized loss weights will improve performance over default configuration

## Methodology

### Model Configuration
- **Architecture**: YOLOv5n (nano variant for lightweight deployment)
- **Initialization**: Pre-trained COCO weights (yolov5n.pt)
- **Input Resolution**: 640x640 pixels
- **Batch Size**: 8 (fixed for training stability)

### Training Configuration (Optimized)

Based on recovery analysis findings, the following optimizations are applied:

#### Optimizer Settings
```yaml
optimizer: SGD                    # SGD outperformed AdamW in all tests
lr0: 0.01                        # Optimal initial learning rate
lrf: 0.01                        # Final learning rate factor
momentum: 0.937                  # Standard SGD momentum
weight_decay: 0.0005             # L2 regularization
```

#### Loss Function Weights
```yaml
cls: 0.3                         # Class loss weight (optimal ratio)
obj: 0.7                         # Object loss weight (optimal ratio)
box: 0.05                        # Box regression loss weight
```

#### Training Duration
```yaml
epochs: 50                       # Updated from 100 based on convergence analysis
patience: 300                    # Early stopping patience
```

#### Data Augmentation (Phase 1 Requirement)
```yaml
# ALL AUGMENTATION DISABLED FOR PHASE 1 BASELINE
hsv_h: 0.0          # HSV-Hue augmentation
hsv_s: 0.0          # HSV-Saturation augmentation  
hsv_v: 0.0          # HSV-Value augmentation
degrees: 0.0        # Rotation augmentation
translate: 0.0      # Translation augmentation
scale: 0.0          # Scale augmentation
shear: 0.0          # Shear augmentation
perspective: 0.0    # Perspective augmentation
flipud: 0.0         # Vertical flip augmentation
fliplr: 0.0         # Horizontal flip augmentation
mosaic: 0.0         # Mosaic augmentation
mixup: 0.0          # Mixup augmentation
```

### Dataset Configuration

#### VisDrone Dataset
- **Training Images**: 6,471 images with bounding box annotations
- **Validation Images**: 548 images for training validation
- **Test Images**: 1,610 images for final evaluation
- **Classes**: 10 object categories (pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor)
- **Annotation Format**: YOLO format (normalized coordinates)

#### Synthetic Test Datasets
- **Clean Baseline**: Original VisDrone test-dev dataset
- **Fog Synthetic**: Synthetic fog augmented test images
- **Rain Synthetic**: Synthetic rain augmented test images  
- **Night Synthetic**: Synthetic low-light/night test images
- **Mixed Conditions**: Combined weather effects test images

## Experimental Workflow

### Phase 1: Training
1. **Environment Setup**: Validate virtual environment and dependencies
2. **Configuration Validation**: Verify all paths and hyperparameters
3. **Training Execution**: Run optimized training for 50 epochs
4. **Checkpoint Management**: Save best and last model weights
5. **Training Analysis**: Log loss curves, learning rate schedule, and convergence

### Phase 2: Validation  
1. **Model Loading**: Load best weights from training
2. **Clean Dataset Testing**: Evaluate on original VisDrone test-dev
3. **Metrics Calculation**: Compute mAP@0.5, mAP@0.5:0.95, precision, recall
4. **Per-Class Analysis**: Generate detailed class-wise performance metrics
5. **Results Documentation**: Save comprehensive validation results

### Phase 3: Weather Conditions Testing
1. **Sequential Testing**: Test on each synthetic weather dataset
2. **Metrics Consistency**: Use identical evaluation protocol for all conditions
3. **Degradation Analysis**: Calculate performance drops relative to baseline
4. **Comparative Analysis**: Generate weather robustness comparison tables
5. **Results Compilation**: Document comprehensive weather testing results

### Phase 4: Results Analysis
1. **Performance Summary**: Compile all experimental results
2. **Baseline Establishment**: Confirm baseline performance metrics
3. **Degradation Quantification**: Document weather-induced performance loss
4. **Protocol Validation**: Verify adherence to Phase 1 requirements
5. **Phase 2 Justification**: Use results to justify augmentation strategies

## Success Criteria

### Training Success Criteria
- [ ] Training completes without errors for 50 epochs
- [ ] Model convergence achieved (loss stabilization)
- [ ] Best weights saved successfully
- [ ] Training logs generated completely
- [ ] No augmentation applied (protocol compliance)

### Validation Success Criteria  
- [ ] mAP@0.5 between 17-25% (based on recovery analysis)
- [ ] mAP@0.5:0.95 > 8.5% (stringent metric threshold)
- [ ] All 10 classes show measurable performance
- [ ] Validation results saved in JSON format
- [ ] Per-class metrics generated successfully

### Weather Testing Success Criteria
- [ ] All 5 conditions tested successfully (clean, fog, rain, night, mixed)
- [ ] Consistent evaluation metrics across all conditions
- [ ] 80-95% performance degradation demonstrated
- [ ] Degradation analysis completed and documented
- [ ] Comparative results table generated

### Overall Experiment Success Criteria
- [ ] Complete workflow executed end-to-end
- [ ] All phases completed successfully
- [ ] Results properly documented and saved
- [ ] Baseline performance established for Phase 2
- [ ] Weather robustness quantified

## Performance Expectations

### Based on Recovery Analysis

#### Training Performance
- **Convergence Epoch**: 35-40 epochs typically optimal
- **Best Performance Range**: 17-25% mAP@0.5
- **Exceptional Performance**: 24.66% mAP@0.5 (achieved in Run 1)
- **Typical Performance**: 17-19% mAP@0.5

#### Validation Performance  
- **Primary Metric**: mAP@0.5 = 18-25%
- **Stringent Metric**: mAP@0.5:0.95 = 8.5-12%
- **Precision**: 30-50%
- **Recall**: 20-25%

#### Weather Degradation Expectations
- **Fog Conditions**: 60-70% performance drop
- **Rain Conditions**: 20-30% performance drop  
- **Night Conditions**: 40-60% performance drop
- **Mixed Conditions**: 70-80% performance drop
- **Overall Range**: 80-95% degradation (matches literature)

## Risk Mitigation

### Technical Risks
1. **Training Failure**: Implement comprehensive error handling and checkpoint recovery
2. **Memory Issues**: Use fixed batch size and monitor GPU memory usage
3. **Path Resolution**: Use absolute paths and validate all file locations
4. **Dependency Issues**: Verify virtual environment and package versions

### Experimental Risks
1. **Poor Baseline Performance**: Quick test validation with 5 epochs to catch configuration issues early
2. **Weather Dataset Missing**: Validate all synthetic datasets before starting experiment
3. **Results Corruption**: Implement multiple backup strategies for results storage
4. **Protocol Deviation**: Automated validation of augmentation settings (must be 0.0)

## Quality Assurance

### Configuration Validation
- All augmentation parameters verified as 0.0
- SGD optimizer and loss weights confirmed
- Dataset paths validated before training
- Virtual environment dependencies checked

### Results Validation
- Multiple output formats (JSON, logs, checkpoints)
- Automated metric extraction and validation
- Cross-referencing between different result files
- Statistical sanity checks on performance metrics

### Documentation Standards
- Comprehensive logging at all stages
- Timestamped results for traceability
- Version control of all configuration files
- Reproducibility documentation

## Implementation Tools

### Scripts Overview
1. **train_phase1_experiment1.py**: Optimized training script with recovery analysis findings
2. **validate_phase1_experiment1.py**: Comprehensive validation with detailed metrics
3. **test_weather_conditions.py**: Sequential weather conditions testing
4. **run_complete_experiment1.py**: Master orchestration script for complete workflow

### Configuration Files
1. **experiment1_training_config.yaml**: Optimized hyperparameters based on recovery analysis
2. **visdrone_experiment1.yaml**: Dataset configuration with proper paths

### Quick Test Option
- **--QuickTest**: 5 epochs training + validation only
- **Purpose**: Rapid configuration validation before full experiment
- **Success Threshold**: >10% mAP@0.5 indicates proper configuration

## References

### Recovery Analysis Sources
- Phase 1 Log Analysis: Detailed analysis of 8 previous training runs
- Phase 1 Trial Analysis: Configuration optimization findings
- Best Performing Run: Run 1 achieving 24.66% mAP@0.5

### Protocol Sources  
- Training Phase Protocol: Multi-phase experimental design
- Training Protocol: Comprehensive YOLO training guidelines
- Methodology: Research methodology and evaluation framework

---

**Document Version**: 1.0  
**Last Updated**: August 12, 2025  
**Experiment ID**: phase1_baseline_experiment1  
**Status**: Implementation Ready