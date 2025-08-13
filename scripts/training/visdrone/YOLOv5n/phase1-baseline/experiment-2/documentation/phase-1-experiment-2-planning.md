YOLOv5n VisDrone Phase 1 - Experiment 2 Design Document

  1. Executive Summary

  This document outlines the comprehensive design for Experiment 2 of the YOLOv5n VisDrone Phase 1
  baseline study. Based on the catastrophic failure of Experiment 1 (7.36% mAP on clean data, 96-97%
  degradation in weather conditions), Experiment 2 implements critical changes while maintaining
  Phase 1 constraints.

  Key Changes:
  - Reduced learning rate (0.01 → 0.001) with cosine annealing
  - Extended training (50 → 100 epochs) with progressive strategy
  - Optimized hyperparameters for VisDrone dataset characteristics
  - Multi-scale training for small object detection
  - Weather-aware validation strategy

  2. Experiment 1 Failure Analysis

  Performance Metrics

  | Metric             | Experiment 1 Result | Target for Experiment 2 |
  |--------------------|---------------------|-------------------------|
  | Clean mAP@0.5      | 7.36%               | 25-30%                  |
  | Fog mAP@0.5        | 0.19%               | 8-10%                   |
  | Rain mAP@0.5       | 0.28%               | 10-12%                  |
  | Night mAP@0.5      | 0.23%               | 8-10%                   |
  | Training Stability | Peaked at epoch 20  | Stable progression      |

  Root Causes Identified

  1. Learning rate too high (0.01): Caused optimization instability
  2. No augmentation: Zero robustness to variations (Phase 1 constraint)
  3. Insufficient epochs: Model undertrained at 50 epochs
  4. Loss imbalance: cls=0.3, obj=0.7 underweighted classification
  5. Fixed image size: Poor multi-scale object handling

  3. Experiment 2 Configuration

  3.1 Core Training Parameters

  # Optimizer Configuration (APPROVED)
  optimizer: SGD  # Maintained per requirements
  lr0: 0.001  # Reduced from 0.01 (10x reduction)
  lrf: 0.01  # Final LR = lr0 * lrf = 0.00001
  momentum: 0.937
  weight_decay: 0.0005
  cos_lr: True  # Enable cosine learning rate schedule

  # Training Duration
  epochs: 100  # Extended from 50
  patience: 30  # Early stopping patience

  # Batch Configuration
  batch_size: 16  # Increased from 8 (if memory allows, else stay at 8)
  workers: 8

  Justification:
  - LR reduction: Experiment 1 showed validation metrics peaked early then degraded, indicating
  overshooting
  - Cosine scheduler: Smooth decay prevents sudden performance drops
  - Extended epochs: More time for convergence with lower LR
  - Batch size increase: Better gradient estimates, especially important with SGD

  3.2 Loss Weights Adjustment

  # Loss Weights (APPROVED)
  cls: 0.5  # Increased from 0.3
  obj: 0.5  # Decreased from 0.7
  box: 0.05  # Standard
  anchor_t: 4.0  # Anchor threshold

  Justification:
  - Balanced cls/obj weights for better classification performance
  - Current 0.3/0.7 split biases toward objectness over classification
  - Critical for distinguishing between similar classes (e.g., car vs van)

  3.3 Multi-Scale Training (APPROVED)

  # Image Size Configuration
  imgsz: 640  # Base size
  multiscale: True  # Enable multi-scale training
  scale: 0.5  # Random scaling ±50%
  # Effective training range: 320-960 pixels

  Justification:
  - VisDrone contains extreme scale variations (10px to 500px objects)
  - Multi-scale training improves robustness to different object sizes
  - Particularly important for small objects (pedestrians, bicycles)

  3.4 Data Augmentation Settings

  # Augmentation Parameters (Phase 1 Constraint: DISABLED)
  hsv_h: 0.0  # Hue variation
  hsv_s: 0.0  # Saturation variation
  hsv_v: 0.0  # Value variation
  degrees: 0.0  # Rotation
  translate: 0.0  # Translation
  scale: 0.0  # Scale (separate from multi-scale training)
  shear: 0.0  # Shear
  perspective: 0.0  # Perspective
  flipud: 0.0  # Vertical flip
  fliplr: 0.0  # Horizontal flip
  mosaic: 0.0  # Mosaic augmentation
  mixup: 0.0  # Mixup augmentation
  copy_paste: 0.0  # Copy-paste augmentation

  Note: Augmentation remains disabled per Phase 1 requirements. This will be enabled in Phase 2.

  4. Progressive Training Strategy (APPROVED - Option A)

  4.1 Implementation Approach

  Three-Phase Manual Checkpoint & Resume Strategy:

  # Phase 2A: Foundation (Epochs 1-30)
  python train.py \
      --epochs 30 \
      --cfg configs/yolov5n.yaml \
      --hyp configs/exp2_phase2a_hyp.yaml \
      --data configs/visdrone_experiment2.yaml \
      --batch-size 16 \
      --name exp2_phase2a

  # Phase 2B: Optimization (Epochs 31-70)
  python train.py \
      --epochs 40 \
      --cfg configs/yolov5n.yaml \
      --hyp configs/exp2_phase2b_hyp.yaml \
      --data configs/visdrone_experiment2.yaml \
      --resume exp2_phase2a/weights/last.pt \
      --batch-size 16 \
      --name exp2_phase2b

  # Phase 2C: Fine-tuning (Epochs 71-100)
  python train.py \
      --epochs 30 \
      --cfg configs/yolov5n.yaml \
      --hyp configs/exp2_phase2c_hyp.yaml \
      --data configs/visdrone_experiment2.yaml \
      --resume exp2_phase2b/weights/last.pt \
      --batch-size 16 \
      --name exp2_phase2c

  4.2 Phase-Specific Configurations

  Phase 2A (Epochs 1-30): Foundation Building
  lr0: 0.001  # Initial learning rate
  cos_lr: False  # Linear decay in first phase
  warmup_epochs: 5  # Warmup for stability
  - Goal: Establish stable feature extraction
  - Focus: Basic pattern recognition
  - Success Metric: >15% mAP on clean data

  Phase 2B (Epochs 31-70): Optimization
  lr0: 0.0005  # Reduced LR
  cos_lr: True  # Enable cosine schedule
  warmup_epochs: 0  # No warmup needed
  - Goal: Refine detection accuracy
  - Focus: Small object detection improvement
  - Success Metric: >20% mAP on clean data

  Phase 2C (Epochs 71-100): Fine-tuning
  lr0: 0.0001  # Fine-tuning LR
  cos_lr: True  # Continue cosine schedule
  label_smoothing: 0.1  # Add label smoothing
  - Goal: Final optimization
  - Focus: Reducing false positives
  - Success Metric: >25% mAP on clean data

  5. Validation Strategy (APPROVED)

  5.1 Multi-Condition Validation

  # Validation Configuration
  val_conditions = {
      'clean': {
          'path': 'data/raw/visdrone/val',
          'weight': 0.4,
          'frequency': 'every_epoch'
      },
      'fog': {
          'path': 'data/synthetic_test/val_fog',
          'weight': 0.2,
          'frequency': 'every_5_epochs'
      },
      'rain': {
          'path': 'data/synthetic_test/val_rain',
          'weight': 0.2,
          'frequency': 'every_5_epochs'
      },
      'night': {
          'path': 'data/synthetic_test/val_night',
          'weight': 0.2,
          'frequency': 'every_5_epochs'
      }
  }

  # Composite metric for model selection
  composite_score = (0.4 * clean_mAP +
                    0.2 * fog_mAP +
                    0.2 * rain_mAP +
                    0.2 * night_mAP)

  Justification:
  - Track robustness throughout training
  - Prevent overfitting to clean data only
  - Early warning for weather performance degradation

  5.2 Checkpointing Strategy

  save_period: 5  # Save checkpoint every 5 epochs
  save_best: True  # Save best model based on composite score
  save_last: True  # Always keep last checkpoint
  keep_checkpoints: 3  # Keep last 3 checkpoints

  6. Additional Optimizations (APPROVED)

  6.1 Native YOLOv5 Features to Enable

  # Advanced Training Features
  rect: True  # Rectangular training for efficiency
  cache: 'ram'  # Cache images in RAM if available
  image_weights: False  # Don't use image weighting (Phase 1)
  multi_scale: True  # Already configured above
  single_cls: False  # Multi-class detection

  # Inference Optimizations (for validation)
  conf_thres: 0.001  # Low threshold for validation
  iou_thres: 0.6  # NMS IoU threshold
  max_det: 300  # Maximum detections per image

  6.2 Hardware Optimization

  # GPU Configuration
  device: 0  # Single GPU
  amp: True  # Automatic mixed precision
  cudnn_benchmark: True  # Enable CuDNN autotuner

  # Memory Management
  batch_size: 16  # Or 8 if OOM occurs
  workers: 8  # Data loading workers
  pin_memory: True  # Pin memory for faster transfer

  7. Monitoring and Metrics

  7.1 Primary Metrics to Track

  metrics = {
      'training': [
          'box_loss',
          'obj_loss',
          'cls_loss',
          'total_loss'
      ],
      'validation': [
          'mAP@0.5',
          'mAP@0.5:0.95',
          'precision',
          'recall',
          'F1-score'
      ],
      'per_class': [
          'AP_per_class',
          'confusion_matrix',
          'class_wise_precision_recall'
      ],
      'weather_specific': [
          'clean_mAP',
          'fog_mAP',
          'rain_mAP',
          'night_mAP',
          'degradation_percentage'
      ]
  }

  7.2 Early Stopping Criteria

  early_stopping = {
      'monitor': 'composite_mAP',
      'patience': 30,
      'min_delta': 0.001,
      'mode': 'max'
  }

  # Additional termination conditions
  if epoch > 30 and clean_mAP < 0.10:
      print("Terminating: No learning progress")
  if weather_avg_mAP < 0.01 and epoch > 50:
      print("Terminating: No weather robustness")

  8. Expected Outcomes

  8.1 Conservative Estimates

  | Metric        | Experiment 1 | Experiment 2 Target | Justification          |
  |---------------|--------------|---------------------|------------------------|
  | Clean mAP@0.5 | 7.36%        | 20-25%              | Better optimization    |
  | Fog mAP@0.5   | 0.19%        | 5-7%                | Improved base features |
  | Rain mAP@0.5  | 0.28%        | 6-8%                | Better generalization  |
  | Night mAP@0.5 | 0.23%        | 5-7%                | Stronger detection     |
  | Training Time | 2.88 hrs     | ~6 hrs              | 2x epochs              |

  8.2 Best Case Scenario

  - Clean mAP@0.5: 30-35%
  - Weather average: 10-12%
  - Small object detection: 15% improvement
  - False positive rate: 30% reduction

  9. Risk Mitigation

  9.1 Potential Risks and Mitigations

  | Risk                             | Probability | Impact | Mitigation                     |
  |----------------------------------|-------------|--------|--------------------------------|
  | OOM with batch_size=16           | Medium      | Low    | Fall back to batch_size=8      |
  | No improvement after 30 epochs   | Low         | High   | Adjust LR, check data pipeline |
  | Overfitting without augmentation | High        | Medium | Strong regularization, dropout |
  | Weather performance still <1%    | Medium      | High   | Consider Phase 2 early start   |

  9.2 Fallback Options

  1. If mAP < 15% at epoch 30:
    - Reduce LR to 0.0005
    - Increase warmup to 10 epochs
  2. If weather mAP < 3% at epoch 50:
    - Document Phase 1 limitations
    - Prepare aggressive Phase 2 plan
  3. If training unstable:
    - Switch to AdamW optimizer (if approved)
    - Implement gradient clipping

  10. Implementation Timeline

  | Phase                 | Epochs | Duration   | Milestone          |
  |-----------------------|--------|------------|--------------------|
  | Setup & Config        | -      | 1 hour     | Scripts ready      |
  | Phase 2A              | 1-30   | 2 hours    | >15% clean mAP     |
  | Checkpoint & Analysis | -      | 30 min     | Evaluate Phase 2A  |
  | Phase 2B              | 31-70  | 2.5 hours  | >20% clean mAP     |
  | Checkpoint & Analysis | -      | 30 min     | Evaluate Phase 2B  |
  | Phase 2C              | 71-100 | 1.5 hours  | >25% clean mAP     |
  | Final Testing         | -      | 1 hour     | Weather conditions |
  | Total                 | 100    | ~8.5 hours | Complete           |

  11. Success Criteria

  Minimum Success (Must Achieve):
  - Clean mAP@0.5 ≥ 20%
  - At least one weather condition ≥ 5% mAP
  - Stable training progression

  Target Success (Should Achieve):
  - Clean mAP@0.5 ≥ 25%
  - All weather conditions ≥ 5% mAP
  - Per-class AP improvement ≥ 10%

  Stretch Goals (Nice to Have):
  - Clean mAP@0.5 ≥ 30%
  - Weather average ≥ 10% mAP
  - Small object AP ≥ 15%

  12. Key Differences from Experiment 1

  | Aspect               | Experiment 1 | Experiment 2  | Expected Impact      |
  |----------------------|--------------|---------------|----------------------|
  | Learning Rate        | 0.01         | 0.001         | +10-15% stability    |
  | LR Schedule          | None         | Cosine        | Better convergence   |
  | Epochs               | 50           | 100           | Complete training    |
  | Loss Weights         | 0.3/0.7      | 0.5/0.5       | Balanced detection   |
  | Multi-scale          | No           | Yes           | +5-10% small objects |
  | Validation           | Clean only   | Multi-weather | Robustness tracking  |
  | Batch Size           | 8            | 16            | Better gradients     |
  | Progressive Training | No           | 3-phase       | Optimal convergence  |

  13. Conclusion

  Experiment 2 addresses the critical failures of Experiment 1 through:
  1. Optimized learning dynamics (10x lower LR, cosine scheduling)
  2. Extended training (100 epochs with progressive strategy)
  3. Balanced objectives (equal cls/obj weights)
  4. Multi-scale robustness (for VisDrone's scale variation)
  5. Comprehensive validation (weather-aware metrics)

  While maintaining Phase 1's no-augmentation constraint limits potential improvements, these
  optimizations should achieve 20-25% mAP on clean data (3x improvement) and 5-10% on weather 
  conditions (25-50x improvement), providing a stronger baseline for Phase 2's augmentation
  experiments.