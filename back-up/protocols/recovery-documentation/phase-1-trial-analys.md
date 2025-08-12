DEVELOPMENT TIMELINE & ITERATIONS

  Phase 1: Initial Setup (First Repository)

  - Configuration: Used SGD optimizer, batch size 8, lr=0.01
  - Results: Achieved 24.5% mAP@0.5 with augmentation enabled
  - Key Files: hyp_visdrone_trial-2_optimized.yaml, yolov5n_visdrone_config.yaml

  Phase 2: Protocol-Compliant Baseline

  - Objective: Create true baseline with NO augmentation (Phase 1 protocol requirement)     
  - Configuration Changes:
    - Switched to AdamW optimizer
    - Disabled ALL augmentation (mosaic=0, mixup=0, etc.)
    - Auto batch sizing enabled
    - Cosine learning rate scheduler
  - Initial Results: 18% mAP@0.5 (lower due to no augmentation)

  Phase 3: Experiment 2 Improvements

  - Goal: Match first repository performance while maintaining protocol compliance
  - Key Changes:
    - Reverted to SGD optimizer (proven better for YOLOv5)
    - Fixed batch size (8) for stability
    - Balanced loss weights (cls=0.3, obj=0.7)
    - Extended patience to 300 epochs
  - Results: 15.7% mAP@0.5 on clean dataset