
# Phase 1 Baseline Validation Summary
## YOLOv5n on VisDrone Dataset (No Augmentation)

**Date**: 2025-08-03 14:17:43
**Model**: YOLOv5n (1.9M parameters)
**Dataset**: VisDrone validation set
**Augmentation**: None (True Baseline)

## Performance Metrics

| Metric | Value |
|--------|-------|
| mAP@0.5 | 0.1800 |
| mAP@0.5:0.95 | 0.0857 |
| Precision | 0.2700 |
| Recall | 0.2230 |
| Validation Loss | 0.0000 |

## Model Specifications
- **Architecture**: YOLOv5n (nano)
- **Input Size**: 640×640 pixels
- **Classes**: 10 (VisDrone objects)
- **Training**: 100 epochs, AdamW optimizer
- **Augmentation**: DISABLED (baseline)

## Next Steps
1. Test on synthetic weather conditions (fog, rain, night, mixed)
2. Measure performance degradation percentages
3. Use results to justify Phase 2 augmentation training

## Files Generated
- Model weights: `runs/phase1/yolov5n_baseline/weights/best.pt`
- Validation results: `runs/phase1/yolov5n_baseline/validation/`
- Confusion matrix: Available in validation output
- PR curves: Available in validation output

---
*This baseline establishes the starting point for weather robustness analysis in Phase 2.*
