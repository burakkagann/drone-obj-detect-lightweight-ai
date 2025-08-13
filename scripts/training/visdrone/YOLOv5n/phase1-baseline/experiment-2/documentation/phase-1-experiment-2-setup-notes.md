# YOLOv5n VisDrone Phase 1 Baseline - Experiment 2

## Overview

Experiment 2 implements critical optimizations to address the catastrophic failure of Experiment 1 (7.36% mAP). This experiment maintains Phase 1's no-augmentation constraint while implementing:

- **10x Learning Rate Reduction**: 0.01 → 0.001 with cosine annealing
- **Progressive Training Strategy**: 3-phase approach over 100 epochs
- **Balanced Loss Weights**: cls=0.5, obj=0.5 (vs 0.3/0.7)
- **Multi-Scale Training**: Addresses VisDrone's extreme scale variation
- **Multi-Weather Validation**: Tracks robustness throughout training

## Expected Improvements

| Metric | Experiment 1 | Experiment 2 Target | Expected Gain |
|--------|--------------|-------------------|---------------|
| Clean mAP@0.5 | 7.36% | 20-25% | 3x improvement |
| Fog mAP@0.5 | 0.19% | 5-7% | 25-35x improvement |
| Rain mAP@0.5 | 0.28% | 6-8% | 20-30x improvement |
| Night mAP@0.5 | 0.23% | 5-7% | 25-30x improvement |

## Quick Start

### Complete Experiment (Recommended)
```bash
# Run full experiment (training + validation + testing)
python run_experiment2.py --mode full --batch-size 16

# With smaller batch size for limited GPU memory
python run_experiment2.py --mode full --batch-size 8
```

### Individual Components

#### Progressive Training Only
```bash
# Run all 3 phases sequentially
python train_progressive.py --phase all --batch-size 16

# Or run phases individually
python train_progressive.py --phase 2a  # Epochs 1-30
python train_progressive.py --phase 2b  # Epochs 31-70
python train_progressive.py --phase 2c  # Epochs 71-100
```

#### Multi-Weather Validation
```bash
# Validate specific phase results
python validate_multi_weather.py --phase 2c

# Or validate custom weights
python validate_multi_weather.py --weights path/to/best.pt
```

#### Weather Testing
```bash
# Test final model on all weather conditions
python test_weather_conditions.py --phase 2c

# Or test specific weights
python test_weather_conditions.py --weights path/to/best.pt --save-images
```

## Progressive Training Strategy

### Phase 2A: Foundation (Epochs 1-30)
- **Learning Rate**: 0.001 (linear decay)
- **Warmup**: 5 epochs
- **Goal**: Establish stable feature extraction
- **Target**: >15% mAP

### Phase 2B: Optimization (Epochs 31-70)
- **Learning Rate**: 0.0005 (cosine annealing)
- **Label Smoothing**: 0.05
- **Goal**: Refine detection accuracy
- **Target**: >20% mAP

### Phase 2C: Fine-tuning (Epochs 71-100)
- **Learning Rate**: 0.0001 (cosine annealing)
- **Label Smoothing**: 0.1
- **Goal**: Final optimization
- **Target**: >25% mAP

## Configuration Files

### Hyperparameter Configs
- `configs/yolov5n-visdrone/experiment2a_phase1_hyp.yaml` - Phase 2A settings
- `configs/yolov5n-visdrone/experiment2b_phase1_hyp.yaml` - Phase 2B settings
- `configs/yolov5n-visdrone/experiment2c_phase_hyp.yaml` - Phase 2C settings

### Data Config
- `configs/yolov5n-visdrone/visdrone_experiment2.yaml` - Dataset configuration

## Key Optimizations

### 1. Learning Rate Fix
```yaml
# Experiment 1 (Failed)
lr0: 0.01  # Too high - caused instability

# Experiment 2 (Optimized)
lr0: 0.001  # Phase 2A
lr0: 0.0005  # Phase 2B
lr0: 0.0001  # Phase 2C
cos_lr: True  # Smooth decay
```

### 2. Loss Weight Balancing
```yaml
# Experiment 1 (Imbalanced)
cls: 0.3
obj: 0.7  # Over-emphasized objectness

# Experiment 2 (Balanced)
cls: 0.5
obj: 0.5  # Equal weight for better classification
```

### 3. Multi-Scale Training
```yaml
# Enabled for scale variation
imgsz: 640  # Base size
scale: 0.5  # ±50% variation (320-960 pixels)
rect: True  # Rectangular training
```

## Output Structure

```
experiment-2/
├── logs&results/
│   ├── training/
│   │   ├── exp2_phase2a/     # Phase 2A results
│   │   ├── exp2_phase2b/     # Phase 2B results
│   │   └── exp2_phase2c/     # Phase 2C results
│   ├── validation/
│   │   └── multi_weather_*/  # Validation results per phase
│   └── weather_testing/
│       ├── weather_testing_complete_*.json
│       └── weather_comparison_*.json
├── train_progressive.py      # Main training script
├── validate_multi_weather.py # Multi-condition validation
├── test_weather_conditions.py # Weather testing
└── run_experiment2.py        # Complete experiment runner
```

## Monitoring Progress

### Training Metrics
Monitor `logs&results/training/exp2_phase*/results.csv` for:
- mAP@0.5 progression
- Loss convergence
- Precision/Recall balance

### Validation Tracking
Check `logs&results/validation/multi_weather_*/validation_summary.json` for:
- Composite scores (weighted across conditions)
- Per-weather performance
- Degradation percentages

### Weather Testing
Review `logs&results/weather_testing/weather_comparison_*.json` for:
- Baseline performance
- Weather-specific degradation
- Comparison with Experiment 1

## Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size
python run_experiment2.py --mode full --batch-size 8

# Or disable caching in configs
cache: 'disk'  # Instead of 'ram'
```

### Training Instability
- Check learning rate progression in logs
- Verify data loading with smaller subset
- Monitor gradient norms if available

### Poor Weather Performance
- Expected in Phase 1 (no augmentation)
- Document for Phase 2 augmentation strategy
- Focus on baseline improvement first

## Success Criteria

### Minimum Success (Must Achieve)
- Clean mAP@0.5 ≥ 20%
- At least one weather ≥ 5% mAP
- Stable training progression

### Target Success (Should Achieve)
- Clean mAP@0.5 ≥ 25%
- All weather conditions ≥ 5% mAP
- Per-class AP improvement ≥ 10%

### Stretch Goals
- Clean mAP@0.5 ≥ 30%
- Weather average ≥ 10% mAP
- Small object AP ≥ 15%

## Analysis Scripts

### Generate Comprehensive Report
```bash
# After experiment completion
python run_experiment2.py --mode analyze
```

This generates:
- Training progression analysis
- Phase-wise performance comparison
- Weather degradation analysis
- Comparison with Experiment 1
- Recommendations for Phase 2

## Next Steps

Based on Experiment 2 results:

1. **If mAP < 20%**: Consider architectural changes (YOLOv5s/YOLOv8n)
2. **If mAP 20-25%**: Proceed to Phase 2 with augmentation
3. **If mAP > 25%**: Strong baseline for weather augmentation experiments

## References

- [Experiment 1 Analysis](../experiment-1/documentation/experiment1-trial-1-results-analysis.md)
- [Phase 1 Design Document](../experiment-2/EXPERIMENT2_DESIGN.md)
- [YOLOv5 Documentation](https://github.com/ultralytics/yolov5)
- [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset)

## Contact

For issues or questions about this experiment, refer to the main repository documentation or create an issue with the tag `experiment-2`.

---

**Note**: This experiment maintains Phase 1's no-augmentation constraint. Weather robustness improvements are limited without augmentation. Phase 2 will enable augmentation for significant weather performance gains.