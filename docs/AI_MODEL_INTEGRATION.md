# AtlasResQ AI — AI Model Integration

## Model: EfficientNet-B0 (ResQVision)

### Dataset
- **Name**: Kaggle "Satellite Images of Hurricane Damage"
- **Total images**: 14,000 (all 128×128 RGB)
- **Classes**: `damage` (7,000), `no_damage` (7,000) — balanced
- **Split**: Official Kaggle train/val/test (10k/2k/2k)
- **Training label mapping**: `damage=0`, `no_damage=1` (alphabetical sort)

### Architecture (verbatim from notebook STEP 9)
```python
def build_efficientnet_b0():
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_feats = m.classifier[1].in_features   # 1280
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_feats, 1),              # single binary logit
    )
    return m
```

### Training Configuration
| Parameter | Value |
|---|---|
| ResNet-18 baseline epochs | 6 |
| EfficientNet-B0 initial epochs (frozen backbone) | 5 |
| EfficientNet-B0 fine-tune max epochs | 20 |
| Fine-tune LR | 1e-4 |
| Weight decay | 1e-4 |
| Early stopping patience | 5 |
| Primary metric | val_F1 |
| Loss | BCEWithLogitsLoss |
| Optimizer | AdamW |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=2) |
| AMP | Yes (CUDA) |
| Batch size | 64 (Tesla T4, 15.6 GB) |

### Preprocessing (Eval / Inference)
```python
eval_transform = transforms.Compose([
    transforms.Lambda(lambda im: im.convert("RGB")),
    transforms.Resize(255),            # int(224 * 1.14)
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std= [0.229, 0.224, 0.225],
    ),
])
```

### Calibration (notebook STEP 17)
- **Method**: Temperature Scaling (LBFGS-fitted on validation set)
- **Learned temperature**: `T = 1.2596931457519531`
- **Formula**: `calibrated_prob = sigmoid(logit / T)`
- **Brier score before**: 0.004293
- **Brier score after**: 0.004318

### Decision Threshold (notebook STEP 16)
- **Method**: F1 grid search on validation set (range 0.05–0.95)
- **Selected threshold**: `0.375`
- **Validation F1 at threshold**: `0.9925`
- **Usage**: `is_damage = calibrated_prob < threshold`

### Test-Time Augmentation (notebook STEP 21)
- **Variants**: Original, H-flip, V-flip, Rotate(10°) — 4 total
- **Metric computed**: mean and std of calibrated probabilities
- **Interpretation**: `tta_uncertainty` = std = model uncertainty proxy
- **NOT** a guaranteed statistical confidence interval

### Grad-CAM (notebook STEP 22)
- **Target layer**: `model.features[-1]` (last conv block of EfficientNet-B0)
- **Output**: 224×224 heatmap (normalised [0,1])
- **Purpose**: AI attention/evidence visualisation
- **NOT**: segmentation, exact building boundaries, or confirmed damage mask

### Final Test Metrics
| Metric | Value |
|---|---|
| Accuracy | 99.45% |
| Precision | 99.01% |
| Recall | 99.90% |
| F1 | 99.45% |
| ROC-AUC | 99.9773% |
| Damage Recall | 99.00% |
| False Negatives | 10 / 1000 damage images |

### Inference Pipeline
```
image_bytes
    │
    ▼ PIL.open().convert("RGB")
    │
    ▼ Resize(255) → CenterCrop(224) → ToTensor() → Normalize()
    │   Tensor: [1, 3, 224, 224]
    │
    ▼ EfficientNet-B0.forward()
    │   Output: raw_logit (scalar)
    │
    ▼ sigmoid(raw_logit)
    │   damage_probability (uncalibrated)
    │
    ▼ sigmoid(raw_logit / T)    where T = 1.2597
    │   calibrated_probability
    │
    ▼ calibrated_prob < 0.375?
    │   prediction: "Damage" or "No Damage"
    │
    ▼ TTA (4 augmented variants)
    │   tta_mean_probability, tta_uncertainty, confidence_level
    │
    ▼ GradCAM(model.features[-1])
        heatmap_b64 (224×224 PNG, base64-encoded)
```
