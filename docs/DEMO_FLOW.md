# AtlasResQ AI — Demo Flow

> For hackathon presentations. DEMO mode is always explicitly labelled.

## Principle
- LIVE mode uses the actual trained EfficientNet-B0 checkpoint.
- DEMO mode uses deterministic predefined results.
- The system NEVER silently switches from LIVE to DEMO.
- All demo outputs are clearly marked `"mode": "DEMO"`.

## Demo Scenarios

### 1. Damage Detected
```json
{
  "mode": "DEMO",
  "prediction": "Damage",
  "damage_probability": 0.9453,
  "calibrated_probability": 0.9312,
  "threshold": 0.375,
  "tta_mean_probability": 0.9287,
  "tta_uncertainty": 0.0143,
  "confidence_level": "HIGH"
}
```

### 2. No Damage Detected
```json
{
  "mode": "DEMO",
  "prediction": "No Damage",
  "damage_probability": 0.0821,
  "calibrated_probability": 0.0694,
  "threshold": 0.375,
  "tta_mean_probability": 0.0712,
  "tta_uncertainty": 0.0089,
  "confidence_level": "HIGH"
}
```

## Triggering Demo Mode
```bash
# Via API
curl -X POST "http://localhost:8000/api/infer/demo?scenario=damage"
curl -X POST "http://localhost:8000/api/infer/demo?scenario=no_damage"
```

## Important Constraints
- Demo values are NOT connected to the real model.
- Demo mode does NOT produce Grad-CAM heatmaps.
- The UI (future) must display a prominent DEMO banner when in demo mode.
