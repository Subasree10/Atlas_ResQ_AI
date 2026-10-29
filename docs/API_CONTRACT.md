# AtlasResQ AI — API Contract

## Base URL
`http://localhost:8000` (development)

---

## `GET /health`
Service health check.

**Response 200**
```json
{
  "status": "ok",
  "service": "AtlasResQ AI Backend",
  "model_loaded": true,
  "mode": "LIVE"
}
```

---

## `GET /api/model/status`
Returns real model status. `loaded` is `true` ONLY if the checkpoint loaded successfully.

**Response 200**
```json
{
  "model_name": "EfficientNet-B0",
  "model_version": "1.0.0",
  "loaded": true,
  "load_error": null,
  "device": "cpu",
  "input_size": [224, 224],
  "classes": ["No Damage", "Damage"],
  "threshold": 0.375,
  "calibration_method": "temperature_scaling",
  "temperature": 1.2596931457519531,
  "service_mode": "LIVE",
  "test_metrics": {
    "test_accuracy": 0.9945,
    "test_f1": 0.9945246391239423,
    "test_auc": 0.999773
  }
}
```

---

## `POST /api/infer`
Full LIVE inference pipeline. Requires model to be loaded.

**Request**
- `Content-Type: multipart/form-data`
- `file`: image file (JPEG/PNG/TIFF/BMP, max 20 MB)
- `gradcam` (query param, default `true`): include Grad-CAM heatmap

**Response 200**
```json
{
  "mode": "LIVE",
  "prediction": "Damage",
  "damage_probability": 0.9453,
  "calibrated_probability": 0.9312,
  "threshold": 0.375,
  "tta_mean_probability": 0.9287,
  "tta_uncertainty": 0.0143,
  "confidence_level": "HIGH",
  "model_name": "EfficientNet-B0 (ResQVision)",
  "model_version": "1.0.0",
  "processing_time_ms": 350.2,
  "gradcam_available": true,
  "gradcam_heatmap_b64": "<base64-encoded PNG>",
  "raw_logit": 2.854123,
  "disclaimer": "AI-estimated visible damage assessment. Preliminary — requires human verification. Does NOT establish survivors, casualties, or confirmed rescue requirements."
}
```

**Response 503** — Model not loaded
```json
{
  "error": "LIVE model not loaded. Cannot perform real inference.",
  "mode": "LIVE",
  "loaded": false
}
```

**Response 400** — Unsupported image type  
**Response 413** — Image too large

---

## `POST /api/infer/demo`
DEMO mode inference — deterministic predefined results. Always labelled as DEMO.

**Query params**
- `scenario`: `"damage"` or `"no_damage"` (default: `"damage"`)

**Response 200**
```json
{
  "mode": "DEMO",
  "prediction": "Damage",
  "damage_probability": 0.9453,
  "calibrated_probability": 0.9312,
  "threshold": 0.375,
  ...
}
```

---

## Inference Response Fields

| Field | Type | Description |
|---|---|---|
| `mode` | string | `"LIVE"` or `"DEMO"` — always explicit |
| `prediction` | string | `"Damage"` or `"No Damage"` |
| `damage_probability` | float | `sigmoid(raw_logit)` — uncalibrated |
| `calibrated_probability` | float | `sigmoid(raw_logit / temperature)` |
| `threshold` | float | 0.375 — validation-optimised |
| `tta_mean_probability` | float | Mean over 4 TTA variants (calibrated) |
| `tta_uncertainty` | float | Std over 4 TTA variants — uncertainty proxy |
| `confidence_level` | string | `"HIGH"` / `"MEDIUM"` / `"LOW"` |
| `model_name` | string | Model identifier |
| `model_version` | string | Version string |
| `processing_time_ms` | float | Wall-clock time |
| `gradcam_available` | bool | Whether heatmap is included |
| `gradcam_heatmap_b64` | string\|null | Base64 PNG (224×224 grayscale) |
| `raw_logit` | float | Raw scalar logit from model |
| `disclaimer` | string | Scientific limitations statement |
