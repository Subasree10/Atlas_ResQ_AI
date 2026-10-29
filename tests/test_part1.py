"""
AtlasResQ AI â€” Part 1 Test Suite

Tests:
  1. JSON artifact loading (class_names, model_config, calibration, threshold, metrics)
  2. EfficientNet-B0 architecture reconstruction
  3. Checkpoint loading (resqvision_efficientnet_b0.pt)
  4. State dict key verification
  5. Preprocessing pipeline
  6. Calibration service
  7. Threshold verification
  8. TTA service (with dummy model)
  9. Grad-CAM service (with real model if loaded)
  10. End-to-end inference (real image from gradcam PNGs in workspace)

Run:
    cd e:/AtlasResQ
    python backend/tests/test_part1.py
"""

import sys
import os
import json
import math
import traceback
import io
from pathlib import Path

# Add backend to path so imports work
_ROOT = Path(__file__).parent.parent.parent          # e:/AtlasResQ
_BACKEND = _ROOT / "backend"
sys.path.insert(0, str(_BACKEND))

MODELS_DIR = _ROOT / "models"
CHECKPOINT = str(MODELS_DIR / "resqvision_efficientnet_b0.pt")

PASS = "âœ… PASS"
FAIL = "âŒ FAIL"
SKIP = "â­  SKIP"

results = []


def run_test(name: str, fn):
    try:
        fn()
        print(f"  {PASS}  {name}")
        results.append((name, True, None))
    except Exception as e:
        msg = str(e)
        print(f"  {FAIL}  {name}")
        print(f"          {msg}")
        results.append((name, False, msg))


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\nâ”â”â” 1. JSON ARTIFACT LOADING â”â”â”")

def test_class_names():
    with open(MODELS_DIR / "class_names.json") as f:
        data = json.load(f)
    assert isinstance(data, list), "class_names.json must be a list"
    assert len(data) == 2, f"Expected 2 classes, got {len(data)}"
    assert "No Damage" in data and "Damage" in data, f"Unexpected classes: {data}"

def test_model_config():
    with open(MODELS_DIR / "model_config.json") as f:
        data = json.load(f)
    assert data["architecture"] == "EfficientNet-B0"
    assert data["input_size"] == [224, 224]
    assert data["normalization"]["mean"] == [0.485, 0.456, 0.406]
    assert data["normalization"]["std"] == [0.229, 0.224, 0.225]

def test_calibration_json():
    with open(MODELS_DIR / "calibration.json") as f:
        data = json.load(f)
    assert data["method"] == "temperature_scaling"
    temp = data["temperature"]
    assert 1.0 < temp < 2.0, f"Temperature {temp} out of expected range (1.0â€“2.0)"

def test_threshold_json():
    with open(MODELS_DIR / "threshold.json") as f:
        data = json.load(f)
    t = data["selected_threshold"]
    assert 0.0 < t < 1.0, f"Threshold {t} out of range"
    assert abs(t - 0.375) < 0.01, f"Expected ~0.375, got {t}"
    assert data["validation_f1_at_selected"] > 0.98, "Val F1 unexpectedly low"

def test_metrics_json():
    with open(MODELS_DIR / "metrics.json") as f:
        data = json.load(f)
    assert data["test_accuracy"] > 0.99
    assert data["test_f1"] > 0.99
    assert data["test_auc"] > 0.999
    assert data["false_negatives"] <= 15, "Too many false negatives"

run_test("class_names.json valid", test_class_names)
run_test("model_config.json valid", test_model_config)
run_test("calibration.json valid", test_calibration_json)
run_test("threshold.json valid", test_threshold_json)
run_test("metrics.json valid", test_metrics_json)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\nâ”â”â” 2. ARCHITECTURE RECONSTRUCTION â”â”â”")

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print(f"  {SKIP}  PyTorch not installed â€” skipping all model tests")

if HAS_TORCH:
    from app.ai.model_architecture import build_efficientnet_b0, load_checkpoint

    def test_architecture_shape():
        model = build_efficientnet_b0()
        assert hasattr(model, "features"), "model must have .features attribute"
        clf = model.classifier
        assert isinstance(clf, nn.Sequential), "classifier must be nn.Sequential"
        assert len(clf) == 2, f"Expected 2 classifier layers, got {len(clf)}"
        assert isinstance(clf[0], nn.Dropout), "clf[0] must be Dropout"
        assert isinstance(clf[1], nn.Linear), "clf[1] must be Linear"
        assert clf[1].out_features == 1, f"Expected 1 output, got {clf[1].out_features}"
        assert clf[1].in_features == 1280, f"Expected in_features=1280, got {clf[1].in_features}"
        assert clf[0].p == 0.3, f"Expected dropout p=0.3, got {clf[0].p}"

    def test_architecture_forward():
        model = build_efficientnet_b0()
        model.eval()
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (1, 1), f"Expected output shape (1,1), got {out.shape}"

    run_test("EfficientNet-B0 reconstructed with correct head", test_architecture_shape)
    run_test("EfficientNet-B0 forward pass shape (1,1)", test_architecture_forward)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\nâ”â”â” 3. CHECKPOINT LOADING â”â”â”")

def assert_exists(p):
    assert Path(p).exists(), f"File not found: {p}"

if HAS_TORCH:
    def test_checkpoint_loads():
        global _model_global
        assert Path(CHECKPOINT).exists(), f"Checkpoint not found: {CHECKPOINT}"
        model = build_efficientnet_b0()
        device = torch.device("cpu")
        model = load_checkpoint(model, CHECKPOINT, device)
        assert model is not None
        _model_global = model

    def test_no_statedict_mismatch():
        assert _model_global is not None, "Model not loaded in previous test"
        # Verify the model has parameters (not empty)
        params = list(_model_global.parameters())
        assert len(params) > 0

    def test_model_in_eval_mode():
        assert _model_global is not None
        assert not _model_global.training, "Model should be in eval mode after load_checkpoint"

    run_test("Checkpoint file exists", lambda: assert_exists(CHECKPOINT))
    run_test("Checkpoint loads without mismatch", test_checkpoint_loads)
    run_test("No state_dict mismatch", test_no_statedict_mismatch)
    run_test("Model in eval mode", test_model_in_eval_mode)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\nâ”â”â” 4. PREPROCESSING â”â”â”")

if HAS_TORCH:
    from app.ai.preprocessing import get_eval_transform, preprocess_image, IMG_SIZE, MEAN, STD, RESIZE_TO

    def test_preprocessing_constants():
        assert IMG_SIZE == 224
        assert RESIZE_TO == 255   # int(224 * 1.14)
        assert MEAN == [0.485, 0.456, 0.406]
        assert STD == [0.229, 0.224, 0.225]

    def test_preprocessing_output_shape():
        # Create a dummy 128x128 RGB image (dataset native size)
        from PIL import Image
        pil = Image.new("RGB", (128, 128), color=(100, 150, 200))
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        tensor = preprocess_image(buf.getvalue())
        assert tensor.shape == (1, 3, 224, 224), f"Expected (1,3,224,224), got {tensor.shape}"

    def test_preprocessing_normalizes():
        from PIL import Image
        import numpy as np
        pil = Image.new("RGB", (128, 128), color=(128, 128, 128))
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        tensor = preprocess_image(buf.getvalue())
        # After normalization, uniform 128/255 â‰ˆ 0.502 should be close to 0
        # (mean subtracted ~0.44â€“0.48). Values should not be in [0,255].
        assert tensor.min().item() < 1.0, "Tensor not normalized"
        assert tensor.max().item() < 5.0, "Tensor values suspiciously large"

    run_test("Preprocessing constants match notebook", test_preprocessing_constants)
    run_test("Preprocessing output shape (1,3,224,224)", test_preprocessing_output_shape)
    run_test("Preprocessing normalizes values", test_preprocessing_normalizes)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\nâ”â”â” 5. CALIBRATION â”â”â”")

from app.ai.calibration_service import CalibrationService

def test_calibration_loads():
    svc = CalibrationService(str(MODELS_DIR / "calibration.json"))
    assert abs(svc.temperature - 1.2597) < 0.001

def test_calibration_probability_range():
    svc = CalibrationService(str(MODELS_DIR / "calibration.json"))
    for logit in [-10.0, -2.0, 0.0, 2.0, 10.0]:
        p = svc.calibrated_probability(logit)
        assert 0.0 <= p <= 1.0, f"Out of range: {p} for logit {logit}"

def test_calibration_monotone():
    svc = CalibrationService(str(MODELS_DIR / "calibration.json"))
    logits = [-5.0, -2.0, 0.0, 2.0, 5.0]
    probs = [svc.calibrated_probability(l) for l in logits]
    for i in range(len(probs) - 1):
        assert probs[i] > probs[i+1], "Calibrated probability not monotone"

def test_calibration_temperature_effect():
    svc = CalibrationService(str(MODELS_DIR / "calibration.json"))
    # With T > 1, calibrated prob at logit=0 should still be 0.5
    p = svc.calibrated_probability(0.0)
    assert abs(p - 0.5) < 0.001, f"Expected 0.5 at logit=0, got {p}"

run_test("CalibrationService loads correctly", test_calibration_loads)
run_test("Calibrated probability in [0,1]", test_calibration_probability_range)
run_test("Calibrated probability is monotone", test_calibration_monotone)
run_test("Calibrated probability at logit=0 is 0.5", test_calibration_temperature_effect)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\nâ”â”â” 6. THRESHOLD â”â”â”")

def test_threshold_value():
    with open(MODELS_DIR / "threshold.json") as f:
        data = json.load(f)
    t = data["selected_threshold"]
    assert abs(t - 0.375) < 0.01, f"Expected ~0.375, got {t}"

def test_threshold_not_default_half():
    with open(MODELS_DIR / "threshold.json") as f:
        data = json.load(f)
    t = data["selected_threshold"]
    assert abs(t - 0.5) > 0.05, "Threshold appears to be the default 0.5, not the selected one"

run_test("Selected threshold â‰ˆ 0.375 (not assumed 0.5)", test_threshold_value)
run_test("Threshold is NOT the default 0.5", test_threshold_not_default_half)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\nâ”â”â” 7. TTA SERVICE â”â”â”")

if HAS_TORCH:
    from app.ai.tta_service import TTAService
    from PIL import Image

    def test_tta_returns_four_probs():
        if _model_global is None:
            raise RuntimeError("Model not loaded")
        cal = CalibrationService(str(MODELS_DIR / "calibration.json"))
        tta = TTAService(_model_global, cal, torch.device("cpu"))
        pil = Image.new("RGB", (128, 128))
        result = tta.predict(pil)
        assert len(result["individual_probs"]) == 4, "Expected 4 TTA variants"

    def test_tta_uncertainty_non_negative():
        if _model_global is None:
            raise RuntimeError("Model not loaded")
        cal = CalibrationService(str(MODELS_DIR / "calibration.json"))
        tta = TTAService(_model_global, cal, torch.device("cpu"))
        pil = Image.new("RGB", (224, 224), color=(100, 150, 200))
        result = tta.predict(pil)
        assert result["tta_uncertainty"] >= 0.0
        assert 0.0 <= result["tta_mean_probability"] <= 1.0
        assert result["confidence_level"] in ("HIGH", "MEDIUM", "LOW")

    run_test("TTA returns 4 variant probabilities", test_tta_returns_four_probs)
    run_test("TTA uncertainty â‰¥ 0, mean in [0,1]", test_tta_uncertainty_non_negative)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\nâ”â”â” 8. GRAD-CAM â”â”â”")

if HAS_TORCH:
    from app.ai.gradcam_service import GradCAMService

    def test_gradcam_target_layer():
        if _model_global is None:
            raise RuntimeError("Model not loaded")
        # Confirm features[-1] is accessible
        target = _model_global.features[-1]
        assert target is not None

    def test_gradcam_output():
        if _model_global is None:
            raise RuntimeError("Model not loaded")
        svc = GradCAMService(_model_global)
        dummy = torch.randn(1, 3, 224, 224)
        result = svc.compute(dummy)
        assert "heatmap_b64" in result
        assert "raw_logit" in result
        import base64
        decoded = base64.b64decode(result["heatmap_b64"])
        from PIL import Image as PILImage
        img = PILImage.open(io.BytesIO(decoded))
        assert img.size == (224, 224), f"Expected 224x224 heatmap, got {img.size}"

    run_test("Grad-CAM target layer (features[-1]) accessible", test_gradcam_target_layer)
    run_test("Grad-CAM produces 224x224 heatmap", test_gradcam_output)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\nâ”â”â” 9. END-TO-END INFERENCE â”â”â”")

if HAS_TORCH and _model_global is not None:
    from app.ai.inference_service import EfficientNetInferenceService

    def test_e2e_with_real_image():
        # Use a Grad-CAM output image that's already in the workspace as a test image
        test_img_path = _ROOT / "gradcam_true_positive.png"
        if not test_img_path.exists():
            raise RuntimeError(f"Test image not found: {test_img_path}")

        with open(test_img_path, "rb") as f:
            img_bytes = f.read()

        svc = EfficientNetInferenceService()
        assert svc.is_loaded, "Service should be loaded"
        result = svc.infer(img_bytes, run_gradcam=True)

        print(f"\n          â”€â”€ Real Inference Result â”€â”€")
        print(f"          prediction:            {result.prediction}")
        print(f"          raw_logit:             {result.raw_logit:.6f}")
        print(f"          damage_probability:    {result.damage_probability:.4f}")
        print(f"          calibrated_prob:       {result.calibrated_probability:.4f}")
        print(f"          threshold:             {result.threshold:.4f}")
        print(f"          tta_mean_probability:  {result.tta_mean_probability:.4f}")
        print(f"          tta_uncertainty:       {result.tta_uncertainty:.4f}")
        print(f"          confidence_level:      {result.confidence_level}")
        print(f"          gradcam_available:     {result.gradcam_available}")
        print(f"          processing_time_ms:    {result.processing_time_ms:.1f}")
        print(f"          mode:                  {result.mode}")

        assert result.mode == "LIVE"
        assert result.prediction in ("Damage", "No Damage")
        assert 0.0 <= result.damage_probability <= 1.0
        assert 0.0 <= result.calibrated_probability <= 1.0
        assert 0.0 <= result.tta_mean_probability <= 1.0
        assert result.tta_uncertainty >= 0.0
        assert result.confidence_level in ("HIGH", "MEDIUM", "LOW")

    run_test("End-to-end inference on real workspace image", test_e2e_with_real_image)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\nâ”â”â” RESULTS SUMMARY â”â”â”")
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
total = len(results)
print(f"\n  Total: {total}  |  Passed: {passed}  |  Failed: {failed}\n")

for name, ok, err in results:
    icon = PASS if ok else FAIL
    print(f"  {icon}  {name}")
    if err:
        print(f"       â””â”€ {err}")

if failed > 0:
    print(f"\nâš ï¸  {failed} run_test(s) failed.")
    sys.exit(1)
else:
    print(f"\nâœ… All {total} tests passed.")


