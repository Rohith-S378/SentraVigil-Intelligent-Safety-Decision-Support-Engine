"""Line-delimited JSON bridge for TourShield's trained GRU-VAE ONNX model."""
import json
import os
import sys
import numpy as np
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "gru_vae_best.onnx")
WINDOW_SIZE, THRESHOLD = 20, 0.45
FEATURE_NAMES = ["avg_speed", "inactivity_duration", "route_deviation", "direction_changes", "zone_level", "journey_duration"]
# Telemetry calibration: makes the differently-scaled simulator features comparable.
FEATURE_MEANS = np.array([1.4, 0.0, 3.25, 0.15, 0.0, 30.0], dtype=np.float32)
FEATURE_STDS = np.array([0.35, 10.0, 12.0, 1.5, 0.75, 30.0], dtype=np.float32)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"ONNX model not found: {MODEL_PATH}")
SESSION = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
INPUT_NAME = SESSION.get_inputs()[0].name

def evaluate(window):
    values = np.asarray(window, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES) or len(values) == 0:
        raise ValueError("window must contain one or more six-feature rows")
    if len(values) < WINDOW_SIZE:
        values = np.vstack((np.repeat(values[:1], WINDOW_SIZE - len(values), axis=0), values))
    values = values[-WINDOW_SIZE:]
    normalized = ((values - FEATURE_MEANS) / FEATURE_STDS)[None].astype(np.float32)
    reconstruction, mu, logvar = SESSION.run(None, {INPUT_NAME: normalized})
    feature_errors = np.mean((reconstruction[0] - normalized[0]) ** 2, axis=0)
    error = float(np.mean(feature_errors))
    latest = values[-1]
    # SAFETY-NET RULE (not part of the learned model): force-flag hard physical
    # thresholds the checkpoint was not trained to guarantee catching.
    if latest[4] >= 2.0 or latest[2] > 35.0 or latest[1] > 50.0:
        error = max(error, THRESHOLD + 0.18)
    total = float(feature_errors.sum())
    breakdown = {name: {"error": float(feature_errors[i]), "percentage": float(feature_errors[i] / total * 100 if total else 0), "latest_val": float(latest[i])} for i, name in enumerate(FEATURE_NAMES)}
    return {"reconstruction_error": error, "threshold": THRESHOLD, "is_anomaly": error > THRESHOLD, "feature_breakdown": breakdown, "latent_mu": mu[0].astype(float).tolist(), "latent_logvar": logvar[0].astype(float).tolist()}

for line in sys.stdin:
    request_id = None
    try:
        payload = json.loads(line)
        request_id = payload.get("tourist_id") if isinstance(payload, dict) else None
        result = evaluate(payload["window"] if isinstance(payload, dict) else payload)
        result["tourist_id"] = request_id
    except Exception as exc:
        result = {"tourist_id": request_id, "error": str(exc)}
    print(json.dumps(result), flush=True)
