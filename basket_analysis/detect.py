import base64
import json
import os
import urllib.request
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


ROBOFLOW_API_URL = "https://detect.roboflow.com"


def _require_env_var(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _encode_frame(frame_bgr: np.ndarray) -> bytes:
    success, buffer = cv2.imencode(".jpg", frame_bgr)
    if not success:
        raise RuntimeError("Failed to encode frame as JPEG for Roboflow API")
    return buffer.tobytes()


def detect_frame(
    frame_bgr: np.ndarray,
    confidence: float = 0.2,
    overlap: float = 0.5,
    classes: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    api_key = _require_env_var("ROBOFLOW_API_KEY")
    model_id = _require_env_var("ROBOFLOW_MODEL_ID")
    version = _require_env_var("ROBOFLOW_MODEL_VERSION")

    encoded = _encode_frame(frame_bgr)
    image_b64 = base64.b64encode(encoded).decode("utf-8")

    url = (
        f"{ROBOFLOW_API_URL}/{model_id}/{version}?api_key={api_key}"
        f"&confidence={confidence}&overlap={overlap}"
    )

    request = urllib.request.Request(
        url,
        data=image_b64.encode("utf-8"),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=30) as response:
        payload = response.read().decode("utf-8")

    data = json.loads(payload)
    predictions = data.get("predictions", [])
    results: List[Dict[str, Any]] = []

    for pred in predictions:
        label = pred.get("class")
        if classes and label not in classes:
            continue
        x_center = float(pred["x"])
        y_center = float(pred["y"])
        width = float(pred["width"])
        height = float(pred["height"])
        x1 = x_center - width / 2.0
        y1 = y_center - height / 2.0
        x2 = x_center + width / 2.0
        y2 = y_center + height / 2.0
        results.append(
            {
                "class": label,
                "confidence": float(pred.get("confidence", 0.0)),
                "bbox": [x1, y1, x2, y2],
            }
        )

    return results
