from typing import Dict, Tuple


def validate_setup(setup: Dict) -> None:
    required_keys = ["court_keypoints", "hoop", "team_colors"]
    missing = [key for key in required_keys if key not in setup]
    if missing:
        raise ValueError(f"Missing required setup keys: {', '.join(missing)}")


def pixel_to_meter_scale(setup: Dict) -> float:
    if "pixel_to_meter" not in setup:
        raise ValueError("Missing 'pixel_to_meter' in setup.json for metric conversion")
    return float(setup["pixel_to_meter"])


def pixels_to_meters(distance_px: float, setup: Dict) -> float:
    scale = pixel_to_meter_scale(setup)
    return distance_px * scale


def center_from_bbox(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0
