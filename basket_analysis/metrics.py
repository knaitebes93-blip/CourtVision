from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from geometry import pixels_to_meters


def compute_metrics(
    track_history: Dict[int, List[Tuple[int, float, float]]],
    fps: float,
    setup: Dict,
) -> pd.DataFrame:
    rows = []

    for track_id, samples in track_history.items():
        if len(samples) < 2:
            continue

        samples_sorted = sorted(samples, key=lambda item: item[0])
        positions = np.array([[x, y] for _, x, y in samples_sorted], dtype=np.float32)

        deltas = positions[1:] - positions[:-1]
        distances = np.linalg.norm(deltas, axis=1)
        total_distance_px = float(distances.sum())

        total_time_s = (samples_sorted[-1][0] - samples_sorted[0][0]) / fps
        avg_speed_px_s = total_distance_px / total_time_s if total_time_s > 0 else 0.0

        avg_position_x = float(positions[:, 0].mean())
        avg_position_y = float(positions[:, 1].mean())

        row = {
            "track_id": track_id,
            "distance_px": total_distance_px,
            "avg_speed_px_s": avg_speed_px_s,
            "avg_position_x": avg_position_x,
            "avg_position_y": avg_position_y,
        }

        if "pixel_to_meter" in setup:
            row["distance_m"] = pixels_to_meters(total_distance_px, setup)
            row["avg_speed_m_s"] = pixels_to_meters(avg_speed_px_s, setup)

        rows.append(row)

    return pd.DataFrame(rows)
