from typing import Any, Dict, List

import numpy as np


class ByteTrackWrapper:
    def __init__(self, fps: float, track_buffer: int = 30) -> None:
        try:
            from yolox.tracker.byte_tracker import BYTETracker
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "ByteTrack dependency not found. Install yolox or another ByteTrack "
                "implementation that exposes yolox.tracker.byte_tracker.BYTETracker."
            ) from exc

        self._tracker = BYTETracker(
            args={
                "track_thresh": 0.25,
                "track_buffer": track_buffer,
                "match_thresh": 0.8,
                "frame_rate": fps,
            },
            frame_rate=fps,
        )

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not detections:
            self._tracker.update(np.empty((0, 5)), (0, 0), (0, 0))
            return []

        det_array = np.array(
            [
                [
                    det["bbox"][0],
                    det["bbox"][1],
                    det["bbox"][2],
                    det["bbox"][3],
                    det["confidence"],
                ]
                for det in detections
            ],
            dtype=np.float32,
        )

        tracks = self._tracker.update(det_array, (0, 0), (0, 0))
        results: List[Dict[str, Any]] = []

        for track in tracks:
            tlbr = track.tlbr
            results.append(
                {
                    "track_id": int(track.track_id),
                    "bbox": [float(tlbr[0]), float(tlbr[1]), float(tlbr[2]), float(tlbr[3])],
                }
            )

        return results
