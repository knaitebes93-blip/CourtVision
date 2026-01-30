import json
import os
from pathlib import Path
from typing import Dict, List

import cv2

from detect import detect_frame
from geometry import center_from_bbox, validate_setup
from metrics import compute_metrics
from track import ByteTrackWrapper

BASE_DIR = Path(__file__).resolve().parent
INPUT_VIDEO = BASE_DIR / "input" / "clip.mp4"
SETUP_PATH = BASE_DIR / "config" / "setup.json"
OUTPUT_VIDEO = BASE_DIR / "output" / "annotated.mp4"
OUTPUT_METRICS = BASE_DIR / "output" / "metrics.csv"


def load_setup(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"setup.json not found at {path}")
    with path.open("r", encoding="utf-8") as handle:
        setup = json.load(handle)
    validate_setup(setup)
    return setup


def ensure_output_dir(path: Path) -> None:
    os.makedirs(path.parent, exist_ok=True)


def draw_tracks(frame, tracks: List[Dict]) -> None:
    for track in tracks:
        x1, y1, x2, y2 = [int(coord) for coord in track["bbox"]]
        track_id = track["track_id"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID {track_id}",
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )


def run() -> None:
    setup = load_setup(SETUP_PATH)

    if not INPUT_VIDEO.exists():
        raise FileNotFoundError(f"Input video not found at {INPUT_VIDEO}")

    capture = cv2.VideoCapture(str(INPUT_VIDEO))
    if not capture.isOpened():
        raise RuntimeError("Failed to open input video")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ensure_output_dir(OUTPUT_VIDEO)
    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    tracker = ByteTrackWrapper(fps=fps)
    frame_index = 0
    track_history: Dict[int, List] = {}

    while True:
        success, frame = capture.read()
        if not success:
            break

        detections = detect_frame(frame, classes=["player"])
        tracks = tracker.update(detections)

        for track in tracks:
            center_x, center_y = center_from_bbox(track["bbox"])
            track_history.setdefault(track["track_id"], []).append(
                (frame_index, center_x, center_y)
            )

        draw_tracks(frame, tracks)
        writer.write(frame)
        frame_index += 1

    capture.release()
    writer.release()

    metrics_df = compute_metrics(track_history, fps, setup)
    ensure_output_dir(OUTPUT_METRICS)
    metrics_df.to_csv(OUTPUT_METRICS, index=False)


if __name__ == "__main__":
    run()
