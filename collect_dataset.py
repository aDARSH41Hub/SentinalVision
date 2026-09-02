"""Controlled Phase-1 dataset collection helper.

Keys while the camera window is running:
    B = benign
    S = suspicious
    T = threatening
    Q = quit

Each sample contains:
    dataset/<label>/<timestamp>_<frame_id>.jpg
    dataset/<label>/<timestamp>_<frame_id>.wav   # rolling 2-sec audio ending at capture
    dataset/<label>/<timestamp>_<frame_id>.json  # metadata

Use only controlled/acted scenarios. Do not stage real weapon use.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import cv2

from sentinel_vision import SentinelUnifiedConfig, SentinelVisionUnified

LABEL_KEYS = {
    ord("b"): "benign",
    ord("s"): "suspicious",
    ord("t"): "threatening",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output-dir", default="dataset")
    args = parser.parse_args()

    root = Path(args.output_dir)
    for label in LABEL_KEYS.values():
        (root / label).mkdir(parents=True, exist_ok=True)

    system = SentinelVisionUnified(config=SentinelUnifiedConfig())
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        system.stop()
        raise RuntimeError(f"Could not open camera {args.camera}")

    frame_id = 0
    print("B=benign, S=suspicious, T=threatening, Q=quit")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                continue

            result = system.process_frame(frame, frame_id=frame_id)
            display = result["annotated_frame"]
            cv2.putText(
                display,
                "B=benign S=suspicious T=threatening Q=quit",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow("SentinelVision Dataset Collector", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            label = LABEL_KEYS.get(key)
            if label is None:
                frame_id += 1
                continue

            if system.audio is None:
                print("Cannot collect synchronized audio: audio sensor is unavailable")
                continue

            timestamp = int(time.time() * 1000)
            stem = f"{timestamp}_{frame_id:06d}"
            folder = root / label
            image_path = folder / f"{stem}.jpg"
            audio_path = folder / f"{stem}.wav"
            metadata_path = folder / f"{stem}.json"

            cv2.imwrite(str(image_path), frame)
            system.audio.save_audio_buffer(str(audio_path))

            metadata: Dict[str, Any] = {
                "timestamp_ms": timestamp,
                "frame_id": frame_id,
                "label": label,
                "audio_window_seconds": system.audio.config.BUFFER_SECONDS,
                "audio_window_semantics": "rolling window ending at label capture time",
                "audio_class": result["audio"].get("label"),
                "audio_classifier_confidence": result["audio"].get("confidence", 0.0),
                "audio_threat_score": result["audio"].get("threat_score", 0.0),
                "persons": result["vision"].get("persons", 0),
                "faces": result["vision"].get("faces", 0),
                "fused_score": result["composite"].get("threat_score", 0.0),
                "risk_level_at_capture": result["composite"].get("risk_label"),
                "sensor_coverage": result["composite"].get("sensor_coverage", 0.0),
                "notes": "",
            }
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            print(f"Saved {label}: {stem}")
            frame_id += 1
    finally:
        camera.release()
        cv2.destroyAllWindows()
        system.stop()


if __name__ == "__main__":
    main()
