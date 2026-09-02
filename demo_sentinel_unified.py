"""Interactive SentinelVision Phase-1 demo.

Keys:
    Q - quit
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any, Dict

import cv2
import numpy as np

from sentinel_vision import SentinelUnifiedConfig, SentinelVisionUnified

logger = logging.getLogger(__name__)


STATUS_TEXT = {
    "BENIGN": "[OK] SAFE",
    "SUSPICIOUS": "[!] ALERT",
    "THREATENING": "[!!!] THREAT",
}


def draw_hud(frame: np.ndarray, result: Dict[str, Any]) -> None:
    vision = result["vision"]
    audio = result["audio"]
    composite = result["composite"]
    performance = result["performance"]
    health = result["health"]

    risk = composite["risk_label"]
    hud = [
        STATUS_TEXT.get(risk, risk),
        f"Score: {composite['threat_score']:.3f}",
        f"Persons: {vision['persons']}  Faces: {vision['faces']}",
        f"Audio: {audio.get('label', 'Unknown')}",
        f"Audio threat: {audio.get('threat_score', 0.0):.3f}",
        f"Vision score: {composite.get('raw_vision_score', 0.0):.3f}",
        f"Coverage: {composite.get('sensor_coverage', 0.0):.1f}",
        f"FPS: {performance['frame_fps']:.1f}  Vision: {performance['vision_latency_ms']:.1f}ms",
        f"Audio age: {performance['audio_age_ms']:.0f}ms",
    ]

    y = 30
    for index, text in enumerate(hud):
        scale = 0.85 if index == 0 else 0.58
        thickness = 2 if index == 0 else 1
        cv2.putText(
            frame,
            text,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        y += 28 if index == 0 else 24

    errors = []
    if health["vision_status"] != "OK":
        errors.append(f"VISION {health['vision_status']}: {health['last_vision_error'] or 'unavailable'}")
    if health["audio_status"] != "OK":
        errors.append(f"AUDIO {health['audio_status']}: {health['last_audio_error'] or 'unavailable'}")

    if errors:
        y += 10
        cv2.rectangle(frame, (8, y - 20), (frame.shape[1] - 8, y + 42 * len(errors)), (0, 0, 180), -1)
        for error in errors:
            cv2.putText(
                frame,
                error[:100],
                (15, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y += 36


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    system = SentinelVisionUnified(config=SentinelUnifiedConfig())
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        system.stop()
        raise RuntimeError(f"Could not open camera {args.camera}")

    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_id = 0
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                logger.warning("Camera frame read failed")
                continue

            result = system.process_frame(frame, frame_id=frame_id)
            frame_id += 1
            display = result["annotated_frame"]
            draw_hud(display, result)

            cv2.imshow("SentinelVision - Unified Phase 1", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        system.stop()


if __name__ == "__main__":
    main()
