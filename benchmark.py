"""Structured continuous benchmark for the SentinelVision Phase-1 pipeline.

Default:
    300 seconds (5 minutes)

Outputs:
    CSV with one row per processed frame.
    JSON summary with latency/FPS/error statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2

from sentinel_vision import SentinelUnifiedConfig, SentinelVisionUnified


CSV_FIELDS = [
    "timestamp",
    "frame_id",
    "yolo_ms",
    "haar_ms",
    "ast_ms",
    "vision_ms",
    "audio_age_ms",
    "fused_score",
    "risk_level",
    "persons",
    "faces",
    "audio_class",
    "audio_threat",
    "sensor_coverage",
    "vision_status",
    "audio_status",
    "frame_read_ok",
]


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = int(round((percentile / 100.0) * (len(values) - 1)))
    return values[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=300.0)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output-dir", default="benchmarks")
    parser.add_argument("--no-window", action="store_true")
    args = parser.parse_args()

    if args.seconds <= 0:
        raise ValueError("--seconds must be positive")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"benchmark_{stamp}.csv"
    json_path = out_dir / f"benchmark_{stamp}_summary.json"

    system = SentinelVisionUnified(config=SentinelUnifiedConfig())
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        system.stop()
        raise RuntimeError(f"Could not open camera {args.camera}")

    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    started = time.time()
    frame_id = 0
    processed = 0
    read_failures = 0
    end_to_end_ms: List[float] = []
    fps_values: List[float] = []
    scores: List[float] = []
    risks: Dict[str, int] = {"BENIGN": 0, "SUSPICIOUS": 0, "THREATENING": 0}

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()

        try:
            while time.time() - started < args.seconds:
                read_start = time.perf_counter()
                ok, frame = camera.read()
                if not ok:
                    read_failures += 1
                    writer.writerow({
                        "timestamp": time.time(),
                        "frame_id": frame_id,
                        "frame_read_ok": 0,
                        **{field: "" for field in CSV_FIELDS if field not in {"timestamp", "frame_id", "frame_read_ok"}},
                    })
                    continue

                result = system.process_frame(frame, frame_id=frame_id)
                elapsed_ms = (time.perf_counter() - read_start) * 1000.0
                frame_id += 1
                processed += 1
                end_to_end_ms.append(elapsed_ms)

                composite = result["composite"]
                vision = result["vision"]
                audio = result["audio"]
                performance = result["performance"]
                health = result["health"]
                risk = composite["risk_label"]
                risks[risk] = risks.get(risk, 0) + 1
                scores.append(float(composite["threat_score"]))
                fps_values.append(float(performance["fps"]))

                writer.writerow({
                    "timestamp": result["timestamp"],
                    "frame_id": result["frame_id"],
                    "yolo_ms": performance["yolo_ms"],
                    "haar_ms": performance["haar_ms"],
                    "ast_ms": performance["ast_ms"],
                    "vision_ms": performance["vision_latency_ms"],
                    "audio_age_ms": performance["audio_age_ms"],
                    "fused_score": composite["threat_score"],
                    "risk_level": risk,
                    "persons": vision["persons"],
                    "faces": vision["faces"],
                    "audio_class": audio.get("label", "Unknown"),
                    "audio_threat": audio.get("threat_score", 0.0),
                    "sensor_coverage": composite.get("sensor_coverage", 0.0),
                    "vision_status": health["vision_status"],
                    "audio_status": health["audio_status"],
                    "frame_read_ok": 1,
                })

                if not args.no_window:
                    display = result["annotated_frame"]
                    cv2.putText(
                        display,
                        f"BENCHMARK {time.time() - started:.0f}/{args.seconds:.0f}s  {risk} {composite['threat_score']:.3f}",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("SentinelVision Benchmark", display)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            camera.release()
            cv2.destroyAllWindows()
            system.stop()

    elapsed_seconds = max(time.time() - started, 1e-9)
    summary: Dict[str, Any] = {
        "requested_seconds": args.seconds,
        "actual_seconds": elapsed_seconds,
        "processed_frames": processed,
        "frame_read_failures": read_failures,
        "observed_fps": processed / elapsed_seconds,
        "e2e_latency_ms": {
            "mean": statistics.mean(end_to_end_ms) if end_to_end_ms else 0.0,
            "median": statistics.median(end_to_end_ms) if end_to_end_ms else 0.0,
            "p95": _percentile(end_to_end_ms, 95),
            "min": min(end_to_end_ms) if end_to_end_ms else 0.0,
            "max": max(end_to_end_ms) if end_to_end_ms else 0.0,
        },
        "reported_fps": {
            "min": min(fps_values) if fps_values else 0.0,
            "max": max(fps_values) if fps_values else 0.0,
        },
        "mean_fused_score": statistics.mean(scores) if scores else 0.0,
        "risk_counts": risks,
        "sensor_health": system.get_status(),
        "csv": str(csv_path),
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"CSV: {csv_path}")
    print(f"Summary: {json_path}")


if __name__ == "__main__":
    main()
