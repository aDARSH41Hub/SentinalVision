"""Structured benchmark harness for SentinelVision Phase-1.

Outputs:
  - benchmark_results_<timestamp>.csv   (per-frame observations)
  - benchmark_summary_<timestamp>.json  (aggregate metrics + run metadata)

Design goals:
  - Low overhead timing (perf_counter, minimal work inside loop)
  - Robust to degraded sensors and annotated_frame=None
  - Bounded camera-read retry (no infinite spin)
  - Clean shutdown (camera release + system.stop())
  - End-to-end latency measured: frame read -> DetectionResult ready
  - No scientific claims beyond what is measured
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

from sentinel_vision import SentinelVisionUnified


# -----------------------------
# Config
# -----------------------------

@dataclass(frozen=True)
class BenchmarkConfig:
    """Benchmark runtime configuration."""

    duration_sec: float = 300.0          # 5-minute default structured benchmark
    camera_index: int = 0
    warmup_sec: float = 2.0              # allow pipelines to stabilize
    camera_retry_delay_sec: float = 0.05 # avoid hot spin
    show_preview: bool = False           # keep False for accurate benchmarking
    output_dir: str = "benchmark_outputs"


# -----------------------------
# Helpers
# -----------------------------

def _now_ms() -> float:
    """Return current time in milliseconds."""
    return time.perf_counter() * 1000.0


def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _percentile(sorted_values: List[float], p: float) -> Optional[float]:
    """Compute percentile from a sorted list; returns None if empty."""
    if not sorted_values:
        return None
    if p <= 0:
        return sorted_values[0]
    if p >= 100:
        return sorted_values[-1]

    k = (len(sorted_values) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    return d0 + d1


def _safe_get(mapping: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Safe dict getter with default."""
    if not isinstance(mapping, dict):
        return default
    return mapping.get(key, default)


# -----------------------------
# Benchmark core
# -----------------------------

def run_benchmark(config: BenchmarkConfig) -> Dict[str, Any]:
    """Run benchmark and return summary dict."""
    _ensure_dir(config.output_dir)

    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    csv_path = Path(config.output_dir) / f"benchmark_results_{timestamp_str}.csv"
    json_path = Path(config.output_dir) / f"benchmark_summary_{timestamp_str}.json"

    system = SentinelVisionUnified()
    cap = cv2.VideoCapture(config.camera_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {config.camera_index}")

    # CSV header aligned to current DetectionResult schema + E2E latency
    fieldnames = [
        "timestamp_epoch",
        "frame_id",
        "risk_label",
        "threat_score",
        "sensor_coverage",
        "audio_available",
        "vision_available",
        "persons",
        "faces",
        "audio_class",
        "audio_threat_score",
        "vision_latency_ms",
        "audio_age_ms",
        "frame_fps",
        "vision_status",
        "audio_status",
        "last_vision_error",
        "last_audio_error",
        "camera_read_ok",
        "loop_latency_ms",
        "e2e_latency_ms",
    ]

    rows: List[Dict[str, Any]] = []

    frame_id = 0
    camera_failures = 0
    processed_frames = 0

    # Warmup (not recorded)
    warmup_end = time.perf_counter() + config.warmup_sec
    while time.perf_counter() < warmup_end:
        ok, frame = cap.read()
        if not ok:
            time.sleep(config.camera_retry_delay_sec)
            continue
        _ = system.process_frame(frame, frame_id=frame_id)
        frame_id += 1

    start_time = time.perf_counter()
    end_time = start_time + config.duration_sec

    try:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            while time.perf_counter() < end_time:
                loop_start = _now_ms()

                ok, frame = cap.read()
                if not ok:
                    camera_failures += 1
                    time.sleep(config.camera_retry_delay_sec)
                    continue

                # E2E start: frame captured successfully
                e2e_start = _now_ms()

                result = system.process_frame(frame, frame_id=frame_id)

                # E2E end: DetectionResult ready
                e2e_end = _now_ms()
                e2e_latency_ms = e2e_end - e2e_start

                loop_end = _now_ms()
                loop_latency_ms = loop_end - loop_start

                if result is None:
                    row = {
                        "timestamp_epoch": time.time(),
                        "frame_id": frame_id,
                        "risk_label": "ERROR",
                        "threat_score": None,
                        "sensor_coverage": None,
                        "audio_available": None,
                        "vision_available": None,
                        "persons": None,
                        "faces": None,
                        "audio_class": None,
                        "audio_threat_score": None,
                        "vision_latency_ms": None,
                        "audio_age_ms": None,
                        "frame_fps": None,
                        "vision_status": "ERROR",
                        "audio_status": "UNKNOWN",
                        "last_vision_error": "process_frame returned None",
                        "last_audio_error": None,
                        "camera_read_ok": True,
                        "loop_latency_ms": loop_latency_ms,
                        "e2e_latency_ms": e2e_latency_ms,
                    }
                    writer.writerow(row)
                    rows.append(row)
                    frame_id += 1
                    continue

                composite = _safe_get(result, "composite", {}) or {}
                vision = _safe_get(result, "vision", {}) or {}
                audio = _safe_get(result, "audio", {}) or {}
                performance = _safe_get(result, "performance", {}) or {}
                health = _safe_get(result, "health", {}) or {}

                row = {
                    "timestamp_epoch": time.time(),
                    "frame_id": frame_id,
                    "risk_label": _safe_get(composite, "risk_label"),
                    "threat_score": _safe_get(composite, "threat_score"),
                    "sensor_coverage": _safe_get(composite, "sensor_coverage"),
                    "audio_available": _safe_get(composite, "audio_available"),
                    "vision_available": _safe_get(composite, "vision_available"),
                    "persons": _safe_get(vision, "persons"),
                    "faces": _safe_get(vision, "faces"),
                    "audio_class": _safe_get(audio, "label"),
                    "audio_threat_score": _safe_get(audio, "threat_score"),
                    "vision_latency_ms": _safe_get(performance, "vision_latency_ms"),
                    "audio_age_ms": _safe_get(performance, "audio_age_ms"),
                    "frame_fps": _safe_get(performance, "frame_fps"),
                    "vision_status": _safe_get(health, "vision_status"),
                    "audio_status": _safe_get(health, "audio_status"),
                    "last_vision_error": _safe_get(health, "last_vision_error"),
                    "last_audio_error": _safe_get(health, "last_audio_error"),
                    "camera_read_ok": True,
                    "loop_latency_ms": loop_latency_ms,
                    "e2e_latency_ms": e2e_latency_ms,
                }

                writer.writerow(row)
                rows.append(row)

                if config.show_preview:
                    annotated = _safe_get(vision, "annotated_frame", None)
                    if annotated is not None:
                        cv2.imshow("SentinelVision Benchmark", annotated)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                frame_id += 1
                processed_frames += 1

    finally:
        cap.release()
        if config.show_preview:
            cv2.destroyAllWindows()
        system.stop()

    # -----------------------------
    # Summary computation
    # -----------------------------
    def collect_float(key: str) -> List[float]:
        vals: List[float] = []
        for r in rows:
            v = r.get(key)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        return vals

    vision_lat = sorted(collect_float("vision_latency_ms"))
    audio_age = sorted(collect_float("audio_age_ms"))
    loop_lat = sorted(collect_float("loop_latency_ms"))
    e2e_lat = sorted(collect_float("e2e_latency_ms"))
    fps_vals = sorted(collect_float("frame_fps"))

    def stats(values: List[float]) -> Dict[str, Optional[float]]:
        if not values:
            return {"mean": None, "median": None, "p95": None, "min": None, "max": None}
        return {
            "mean": sum(values) / len(values),
            "median": _percentile(values, 50),
            "p95": _percentile(values, 95),
            "min": values[0],
            "max": values[-1],
        }

    risk_counts: Dict[str, int] = {}
    for r in rows:
        label = r.get("risk_label")
        if label is None:
            continue
        risk_counts[label] = risk_counts.get(label, 0) + 1

    actual_duration = time.perf_counter() - start_time

    summary: Dict[str, Any] = {
        "run_metadata": {
            "start_time_epoch": start_time,
            "duration_requested_sec": config.duration_sec,
            "duration_actual_sec": actual_duration,
            "warmup_sec": config.warmup_sec,
            "camera_index": config.camera_index,
            "processed_frames": processed_frames,
            "camera_failures": camera_failures,
            "csv_path": str(csv_path),
        },
        "latency_ms": {
            "vision": stats(vision_lat),
            "audio_age": stats(audio_age),
            "loop": stats(loop_lat),
            "e2e": stats(e2e_lat),
        },
        "fps": {
            "values": stats(fps_vals),
            "effective_avg_fps": (processed_frames / actual_duration) if actual_duration > 0 else None,
        },
        "risk_distribution": risk_counts,
        "notes": [
            "e2e_latency_ms is measured from successful camera read to DetectionResult ready.",
            "Threat score is heuristic and uncalibrated.",
            "sensor_coverage=0 means no usable evidence; interpret as unknown, not safe.",
            "This benchmark measures pipeline performance, not real-world threat accuracy.",
        ],
    }

    with json_path.open("w", encoding="utf-8") as jf:
        json.dump(summary, jf, indent=2)

    return summary


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SentinelVision structured benchmark")
    parser.add_argument("--duration", type=float, default=300.0, help="Benchmark duration in seconds")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--warmup", type=float, default=2.0, help="Warmup seconds (not recorded)")
    parser.add_argument("--show", action="store_true", help="Show preview (not recommended while benchmarking)")
    parser.add_argument("--outdir", type=str, default="benchmark_outputs", help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BenchmarkConfig(
        duration_sec=args.duration,
        camera_index=args.camera,
        warmup_sec=args.warmup,
        show_preview=args.show,
        output_dir=args.outdir,
    )
    summary = run_benchmark(config)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()