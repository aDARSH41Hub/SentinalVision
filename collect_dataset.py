"""Controlled dataset collector for SentinelVision Phase-1.

Captures on keypress:
  - frame image (jpg)
  - ~2-second audio clip (wav)
  - metadata JSON (includes latest unified inference outputs)

Keyboard controls (focus preview window):
  B = benign
  S = suspicious
  T = threatening
  Q = quit

Output layout:
  dataset_root/
    images/
    audio/
    metadata/
    manifest.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf

from sentinel_vision import SentinelVisionUnified


# -----------------------------
# Config
# -----------------------------

@dataclass(frozen=True)
class CollectorConfig:
    """Dataset collector configuration."""

    camera_index: int = 0
    output_dir: str = "dataset_collected"
    sample_rate: int = 16000
    channels: int = 1
    audio_clip_sec: float = 2.0
    camera_retry_delay_sec: float = 0.05
    show_preview: bool = True


# -----------------------------
# Helpers
# -----------------------------

def _ensure_dirs(root: Path) -> Dict[str, Path]:
    dirs = {
        "root": root,
        "images": root / "images",
        "audio": root / "audio",
        "metadata": root / "metadata",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def _timestamp_str() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _safe_get(mapping: Dict[str, Any], key: str, default: Any = None) -> Any:
    if not isinstance(mapping, dict):
        return default
    return mapping.get(key, default)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _append_manifest(manifest_path: Path, record: Dict[str, Any]) -> None:
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


class RollingAudioRecorder:
    """Simple rolling buffer recorder using sounddevice (mono)."""

    def __init__(self, sample_rate: int, channels: int, clip_sec: float) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.clip_sec = clip_sec
        self.blocksize = int(sample_rate * 0.1)  # 100ms blocks
        self._buffer = np.zeros((0,), dtype=np.float32)
        self._max_samples = int(sample_rate * clip_sec)
        self._stream: Optional[sd.InputStream] = None

    def _callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        if status:
            # Keep it quiet; collector is a utility
            pass
        mono = indata[:, 0].astype(np.float32) if indata.ndim == 2 else indata.astype(np.float32)
        self._buffer = np.concatenate((self._buffer, mono))
        if self._buffer.size > self._max_samples:
            self._buffer = self._buffer[-self._max_samples:]

    def start(self) -> None:
        if self._stream is not None:
            return
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            blocksize=self.blocksize,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._stream = None

    def get_clip(self) -> np.ndarray:
        """Return a copy of the current rolling buffer."""
        return self._buffer.copy()


# -----------------------------
# Collector
# -----------------------------

def run_collector(config: CollectorConfig) -> None:
    root = Path(config.output_dir)
    dirs = _ensure_dirs(root)
    manifest_path = dirs["root"] / "manifest.jsonl"

    system = SentinelVisionUnified()
    recorder = RollingAudioRecorder(
        sample_rate=config.sample_rate,
        channels=config.channels,
        clip_sec=config.audio_clip_sec,
    )
    recorder.start()

    cap = cv2.VideoCapture(config.camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        recorder.stop()
        system.stop()
        raise RuntimeError(f"Unable to open camera index {config.camera_index}")

    frame_id = 0
    sample_index = 0

    print("\nDataset collector started.")
    print("Controls (focus preview window): B=benign, S=suspicious, T=threatening, Q=quit")
    print("Tip: keep scenarios controlled/acted; do not use real weapons.\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(config.camera_retry_delay_sec)
                continue

            result = system.process_frame(frame, frame_id=frame_id)
            annotated = None
            if result is not None:
                vision = _safe_get(result, "vision", {}) or {}
                annotated = _safe_get(vision, "annotated_frame", None)

            display = annotated if annotated is not None else frame
            if config.show_preview:
                cv2.imshow("SentinelVision Dataset Collector", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            label: Optional[str] = None
            if key == ord("b"):
                label = "BENIGN"
            elif key == ord("s"):
                label = "SUSPICIOUS"
            elif key == ord("t"):
                label = "THREATENING"

            if label is None:
                frame_id += 1
                continue

            note = input("Note (optional, press Enter to skip): ").strip()

            ts = _timestamp_str()
            img_path = dirs["images"] / f"{sample_index:06d}_{label}_{ts}.jpg"
            wav_path = dirs["audio"] / f"{sample_index:06d}_{label}_{ts}.wav"
            meta_path = dirs["metadata"] / f"{sample_index:06d}_{label}_{ts}.json"

            write_ok = cv2.imwrite(str(img_path), display)
            if not write_ok:
                print(f"[WARN] Failed to write image: {img_path}")
                frame_id += 1
                continue

            audio_clip = recorder.get_clip()
            audio_saved = False
            audio_error = None
            if audio_clip.size == 0:
                audio_error = "Audio buffer empty."
            else:
                try:
                    sf.write(str(wav_path), audio_clip, config.sample_rate)
                    audio_saved = True
                except Exception as e:
                    audio_error = f"{type(e).__name__}: {e}"

            composite = (_safe_get(result, "composite", {}) or {}) if result else {}
            vision = (_safe_get(result, "vision", {}) or {}) if result else {}
            audio = (_safe_get(result, "audio", {}) or {}) if result else {}
            performance = (_safe_get(result, "performance", {}) or {}) if result else {}
            health = (_safe_get(result, "health", {}) or {}) if result else {}

            metadata: Dict[str, Any] = {
                "timestamp": ts,
                "sample_index": sample_index,
                "frame_id": frame_id,
                "label": label,
                "note": note if note else None,
                "consent": "controlled/acted scenario; confirm consent in protocol",
                "image_path": str(img_path),
                "audio_path": str(wav_path) if audio_saved else None,
                "audio_saved": audio_saved,
                "audio_error": audio_error,
                "composite": composite,
                "vision": {
                    "persons": _safe_get(vision, "persons"),
                    "faces": _safe_get(vision, "faces"),
                    "objects": _safe_get(vision, "objects"),
                    "fps": _safe_get(vision, "fps"),
                },
                "audio": {
                    "label": _safe_get(audio, "label"),
                    "confidence": _safe_get(audio, "confidence"),
                    "threat_score": _safe_get(audio, "threat_score"),
                    "risk_label": _safe_get(audio, "risk_label"),
                },
                "performance": performance,
                "health": health,
            }

            _write_json(meta_path, metadata)

            manifest_record = {
                "sample_index": sample_index,
                "timestamp": ts,
                "label": label,
                "note": metadata["note"],
                "image_path": metadata["image_path"],
                "audio_path": metadata["audio_path"],
                "audio_saved": audio_saved,
                "metadata_path": str(meta_path),
            }
            _append_manifest(manifest_path, manifest_record)

            print(f"[SAVED] {sample_index:06d} label={label} audio_saved={audio_saved} note={note if note else '-'}")
            sample_index += 1
            frame_id += 1

    finally:
        cap.release()
        recorder.stop()
        if config.show_preview:
            cv2.destroyAllWindows()
        system.stop()

    print(f"\nDone. Samples saved: {sample_index}")
    print(f"Output root: {root.resolve()}")


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SentinelVision controlled dataset collector")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--outdir", type=str, default="dataset_collected", help="Output directory")
    parser.add_argument("--no-preview", action="store_true", help="Disable preview window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CollectorConfig(
        camera_index=args.camera,
        output_dir=args.outdir,
        show_preview=not args.no_preview,
    )
    run_collector(config)


if __name__ == "__main__":
    main()
