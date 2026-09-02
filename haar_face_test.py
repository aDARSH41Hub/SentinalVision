"""Haar cascade face detector -- standalone test harness.

Useful on its own for verifying webcam + Haar cascade work before wiring into
the full SentinelVision pipeline. Also doubles as a quick tool for collecting
labeled frames for the dataset (--save-dir) and for testing the privacy
face-blur feature in isolation (--blur).

Usage:
    python haar_face_test.py
    python haar_face_test.py --camera 1 --resize-width 480
    python haar_face_test.py --save-dir captures
    python haar_face_test.py --blur
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Haar cascade face detection test harness")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--resize-width", type=int, default=640,
                         help="Downscale frames to this width before detection, for speed (default: 640)")
    parser.add_argument("--scale-factor", type=float, default=1.05)
    parser.add_argument("--min-neighbors", type=int, default=3)
    parser.add_argument("--min-size", type=int, default=40, help="Minimum face size in pixels")
    parser.add_argument("--blur", action="store_true",
                         help="Blur detected faces instead of boxing them (privacy mode)")
    parser.add_argument("--save-dir", type=Path, default=None,
                         help="If set, press 's' to save the current frame here (for dataset collection)")
    return parser.parse_args()


def load_face_cascade() -> cv2.CascadeClassifier:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError(f"Could not load Haar cascade from {cascade_path}")
    return cascade


def main() -> None:
    args = parse_args()

    face_cascade = load_face_cascade()

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open webcam at index {args.camera}")

    # Optimize camera for low-latency, real-time capture
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize frame buffering
    camera.set(cv2.CAP_PROP_FPS, 30)        # Request 30 FPS
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    print("Haar face detection started (optimized). Press Q to quit" +
          (", S to save a frame" if args.save_dir else "") + ".")

    prev_time = time.time()
    fps = 0.0
    frame_count = 0
    frame_read_failures = 0
    max_failures = 10
    gray_buffer = None  # Reuse grayscale buffer across frames (memory efficiency)

    # try/finally guarantees the camera is released and windows closed even if
    # detection throws partway through -- without this, a crash mid-loop leaves
    # the webcam LED on and the device locked until the process is killed.
    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                frame_read_failures += 1
                if frame_read_failures > max_failures:
                    print(f"ERROR: Could not read webcam frame {max_failures} times. Exiting.")
                    break
                continue
            
            frame_read_failures = 0  # Reset counter on successful read

            # Haar cascades get noticeably slower as resolution grows. Detecting
            # on a downscaled copy and mapping boxes back to the full-res frame
            # keeps the display sharp while keeping detection fast -- this is
            # the single biggest lever for real-time performance here.
            orig_h, orig_w = frame.shape[:2]
            scale = args.resize_width / orig_w if orig_w > args.resize_width else 1.0
            small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR) if scale != 1.0 else frame

            # Reuse grayscale buffer for memory efficiency (no allocation per frame)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY, dst=gray_buffer)
            if gray_buffer is None:
                gray_buffer = gray  # Save reference for next iteration
            
            gray = cv2.equalizeHist(gray)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=args.scale_factor,
                minNeighbors=args.min_neighbors,
                minSize=(args.min_size, args.min_size),
            )

            for (x, y, w, h) in faces:
                # Map detection coords from the downscaled frame back to full-res
                x, y, w, h = (int(v / scale) for v in (x, y, w, h))

                if args.blur:
                    roi = frame[y:y + h, x:x + w]
                    if roi.size:
                        frame[y:y + h, x:x + w] = cv2.GaussianBlur(roi, (35, 35), 0)
                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, "Face", (x, max(y - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Rolling 1-second FPS, not just a guess -- you'll want this exact
            # number later for the evaluation section of the paper.
            frame_count += 1
            now = time.time()
            if now - prev_time >= 1.0:
                fps = frame_count / (now - prev_time)
                frame_count = 0
                prev_time = now

            cv2.putText(frame, f"Faces: {len(faces)}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("SentinelVision - Haar Face Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if args.save_dir and key == ord("s"):
                filename = args.save_dir / f"frame_{int(time.time() * 1000)}.jpg"
                cv2.imwrite(str(filename), frame)
                print(f"Saved {filename}")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()