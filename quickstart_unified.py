#!/usr/bin/env python3
"""Quick Start Guide for SentinelVisionUnified.

This file demonstrates the simplest way to get started with the unified
vision + audio surveillance system.

Run:
    python quickstart_unified.py

Controls:
    Q - Quit the live demo
"""

import logging

import cv2

from sentinel_vision import SentinelVisionUnified, SentinelUnifiedConfig
from audio import AudioDetectorConfig
from vision import VisionDetectorConfig


def main() -> None:
    """Run the live SentinelVision unified demo."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("SentinelVisionUnified - Live Demo")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Create configuration
    # ------------------------------------------------------------------
    logger.info("\n[1/5] Creating configuration...")

    # SentinelUnifiedConfig owns orchestration/runtime behavior.
    # FusionConfig remains the single authoritative source for fusion
    # weights and thresholds.
    config = SentinelUnifiedConfig(
        graceful_degradation=True,
    )

    # Configure fusion weights through the authoritative FusionConfig.
    config.fusion = config.fusion.__class__(
        **{
            **config.fusion.__dict__,
            "audio_weight": 0.60,
            "vision_weight": 0.40,
        }
    )

    logger.info(
        "✓ Config created with audio_weight=%.1f%%, vision_weight=%.1f%%",
        config.fusion.normalized_weights[0] * 100,
        config.fusion.normalized_weights[1] * 100,
    )

    # ------------------------------------------------------------------
    # Step 2: Initialize unified detector
    # ------------------------------------------------------------------
    logger.info("\n[2/5] Initializing unified detector...")

    detector = None
    camera = None

    try:
        detector = SentinelVisionUnified(
            vision_config=VisionDetectorConfig(),
            audio_config=AudioDetectorConfig(),
            config=config,
        )

        logger.info("✓ Detector initialized successfully")

        # ------------------------------------------------------------------
        # Step 3: Check system status
        # ------------------------------------------------------------------
        logger.info("\n[3/5] Checking system status...")

        status = detector.get_status()

        logger.info(
            "  Vision:  %s",
            "✓ Available"
            if status["vision_available"]
            else "✗ Unavailable",
        )

        logger.info(
            "  Audio:   %s",
            "✓ Available"
            if status["audio_available"]
            else "✗ Unavailable",
        )

        logger.info(
            "  Weights: Audio=%.0f%%, Vision=%.0f%%",
            status["audio_weight"] * 100,
            status["vision_weight"] * 100,
        )

        if (
            not status["vision_available"]
            and not status["audio_available"]
        ):
            logger.error(
                "✗ Both sensors unavailable, cannot continue"
            )
            return

        # ------------------------------------------------------------------
        # Step 4: Open camera
        # ------------------------------------------------------------------
        logger.info("\n[4/5] Opening camera...")

        camera = cv2.VideoCapture(0)

        if not camera.isOpened():
            logger.error("✗ Could not open camera. Make sure:")
            logger.error("  • Camera is connected")
            logger.error("  • No other application is using it")
            return

        # Camera configuration
        camera.set(cv2.CAP_PROP_FPS, 30)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        logger.info(
            "✓ Camera opened at requested 30 FPS, 1280x720"
        )

        # ------------------------------------------------------------------
        # Step 5: Live frame processing
        # ------------------------------------------------------------------
        logger.info("\n[5/5] Starting live processing...")
        logger.info("  Press 'Q' in the camera window to quit\n")

        frame_id = 0

        # Run continuously until the user presses Q.
        while True:

            ret, frame = camera.read()

            if not ret:
                logger.warning("Failed to read frame")
                break

            # Process the frame through the unified pipeline.
            result = detector.process_frame(
                frame,
                frame_id,
            )

            if result is None:
                logger.warning(
                    "Frame %d: Processing failed",
                    frame_id,
                )

                frame_id += 1
                continue

            # --------------------------------------------------------------
            # Extract fusion results
            # --------------------------------------------------------------
            composite = result["composite"]
            performance = result["performance"]

            threat_score = composite["threat_score"]
            risk_label = composite["risk_label"]
            reasoning = composite["reasoning"]

            fps = performance["frame_fps"]
            vision_latency = performance["vision_latency_ms"]
            audio_age = performance["audio_age_ms"]

            # Log the frame result.
            logger.info(
                "Frame %4d | Threat: %s (%.0f%%) | FPS: %.1f | "
                "Vision: %.1fms | Audio age: %.1fms | %s",
                frame_id,
                risk_label,
                threat_score * 100,
                fps,
                vision_latency,
                audio_age,
                reasoning[:70] + "..."
                if len(reasoning) > 70
                else reasoning,
            )

            # --------------------------------------------------------------
            # Display annotated frame
            # --------------------------------------------------------------
            display_frame = result["vision"].get(
                "annotated_frame"
            )

            # Fallback to the original frame if the vision module does
            # not return an annotated frame.
            if display_frame is None:
                display_frame = frame.copy()

            # Risk indicator color.
            if risk_label == "THREATENING":
                indicator_color = (0, 0, 255)      # Red

            elif risk_label == "SUSPICIOUS":
                indicator_color = (0, 165, 255)    # Orange

            else:
                indicator_color = (0, 255, 0)      # Green

            # Display risk label.
            cv2.putText(
                display_frame,
                f"{risk_label} ({threat_score:.0%})",
                (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                indicator_color,
                2,
            )

            # Display FPS.
            cv2.putText(
                display_frame,
                f"FPS: {fps:.1f}",
                (30, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            # Display sensor availability.
            sensor_text = (
                f"Vision: {'OK' if status['vision_available'] else 'OFF'} | "
                f"Audio: {'OK' if status['audio_available'] else 'OFF'}"
            )

            cv2.putText(
                display_frame,
                sensor_text,
                (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            # Display controls.
            cv2.putText(
                display_frame,
                "Press Q to quit",
                (30, display_frame.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
            )

            cv2.imshow(
                "SentinelVision Unified",
                display_frame,
            )

            # --------------------------------------------------------------
            # Keyboard handling
            # --------------------------------------------------------------
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q")):
                logger.info("Quit requested")
                break

            frame_id += 1

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------
        logger.info("\n" + "=" * 70)
        logger.info("Processing Complete!")
        logger.info("=" * 70)

        alerts = detector.get_recent_alerts(count=5)
        status = detector.get_status()

        logger.info(
            "Frames processed: %d",
            frame_id + 1,
        )

        logger.info(
            "Average FPS: %.1f",
            status["fps"],
        )

        logger.info(
            "Total alerts: %d",
            len(alerts),
        )

        if alerts:

            logger.info("\nRecent Alerts:")

            for i, alert in enumerate(alerts, 1):

                logger.info(
                    "  [%d] Frame %d: %s (%.2f) - %s",
                    i,
                    alert.frame_id,
                    alert.risk_label,
                    alert.threat_score,
                    alert.reasoning[:60],
                )

    except KeyboardInterrupt:

        logger.info("Interrupted by user")

    except Exception as exc:

        logger.error(
            "✗ Quick-start demo failed: %s",
            exc,
            exc_info=True,
        )

    finally:

        logger.info("\n" + "=" * 70)
        logger.info("Shutting down...")

        # Release camera resources.
        if camera is not None:
            camera.release()

        # Close OpenCV windows.
        cv2.destroyAllWindows()

        # Stop detector and background resources.
        if detector is not None:
            detector.stop()

        logger.info("✓ Resources released")
        logger.info("=" * 70)


if __name__ == "__main__":
    main()