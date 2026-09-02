#!/usr/bin/env python3
"""Quick Start Guide for SentinelVisionUnified

This file demonstrates the simplest way to get started with the unified
vision+audio surveillance system.

Run: python quickstart_unified.py
"""

import logging
import cv2
from sentinel_vision import SentinelVisionUnified, SentinelUnifiedConfig
from audio import AudioDetectorConfig
from vision import VisionDetectorConfig


def main():
    """Minimal example to verify unified system works."""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 70)
    logger.info("SentinelVisionUnified - Quick Start Demo")
    logger.info("=" * 70)
    
    # Step 1: Create configuration
    logger.info("\n[1/5] Creating configuration...")
    config = SentinelUnifiedConfig(
        audio_weight=0.6,      # 60% weight to audio threats
        vision_weight=0.4,     # 40% weight to vision threats
        graceful_degradation=True,  # Continue if one sensor fails
    )
    logger.info("✓ Config created with audio_weight=0.6, vision_weight=0.4")
    
    # Step 2: Initialize detector
    logger.info("\n[2/5] Initializing unified detector...")
    try:
        detector = SentinelVisionUnified(
            vision_config=VisionDetectorConfig(),
            audio_config=AudioDetectorConfig(),
            config=config,
        )
        logger.info("✓ Detector initialized successfully")
    except Exception as e:
        logger.error("✗ Failed to initialize detector: %s", e)
        logger.error("  (This is expected if audio hardware unavailable)")
        return
    
    # Step 3: Check system status
    logger.info("\n[3/5] Checking system status...")
    status = detector.get_status()
    logger.info("  Vision:  %s", "✓ Available" if status["vision_available"] else "✗ Unavailable")
    logger.info("  Audio:   %s", "✓ Available" if status["audio_available"] else "✗ Unavailable")
    logger.info("  Weights: Audio=%.0f%%, Vision=%.0f%%",
                status["audio_weight"] * 100,
                status["vision_weight"] * 100)
    
    if not status["vision_available"] and not status["audio_available"]:
        logger.error("✗ Both sensors unavailable, cannot continue")
        return
    
    # Step 4: Open camera
    logger.info("\n[4/5] Opening camera...")
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        logger.error("✗ Could not open camera. Make sure:")
        logger.error("  • Camera is connected")
        logger.error("  • No other application is using it")
        detector.stop()
        return
    
    # Optimize camera settings
    camera.set(cv2.CAP_PROP_FPS, 30)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    logger.info("✓ Camera opened at 30 FPS, 1280x720")
    
    # Step 5: Process frames
    logger.info("\n[5/5] Processing frames (20 frames)...")
    logger.info("  Press 'Q' to quit early\n")
    
    frame_id = 0
    frames_to_process = 20
    
    try:
        while frame_id < frames_to_process:
            ret, frame = camera.read()
            
            if not ret:
                logger.warning("Failed to read frame")
                break
            
            # Process frame through unified detector
            result = detector.process_frame(frame, frame_id)
            
            if result is None:
                logger.warning("Frame %d: Processing failed", frame_id)
                frame_id += 1
                continue
            
            # Display results
            threat_score = result['composite']['threat_score']
            risk_label = result['composite']['risk_label']
            reasoning = result['composite']['reasoning']
            
            fps = result['performance']['frame_fps']
            vision_lat = result['performance']['vision_latency_ms']
            audio_lat = result['performance']['audio_latency_ms']
            
            logger.info(
                "Frame %3d | Threat: %s (%.0f%%) | FPS: %.1f | "
                "Vision: %.1fms | Audio: %.1fms | %s",
                frame_id,
                risk_label,
                threat_score * 100,
                fps,
                vision_lat,
                audio_lat,
                reasoning[:50] + "..." if len(reasoning) > 50 else reasoning,
            )
            
            # Show frame with threat visualization
            display_frame = result['vision'].get('annotated_frame')
            if display_frame is not None:
                # Draw threat indicator in corner
                color = (0, 0, 255) if risk_label == "THREATENING" else \
                        (0, 165, 255) if risk_label == "SUSPICIOUS" else (0, 255, 0)
                
                cv2.putText(
                    display_frame,
                    f"{risk_label} ({threat_score:.0%})",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    color,
                    2,
                )
                
                cv2.imshow("SentinelVision Unified", display_frame)
            
            # Check for quit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                logger.info("Quit requested")
                break
            
            frame_id += 1
        
        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("Processing Complete!")
        logger.info("=" * 70)
        
        alerts = detector.get_recent_alerts(count=5)
        status = detector.get_status()
        
        logger.info(f"Frames processed: {frame_id}")
        logger.info(f"Average FPS: {status['fps']:.1f}")
        logger.info(f"Total alerts: {len(alerts)}")
        
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
    
    finally:
        # Cleanup
        logger.info("\n" + "=" * 70)
        logger.info("Shutting down...")
        
        camera.release()
        cv2.destroyAllWindows()
        detector.stop()
        
        logger.info("✓ Resources released")
        logger.info("=" * 70)


if __name__ == "__main__":
    main()
