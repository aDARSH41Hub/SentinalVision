"""Live audio detection test using the AudioDetector class.

This script demonstrates real-time audio threat detection using the
AudioDetector class. It monitors microphone input continuously and
displays inference results, threat scores, and predictions.

The detector automatically runs in a background thread, making this
script suitable for integration with other real-time systems.

Usage:
    python audio_live_test.py
    
    Press Ctrl+C to stop.
"""

import logging
import time
from typing import Dict, Any

from audio import AudioDetector, AudioDetectorConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def format_result(result: Dict[str, Any]) -> None:
    """Print formatted detection result.
    
    Args:
        result: Detection result dictionary from AudioDetector
    """
    print("\n" + "-" * 60)
    
    # Top prediction
    print(
        f"Top class      : {result['label']} ({result['confidence']:.2%})"
    )
    
    # Threat score
    print(f"Threat score   : {result['threat_score']:.3f}")
    
    # Risk classification
    risk_color = {
        "BENIGN": "✓",
        "SUSPICIOUS": "⚠",
        "THREATENING": "✗",
    }
    print(f"Risk           : {result['risk_label']} {risk_color.get(result['risk_label'], '?')}")
    
    # Audio metrics
    print(f"Peak amplitude : {result['peak']:.4f}")
    print(f"RMS amplitude  : {result['rms']:.4f}")
    
    # Threat predictions (if any)
    if result["threat_predictions"]:
        print("\nDetected threats:")
        for label, confidence, weight, score in result["threat_predictions"]:
            print(
                f"  • {label:<35} {confidence:.3f} × {weight:.2f} = {score:.3f}"
            )


def main() -> None:
    """Run live audio detector.
    
    Raises:
        RuntimeError: If detector initialization fails
    """
    logger.info("Initializing AudioDetector...")
    
    try:
        config = AudioDetectorConfig()
        detector = AudioDetector(config=config)
    except Exception as e:
        logger.error("Failed to initialize detector: %s", e, exc_info=True)
        raise RuntimeError(f"Could not initialize detector: {e}") from e
    
    logger.info("AudioDetector initialized successfully")
    logger.info("Detector running. Press Ctrl+C to stop.")
    
    print("\n" + "=" * 60)
    print("SentinelVision - Live Audio Detection")
    print("=" * 60)
    print("Listening for threats...")
    print("Try clapping, shouting, or making loud impacts.")
    print("Press Ctrl+C to stop.\n")
    
    last_risk = "BENIGN"
    
    try:
        while True:
            result = detector.get_result()
            
            # Log important state changes
            if result["risk_label"] != last_risk:
                logger.warning(
                    "Risk level changed: %s → %s (score=%.3f)",
                    last_risk,
                    result["risk_label"],
                    result["threat_score"],
                )
                last_risk = result["risk_label"]
            
            # Display results
            format_result(result)
            
            # Update interval (1 second to match detector interval)
            time.sleep(1.0)
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        print("\n\nStopping detector...")
    
    finally:
        logger.info("Cleaning up...")
        detector.stop()
        logger.info("Detector stopped")
        print("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("Test failed: %s", e, exc_info=True)
        exit(1)